#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "cmd.h"
#include "ipu-util.h"

#define N 1024
typedef uint32_t dataType;

// Kernel function to add scalar to each element of the vector
__global__ void addScalar(uint32_t *vec, uint32_t scalar, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        vec[idx] += scalar;
    }
}

int main(int argc, char **argv) {
    //////////////////////////////HIP///////////////////////////////////////

    // HSA buffer creation and hip registration
    // On intergrated systems hipmalloc is zerocopy between cpu-gpu:
    // https://rocblas.readthedocs.io/en/rocm-6.0.0/conceptual/gpu-memory.html
    hipError_t err;

    dataType *hip_managed_ptr_input = nullptr;
    err = hipMalloc(&hip_managed_ptr_input, N * sizeof(dataType));
    if (err != hipSuccess) {
        std::cout << "hipMalloc failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    dataType *hip_managed_ptr_output = nullptr;
    err = hipMalloc(&hip_managed_ptr_output, N * sizeof(dataType));
    if (err != hipSuccess) {
        std::cout << "hipMalloc failed: " << hipGetErrorString(err) << std::endl;
        err = hipFree(hip_managed_ptr_input);
        return -1;
    }

    // init
    for (int i = 0; i < N; i++) {
        hip_managed_ptr_input[i] = 42;
    }

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addScalar<<<blocksPerGrid, threadsPerBlock>>>(hip_managed_ptr_input, 3, N);

    // Check for kernel launch errors
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        err = hipFree(hip_managed_ptr_input);
        err = hipFree(hip_managed_ptr_output);
        return -1;
    }

    // Synchronize to wait for the kernel to finish
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "hipDeviceSynchronize failed: " << hipGetErrorString(err) << std::endl;
        err = hipFree(hip_managed_ptr_input);
        err = hipFree(hip_managed_ptr_output);
        return -1;
    }

    // Export input device memory as DMA-BUF
    hsa_status_t status;

    int dmabuf_fd_input;
    uint64_t offset_input;

    status =
        hsa_amd_portable_export_dmabuf(hip_managed_ptr_input, N * sizeof(dataType), &dmabuf_fd_input, &offset_input);
    if (status == HSA_STATUS_SUCCESS) {
        std::cout << "DMA-BUF export successful. FD: " << dmabuf_fd_input << ", Offset: " << offset_input << std::endl;
    } else {
        std::cerr << "Failed to export DMA-BUF. Status: " << status << std::endl;
        return -1;
    }

    // Verify FD is valid
    if (dmabuf_fd_input < 0) {
        std::cout << "Invalid DMA-BUF FD: " << dmabuf_fd_input << std::endl;
        return -1;
    }

    // Export output device memory as DMA-BUF
    int dmabuf_fd_output;
    uint64_t offset_output;
    status =
        hsa_amd_portable_export_dmabuf(hip_managed_ptr_output, N * sizeof(dataType), &dmabuf_fd_output, &offset_output);
    if (status == HSA_STATUS_SUCCESS) {
        std::cout << "DMA-BUF export successful. FD: " << dmabuf_fd_output << ", Offset: " << offset_output
                  << std::endl;
    } else {
        std::cerr << "Failed to export DMA-BUF. Status: " << status << std::endl;
        return -1;
    }

    // Verify FD is valid
    if (dmabuf_fd_output < 0) {
        std::cout << "Invalid DMA-BUF FD: " << dmabuf_fd_output << std::endl;
        return -1;
    }

    /////////////////////////////////////////////////////////XDNA//////////////////////////////////////////////

    int drv_fd;
    int ret;
    const char drv_path[] = "/dev/accel/accel0";
    const char dpu_inst_path[] = "./add_one_insts.txt";
    const char pdi_path[] = "./add_one.pdi"; // Add one kernel
    void *queue_buf;
    __u32 queue_desc_handle;
    __u32 completion_signal_handle;
    __u32 queue_buf_handle;
    __u32 heap_handle;
    __u32 queue_handle;
    __u32 major, minor;

    if (argc != 1) {
        printf("Usage: ./zero_copy_buffer.elf\n");
        return -1;
    }

    // open the driver
    drv_fd = open(drv_path, O_RDWR);

    if (drv_fd < 0) {
        printf("Error %i opening %s\n", drv_fd, drv_path);
        return -1;
    }

    printf("%s open\n", drv_path);

    // get driver version
    if (get_driver_version(drv_fd, &major, &minor) < 0) {
        printf("Error getting driver version\n");
        printf("Closing\n");
        close(drv_fd);
        printf("Done\n");
        return -1;
    }

    printf("Driver version %u.%u\n", major, minor);

    /////////////////////////////////////////////////////////////////////////////////
    // Step 0: Allocate the necessary BOs. This includes:
    // 1. The operands for the two kernels that will be launched
    // 2. A heap which contains:
    //  a. A PDI for the design that will be run
    //  b. Instruction sequences for both runs

    // reserve some device memory for the heap
    if (alloc_heap(drv_fd, 48 * 1024 * 1024, &heap_handle) < 0) {
        perror("Error allocating device heap");
        printf("Closing\n");
        close(drv_fd);
        printf("Done\n");
        return -1;
    }

    uint64_t pdi_vaddr;
    uint64_t pdi_xdna_vaddr;
    __u32 pdi_handle;
    printf("Loading pdi\n");
    ret = load_pdi(drv_fd, &pdi_vaddr, &pdi_xdna_vaddr, &pdi_handle, pdi_path);
    if (ret < 0) {
        printf("Error %i loading pdi\n", ret);
        printf("Closing\n");
        close(drv_fd);
        printf("Done\n");
        return -1;
    }

    uint64_t dpu_0_vaddr;
    uint64_t dpu_0_xdna_vaddr;
    __u32 dpu_0_handle;
    __u32 num_dpu_0_insts;
    printf("Loading dpu inst\n");
    ret = load_instructions(drv_fd, &dpu_0_vaddr, &dpu_0_xdna_vaddr, &dpu_0_handle, dpu_inst_path, &num_dpu_0_insts);
    if (ret < 0) {
        printf("Error %i loading dpu instructions\n", ret);
        printf("Closing\n");
        close(drv_fd);
        printf("Done\n");
        return -1;
    }

    printf("DPU 0 instructions @:             %p\n", (void *)dpu_0_vaddr);
    printf("PDI file @:                     %p\n", (void *)pdi_vaddr);
    printf("PDI handle @:                     %d\n", pdi_handle);

    /////////////// import input///////////////
    uint64_t input_0;
    uint64_t input_0_xdna_vaddr;
    __u32 input_0_handle;

    // import dma-buf to xdna
    drm_prime_handle prime_params_input = {0, 0, dmabuf_fd_input};
    if (ioctl(drv_fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &prime_params_input) < 0) {
        std::cerr << "Failed to import DMA-BUF: " << strerror(errno) << std::endl;
        close(drv_fd);
        return -1;
    }
    std::cout << "Successfully imported DMA-BUF. Handle: " << prime_params_input.handle << std::endl;

    struct amdxdna_drm_get_bo_info get_bo_info = {.handle = prime_params_input.handle};
    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info);
    if (ret != 0) {
        perror("Failed to get BO info");
        return -2;
    }
    std::cout << "Imported Buffer MapOffset: " << get_bo_info.map_offset << std::endl;

    // input_0 = (__u64)hip_managed_ptr_input; 
    input_0 = (__u64)mmap(0, N * sizeof(dataType), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, drv_fd, get_bo_info.map_offset);
    input_0_handle = prime_params_input.handle;
    
    // Functional Print Statement
    // touch buffer so that xdna can see it? (removing it will fail tests)
    // still needed XDNA >= 2.18.0
    std::cout << "Functional Print: " << *((dataType *)input_0) << std::endl; 

    /////////////// import output///////////////
    uint64_t output_0;
    uint64_t output_0_xdna_vaddr;
    __u32 output_0_handle;

    // import dma-buf to xdna
    drm_prime_handle prime_params_output = {0, 0, dmabuf_fd_output};
    if (ioctl(drv_fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &prime_params_output) < 0) {
        std::cerr << "Failed to import DMA-BUF: " << strerror(errno) << std::endl;
        close(drv_fd);
        return -1;
    }
    std::cout << "Successfully imported DMA-BUF. Handle: " << prime_params_output.handle << std::endl;
    // can have the same handle due the the underlying memory implementation?

    // Functional Print Statement
    // touch buffer so that xdna can see it? (removing it will fail tests)
    // still needed XDNA >= 2.18.0
    // for output this is not needed -> need further investigation
    // std::cout << hip_managed_ptr_input[0] << std::endl;

    output_0 = (__u64)hip_managed_ptr_output;
    output_0_handle = prime_params_output.handle;

    for (int i = 0; i < N; i++) {
        *((dataType *)output_0 + i) = 0x0;
    }

    // Performing a sync on the queue descriptor, completion signal, queue buffer
    // and config cu bo.
    sync_bo(drv_fd, dpu_0_handle);
    sync_bo(drv_fd, pdi_handle);
    // We can sync imported buffers with XDNA >= 2.18.0
    sync_bo(drv_fd, input_0_handle);
    sync_bo(drv_fd, output_0_handle);

    struct amdxdna_drm_create_hwctx create_hw_ctx;
    struct amdxdna_drm_exec_cmd exec_cmd_0;
    // the create_cmd_packet_chain will not submit the input_0_handle to the amdxdna_drm_exec_cmd.bo_args -> currently
    // hardcoded
    ret = create_cmd_packet_chain(drv_fd, pdi_handle, dpu_0_xdna_vaddr, dpu_0_handle, input_0, output_0, input_0_handle,
                                  output_0_handle, create_hw_ctx, exec_cmd_0);

    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_EXEC_CMD, &exec_cmd_0);
    if (ret != 0) {
        perror("Failed to submit work");
        return -1;
    }

    /////////////////////////////////////////////////////////////////////////////////
    // Step 4: Wait for the output
    // Use the wait IOCTL to wait for our submission to complete
    struct amdxdna_drm_wait_cmd wait_cmd = {
        .hwctx = create_hw_ctx.handle,
        .timeout = 50, // 50ms timeout
        .seq = exec_cmd_0.seq,
    };

    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_WAIT_CMD, &wait_cmd);
    if (ret != 0) {
        perror("Failed to wait");
        return -1;
    }

    /////////////////////////////////////////////////////////////////////////////////
    // Step 5: Verify output

    // Reading the user buffers
    // sync_bo(drv_fd, input_0_handle);
    sync_bo(drv_fd, output_0_handle);

    int errors = 0;
    std::cout << "Sanity Check for first element src-result 42(init)+3(gpu)+1(npu): " << *((dataType *)output_0)
              << std::endl;
    printf("Checking run 0:\n");
    for (int i = 0; i < N; i++) {
        dataType src = *((dataType *)input_0 + i);
        dataType dst = *((dataType *)output_0 + i);
        // printf("src: 0x%x\n", src);
        // printf("dst: 0x%x\n", dst);
        if (src + 1 != dst && i < 16) {
            printf("[ERROR] %d: %d + 1 != %d\n", i, src, dst);
            errors++;
        }
    }

    if (!errors) {
        printf("PASS!\n");
    } else {
        printf("FAIL! %d/1024\n", errors);
    }

    printf("Closing\n");
    close(drv_fd);
    err = hipFree(hip_managed_ptr_input);
    err = hipFree(hip_managed_ptr_output);
    printf("Done\n");
    return 0;
}
