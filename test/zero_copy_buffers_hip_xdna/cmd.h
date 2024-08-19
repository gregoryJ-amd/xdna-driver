#pragma once

#include "ipu-util.h"

// These packets are variable width but using this as a
// maximum size for now
#define PACKET_SIZE 64

/*
 * Interpretation of the beginning of data payload for ERT_CMD_CHAIN in
 * amdxdna_cmd. The rest of the payload in amdxdna_cmd is cmd BO handles.
 */
struct amdxdna_cmd_chain {
    __u32 command_count;
    __u32 submit_index;
    __u32 error_index;
    __u32 reserved[3];
    __u64 data[] __counted_by(command_count);
};

/* Exec buffer command header format */
struct amdxdna_cmd {
    union {
        struct {
            __u32 state : 4;
            __u32 unused : 6;
            __u32 extra_cu_masks : 2;
            __u32 count : 11;
            __u32 opcode : 5;
            __u32 reserved : 4;
        };
        __u32 header;
    };
    __u32 data[] __counted_by(count);
};

int create_cmd_packet_chain(int drv_fd, __u32 pdi_handle, uint64_t dpu_0_sram_vaddr, __u32 dpu_0_handle,
                            uint64_t input_0, uint64_t output_0, __u32 input_0_handle, __u32 output_0_handle,
                            struct amdxdna_drm_create_hwctx &hw_ctx, struct amdxdna_drm_exec_cmd &cmd) {
    int ret;

    // Allocating a structure to store QOS information
    struct amdxdna_qos_info *qos = (struct amdxdna_qos_info *)malloc(sizeof(struct amdxdna_qos_info));
    qos->gops = 0;
    qos->fps = 0;
    qos->dma_bandwidth = 0;
    qos->latency = 0;
    qos->frame_exec_time = 0;
    qos->priority = 0;

    // This is the structure that we pass
    struct amdxdna_drm_create_hwctx create_hw_ctx = {
        .ext = 0,
        .ext_flags = 0,
        .qos_p = (__u64)qos,
        .umq_bo = 0,
        .log_buf_bo = 0,
        .max_opc = 0x800, // Not sure what this is but this was the value used
        .num_tiles = 4,
        .mem_size = 0,
        .umq_doorbell = 0,
    };
    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_CREATE_HWCTX, &create_hw_ctx);
    if (ret != 0) {
        perror("Failed to create hwctx");
        return -1;
    }
    hw_ctx = create_hw_ctx;

    // Creating a structure to configure the CU
    struct amdxdna_cu_config cu_config = {
        .cu_bo = pdi_handle,
        .cu_func = 0,
    };

    // Creating a structure to configure the hardware context
    struct amdxdna_hwctx_param_config_cu param_config_cu;
    param_config_cu.num_cus = 1;
    param_config_cu.cu_configs[0] = cu_config;

    printf("Size of param_config_cu: 0x%lx\n", sizeof(param_config_cu));

    // Configuring the hardware context with the PDI
    struct amdxdna_drm_config_hwctx config_hw_ctx = {
        .handle = create_hw_ctx.handle,
        .param_type = DRM_AMDXDNA_HWCTX_CONFIG_CU,
        .param_val = (__u64)&param_config_cu, // Pass in the pointer to the param value
        .param_val_size = 0x10,               // Size of param config CU is 16B
    };
    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, &config_hw_ctx);
    if (ret != 0) {
        perror("Failed to config hwctx");
        return -1;
    }

    /////////////////////////////////////////////////////////////////////////////////
    // Step 2: Configuring the CMD BOs with the different instruction sequences
    struct amdxdna_drm_create_bo create_cmd_bo_0 = {
        .type = AMDXDNA_BO_CMD,
        .size = PACKET_SIZE,
    };
    int cmd_bo_ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_cmd_bo_0);
    if (cmd_bo_ret != 0) {
        perror("Failed to create cmd_0");
        return -1;
    }

    struct amdxdna_drm_get_bo_info cmd_bo_0_get_bo_info = {.handle = create_cmd_bo_0.handle};
    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &cmd_bo_0_get_bo_info);
    if (ret != 0) {
        perror("Failed to get cmd BO 0 info");
        return -2;
    }

    // Writing the first packet to the queue
    struct amdxdna_cmd *cmd_0 = (struct amdxdna_cmd *)mmap(0, PACKET_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, drv_fd,
                                                           cmd_bo_0_get_bo_info.map_offset);
    cmd_0->state = 1; // ERT_CMD_STATE_NEW;
    cmd_0->extra_cu_masks = 0;
    cmd_0->count = 0xF;   // NOTE: For some reason this needs to be larger
    cmd_0->opcode = 0x0;  // ERT_START_CU;
    cmd_0->data[0] = 0x3; // NOTE: This one seems to be skipped
    cmd_0->data[1] = 0x3; // Transaction opcode
    cmd_0->data[2] = 0x0;
    cmd_0->data[3] = dpu_0_sram_vaddr;
    cmd_0->data[4] = 0x0;
    cmd_0->data[5] = 0x44;                          // Size of DPU instruction
    cmd_0->data[6] = input_0 & 0xFFFFFFFF;          // Input low
    cmd_0->data[7] = (input_0 >> 32) & 0xFFFFFFFF;  // Input high
    cmd_0->data[8] = output_0 & 0xFFFFFFFF;         // Output low
    cmd_0->data[9] = (output_0 >> 32) & 0xFFFFFFFF; // Output high

    // Allocate a command chain
    void *bo_cmd_chain_buf = NULL;
    cmd_bo_ret = posix_memalign(&bo_cmd_chain_buf, 4096, 4096);
    if (cmd_bo_ret != 0 || bo_cmd_chain_buf == NULL) {
        printf("[ERROR] Failed to allocate cmd_bo buffer of size %d\n", 4096);
    }

    struct amdxdna_drm_create_bo create_cmd_chain_bo = {
        .type = AMDXDNA_BO_CMD,
        .size = 4096,
    };
    cmd_bo_ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_cmd_chain_bo);
    if (cmd_bo_ret != 0) {
        perror("Failed to create command chain BO");
        return -1;
    }

    struct amdxdna_drm_get_bo_info cmd_chain_bo_get_bo_info = {.handle = create_cmd_chain_bo.handle};
    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &cmd_chain_bo_get_bo_info);
    if (ret != 0) {
        perror("Failed to get cmd BO 0 info");
        return -2;
    }

    struct amdxdna_cmd *cmd_chain = (struct amdxdna_cmd *)mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, drv_fd,
                                                               cmd_chain_bo_get_bo_info.map_offset);

    // Writing information to the command buffer
    struct amdxdna_cmd_chain *cmd_chain_payload = (struct amdxdna_cmd_chain *)(cmd_chain->data);
    cmd_chain->state = 1; // ERT_CMD_STATE_NEW;
    cmd_chain->extra_cu_masks = 0;
    cmd_chain->count = 0xA;   // TODO: Why is this the value?
    cmd_chain->opcode = 0x13; // ERT_CMD_CHAIN
    cmd_chain_payload->command_count = 1;
    cmd_chain_payload->submit_index = 0;
    cmd_chain_payload->error_index = 0;
    cmd_chain_payload->data[0] = create_cmd_bo_0.handle;

    // Reading the user buffers -> failes due to bo's not being the correct type
    // sync_bo(drv_fd, create_cmd_chain_bo.handle);
    // sync_bo(drv_fd, create_cmd_bo_0.handle);

    // Perform a submit cmd
    uint32_t bo_args[1] = {dpu_0_handle};
    // uint32_t bo_args[3] = {dpu_0_handle, input_0_handle, output_0_handle};
    struct amdxdna_drm_exec_cmd exec_cmd_0 = {
        .ext = 0,
        .ext_flags = 0,
        .hwctx = create_hw_ctx.handle,
        .type = AMDXDNA_CMD_SUBMIT_EXEC_BUF,
        .cmd_handles = create_cmd_chain_bo.handle,
        .args = (__u64)bo_args,
        .cmd_count = 1,
        .arg_count = sizeof(bo_args) / sizeof(uint32_t),
    };

    cmd = exec_cmd_0;

    return 0;
}
