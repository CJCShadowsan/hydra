#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <sys/sysctl.h>

#define PREFILL_MATMUL_SIZE 4096
#define PREFILL_UBATCH_TOKENS 512
#define PREFILL_MOE_M 1024
#define PREFILL_MOE_N 512
#define PREFILL_MOE_K 2048
#define PREFILL_MOE_EXPERTS 8

typedef struct {
    const char *key;
    const char *variant;
    double gbps;
} ChipBandwidth;

static const ChipBandwidth RATED_BANDWIDTH[] = {
    {"M5", "all", 153},       {"M5 Pro", "all", 307},
    {"M5 Max", "18-core CPU / 32-core GPU", 460},
    {"M5 Max", "18-core CPU / 40-core GPU", 614},
    {"M4", "all", 120},       {"M4 Pro", "all", 273},
    {"M4 Max", "14-core CPU / 32-core GPU", 410},
    {"M4 Max", "16-core CPU / 40-core GPU", 546},
    {"M3", "all", 100},       {"M3 Pro", "all", 150},
    {"M3 Max", "14-core CPU / 30-core GPU", 300},
    {"M3 Max", "16-core CPU / 40-core GPU", 400},
    {"M3 Ultra", "all", 819}, {"M2", "all", 100},
    {"M2 Pro", "all", 200},   {"M2 Max", "all", 400},
    {"M2 Ultra", "all", 800}, {"M1", "all", 68},
    {"M1 Pro", "all", 200},   {"M1 Max", "all", 400},
    {"M1 Ultra", "all", 800},
};

static int physical_cpu_count(void) {
    int32_t count = 0;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.physicalcpu", &count, &size, NULL, 0) != 0) {
        return 0;
    }
    return (int)count;
}

static bool rated_for(NSString *device_name, double *gbps_out, bool *estimated_out) {
    NSUInteger best_len = 0;
    for (size_t i = 0; i < sizeof(RATED_BANDWIDTH) / sizeof(RATED_BANDWIDTH[0]); ++i) {
        NSString *key = [NSString stringWithUTF8String:RATED_BANDWIDTH[i].key];
        if ([device_name containsString:key] && [key length] > best_len) {
            best_len = [key length];
        }
    }
    if (best_len == 0) {
        return false;
    }

    bool all_variants = true;
    for (size_t i = 0; i < sizeof(RATED_BANDWIDTH) / sizeof(RATED_BANDWIDTH[0]); ++i) {
        NSString *key = [NSString stringWithUTF8String:RATED_BANDWIDTH[i].key];
        if ([device_name containsString:key] && [key length] == best_len &&
            strcmp(RATED_BANDWIDTH[i].variant, "all") != 0) {
            all_variants = false;
            break;
        }
    }

    if (all_variants) {
        for (size_t i = 0; i < sizeof(RATED_BANDWIDTH) / sizeof(RATED_BANDWIDTH[0]); ++i) {
            NSString *key = [NSString stringWithUTF8String:RATED_BANDWIDTH[i].key];
            if ([device_name containsString:key] && [key length] == best_len) {
                *gbps_out = RATED_BANDWIDTH[i].gbps;
                *estimated_out = false;
                return true;
            }
        }
    }

    int cpu_count = physical_cpu_count();
    NSString *cpu_pattern = [NSString stringWithFormat:@"%d-core CPU", cpu_count];
    int matches = 0;
    double matched_gbps = 0.0;
    double lowest_gbps = DBL_MAX;

    for (size_t i = 0; i < sizeof(RATED_BANDWIDTH) / sizeof(RATED_BANDWIDTH[0]); ++i) {
        NSString *key = [NSString stringWithUTF8String:RATED_BANDWIDTH[i].key];
        if (![device_name containsString:key] || [key length] != best_len) {
            continue;
        }

        NSString *variant = [NSString stringWithUTF8String:RATED_BANDWIDTH[i].variant];
        if ([variant containsString:cpu_pattern]) {
            matches += 1;
            matched_gbps = RATED_BANDWIDTH[i].gbps;
        }
        if (RATED_BANDWIDTH[i].gbps < lowest_gbps) {
            lowest_gbps = RATED_BANDWIDTH[i].gbps;
        }
    }

    if (matches == 1) {
        *gbps_out = matched_gbps;
        *estimated_out = false;
        return true;
    }

    *gbps_out = lowest_gbps;
    *estimated_out = true;
    return true;
}

static char *copy_c_string(NSString *value) {
    const char *utf8 = [value UTF8String];
    char *copy = malloc(strlen(utf8) + 1);
    if (copy != NULL) {
        strcpy(copy, utf8);
    }
    return copy;
}

static double percentile_value(NSMutableArray<NSNumber *> *values, NSUInteger index) {
    [values sortUsingSelector:@selector(compare:)];
    return [values[index] doubleValue];
}

static double run_memread(id<MTLCommandQueue> queue,
                          id<MTLComputePipelineState> pso,
                          id<MTLBuffer> buf,
                          id<MTLBuffer> sink,
                          MTLSize grid,
                          MTLSize tpg) {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    __block double elapsed = 0.0;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> b) {
      elapsed = [b GPUEndTime] - [b GPUStartTime];
    }];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBuffer:sink offset:0 atIndex:1];
    [enc dispatchThreads:grid threadsPerThreadgroup:tpg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return elapsed;
}

static double run_compute(id<MTLCommandQueue> queue,
                          id<MTLComputePipelineState> pso,
                          id<MTLBuffer> compute_sink,
                          uint32_t compute_thread_count,
                          uint32_t compute_iters,
                          MTLSize grid,
                          MTLSize tpg,
                          double flops_per_thread_per_iter) {
    memset([compute_sink contents], 0, compute_thread_count * sizeof(float));
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    __block double elapsed = 0.0;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> b) {
      elapsed = [b GPUEndTime] - [b GPUStartTime];
    }];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:compute_sink offset:0 atIndex:0];
    [enc setBytes:&compute_iters length:sizeof(compute_iters) atIndex:1];
    [enc dispatchThreads:grid threadsPerThreadgroup:tpg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    double total_flops = (double)compute_thread_count * (double)compute_iters * flops_per_thread_per_iter;
    return total_flops / elapsed / 1e12;
}

static double run_decode_like_memread(id<MTLCommandQueue> queue,
                                      id<MTLComputePipelineState> pso,
                                      id<MTLBuffer> buf,
                                      id<MTLBuffer> sink,
                                      MTLSize grid,
                                      MTLSize tpg,
                                      NSUInteger chunk_bytes,
                                      NSUInteger dispatches) {
    NSDate *start = [NSDate date];
    for (NSUInteger i = 0; i < dispatches; ++i) {
        (void)run_memread(queue, pso, buf, sink, grid, tpg);
    }
    double elapsed = [[NSDate date] timeIntervalSinceDate:start];
    return ((double)chunk_bytes * (double)dispatches) / elapsed / 1e9;
}

static double run_empty_dispatch_overhead_ms(id<MTLCommandQueue> queue,
                                             id<MTLComputePipelineState> pso,
                                             id<MTLBuffer> sink,
                                             MTLSize grid,
                                             MTLSize tpg,
                                             NSUInteger dispatches) {
    NSDate *start = [NSDate date];
    for (NSUInteger i = 0; i < dispatches; ++i) {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:sink offset:0 atIndex:0];
        [enc dispatchThreads:grid threadsPerThreadgroup:tpg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    double elapsed = [[NSDate date] timeIntervalSinceDate:start];
    return elapsed * 1000.0 / (double)dispatches;
}

static double run_mps_matmul(id<MTLCommandQueue> queue,
                             MPSMatrixMultiplication *gemm,
                             MPSMatrix *left,
                             MPSMatrix *right,
                             MPSMatrix *result,
                             double flops) {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    __block double elapsed = 0.0;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> b) {
      elapsed = [b GPUEndTime] - [b GPUStartTime];
    }];
    [gemm encodeToCommandBuffer:cmd leftMatrix:left rightMatrix:right resultMatrix:result];
    [cmd commit];
    [cmd waitUntilCompleted];
    return flops / elapsed / 1e12;
}

static double run_mps_moe_matmul(id<MTLCommandQueue> queue,
                                 MPSMatrixMultiplication *gemm,
                                 NSArray<MPSMatrix *> *left,
                                 NSArray<MPSMatrix *> *right,
                                 NSArray<MPSMatrix *> *result,
                                 double flops) {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    __block double elapsed = 0.0;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> b) {
      elapsed = [b GPUEndTime] - [b GPUStartTime];
    }];
    for (NSUInteger expert = 0; expert < [left count]; ++expert) {
        [gemm encodeToCommandBuffer:cmd
                         leftMatrix:left[expert]
                        rightMatrix:right[expert]
                       resultMatrix:result[expert]];
    }
    [cmd commit];
    [cmd waitUntilCompleted];
    return flops / elapsed / 1e12;
}

char *mesh_llm_gpu_bench_metal_json(char **error_out) {
    @autoreleasepool {
        @try {
        if (error_out != NULL) {
            *error_out = NULL;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            if (error_out != NULL) {
                *error_out = copy_c_string(@"Metal device not available");
            }
            return NULL;
        }

        NSString *shader_source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void memread(device const float4* src [[buffer(0)]], device float* sink [[buffer(1)]], uint id [[thread_position_in_grid]]) { float4 v = src[id]; if (v.x == 9999999.0f) sink[0] = v.x; }\n"
             "kernel void empty_dispatch(device float *sink [[buffer(0)]], uint id [[thread_position_in_grid]]) { if (id == 0) sink[0] += 0.0f; }\n"
             "kernel void compute_fp32(device float *sink [[buffer(0)]], constant uint &iters [[buffer(1)]], uint id [[thread_position_in_grid]]) { float a0 = 1.0f + 0.0001f * float(id + 1); float a1 = a0 + 1.0f; float a2 = a0 + 2.0f; float a3 = a0 + 3.0f; constexpr float b0 = 1.000001f; constexpr float b1 = 0.999991f; constexpr float c0 = 0.500001f; constexpr float c1 = 0.250001f; for (uint i = 0; i < iters; ++i) { a0 = fma(a0, b0, c0); a1 = fma(a1, b1, c1); a2 = fma(a2, b0, c1); a3 = fma(a3, b1, c0); a0 = fma(a0, b1, c1); a1 = fma(a1, b0, c0); a2 = fma(a2, b1, c0); a3 = fma(a3, b0, c1); } sink[id] = a0 + a1 + a2 + a3; }\n"
             "kernel void compute_fp16(device float *sink [[buffer(0)]], constant uint &iters [[buffer(1)]], uint id [[thread_position_in_grid]]) { half seed = half(1.0f + 0.0001f * float(id + 1)); half2 a0 = half2(seed, seed + half(1.0)); half2 a1 = half2(seed + half(2.0), seed + half(3.0)); half2 a2 = half2(seed + half(4.0), seed + half(5.0)); half2 a3 = half2(seed + half(6.0), seed + half(7.0)); constexpr half2 b0 = half2(half(1.0009765625), half(0.9990234375)); constexpr half2 b1 = half2(half(0.99951171875), half(1.00048828125)); constexpr half2 c0 = half2(half(0.1875), half(0.3125)); constexpr half2 c1 = half2(half(0.4375), half(0.5625)); for (uint i = 0; i < iters; ++i) { a0 = fma(a0, b0, c0); a1 = fma(a1, b1, c1); a2 = fma(a2, b0, c1); a3 = fma(a3, b1, c0); a0 = fma(a0, b1, c1); a1 = fma(a1, b0, c0); a2 = fma(a2, b1, c0); a3 = fma(a3, b0, c1); } float2 s0 = float2(a0); float2 s1 = float2(a1); float2 s2 = float2(a2); float2 s3 = float2(a3); sink[id] = s0.x + s0.y + s1.x + s1.y + s2.x + s2.y + s3.x + s3.y; }\n";

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:shader_source options:nil error:&error];
        if (library == nil) {
            if (error_out != NULL) {
                *error_out = copy_c_string([NSString stringWithFormat:@"failed to compile Metal benchmark library: %@", error]);
            }
            return NULL;
        }

        id<MTLFunction> memread = [library newFunctionWithName:@"memread"];
        id<MTLFunction> empty_dispatch = [library newFunctionWithName:@"empty_dispatch"];
        id<MTLFunction> compute_fp32 = [library newFunctionWithName:@"compute_fp32"];
        id<MTLFunction> compute_fp16 = [library newFunctionWithName:@"compute_fp16"];
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:memread error:&error];
        id<MTLComputePipelineState> pso_empty = [device newComputePipelineStateWithFunction:empty_dispatch error:&error];
        id<MTLComputePipelineState> pso_fp32 = [device newComputePipelineStateWithFunction:compute_fp32 error:&error];
        id<MTLComputePipelineState> pso_fp16 = [device newComputePipelineStateWithFunction:compute_fp16 error:&error];
        if (pso == nil || pso_empty == nil || pso_fp32 == nil || pso_fp16 == nil) {
            if (error_out != NULL) {
                *error_out = copy_c_string([NSString stringWithFormat:@"failed to create Metal benchmark pipeline: %@", error]);
            }
            return NULL;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> sink = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
        const NSUInteger buffer_bytes = 512 * 1024 * 1024;
        const NSUInteger float4_bytes = sizeof(float) * 4;
        const NSUInteger element_count = buffer_bytes / float4_bytes;
        id<MTLBuffer> buf = [device newBufferWithLength:buffer_bytes options:MTLResourceStorageModeShared];
        if (queue == nil || sink == nil || buf == nil) {
            if (error_out != NULL) {
                *error_out = copy_c_string(@"failed to allocate Metal benchmark resources");
            }
            return NULL;
        }

        float *ptr = (float *)[buf contents];
        NSUInteger float_count = buffer_bytes / sizeof(float);
        NSUInteger step = MAX((NSUInteger)1, float_count / 1024);
        for (NSUInteger i = 0; i < float_count; i += step) {
            ptr[i] = (float)(i % 256);
        }

        MTLSize tpg = MTLSizeMake([pso maxTotalThreadsPerThreadgroup], 1, 1);
        MTLSize grid = MTLSizeMake(element_count, 1, 1);
        // Decode is a loop of matmul launches, but model-fit consumes this as
        // model-weight bandwidth. A tiny 8 MiB pass can measure cache bandwidth
        // on GPUs with large L2/SLC and then overstate throughput for normal
        // GGUF weight streams. Use a still-smaller-than-full-buffer pass so
        // launch overhead remains visible, then cap the reported decode value
        // by the full-buffer p90 measurement below.
        const NSUInteger decode_chunk_bytes = 256 * 1024 * 1024;
        const NSUInteger decode_dispatches = 8;
        const NSUInteger fixed_overhead_dispatches = 256;
        MTLSize decode_grid = MTLSizeMake(decode_chunk_bytes / float4_bytes, 1, 1);
        MTLSize empty_grid = MTLSizeMake(1, 1, 1);
        MTLSize empty_tpg = MTLSizeMake(1, 1, 1);
        const uint32_t compute_iters = 16384;
        const uint32_t compute_thread_count = 262144;
        id<MTLBuffer> compute_sink =
            [device newBufferWithLength:compute_thread_count * sizeof(float) options:MTLResourceStorageModeShared];
        if (compute_sink == nil) {
            if (error_out != NULL) {
                *error_out = copy_c_string(@"failed to allocate Metal compute benchmark resources");
            }
            return NULL;
        }

        MTLSize compute_grid = MTLSizeMake(compute_thread_count, 1, 1);
        MTLSize compute_tpg_fp32 = MTLSizeMake([pso_fp32 maxTotalThreadsPerThreadgroup], 1, 1);
        MTLSize compute_tpg_fp16 = MTLSizeMake([pso_fp16 maxTotalThreadsPerThreadgroup], 1, 1);

        const NSUInteger half_bytes = sizeof(uint16_t);
        const NSUInteger dense_row_bytes = PREFILL_MATMUL_SIZE * half_bytes;
        const NSUInteger dense_matrix_bytes = dense_row_bytes * PREFILL_MATMUL_SIZE;
        id<MTLBuffer> dense_a = [device newBufferWithLength:dense_matrix_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> dense_b = [device newBufferWithLength:dense_matrix_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> dense_c = [device newBufferWithLength:dense_matrix_bytes options:MTLResourceStorageModeShared];
        if (dense_a == nil || dense_b == nil || dense_c == nil) {
            if (error_out != NULL) {
                *error_out = copy_c_string(@"failed to allocate Metal dense matmul benchmark resources");
            }
            return NULL;
        }
        memset([dense_a contents], 1, dense_matrix_bytes);
        memset([dense_b contents], 1, dense_matrix_bytes);
        memset([dense_c contents], 0, dense_matrix_bytes);

        MPSMatrixDescriptor *dense_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:PREFILL_MATMUL_SIZE
                                                  columns:PREFILL_MATMUL_SIZE
                                                 rowBytes:dense_row_bytes
                                                 dataType:MPSDataTypeFloat16];
        MPSMatrix *dense_left = [[MPSMatrix alloc] initWithBuffer:dense_a descriptor:dense_desc];
        MPSMatrix *dense_right = [[MPSMatrix alloc] initWithBuffer:dense_b descriptor:dense_desc];
        MPSMatrix *dense_result = [[MPSMatrix alloc] initWithBuffer:dense_c descriptor:dense_desc];
        MPSMatrixMultiplication *dense_gemm =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                              transposeLeft:false
                                             transposeRight:false
                                                 resultRows:PREFILL_MATMUL_SIZE
                                              resultColumns:PREFILL_MATMUL_SIZE
                                            interiorColumns:PREFILL_MATMUL_SIZE
                                                      alpha:1.0
                                                       beta:0.0];

        const NSUInteger ubatch_row_bytes = PREFILL_UBATCH_TOKENS * half_bytes;
        const NSUInteger ubatch_matrix_bytes =
            (NSUInteger)PREFILL_MATMUL_SIZE * PREFILL_UBATCH_TOKENS * half_bytes;
        id<MTLBuffer> ubatch_b = [device newBufferWithLength:ubatch_matrix_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> ubatch_c = [device newBufferWithLength:ubatch_matrix_bytes options:MTLResourceStorageModeShared];
        if (ubatch_b == nil || ubatch_c == nil) {
            if (error_out != NULL) {
                *error_out = copy_c_string(@"failed to allocate Metal ubatch matmul benchmark resources");
            }
            return NULL;
        }
        memset([ubatch_b contents], 1, ubatch_matrix_bytes);
        memset([ubatch_c contents], 0, ubatch_matrix_bytes);
        MPSMatrixDescriptor *ubatch_b_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:PREFILL_MATMUL_SIZE
                                                  columns:PREFILL_UBATCH_TOKENS
                                                 rowBytes:ubatch_row_bytes
                                                 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *ubatch_c_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:PREFILL_MATMUL_SIZE
                                                  columns:PREFILL_UBATCH_TOKENS
                                                 rowBytes:ubatch_row_bytes
                                                 dataType:MPSDataTypeFloat16];
        MPSMatrix *ubatch_right = [[MPSMatrix alloc] initWithBuffer:ubatch_b descriptor:ubatch_b_desc];
        MPSMatrix *ubatch_result = [[MPSMatrix alloc] initWithBuffer:ubatch_c descriptor:ubatch_c_desc];
        MPSMatrixMultiplication *ubatch_gemm =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                              transposeLeft:false
                                             transposeRight:false
                                                 resultRows:PREFILL_MATMUL_SIZE
                                              resultColumns:PREFILL_UBATCH_TOKENS
                                            interiorColumns:PREFILL_MATMUL_SIZE
                                                      alpha:1.0
                                                       beta:0.0];

        const NSUInteger moe_a_bytes = (NSUInteger)PREFILL_MOE_M * PREFILL_MOE_K * half_bytes;
        const NSUInteger moe_b_bytes = (NSUInteger)PREFILL_MOE_K * PREFILL_MOE_N * half_bytes;
        const NSUInteger moe_c_bytes = (NSUInteger)PREFILL_MOE_M * PREFILL_MOE_N * half_bytes;
        MPSMatrixDescriptor *moe_a_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:PREFILL_MOE_M
                                                  columns:PREFILL_MOE_K
                                                 rowBytes:PREFILL_MOE_K * half_bytes
                                                 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *moe_b_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:PREFILL_MOE_K
                                                  columns:PREFILL_MOE_N
                                                 rowBytes:PREFILL_MOE_N * half_bytes
                                                 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *moe_c_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:PREFILL_MOE_M
                                                  columns:PREFILL_MOE_N
                                                 rowBytes:PREFILL_MOE_N * half_bytes
                                                 dataType:MPSDataTypeFloat16];
        id<MTLBuffer> moe_a_buffers[PREFILL_MOE_EXPERTS];
        id<MTLBuffer> moe_b_buffers[PREFILL_MOE_EXPERTS];
        id<MTLBuffer> moe_c_buffers[PREFILL_MOE_EXPERTS];
        NSMutableArray<MPSMatrix *> *moe_left = [NSMutableArray arrayWithCapacity:PREFILL_MOE_EXPERTS];
        NSMutableArray<MPSMatrix *> *moe_right = [NSMutableArray arrayWithCapacity:PREFILL_MOE_EXPERTS];
        NSMutableArray<MPSMatrix *> *moe_result = [NSMutableArray arrayWithCapacity:PREFILL_MOE_EXPERTS];
        for (NSUInteger expert = 0; expert < PREFILL_MOE_EXPERTS; ++expert) {
            moe_a_buffers[expert] = [device newBufferWithLength:moe_a_bytes options:MTLResourceStorageModeShared];
            moe_b_buffers[expert] = [device newBufferWithLength:moe_b_bytes options:MTLResourceStorageModeShared];
            moe_c_buffers[expert] = [device newBufferWithLength:moe_c_bytes options:MTLResourceStorageModeShared];
            if (moe_a_buffers[expert] == nil || moe_b_buffers[expert] == nil || moe_c_buffers[expert] == nil) {
                if (error_out != NULL) {
                    *error_out = copy_c_string(@"failed to allocate Metal MoE matmul benchmark resources");
                }
                return NULL;
            }
            memset([moe_a_buffers[expert] contents], 1, moe_a_bytes);
            memset([moe_b_buffers[expert] contents], 1, moe_b_bytes);
            memset([moe_c_buffers[expert] contents], 0, moe_c_bytes);
            [moe_left addObject:[[MPSMatrix alloc] initWithBuffer:moe_a_buffers[expert] descriptor:moe_a_desc]];
            [moe_right addObject:[[MPSMatrix alloc] initWithBuffer:moe_b_buffers[expert] descriptor:moe_b_desc]];
            [moe_result addObject:[[MPSMatrix alloc] initWithBuffer:moe_c_buffers[expert] descriptor:moe_c_desc]];
        }
        MPSMatrixMultiplication *moe_gemm =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                              transposeLeft:false
                                             transposeRight:false
                                                 resultRows:PREFILL_MOE_M
                                              resultColumns:PREFILL_MOE_N
                                            interiorColumns:PREFILL_MOE_K
                                                      alpha:1.0
                                                       beta:0.0];
        const double dense_flops = 2.0
            * (double)PREFILL_MATMUL_SIZE
            * (double)PREFILL_MATMUL_SIZE
            * (double)PREFILL_MATMUL_SIZE;
        const double ubatch_flops = 2.0
            * (double)PREFILL_MATMUL_SIZE
            * (double)PREFILL_UBATCH_TOKENS
            * (double)PREFILL_MATMUL_SIZE;
        const double moe_flops = 2.0
            * (double)PREFILL_MOE_M
            * (double)PREFILL_MOE_N
            * (double)PREFILL_MOE_K
            * (double)PREFILL_MOE_EXPERTS;

        for (int i = 0; i < 3; ++i) {
            (void)run_memread(queue, pso, buf, sink, grid, tpg);
        }
        for (int i = 0; i < 3; ++i) {
            (void)run_compute(queue, pso_fp32, compute_sink, compute_thread_count, compute_iters,
                              compute_grid, compute_tpg_fp32, 16.0);
            (void)run_compute(queue, pso_fp16, compute_sink, compute_thread_count, compute_iters,
                              compute_grid, compute_tpg_fp16, 32.0);
            (void)run_decode_like_memread(queue, pso, buf, sink, decode_grid, tpg,
                                          decode_chunk_bytes, decode_dispatches);
            (void)run_empty_dispatch_overhead_ms(queue, pso_empty, sink, empty_grid, empty_tpg,
                                                 fixed_overhead_dispatches);
            (void)run_mps_matmul(queue, dense_gemm, dense_left, dense_right, dense_result,
                                 dense_flops);
            (void)run_mps_matmul(queue, ubatch_gemm, dense_left, ubatch_right, ubatch_result,
                                 ubatch_flops);
            (void)run_mps_moe_matmul(queue, moe_gemm, moe_left, moe_right, moe_result,
                                     moe_flops);
            (void)run_empty_dispatch_overhead_ms(queue, pso_empty, sink, empty_grid, empty_tpg, 1);
        }

        const int runs = 20;
        NSDate *start = [NSDate date];
        NSMutableArray<NSNumber *> *gbps = [NSMutableArray arrayWithCapacity:runs];
        NSMutableArray<NSNumber *> *fp32_samples = [NSMutableArray arrayWithCapacity:runs];
        NSMutableArray<NSNumber *> *fp16_samples = [NSMutableArray arrayWithCapacity:runs];
        NSMutableArray<NSNumber *> *prefill_matmul_samples = [NSMutableArray arrayWithCapacity:runs];
        NSMutableArray<NSNumber *> *prefill_ubatch_matmul_samples = [NSMutableArray arrayWithCapacity:runs];
        NSMutableArray<NSNumber *> *prefill_moe_matmul_samples = [NSMutableArray arrayWithCapacity:runs];
        NSMutableArray<NSNumber *> *post_prefill_decode_samples = [NSMutableArray arrayWithCapacity:runs];
        NSMutableArray<NSNumber *> *decode_effective_samples = [NSMutableArray arrayWithCapacity:runs];
        NSMutableArray<NSNumber *> *fixed_overhead_samples = [NSMutableArray arrayWithCapacity:runs];

        for (int i = 0; i < runs; ++i) {
            double elapsed = run_memread(queue, pso, buf, sink, grid, tpg);
            [gbps addObject:@((double)buffer_bytes / elapsed / 1e9)];
            [fp32_samples addObject:@(run_compute(queue, pso_fp32, compute_sink, compute_thread_count, compute_iters,
                                                  compute_grid, compute_tpg_fp32, 16.0))];
            [fp16_samples addObject:@(run_compute(queue, pso_fp16, compute_sink, compute_thread_count, compute_iters,
                                                  compute_grid, compute_tpg_fp16, 32.0))];
            [prefill_matmul_samples addObject:@(run_mps_matmul(queue, dense_gemm,
                                                               dense_left, dense_right,
                                                               dense_result, dense_flops))];
            [prefill_ubatch_matmul_samples addObject:@(run_mps_matmul(queue, ubatch_gemm,
                                                                      dense_left, ubatch_right,
                                                                      ubatch_result,
                                                                      ubatch_flops))];
            [prefill_moe_matmul_samples addObject:@(run_mps_moe_matmul(queue, moe_gemm,
                                                                       moe_left, moe_right,
                                                                       moe_result,
                                                                       moe_flops))];
            (void)run_mps_matmul(queue, dense_gemm, dense_left, dense_right, dense_result,
                                 dense_flops);
            [post_prefill_decode_samples addObject:@(run_empty_dispatch_overhead_ms(queue,
                                                                                   pso_empty,
                                                                                   sink,
                                                                                   empty_grid,
                                                                                   empty_tpg,
                                                                                   1))];
            [decode_effective_samples addObject:@(run_decode_like_memread(queue, pso, buf, sink,
                                                                          decode_grid, tpg,
                                                                          decode_chunk_bytes,
                                                                          decode_dispatches))];
            [fixed_overhead_samples addObject:@(run_empty_dispatch_overhead_ms(queue, pso_empty, sink,
                                                                               empty_grid, empty_tpg,
                                                                               fixed_overhead_dispatches))];
        }

        double p50 = percentile_value(gbps, runs / 2);
        double p90 = percentile_value(gbps, (NSUInteger)((double)runs * 0.90) - 1);
        double noise = (p90 - p50) / p90 * 100.0;
        double fp32_measured = percentile_value(fp32_samples, (NSUInteger)((double)runs * 0.90) - 1);
        double fp16_measured = percentile_value(fp16_samples, (NSUInteger)((double)runs * 0.90) - 1);
        double prefill_matmul_measured =
            percentile_value(prefill_matmul_samples, (NSUInteger)((double)runs * 0.90) - 1);
        double prefill_ubatch_matmul_measured =
            percentile_value(prefill_ubatch_matmul_samples, (NSUInteger)((double)runs * 0.90) - 1);
        double prefill_moe_matmul_measured =
            percentile_value(prefill_moe_matmul_samples, (NSUInteger)((double)runs * 0.90) - 1);
        double post_prefill_decode_measured = percentile_value(post_prefill_decode_samples, runs / 2);
        double decode_effective = percentile_value(decode_effective_samples, (NSUInteger)((double)runs * 0.90) - 1);
        decode_effective = MIN(decode_effective, p90);
        double fixed_overhead = percentile_value(fixed_overhead_samples, runs / 2);
        double runtime_secs = [[NSDate date] timeIntervalSinceDate:start];

        NSString *device_name = [device name];
        double rated = 0.0;
        bool estimated = false;
        bool has_rated = rated_for(device_name, &rated, &estimated);
        double efficiency = has_rated ? p90 / rated * 100.0 : 0.0;

        NSMutableString *json = [NSMutableString stringWithFormat:
            @"[{\"device\":\"%@\",\"buffer_mb\":512,\"runs\":%d,\"p50_gbps\":%.2f,\"p90_gbps\":%.2f,\"noise_pct\":%.2f,\"runtime_s\":%.3f,\"decode_effective_gbps\":%.2f,\"decode_fixed_overhead_ms\":%.4f,\"post_prefill_decode_overhead_ms\":%.4f,\"compute_tflops_fp32\":%.2f,\"compute_tflops_fp16\":%.2f,\"prefill_matmul_tflops_fp16\":%.2f,\"prefill_ubatch_matmul_tflops_fp16\":%.2f,\"prefill_moe_matmul_tflops_fp16\":%.2f",
            device_name, runs, p50, p90, noise, runtime_secs, decode_effective, fixed_overhead,
            post_prefill_decode_measured, fp32_measured, fp16_measured, prefill_matmul_measured,
            prefill_ubatch_matmul_measured, prefill_moe_matmul_measured];
        if (has_rated) {
            [json appendFormat:@",\"rated_gbps\":%.0f,\"rated_estimated\":%@", rated, estimated ? @"true" : @"false"];
            [json appendFormat:@",\"efficiency_pct\":%.2f", efficiency];
        }
        [json appendString:@"}]"];
        return copy_c_string(json);
        } @catch (NSException *exception) {
            if (error_out != NULL) {
                *error_out = copy_c_string([NSString stringWithFormat:
                    @"Metal benchmark exception: %@: %@",
                    [exception name],
                    [exception reason]]);
            }
            return NULL;
        }
    }
}

void mesh_llm_gpu_bench_free(void *ptr) {
    free(ptr);
}
