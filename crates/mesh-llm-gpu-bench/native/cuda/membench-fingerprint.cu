// membench-fingerprint.cu — Memory bandwidth fingerprint for NVIDIA GPUs
// Compiled into mesh-llm-gpu-bench for CUDA-flavored mesh-llm builds.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif
#include <algorithm>

#define BUFFER_BYTES (512 * 1024 * 1024)  // 512 MB — safely above L2/LLC on all current NVIDIA GPUs
#define WARMUP_RUNS  3
#define TIMED_RUNS   20
#define BLOCK_SIZE   256
#define COMPUTE_ITERS 16384
#define COMPUTE_BLOCKS_PER_SM 16
// Decode spends its time in many matmul launches, but the bytes that model-fit
// predicts are model-weight bytes, not a tiny cache-resident working set. Keep
// this chunk below the full streaming pass so launch overhead still appears,
// but large enough that a discrete GPU cannot satisfy the measurement entirely
// from L2. The final value is also capped by the full-buffer p90 measurement
// before it is reported as decode-effective bandwidth.
#define DECODE_CHUNK_BYTES (256 * 1024 * 1024)
#define DECODE_DISPATCHES 8
#define FIXED_OVERHEAD_DISPATCHES 256

__global__ void empty_kernel(float* sink) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sink[0] += 0.0f;
    }
}

__global__ void memread(const float4* __restrict__ src, float* sink, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float4 v = src[id];
        if (v.x == 9999999.0f) atomicAdd(sink, v.x);
    }
}

__global__ void compute_fp32_kernel(float* sink, int iters) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float seed = 1.0f + 0.0001f * (float)(id + 1);
    float a0 = seed;
    float a1 = seed + 1.0f;
    float a2 = seed + 2.0f;
    float a3 = seed + 3.0f;
    const float b0 = 1.000001f;
    const float b1 = 0.999991f;
    const float c0 = 0.500001f;
    const float c1 = 0.250001f;
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        a0 = fmaf(a0, b0, c0);
        a1 = fmaf(a1, b1, c1);
        a2 = fmaf(a2, b0, c1);
        a3 = fmaf(a3, b1, c0);
        a0 = fmaf(a0, b1, c1);
        a1 = fmaf(a1, b0, c0);
        a2 = fmaf(a2, b1, c0);
        a3 = fmaf(a3, b0, c1);
    }
    sink[id] = a0 + a1 + a2 + a3;
}

__device__ __forceinline__ __half2 half2_fma_compat(__half2 a, __half2 b, __half2 c) {
    return __hadd2(__hmul2(a, b), c);
}

__global__ void compute_fp16_kernel(float* sink, int iters) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float seed = 1.0f + 0.0001f * (float)(id + 1);
    __half2 a0 = __floats2half2_rn(seed, seed + 1.0f);
    __half2 a1 = __floats2half2_rn(seed + 2.0f, seed + 3.0f);
    const __half2 b0 = __floats2half2_rn(1.000001f, 0.999991f);
    const __half2 b1 = __floats2half2_rn(0.999983f, 1.000013f);
    const __half2 c0 = __floats2half2_rn(0.500001f, 0.250001f);
    const __half2 c1 = __floats2half2_rn(0.125001f, 0.062501f);
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        a0 = half2_fma_compat(a0, b0, c0);
        a1 = half2_fma_compat(a1, b1, c1);
        a0 = half2_fma_compat(a0, b1, c1);
        a1 = half2_fma_compat(a1, b0, c0);
        a0 = half2_fma_compat(a0, b0, c1);
        a1 = half2_fma_compat(a1, b1, c0);
        a0 = half2_fma_compat(a0, b1, c0);
        a1 = half2_fma_compat(a1, b0, c1);
    }
    sink[id] = __low2float(a0) + __high2float(a0) + __low2float(a1) + __high2float(a1);
}

static void check(cudaError_t err, const char* ctx) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", ctx, cudaGetErrorString(err));
        exit(1);
    }
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

static double steady_seconds() {
#ifdef _WIN32
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
#endif
}

int main(int argc, char** argv) {
    int jsonMode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) jsonMode = 1;
    }

    int deviceCount = 0;
    check(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        if (jsonMode) printf("{\"error\":\"No CUDA devices found\"}\n");
        else          printf("No CUDA devices found\n");
        return 1;
    }

    for (int dev = 0; dev < deviceCount; dev++) {
        check(cudaSetDevice(dev), "cudaSetDevice");

        cudaDeviceProp props;
        check(cudaGetDeviceProperties(&props, dev), "cudaGetDeviceProperties");

        int memClockKHz = 0;
        check(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev), "memClockRate");
        double ratedGBps = (double)props.memoryBusWidth
                         * (double)memClockKHz
                         * 2.0 / 8.0 / 1e6;

        int elementCount = BUFFER_BYTES / sizeof(float4);
        int gridSize     = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int computeBlocks = props.multiProcessorCount > 0 ? props.multiProcessorCount * COMPUTE_BLOCKS_PER_SM : 256;
        int computeThreads = computeBlocks * BLOCK_SIZE;

        float4* dSrc;
        float*  dSink;
        float*  dComputeSink;
        check(cudaMalloc(&dSrc,  BUFFER_BYTES), "cudaMalloc src");
        check(cudaMalloc(&dSink, sizeof(float)), "cudaMalloc sink");
        check(cudaMalloc(&dComputeSink, sizeof(float) * computeThreads), "cudaMalloc compute sink");
        check(cudaMemset(dSrc,  0, BUFFER_BYTES), "cudaMemset src");
        check(cudaMemset(dSink, 0, sizeof(float)), "cudaMemset sink");
        check(cudaMemset(dComputeSink, 0, sizeof(float) * computeThreads), "cudaMemset compute sink");

        cudaEvent_t evStart, evStop;
        check(cudaEventCreate(&evStart), "eventCreate start");
        check(cudaEventCreate(&evStop),  "eventCreate stop");

        auto dispatch = [&]() -> double {
            check(cudaEventRecord(evStart), "eventRecord start");
            memread<<<gridSize, BLOCK_SIZE>>>(dSrc, dSink, elementCount);
            check(cudaGetLastError(), "memread launch");
            check(cudaEventRecord(evStop), "eventRecord stop");
            check(cudaEventSynchronize(evStop), "eventSync");
            float ms = 0.0f;
            check(cudaEventElapsedTime(&ms, evStart, evStop), "eventElapsed");
            return (double)BUFFER_BYTES / (ms / 1000.0) / 1e9;
        };

        auto measure_compute_fp32 = [&]() -> double {
            check(cudaMemset(dComputeSink, 0, sizeof(float) * computeThreads), "cudaMemset compute fp32");
            check(cudaEventRecord(evStart), "eventRecord compute fp32 start");
            compute_fp32_kernel<<<computeBlocks, BLOCK_SIZE>>>(dComputeSink, COMPUTE_ITERS);
            check(cudaGetLastError(), "compute fp32 launch");
            check(cudaEventRecord(evStop), "eventRecord compute fp32 stop");
            check(cudaEventSynchronize(evStop), "eventSync compute fp32");
            float ms = 0.0f;
            check(cudaEventElapsedTime(&ms, evStart, evStop), "eventElapsed compute fp32");
            double totalFlops = (double)computeThreads * (double)COMPUTE_ITERS * 16.0;
            return totalFlops / (ms / 1000.0) / 1e12;
        };

        auto measure_compute_fp16 = [&]() -> double {
            check(cudaMemset(dComputeSink, 0, sizeof(float) * computeThreads), "cudaMemset compute fp16");
            check(cudaEventRecord(evStart), "eventRecord compute fp16 start");
            compute_fp16_kernel<<<computeBlocks, BLOCK_SIZE>>>(dComputeSink, COMPUTE_ITERS);
            check(cudaGetLastError(), "compute fp16 launch");
            check(cudaEventRecord(evStop), "eventRecord compute fp16 stop");
            check(cudaEventSynchronize(evStop), "eventSync compute fp16");
            float ms = 0.0f;
            check(cudaEventElapsedTime(&ms, evStart, evStop), "eventElapsed compute fp16");
            double totalFlops = (double)computeThreads * (double)COMPUTE_ITERS * 32.0;
            return totalFlops / (ms / 1000.0) / 1e12;
        };

        auto measure_decode_effective_gbps = [&]() -> double {
            int chunkElements = DECODE_CHUNK_BYTES / sizeof(float4);
            int chunkGridSize = (chunkElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            check(cudaEventRecord(evStart), "eventRecord decode effective start");
            for (int i = 0; i < DECODE_DISPATCHES; ++i) {
                memread<<<chunkGridSize, BLOCK_SIZE>>>(dSrc, dSink, chunkElements);
            }
            check(cudaGetLastError(), "decode effective launch");
            check(cudaEventRecord(evStop), "eventRecord decode effective stop");
            check(cudaEventSynchronize(evStop), "eventSync decode effective");
            float ms = 0.0f;
            check(cudaEventElapsedTime(&ms, evStart, evStop), "eventElapsed decode effective");
            double totalBytes = (double)DECODE_CHUNK_BYTES * (double)DECODE_DISPATCHES;
            return totalBytes / (ms / 1000.0) / 1e9;
        };

        auto measure_fixed_overhead_ms = [&]() -> double {
            check(cudaEventRecord(evStart), "eventRecord fixed overhead start");
            for (int i = 0; i < FIXED_OVERHEAD_DISPATCHES; ++i) {
                empty_kernel<<<1, 1>>>(dSink);
            }
            check(cudaGetLastError(), "fixed overhead launch");
            check(cudaEventRecord(evStop), "eventRecord fixed overhead stop");
            check(cudaEventSynchronize(evStop), "eventSync fixed overhead");
            float ms = 0.0f;
            check(cudaEventElapsedTime(&ms, evStart, evStop), "eventElapsed fixed overhead");
            return (double)ms / (double)FIXED_OVERHEAD_DISPATCHES;
        };

        for (int i = 0; i < WARMUP_RUNS; i++) dispatch();
        for (int i = 0; i < WARMUP_RUNS; i++) {
            (void)measure_compute_fp32();
            (void)measure_compute_fp16();
            (void)measure_decode_effective_gbps();
            (void)measure_fixed_overhead_ms();
        }

        double wallStart = steady_seconds();

        double samples[TIMED_RUNS];
        double fp32Samples[TIMED_RUNS];
        double fp16Samples[TIMED_RUNS];
        double decodeEffectiveSamples[TIMED_RUNS];
        double fixedOverheadSamples[TIMED_RUNS];
        for (int i = 0; i < TIMED_RUNS; i++) {
            samples[i] = dispatch();
            fp32Samples[i] = measure_compute_fp32();
            fp16Samples[i] = measure_compute_fp16();
            decodeEffectiveSamples[i] = measure_decode_effective_gbps();
            fixedOverheadSamples[i] = measure_fixed_overhead_ms();
        }

        double wallEnd = steady_seconds();
        double runtimeSecs = wallEnd - wallStart;

        qsort(samples, TIMED_RUNS, sizeof(double), cmp_double);
        qsort(fp32Samples, TIMED_RUNS, sizeof(double), cmp_double);
        qsort(fp16Samples, TIMED_RUNS, sizeof(double), cmp_double);
        qsort(decodeEffectiveSamples, TIMED_RUNS, sizeof(double), cmp_double);
        qsort(fixedOverheadSamples, TIMED_RUNS, sizeof(double), cmp_double);
        double p50      = samples[TIMED_RUNS / 2];
        double p90      = samples[(int)(TIMED_RUNS * 0.90) - 1];
        double tf32P90  = fp32Samples[(int)(TIMED_RUNS * 0.90) - 1];
        double tf16P90  = fp16Samples[(int)(TIMED_RUNS * 0.90) - 1];
        double decodeEffectiveP90 = decodeEffectiveSamples[(int)(TIMED_RUNS * 0.90) - 1];
        decodeEffectiveP90 = std::min(decodeEffectiveP90, p90);
        double fixedOverheadP50 = fixedOverheadSamples[TIMED_RUNS / 2];
        double noisePct = (p90 - p50) / p90 * 100.0;
        double effPct   = p90 / ratedGBps * 100.0;

        if (jsonMode) {
            if (dev == 0) printf("[");
            printf("{\"device\":\"%s\"," 
                   "\"buffer_mb\":512,"
                   "\"runs\":%d,"
                   "\"p50_gbps\":%.2f,"
                   "\"p90_gbps\":%.2f,"
                   "\"noise_pct\":%.2f,"
                   "\"runtime_s\":%.3f,"
                   "\"rated_gbps\":%.0f,"
                   "\"rated_estimated\":false,"
                   "\"efficiency_pct\":%.2f,"
                   "\"bus_width_bits\":%d,"
                   "\"mem_clock_mhz\":%.0f,"
                   "\"decode_effective_gbps\":%.2f,"
                   "\"decode_fixed_overhead_ms\":%.4f,"
                   "\"compute_tflops_fp32\":%.2f,"
                   "\"compute_tflops_fp16\":%.2f}",
                   props.name, TIMED_RUNS,
                   p50, p90, noisePct, runtimeSecs,
                   ratedGBps, effPct,
                   props.memoryBusWidth,
                   memClockKHz / 1000.0,
                   decodeEffectiveP90,
                   fixedOverheadP50,
                   tf32P90, tf16P90);
            if (dev < deviceCount - 1) printf(",");
            else printf("]\n");
        } else {
            printf("=== Memory Bandwidth Fingerprint ===\n");
            printf("Device : %s  (%.0f GB/s rated)\n", props.name, ratedGBps);
            printf("Bus    : %d-bit @ %.0f MHz\n",
                   props.memoryBusWidth, memClockKHz / 1000.0);
            printf("Buffer : 512 MB read-only  (%d runs)\n", TIMED_RUNS);
            printf("p50    : %.1f GB/s\n", p50);
            printf("p90    : %.1f GB/s  efficiency: %.1f%%\n", p90, effPct);
            printf("tf32   : %.2f TFLOPS\n", tf32P90);
            printf("tf16   : %.2f TFLOPS\n", tf16P90);
            printf("decode : %.1f GB/s effective, %.4f ms fixed dispatch\n",
                   decodeEffectiveP90, fixedOverheadP50);
            printf("noise  : %.1f%%  (p90-p50 spread -- lower is better)\n", noisePct);
            printf("runtime: %.2fs\n", runtimeSecs);
            if (dev < deviceCount - 1) printf("\n");
        }

        cudaFree(dSrc);
        cudaFree(dSink);
        cudaFree(dComputeSink);
        cudaEventDestroy(evStart);
        cudaEventDestroy(evStop);
    }

    return 0;
}
