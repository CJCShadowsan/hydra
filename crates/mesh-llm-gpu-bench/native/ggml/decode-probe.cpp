#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#if defined(MESH_LLM_GGML_PROBE_METAL)
#include "ggml-metal.h"
#endif

#if defined(MESH_LLM_GGML_PROBE_CUDA)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int64_t DECODE_ROWS = 4096;
constexpr int64_t DECODE_COLS = 4096;
constexpr int WARMUP_RUNS = 3;
constexpr int TIMED_RUNS = 20;

enum ProbeBackend {
    PROBE_BACKEND_METAL = 0,
    PROBE_BACKEND_CUDA = 1,
    PROBE_BACKEND_HIP = 2,
};

struct ProbeResult {
    const char * name;
    const char * tensor_type;
    double effective_gbps;
    double tflops;
};

char * copy_c_string(const std::string & value) {
    char * out = static_cast<char *>(std::malloc(value.size() + 1));
    if (out != nullptr) {
        std::memcpy(out, value.c_str(), value.size() + 1);
    }
    return out;
}

void set_error(char ** error_out, const std::string & message) {
    if (error_out != nullptr) {
        *error_out = copy_c_string(message);
    }
}

ggml_backend_t init_backend(int backend_kind) {
    switch (backend_kind) {
        case PROBE_BACKEND_METAL:
#if defined(MESH_LLM_GGML_PROBE_METAL)
            return ggml_backend_metal_init();
#else
            return nullptr;
#endif
        case PROBE_BACKEND_CUDA:
#if defined(MESH_LLM_GGML_PROBE_CUDA)
            return ggml_backend_cuda_init(0);
#else
            return nullptr;
#endif
        case PROBE_BACKEND_HIP:
#if defined(MESH_LLM_GGML_PROBE_CUDA)
            return ggml_backend_cuda_init(0);
#else
            return nullptr;
#endif
        default:
            return nullptr;
    }
}

std::vector<float> deterministic_f32(int64_t count, uint32_t salt) {
    std::vector<float> values(static_cast<size_t>(count));
    uint32_t state = 0x9e3779b9u ^ salt;
    for (int64_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        const float centered = static_cast<float>((state >> 8) & 0xffffu) / 32768.0f - 1.0f;
        values[static_cast<size_t>(i)] = centered * 0.125f;
    }
    return values;
}

std::vector<uint8_t> encode_weights(enum ggml_type type, const std::vector<float> & weights) {
    const size_t encoded_bytes = ggml_row_size(type, DECODE_COLS) * DECODE_ROWS;
    std::vector<uint8_t> encoded(encoded_bytes);
    if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(
            weights.data(),
            reinterpret_cast<ggml_fp16_t *>(encoded.data()),
            static_cast<int64_t>(weights.size()));
        return encoded;
    }
    ggml_quantize_chunk(type, weights.data(), encoded.data(), 0, DECODE_ROWS, DECODE_COLS, nullptr);
    return encoded;
}

double percentile_p90(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(TIMED_RUNS * 0.90) - 1;
    return values[index];
}

bool run_probe(
    ggml_backend_t backend,
    enum ggml_type type,
    const char * name,
    const char * tensor_type,
    ProbeResult & result) {
    const size_t context_bytes = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * weights = ggml_new_tensor_2d(ctx, type, DECODE_COLS, DECODE_ROWS);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DECODE_COLS, 1);
    ggml_tensor * output = ggml_mul_mat(ctx, weights, input);
    ggml_set_name(weights, "ggml_decode_probe_weights");
    ggml_set_name(input, "ggml_decode_probe_input");
    ggml_set_name(output, "ggml_decode_probe_output");
    ggml_set_output(output);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    if (!ggml_backend_supports_op(backend, output)) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        return false;
    }

    std::vector<float> weight_f32 = deterministic_f32(DECODE_ROWS * DECODE_COLS, 17);
    std::vector<uint8_t> weight_encoded = encode_weights(type, weight_f32);
    std::vector<float> input_f32 = deterministic_f32(DECODE_COLS, 29);
    ggml_backend_tensor_set(weights, weight_encoded.data(), 0, weight_encoded.size());
    ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
    ggml_backend_synchronize(backend);

    auto compute_once = [&]() -> double {
        const auto started = std::chrono::steady_clock::now();
        enum ggml_status status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        const auto finished = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            return 0.0;
        }
        return std::chrono::duration<double>(finished - started).count();
    };

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        if (compute_once() <= 0.0) {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            return false;
        }
    }

    std::vector<double> seconds;
    seconds.reserve(TIMED_RUNS);
    for (int i = 0; i < TIMED_RUNS; ++i) {
        const double elapsed = compute_once();
        if (elapsed <= 0.0) {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            return false;
        }
        seconds.push_back(elapsed);
    }

    const double p90_seconds = percentile_p90(seconds);
    const double bytes = static_cast<double>(weight_encoded.size())
        + static_cast<double>(input_f32.size() * sizeof(float))
        + static_cast<double>(DECODE_ROWS * sizeof(float));
    const double flops = 2.0 * static_cast<double>(DECODE_ROWS) * static_cast<double>(DECODE_COLS);
    result = ProbeResult{
        name,
        tensor_type,
        bytes / p90_seconds / 1e9,
        flops / p90_seconds / 1e12,
    };

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

std::string results_json(const std::vector<ProbeResult> & results) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < results.size(); ++i) {
        const ProbeResult & result = results[i];
        if (i > 0) {
            out << ",";
        }
        out << "{\"name\":\"" << result.name << "\","
            << "\"tensor_type\":\"" << result.tensor_type << "\","
            << "\"rows\":" << DECODE_ROWS << ","
            << "\"cols\":" << DECODE_COLS << ","
            << "\"batch_tokens\":1,"
            << "\"effective_gbps\":" << result.effective_gbps << ","
            << "\"tflops\":" << result.tflops << ","
            << "\"runs\":" << TIMED_RUNS << "}";
    }
    out << "]";
    return out.str();
}

} // namespace

extern "C" char * mesh_llm_gpu_bench_ggml_decode_probe_json(int backend_kind, char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML decode probe backend is not available");
        return nullptr;
    }

    std::vector<ProbeResult> results;
    ProbeResult result{};
    if (run_probe(backend, GGML_TYPE_F16, "ggml_decode_f16_matvec", "f16", result)) {
        results.push_back(result);
    }
    if (run_probe(backend, GGML_TYPE_Q8_0, "ggml_decode_q8_0_matvec", "q8_0", result)) {
        results.push_back(result);
    }
    if (run_probe(backend, GGML_TYPE_Q4_K, "ggml_decode_q4_k_matvec", "q4_k", result)) {
        results.push_back(result);
    }

    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML decode probe did not produce supported matvec results");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" void mesh_llm_gpu_bench_ggml_decode_probe_free(void * ptr) {
    std::free(ptr);
}
