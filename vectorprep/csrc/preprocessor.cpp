/*
 * preprocessor.cpp: C ABI kernels for VectorPrep (loaded with ctypes).
 *
 * v2 changes vs v1:
 *   - 64-bit sizes (v1 used int, which overflows past 2^31 elements);
 *   - double accumulators for sums of squares and dot products (v1 accumulated in
 *     float, losing ~3-4 digits on long vectors);
 *   - all kernels operate on caller-owned buffers; the Python layer copies first
 *     unless the caller explicitly asks for in-place operation.
 */
#include <cmath>
#include <cstdint>

#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

EXPORT int32_t vectorprep_abi_version() { return 2; }

EXPORT void scale_minmax(float* data, int64_t size, float target_min, float target_max) {
    if (size <= 0 || !data) return;
    float lo = data[0], hi = data[0];
    for (int64_t i = 1; i < size; ++i) {
        const float v = data[i];
        lo = v < lo ? v : lo;
        hi = v > hi ? v : hi;
    }
    const double range = static_cast<double>(hi) - lo;
    if (range == 0.0) {
        for (int64_t i = 0; i < size; ++i) data[i] = target_min;
        return;
    }
    const double k = (static_cast<double>(target_max) - target_min) / range;
    for (int64_t i = 0; i < size; ++i) data[i] = static_cast<float>(target_min + (data[i] - lo) * k);
}

EXPORT void normalize_l2(float* data, int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0 || !data) return;
    for (int64_t r = 0; r < rows; ++r) {
        float* row = data + r * cols;
        double ss = 0.0;
        for (int64_t c = 0; c < cols; ++c) ss += static_cast<double>(row[c]) * row[c];
        if (ss > 0.0) {
            const float inv = static_cast<float>(1.0 / std::sqrt(ss));
            for (int64_t c = 0; c < cols; ++c) row[c] *= inv;
        }
    }
}

EXPORT double vector_dot_product(const float* a, const float* b, int64_t size) {
    if (size <= 0 || !a || !b) return 0.0;
    double dot = 0.0;
    for (int64_t i = 0; i < size; ++i) dot += static_cast<double>(a[i]) * b[i];
    return dot;
}

}  // extern "C"
