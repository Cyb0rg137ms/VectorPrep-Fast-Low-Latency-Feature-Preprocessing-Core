/*
 * preprocessor.cpp
 * ================
 * High-performance C++ vector and matrix preprocessing library.
 * Exposes C-compatible bindings for MinMax scaling, L2 normalization, and dot products.
 */

#include <cmath>
#include <iostream>

#if defined(_MSC_VER)
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

    /**
     * Scales an array in-place to the range [min_val, max_val] (MinMax scaling).
     */
    EXPORT void scale_minmax(float* data, int size, float target_min, float target_max) {
        if (size <= 0 || !data) return;

        float current_min = data[0];
        float current_max = data[0];

        // Find min and max
        for (int i = 1; i < size; ++i) {
            if (data[i] < current_min) current_min = data[i];
            if (data[i] > current_max) current_max = data[i];
        }

        float range = current_max - current_min;
        if (range == 0.0f) {
            for (int i = 0; i < size; ++i) {
                data[i] = target_min;
            }
            return;
        }

        float target_range = target_max - target_min;

        // Perform in-place scale transformation
        for (int i = 0; i < size; ++i) {
            data[i] = target_min + ((data[i] - current_min) / range) * target_range;
        }
    }

    /**
     * Normalizes each row of a matrix to have L2 norm = 1 (L2 normalization).
     */
    EXPORT void normalize_l2(float* data, int rows, int cols) {
        if (rows <= 0 || cols <= 0 || !data) return;

        for (int r = 0; r < rows; ++r) {
            float* row = data + (r * cols);
            
            // Calculate sum of squares
            float sum_sq = 0.0f;
            for (int c = 0; c < cols; ++c) {
                sum_sq += row[c] * row[c];
            }

            float norm = std::sqrt(sum_sq);
            if (norm > 0.0f) {
                for (int c = 0; c < cols; ++c) {
                    row[c] /= norm;
                }
            }
        }
    }

    /**
     * Computes the dot product of two arrays.
     */
    EXPORT float vector_dot_product(const float* a, const float* b, int size) {
        if (size <= 0 || !a || !b) return 0.0f;

        float dot = 0.0f;
        for (int i = 0; i < size; ++i) {
            dot += a[i] * b[i];
        }
        return dot;
    }
}
