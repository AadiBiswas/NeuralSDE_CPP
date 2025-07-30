#ifndef NEURAL_HPP
#define NEURAL_HPP

#include <tiny_dnn/tiny_dnn.h>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>

using VecF = std::vector<float>;

// Simple feature scaler (per-feature mean/std)
struct StandardScaler {
    std::vector<float> mean;
    std::vector<float> std;

    void fit(const std::vector<tiny_dnn::vec_t>& X) {
        if (X.empty()) return;
        size_t n = X.size();
        size_t d = X[0].size();
        mean.assign(d, 0.f);
        std.assign(d, 0.f);

        // mean
        for (const auto& row : X) {
            for (size_t j = 0; j < d; ++j) mean[j] += static_cast<float>(row[j]);
        }
        for (size_t j = 0; j < d; ++j) mean[j] /= static_cast<float>(n);

        // std
        for (const auto& row : X) {
            for (size_t j = 0; j < d; ++j) {
                float diff = static_cast<float>(row[j]) - mean[j];
                std[j] += diff * diff;
            }
        }
        for (size_t j = 0; j < d; ++j) {
            std[j] = std[j] > 0.f ? std::sqrt(std[j] / static_cast<float>(n)) : 1.f;
        }
    }

    void transform(std::vector<tiny_dnn::vec_t>& X) const {
        if (X.empty()) return;
        size_t d = X[0].size();
        for (auto& row : X) {
            for (size_t j = 0; j < d; ++j) {
                row[j] = (static_cast<float>(row[j]) - mean[j]) / (std[j] == 0.f ? 1.f : std[j]);
            }
        }
    }

    // For single row
    tiny_dnn::vec_t transform(const tiny_dnn::vec_t& row) const {
        tiny_dnn::vec_t out(row.size());
        for (size_t j = 0; j < row.size(); ++j) {
            out[j] = (static_cast<float>(row[j]) - mean[j]) / (std[j] == 0.f ? 1.f : std[j]);
        }
        return out;
    }
};

// Create a simple MLP for regression: input -> hidden... -> 1
inline tiny_dnn::network<tiny_dnn::sequential>
make_mlp(size_t input_dim,
         const std::vector<size_t>& hidden = {64, 64},
         tiny_dnn::activation_type act = tiny_dnn::activation_type::relu) {

    using namespace tiny_dnn;
    network<sequential> net;
    size_t prev = input_dim;

    for (size_t h : hidden) {
        net << fully_connected_layer(prev, h)
            << activation_layer(act);
        prev = h;
    }

    net << fully_connected_layer(prev, 1); // linear output for regression
    return net;
}

#endif
