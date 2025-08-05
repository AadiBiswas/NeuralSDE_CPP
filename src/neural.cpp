#include "../include/neural.hpp"

// Convert raw dataset to standardized (x, y) pairs using provided scalers
Dataset standardize_dataset(const Dataset& raw,
                            StandardScaler& xscaler,
                            StandardScaler& yscaler) {
    Dataset std_ds = raw;
    std::vector<tiny_dnn::vec_t> X, y;

    for (const auto& p : raw) {
        X.push_back(p.first);
        y.push_back({p.second});
    }

    xscaler.fit(X);
    yscaler.fit(y);

    xscaler.transform(X);
    yscaler.transform(y);

    for (size_t i = 0; i < std_ds.size(); ++i) {
        std_ds[i].first = X[i];
        std_ds[i].second = y[i][0];
    }

    return std_ds;
}

// Save scaler stats to file
void save_scaler(const std::string& filename, const StandardScaler& scaler) {
    std::ofstream out(filename);
    out << "mean,std\n";
    for (size_t i = 0; i < scaler.mean.size(); ++i) {
        out << scaler.mean[i] << "," << scaler.std[i] << "\n";
    }
}

// Build MLP network (same logic as make_mlp, but exposed to caller)
tiny_dnn::network<tiny_dnn::sequential> build_mlp(size_t input_dim,
                                                  const std::vector<size_t>& hidden,
                                                  const std::string& activation) {
    return make_mlp(input_dim, hidden, activation);
}
