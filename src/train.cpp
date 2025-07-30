#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <algorithm>

#include "../include/data_loader.hpp"
#include <tiny_dnn/tiny_dnn.h>
#include "../include/neural.hpp"

// Utility: split into train/val
template<typename T>
void train_val_split(const std::vector<T>& X, const std::vector<T>& y,
                     float val_ratio,
                     std::vector<T>& Xtr, std::vector<T>& ytr,
                     std::vector<T>& Xv,  std::vector<T>& yv) {
    size_t n = X.size();
    size_t n_val = static_cast<size_t>(val_ratio * n);
    size_t n_train = n - n_val;
    Xtr.assign(X.begin(), X.begin() + n_train);
    ytr.assign(y.begin(), y.begin() + n_train);
    Xv.assign(X.begin() + n_train, X.end());
    yv.assign(y.begin() + n_train, y.end());
}

// Convert Dataset (vector<pair<Sample, float>>) to tiny-dnn tensors
void dataset_to_tensors(const Dataset& ds,
                        std::vector<tiny_dnn::vec_t>& X,
                        std::vector<tiny_dnn::vec_t>& y) {
    X.clear(); y.clear();
    X.reserve(ds.size()); y.reserve(ds.size());
    for (const auto& ex : ds) {
        tiny_dnn::vec_t xi(ex.first.begin(), ex.first.end());
        tiny_dnn::vec_t yi(1);
        yi[0] = ex.second;
        X.push_back(std::move(xi));
        y.push_back(std::move(yi));
    }
}

// Basic CLI parsing
struct Args {
    std::string data = "gbm_path.csv";
    int window = 20;
    int epochs = 50;
    int batch = 32;
    float lr = 1e-3f;
    float val_ratio = 0.2f;
    std::string model_out = "models/gbm_mlp.tnn";
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](float& v){ if (i+1<argc) v = std::stof(argv[++i]); };
        auto nexti = [&](int& v){ if (i+1<argc) v = std::stoi(argv[++i]); };
        auto nexts = [&](std::string& v){ if (i+1<argc) v = std::string(argv[++i]); };

        if (k == "--data") nexts(a.data);
        else if (k == "--window") nexti(a.window);
        else if (k == "--epochs") nexti(a.epochs);
        else if (k == "--batch") nexti(a.batch);
        else if (k == "--lr") next(a.lr);
        else if (k == "--val_ratio") next(a.val_ratio);
        else if (k == "--model_out") nexts(a.model_out);
    }
    return a;
}

int main(int argc, char** argv) {
    using namespace tiny_dnn;

    Args args = parse_args(argc, argv);
    std::cout << "Training with:\n"
              << "  data=" << args.data << "\n"
              << "  window=" << args.window << "\n"
              << "  epochs=" << args.epochs << "\n"
              << "  batch=" << args.batch << "\n"
              << "  lr=" << args.lr << "\n"
              << "  val_ratio=" << args.val_ratio << "\n"
              << "  model_out=" << args.model_out << "\n";

    // 1) Load dataset
    Dataset ds = load_dataset(args.data, args.window);
    if (ds.empty()) {
        std::cerr << "No data loaded. Exiting.\n";
        return 1;
    }

    // 2) Convert to tensors
    std::vector<vec_t> X, y;
    dataset_to_tensors(ds, X, y);

    // 3) Train/Val split
    std::vector<vec_t> Xtr, ytr, Xv, yv;
    train_val_split(X, y, args.val_ratio, Xtr, ytr, Xv, yv);
    std::cout << "Train samples: " << Xtr.size() << ", Val samples: " << Xv.size() << "\n";

    // 4) Standardize inputs (and optionally outputs)
    StandardScaler xscaler;
    xscaler.fit(Xtr);
    xscaler.transform(Xtr);
    xscaler.transform(Xv);

    // Optional: scale outputs (helps stability for certain SDEs)
    StandardScaler yscaler;
    yscaler.fit(ytr);
    yscaler.transform(ytr);
    yscaler.transform(yv);

    // 5) Build model
    const size_t input_dim = static_cast<size_t>(args.window);
    auto net = make_mlp(input_dim, {64, 64}, activation_type::relu);

    // 6) Optimizer + loss
    adam optimizer;
    optimizer.alpha = args.lr;

    // tiny-dnn fit() expects data as vector<vec_t>
    // Use MSE loss for regression
    auto mse = [](const vec_t& y_pred, const vec_t& y_true) {
        float loss = 0.f;
        for (size_t i = 0; i < y_pred.size(); ++i) {
            float d = static_cast<float>(y_pred[i]) - static_cast<float>(y_true[i]);
            loss += d * d;
        }
        return loss / static_cast<float>(y_pred.size());
    };

    // 7) Training loop (manual epochs to log val loss)
    size_t steps_per_epoch = (Xtr.size() + args.batch - 1) / args.batch;

    for (int epoch = 1; epoch <= args.epochs; ++epoch) {
        // Shuffle indices
        std::vector<size_t> idx(Xtr.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), std::mt19937(std::random_device{}()));

        // Mini-batch SGD
        float epoch_loss = 0.f;
        size_t seen = 0;
        for (size_t step = 0; step < steps_per_epoch; ++step) {
            size_t start = step * args.batch;
            size_t end = std::min(start + (size_t)args.batch, Xtr.size());
            if (start >= end) break;

            std::vector<vec_t> bx, by;
            bx.reserve(end - start);
            by.reserve(end - start);
            for (size_t k = start; k < end; ++k) {
                bx.push_back(Xtr[idx[k]]);
                by.push_back(ytr[idx[k]]);
            }

            // Forward + backward with mean_squared_error
            net.fit<mse_loss>(optimizer, bx, by, 1, 1); // one batch, one epoch inside

            // Track batch loss (compute explicitly)
            float batch_loss = 0.f;
            for (size_t i = 0; i < bx.size(); ++i) {
                vec_t pred = net.predict(bx[i]);
                batch_loss += mse(pred, by[i]);
            }
            epoch_loss += batch_loss;
            seen += (end - start);
        }
        epoch_loss /= static_cast<float>(seen);

        // Validation loss
        float val_loss = 0.f;
        for (size_t i = 0; i < Xv.size(); ++i) {
            vec_t pred = net.predict(Xv[i]);
            val_loss += mse(pred, yv[i]);
        }
        val_loss /= static_cast<float>(std::max<size_t>(1, Xv.size()));

        std::cout << "Epoch " << epoch
                  << " | train_mse=" << epoch_loss
                  << " | val_mse=" << val_loss << "\n";
    }

    // 8) Save model and (optionally) scalers
    // tiny-dnn supports save/load; we save to .tnn
    // Ensure output directory exists or use current dir.
    try {
        net.save(args.model_out);
        std::cout << "Model saved to " << args.model_out << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Failed to save model: " << e.what() << "\n";
    }

    // Save scaler stats (simple CSV)
    std::ofstream fx("xscaler_stats.csv");
    fx << "mean,std\n";
    for (size_t j = 0; j < xscaler.mean.size(); ++j) {
        fx << xscaler.mean[j] << "," << xscaler.std[j] << "\n";
    }
    fx.close();

    std::ofstream fy("yscaler_stats.csv");
    fy << "mean,std\n";
    for (size_t j = 0; j < yscaler.mean.size(); ++j) {
        fy << yscaler.mean[j] << "," << yscaler.std[j] << "\n";
    }
    fy.close();

    std::cout << "Scaler stats saved to xscaler_stats.csv and yscaler_stats.csv\n";
    return 0;
}
