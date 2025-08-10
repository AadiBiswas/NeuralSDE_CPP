#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>
#include <limits>

#include "../include/data_loader.hpp"
#include <tiny_dnn/tiny_dnn.h>
#include "../include/neural.hpp"

namespace fs = std::filesystem;

// -------- Utility: split into train/val --------
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

// -------- Convert Dataset to tiny-dnn tensors --------
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

// -------- Parse helpers --------
static inline bool to_bool(const std::string& s, bool def=false) {
    if (s == "1" || s == "true"  || s == "True" || s == "TRUE")  return true;
    if (s == "0" || s == "false" || s == "False"|| s == "FALSE") return false;
    return def;
}

static inline std::vector<size_t> parse_hidden(const std::string& s, const std::vector<size_t>& dflt={64,64}) {
    if (s.empty()) return dflt;
    std::vector<size_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(static_cast<size_t>(std::stoul(tok)));
    }
    if (out.empty()) return dflt;
    return out;
}

// -------- Compute validation loss (MSE) over val set --------
float compute_val_loss(tiny_dnn::network<tiny_dnn::sequential>& net,
                       const std::vector<tiny_dnn::vec_t>& Xv,
                       const std::vector<tiny_dnn::vec_t>& yv) {
    float loss = 0.0f;
    for (size_t i = 0; i < Xv.size(); ++i) {
        auto y_pred = net.predict(Xv[i]);
        float diff = y_pred[0] - yv[i][0];
        loss += diff * diff;
    }
    return loss / static_cast<float>(Xv.size());
}

// -------- Build standardized feature vector from RAW window using x-scaler --------
static inline tiny_dnn::vec_t standardize_window(const std::vector<float>& raw_win,
                                                 const StandardScaler& xscaler) {
    tiny_dnn::vec_t v(raw_win.size());
    for (size_t j = 0; j < raw_win.size(); ++j) {
        float m = (j < xscaler.mean.size() ? xscaler.mean[j] : 0.f);
        float s = (j < xscaler.std.size()  ? xscaler.std[j]  : 1.f);
        v[j] = (raw_win[j] - m) / (s == 0.f ? 1.f : s);
    }
    return v;
}

// -------- Args --------
struct Args {
    std::string data = "gbm_path.csv";
    int window = 20;
    int epochs = 50;
    int batch = 32;
    float lr = 1e-3f;
    float val_ratio = 0.2f;
    std::string model_out = "models/gbm_mlp.tnn";
    std::string hidden = "64,64";
    std::string act = "relu";
    bool early_stop = true;
    int patience = 10;
    float lr_decay = 0.5f;
    int lr_patience = 5;
    float min_lr = 1e-5f;
    std::string pred_out = "predictions/gbm_val_preds.csv";
    int forecast_horizon = 100;
    std::string forecast_out = "predictions/gbm_val_forecast.csv";
    std::string run_id = ""; // NEW: Optional run ID
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nextf = [&](float& v){ if (i+1<argc) v = std::stof(argv[++i]); };
        auto nexti = [&](int& v){ if (i+1<argc) v = std::stoi(argv[++i]); };
        auto nexts = [&](std::string& v){ if (i+1<argc) v = std::string(argv[++i]); };

        if (k == "--data") nexts(a.data);
        else if (k == "--window") nexti(a.window);
        else if (k == "--epochs") nexti(a.epochs);
        else if (k == "--batch") nexti(a.batch);
        else if (k == "--lr") nextf(a.lr);
        else if (k == "--val_ratio") nextf(a.val_ratio);
        else if (k == "--model_out") nexts(a.model_out);
        else if (k == "--hidden") nexts(a.hidden);
        else if (k == "--act")    nexts(a.act);
        else if (k == "--early_stop") { std::string v; nexts(v); a.early_stop = to_bool(v, true); }
        else if (k == "--patience") nexti(a.patience);
        else if (k == "--lr_decay") nextf(a.lr_decay);
        else if (k == "--lr_patience") nexti(a.lr_patience);
        else if (k == "--min_lr") nextf(a.min_lr);
        else if (k == "--pred_out") nexts(a.pred_out);
        else if (k == "--forecast_horizon") nexti(a.forecast_horizon);
        else if (k == "--forecast_out") nexts(a.forecast_out);
        else if (k == "--run_id") nexts(a.run_id); // NEW
    }
    return a;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    Dataset raw_ds = load_dataset(args.data, args.window);
    StandardScaler xscaler, yscaler;
    Dataset ds = standardize_dataset(raw_ds, xscaler, yscaler);
    save_scaler("logs/xscaler_stats.csv", xscaler);
    save_scaler("logs/yscaler_stats.csv", yscaler);

    std::vector<tiny_dnn::vec_t> X, y;
    dataset_to_tensors(ds, X, y);
    std::vector<tiny_dnn::vec_t> Xtr, ytr, Xv, yv;
    train_val_split(X, y, args.val_ratio, Xtr, ytr, Xv, yv);

    auto net = build_mlp(X[0].size(), parse_hidden(args.hidden), args.act);

    tiny_dnn::adam optimizer;
    optimizer.alpha = args.lr;

    std::ofstream val_log("logs/val_loss_log.csv");
    val_log << "epoch,val_loss\n";

    std::ofstream ptrack("logs/full_val_preds.csv", std::ios::app);
    ptrack << "epoch,idx,true,pred\n";

    for (int epoch = 1; epoch <= args.epochs; ++epoch) {
        net.train<tiny_dnn::mse>(optimizer, Xtr, ytr, args.batch, 1);

        float val_loss = compute_val_loss(net, Xv, yv);
        val_log << epoch << "," << val_loss << "\n";

        for (size_t i = 0; i < Xv.size(); ++i) {
            auto yhat = net.predict(Xv[i]);
            float y_true = yscaler.inverse_single(yv[i][0]);
            float y_pred = yscaler.inverse_single(yhat[0]);
            ptrack << epoch << "," << i << "," << y_true << "," << y_pred << "\n";
        }
    }

    val_log.close();
    ptrack.close();

    net.save(args.model_out);

    // -------- If run_id is provided, move outputs to dashboard-friendly structure --------
    if (!args.run_id.empty()) {
        fs::create_directories("logs/" + args.run_id);
        fs::copy_file("logs/val_loss_log.csv", "logs/" + args.run_id + "/val_loss_log.csv", fs::copy_options::overwrite_existing);
        fs::copy_file("logs/full_val_preds.csv", "logs/" + args.run_id + "/full_val_preds.csv", fs::copy_options::overwrite_existing);

        if (fs::exists(args.forecast_out)) {
            std::string forecast_renamed = "predictions/" + args.run_id + "_forecast.csv";
            fs::copy_file(args.forecast_out, forecast_renamed, fs::copy_options::overwrite_existing);
        }
    }

    return 0;
}
