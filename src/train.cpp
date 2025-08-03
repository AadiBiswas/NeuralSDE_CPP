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

// -------- Build standardized feature vector from RAW window using x-scaler --------
static inline tiny_dnn::vec_t standardize_window(const std::vector<float>& raw_win,
                                                 const StandardScaler& xscaler) {
    tiny_dnn::vec_t v(raw_win.size());
    // xscaler.mean/std are per-column stats (length == window)
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

    // 2.3
    std::string hidden = "64,64";
    std::string act = "relu";
    bool early_stop = true;
    int patience = 10;
    float lr_decay = 0.5f;
    int lr_patience = 5;
    float min_lr = 1e-5f;
    std::string pred_out = "predictions/gbm_val_preds.csv";

    // 2.4
    int forecast_horizon = 100;
    std::string forecast_out = "predictions/gbm_val_forecast.csv";
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
              << "  model_out=" << args.model_out << "\n"
              << "  hidden=" << args.hidden << "\n"
              << "  act=" << args.act << "\n"
              << "  early_stop=" << (args.early_stop ? "true" : "false") << "\n"
              << "  patience=" << args.patience << "\n"
              << "  lr_decay=" << args.lr_decay << "\n"
              << "  lr_patience=" << args.lr_patience << "\n"
              << "  min_lr=" << args.min_lr << "\n"
              << "  pred_out=" << args.pred_out << "\n"
              << "  forecast_horizon=" << args.forecast_horizon << "\n"
              << "  forecast_out=" << args.forecast_out << "\n";

    // Ensure output directories exist
    if (!args.model_out.empty()) {
        fs::path mp(args.model_out);
        if (!mp.parent_path().empty()) fs::create_directories(mp.parent_path());
    }
    if (!args.pred_out.empty()) {
        fs::path pp(args.pred_out);
        if (!pp.parent_path().empty()) fs::create_directories(pp.parent_path());
    }
    if (!args.forecast_out.empty()) {
        fs::path fp(args.forecast_out);
        if (!fp.parent_path().empty()) fs::create_directories(fp.parent_path());
    }

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
    const size_t n_train = Xtr.size();
    std::cout << "Train samples: " << Xtr.size() << ", Val samples: " << Xv.size() << "\n";

    // 4) Standardize inputs and outputs
    StandardScaler xscaler;
    xscaler.fit(Xtr);
    xscaler.transform(Xtr);
    xscaler.transform(Xv);

    StandardScaler yscaler;
    yscaler.fit(ytr);
    yscaler.transform(ytr);
    yscaler.transform(yv);

    // 5) Build model
    const size_t input_dim = static_cast<size_t>(args.window);
    auto hidden = parse_hidden(args.hidden, {64,64});
    auto net = make_mlp(input_dim, hidden, args.act);

    // 6) Optimizer
    adam optimizer;
    optimizer.alpha = args.lr;

    auto mse_metric = [](const vec_t& y_pred, const vec_t& y_true) {
        float loss = 0.f;
        for (size_t i = 0; i < y_pred.size(); ++i) {
            float d = static_cast<float>(y_pred[i]) - static_cast<float>(y_true[i]);
            loss += d * d;
        }
        return loss / static_cast<float>(y_pred.size());
    };

    // 7) Training loop with early stopping + LR decay
    size_t steps_per_epoch = (Xtr.size() + args.batch - 1) / args.batch;
    float best_val = std::numeric_limits<float>::infinity();
    int no_improve = 0;
    int lr_wait = 0;

    for (int epoch = 1; epoch <= args.epochs; ++epoch) {
        // Shuffle
        std::vector<size_t> idx(Xtr.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), std::mt19937(std::random_device{}()));

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

            net.fit<mse>(optimizer, bx, by, 1, 1);

            float batch_loss = 0.f;
            for (size_t i = 0; i < bx.size(); ++i) {
                vec_t pred = net.predict(bx[i]);
                batch_loss += mse_metric(pred, by[i]);
            }
            epoch_loss += batch_loss;
            seen += (end - start);
        }
        epoch_loss /= static_cast<float>(std::max<size_t>(1, seen));

        // Validation loss
        float val_loss = 0.f;
        for (size_t i = 0; i < Xv.size(); ++i) {
            vec_t pred = net.predict(Xv[i]);
            val_loss += mse_metric(pred, yv[i]);
        }
        val_loss /= static_cast<float>(std::max<size_t>(1, Xv.size()));

        bool improved = val_loss + 1e-12f < best_val;
        if (improved) {
            best_val = val_loss;
            no_improve = 0;
            lr_wait = 0;
            try { net.save(args.model_out); } catch (...) {}
        } else {
            no_improve++;
            lr_wait++;
        }

        if (lr_wait >= args.lr_patience && optimizer.alpha > args.min_lr) {
            optimizer.alpha = std::max(args.min_lr, optimizer.alpha * args.lr_decay);
            lr_wait = 0;
            std::cout << "  lr decayed to " << optimizer.alpha << "\n";
        }

        std::cout << "Epoch " << epoch
                  << " | train_mse=" << epoch_loss
                  << " | val_mse=" << val_loss
                  << (improved ? "  (best ✓)" : "") << "\n";

        if (args.early_stop && no_improve >= args.patience) {
            std::cout << "Early stopping triggered (patience=" << args.patience << ")\n";
            break;
        }
    }

    // Save scalers next to model
    try {
        fs::path mp(args.model_out);
        fs::path mdir = mp.parent_path().empty() ? fs::path(".") : mp.parent_path();

        std::ofstream fx(mdir / "xscaler_stats.csv");
        fx << "mean,std\n";
        for (size_t j = 0; j < xscaler.mean.size(); ++j) {
            fx << xscaler.mean[j] << "," << xscaler.std[j] << "\n";
        }
        fx.close();

        std::ofstream fy(mdir / "yscaler_stats.csv");
        fy << "mean,std\n";
        for (size_t j = 0; j < yscaler.mean.size(); ++j) {
            fy << yscaler.mean[j] << "," << yscaler.std[j] << "\n";
        }
        fy.close();

        std::cout << "Scaler stats saved next to model\n";
    } catch (...) {}

    // Reload best model (if exists) for evaluation
    try { net.load(args.model_out); } catch (...) {}

    // Export one-step predictions on validation set (inverse-scaled)
    if (!args.pred_out.empty() && !Xv.empty()) {
        fs::path pp(args.pred_out);
        if (!pp.parent_path().empty()) fs::create_directories(pp.parent_path());

        std::ofstream pout(pp);
        pout << "idx,true,pred\n";
        for (size_t i = 0; i < Xv.size(); ++i) {
            vec_t yhat = net.predict(Xv[i]);
            float y_true = yscaler.inverse_single(yv[i][0]);
            float y_pred = yscaler.inverse_single(yhat[0]);
            pout << i << "," << y_true << "," << y_pred << "\n";
        }
        pout.close();
        std::cout << "Validation predictions saved to " << pp << "\n";
    }

    // -------- 2.4: Recursive multi-step forecast over validation tail --------
    if (args.forecast_horizon > 0 && !args.forecast_out.empty() && Xv.size() > 0) {
        // Use the FIRST validation window as starting point, in RAW domain
        // Reconstruct RAW window from ds[n_train].first
        if (n_train < ds.size()) {
            std::vector<float> curr_win_raw = ds[n_train].first; // length == window
            size_t steps = static_cast<size_t>(args.forecast_horizon);
            size_t max_steps_with_truth = std::min(steps, ds.size() - n_train); // we have ground truth for these

            fs::path fp(args.forecast_out);
            if (!fp.parent_path().empty()) fs::create_directories(fp.parent_path());
            std::ofstream fcsv(fp);
            fcsv << "step,true,pred\n";

            for (size_t h = 0; h < steps; ++h) {
                // Standardize current RAW window using x-scaler
                tiny_dnn::vec_t x_std = standardize_window(curr_win_raw, xscaler);
                tiny_dnn::vec_t yhat_std = net.predict(x_std);
                float yhat_raw = yscaler.inverse_single(yhat_std[0]);

                float y_true = std::numeric_limits<float>::quiet_NaN();
                size_t i_val = n_train + h;
                if (i_val < ds.size()) {
                    y_true = ds[i_val].second; // ground truth next point for this window
                }

                fcsv << (h+1) << "," << y_true << "," << yhat_raw << "\n";

                // Roll the window with predicted RAW value
                curr_win_raw.erase(curr_win_raw.begin());
                curr_win_raw.push_back(yhat_raw);
            }

            fcsv.close();
            std::cout << "Recursive forecast saved to " << fp << "\n";
        }
    }

    return 0;
}
