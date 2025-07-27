#include <iostream>
#include <vector>
#include <random>
#include <fstream>

#include "../include/GBM.hpp"

int main() {
    const double dt = 0.01;
    const int steps = 1000;
    const double x0 = 1.0;

    GBM gbm(0.1, 0.2);  // mu = 0.1, sigma = 0.2
    std::vector<double> path(steps);
    path[0] = x0;

    std::default_random_engine gen;
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int i = 1; i < steps; ++i) {
        double t = i * dt;
        double x_prev = path[i - 1];
        double dW = dist(gen) * std::sqrt(dt);
        double dx = gbm.drift(x_prev, t) * dt + gbm.diffusion(x_prev, t) * dW;
        path[i] = x_prev + dx;
    }

    std::ofstream out("gbm_path.csv");
    for (int i = 0; i < steps; ++i) {
        out << i * dt << "," << path[i] << "\n";
    }

    std::cout << "GBM path saved to gbm_path.csv\n";
    return 0;
}
