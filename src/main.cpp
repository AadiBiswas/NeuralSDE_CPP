#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>

#include "../include/GBM.hpp"
#include "../include/OU.hpp"

void simulate(const SDE& process, const std::string& filename, double x0 = 1.0, double dt = 0.01, int steps = 1000) {
    std::vector<double> path(steps);
    path[0] = x0;

    std::default_random_engine gen;
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int i = 1; i < steps; ++i) {
        double t = i * dt;
        double x_prev = path[i - 1];
        double dW = dist(gen) * std::sqrt(dt);
        double dx = process.drift(x_prev, t) * dt + process.diffusion(x_prev, t) * dW;
        path[i] = x_prev + dx;
    }

    std::ofstream out(filename);
    for (int i = 0; i < steps; ++i) {
        out << i * dt << "," << path[i] << "\n";
    }

    std::cout << filename << " written.\n";
}

int main() {
    GBM gbm(0.1, 0.2);                      // mu = 0.1, sigma = 0.2
    OU ou(1.5, 0.5, 0.1);                   // theta = 1.5, mu = 0.5, sigma = 0.1

    simulate(gbm, "gbm_path.csv");
    simulate(ou,  "ou_path.csv");

    return 0;
}
