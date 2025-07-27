#ifndef GBM_HPP
#define GBM_HPP

#include "SDE.hpp"

class GBM : public SDE {
private:
    double mu;
    double sigma;

public:
    GBM(double mu_, double sigma_) : mu(mu_), sigma(sigma_) {}

    double drift(double x, double t) const override {
        return mu * x;
    }

    double diffusion(double x, double t) const override {
        return sigma * x;
    }
};

#endif
