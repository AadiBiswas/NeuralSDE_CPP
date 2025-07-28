#ifndef OU_HPP
#define OU_HPP

#include "SDE.hpp"

class OU : public SDE {
private:
    double theta;  // Mean reversion speed
    double mu;     // Long-term mean
    double sigma;  // Volatility

public:
    OU(double theta_, double mu_, double sigma_)
        : theta(theta_), mu(mu_), sigma(sigma_) {}

    double drift(double x, double t) const override {
        return theta * (mu - x);
    }

    double diffusion(double x, double t) const override {
        return sigma;
    }
};

#endif
