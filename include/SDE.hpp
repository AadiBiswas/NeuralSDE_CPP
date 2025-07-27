#ifndef SDE_HPP
#define SDE_HPP

class SDE {
public:
    virtual double drift(double x, double t) const = 0;
    virtual double diffusion(double x, double t) const = 0;
    virtual ~SDE() = default;
};

#endif