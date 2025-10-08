# Convert the C++ function to Python (NumPy, vectorized), then plot and benchmark.

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numba
import cppyy

def benchmark_func(func, name, elems):
    t0 = time.perf_counter()
    y = [func(x) for x in elems]
    t1 = time.perf_counter()
    N = len(elems)
    return {
        "method": name,
        "N": N,
        "seconds": t1 - t0,
        "M evals/s": (N / (t1 - t0)) / 1e6,
        "checksum": float(np.sum(y))
    }


w = 16
sigma = 2.0
W = w * sigma


def kaiser_bessel(dx, w=16, sigma=2):
    """
    dx: scalar or array
    w: kernel width parameter
    sigma: oversampling factor
    """
    W = w * sigma
    a = (2.0 * dx) / W
    beta = np.pi * w * (1.0 - 1.0 / (2.0 * sigma))
    t = np.sqrt(1.0 - a * a)
    return float(np.i0(beta * t) / np.i0(beta))


def kaiser_bessel_poly(deg=16, samples=4097, domain=(0.0, 1.0)):
    """
    Fit Chebyshev polynomial to kaiser_bessel on [domain[0], domain[1]].
    Returns a numpy.polynomial.Chebyshev object.
    """
    from numpy.polynomial import Chebyshev as Ch
    x = np.linspace(domain[0], domain[1], samples)
    y = [ kaiser_bessel(x) for x in x ]
    return Ch.fit(x, y, deg, domain=domain)

@numba.njit
def i0(x: float) -> float:
    """
    CHATGPT WARNING!
    Modified Bessel function of the first kind, order 0.
    Pure Python, ~1e-15 relative accuracy for real x until exp overflow (~709).
    """
    ax = abs(x)
    if ax == 0.0:
        return 1.0

    # Small/medium x: series  I0(x) = sum_{k>=0} (x^2/4)^k / (k!)^2
    if ax <= 15.0:
        y = 0.25 * ax * ax
        t = 1.0
        s = 1.0
        k = 0
        while True:
            k += 1
            t *= y / (k * k)
            s_next = s + t
            if t < 1e-16 * s_next:
                return s_next
            s = s_next

    # Large x: asymptotic  I0(x) ~ e^x / sqrt(2πx) * sum_{k>=0} ((2k-1)!!)^2 / (k! (8x)^k)
    inv_sqrt2pi = 0.39894228040143267793994605993438  # 1/sqrt(2π)
    s = 1.0
    t = 1.0
    k = 0
    while True:
        k += 1
        t *= ((2*k - 1)**2) / (k * 8.0 * ax)
        s_next = s + t
        if t < 1e-16 * s_next:
            s = s_next
            break
        s = s_next

    return np.exp(ax) * (inv_sqrt2pi / np.sqrt(ax)) * s


@numba.njit
def kaiser_bessel_jit(dx, w=16, sigma=2):
    W = w * sigma
    a = (2.0 * dx) / W
    beta = np.pi * w * (1.0 - 1.0 / (2.0 * sigma))
    t = np.sqrt(1.0 - a * a)
    return i0(beta * t) / i0(beta)

cppyy.cppdef(r"""
#pragma GCC optimize ("O3")
#pragma GCC target("native") 

#include <cmath>
double kaiser_bessel(double dx, double w, double sigma) {
    const double W = w * sigma;
    const double a = (2.0 * dx) / W;
    const double pi = 3.14159265358979323846;
    const double beta = pi * w * (1.0 - 1.0 / (2.0 * sigma));
    const double t = std::sqrt(1.0 - a * a);
    const auto i0 = [](double x) { return std::cyl_bessel_i(0.0, x); };
    return i0(beta * t) / i0(beta);
}
""")


def kaiser_bessel_cpp(dx, w=16.0, sigma=2.0):
    return float(cppyy.gbl.kaiser_bessel(float(dx), float(w), float(sigma)))

cppyy.cppdef(r"""
#pragma GCC optimize ("O3")
#pragma GCC target("native")

#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>

constexpr double PI = 3.14159265358979323846;

template <class Func> class Cheb {
public:
    Cheb(Func F, const int n, const double a = -1, const double b = 1)
        : nodes(n), low(b - a), hi(b + a), coeffs(nodes) {

        std::vector<double> fvals(nodes);

        for (int k = 0; k < nodes; ++k) {
            double theta = (2 * k + 1) * PI / (2 * nodes);
            double xk = std::cos(theta);
            double x_mapped = map_to_domain(xk);
            fvals[k] = F(x_mapped);
        }

        for (int m = 0; m < nodes; ++m) {
            double sum = 0.0;
            for (int k = 0; k < nodes; ++k) {
                double theta = (2 * k + 1) * PI / (2 * nodes);
                sum += fvals[k] * std::cos(m * theta);
            }
            coeffs[m] = (2.0 / nodes) * sum;
        }

        coeffs[0] *= 0.5;
        std::reverse(coeffs.begin(), coeffs.end());
    }

#pragma GCC push_options
#pragma GCC optimize ("O3","fast-math")
#pragma GCC target ("native")
    double operator()(const double pt) const {
        const double x = map_from_domain(pt);
        const double x2 = 2 * x;

        double c0 = coeffs[0];
        double c1 = coeffs[1];

        for (int i = 2; i < nodes; ++i) {
            const double tmp = c1;
            c1 = coeffs[i] - c0;
            c0 = c0 * x2 + tmp;
        }

        return c1 + c0 * x;
    }
#pragma GCC pop_options

private:
    const int nodes;
    double low, hi;
    std::vector<double> coeffs;

    template <class T> constexpr auto map_to_domain(const T x) const { return 0.5 * (low * x + hi); }
    constexpr double map_from_domain(double x) const { return (2.0 * x - hi) / low; }
};

// Expose a concrete instantiation taking a Python callable via std::function
using ChebFn = Cheb<std::function<double(double)>>;

// Factory for convenience (also allows direct ChebFn(...) construction from Python)
inline ChebFn make_cheb(std::function<double(double)> f, int n, double a=-1.0, double b=1.0) {
    return ChebFn(std::move(f), n, a, b);
}
""")

# Python usage
make_cheb = cppyy.gbl.make_cheb

kaiser_bessel_cheb = make_cheb(lambda x: kaiser_bessel(x), 16, 0, 1,)


a = 0
b = 1

# ---- Plot ----
plt.figure()

dx_plot = np.linspace(a, b, 1000)
y_plot = [kaiser_bessel(x) for x in dx_plot]
plt.plot(dx_plot, y_plot, label="kaiser_bessel")


np_cheb = kaiser_bessel_poly()
y_plot = [np_cheb(x) for x in dx_plot]
plt.plot(dx_plot, y_plot, label="numpy cheb")

y_plot = [kaiser_bessel_jit(x) for x in dx_plot]
plt.plot(dx_plot, y_plot, label="kaiser_bessel_jit")

y_plot = [kaiser_bessel_cpp(x) for x in dx_plot]
plt.plot(dx_plot, y_plot, label="kaiser_bessel_cpp")

y_plot = [kaiser_bessel_cheb(x) for x in dx_plot]
plt.plot(dx_plot, y_plot, label="kaiser_bessel_cheb")

plt.xlabel("dx")
plt.ylabel("value")
plt.title("Kaiser–Bessel kernel (w=%.1f, sigma=%.1f)" % (w, sigma))
plt.legend()
plt.show()

# ---- Benchmark ----
results = []

# Vectorized benchmark on 2,000,000 samples within support
N = 10_000

rng = np.random.default_rng(42)
dx = rng.uniform(a, b, size=N)

result = benchmark_func(kaiser_bessel, "numpy", dx)
results.append(result)

result = benchmark_func(np_cheb, "numpy cheb", dx)
results.append(result)

result_jit = benchmark_func(kaiser_bessel_jit, "numba_jit", dx)
results.append(result_jit)

results_cpp = benchmark_func(kaiser_bessel_cpp, "cppyy c++", dx)
results.append(results_cpp)

results_cpp = benchmark_func(kaiser_bessel_cheb, "cppyy cheb", dx)
results.append(results_cpp)


df = pd.DataFrame(results)
print(df)
