#include "cheb.h"

#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <chrono>

#include <nanobench.h>

static constexpr bool debug_print = true; // Set to false to disable printing
static constexpr double error_threshold = 1e-14;

template <typename T, typename V>
void test(V &&f) {
    int n = 16; // Number of Chebyshev nodes (degree = n - 1)
    double a = 0, b = 1;

    T interpolator(f, n, a, b);

    if constexpr (debug_print) {
        std::cout << "Chebyshev interpolation test on random samples:\n";
        std::cout << "Function: f(x) = KB(x), Domain: [" << a << ", " << b << "], Nodes: " << n << "\n\n";
    }

    std::mt19937 rng{42};
    std::uniform_real_distribution<double> dist(a, b);

    if constexpr (debug_print) {
        std::cout << std::setprecision(6) << std::scientific;
        std::cout << "x" << std::setw(20) << "f(x)" << std::setw(20) << "Interp(x)" << std::setw(20) << "Rel. Error\n";
        std::cout << std::string(80, '-') << "\n";
    }

    for (int i = 0; i < 15; ++i) {
        double x = dist(rng);
        double fx = f(x);
        double fx_interp = interpolator(x);
        double err = std::abs(1.0 - fx / fx_interp);
        if constexpr (debug_print) {
            std::cout << x << "\t" << fx << "\t" << fx_interp << "\t" << err << "\n";
        }
        if (err > error_threshold) {
            throw std::runtime_error("Interpolation error above threshold: " + std::to_string(err));
        }
    }
}


template <typename T, typename V>
void bench_interpolation(ankerl::nanobench::Bench &bench, const std::string &name, const size_t n, V &&f) {

    double a = 0, b = 1;
    T interpolator(f, n, a, b);

    std::minstd_rand rng{42};
    std::uniform_real_distribution<double> dist{a, b};

    bench
        .run("N=" + std::to_string(n) + " " + name,
             [&] { ankerl::nanobench::doNotOptimizeAway(interpolator(dist(rng))); });
}


double kaiser_bessel(double dx, double w, double sigma) {
    const double W = w * sigma;
    const double a = (2.0 * dx) / W;
    constexpr double pi = 3.14159265358979323846;
    const double beta = pi * w * (1.0 - 1.0 / (2.0 * sigma));

    const double t = std::sqrt(1.0 - a * a);
    const auto i0 = [](const double x) { return std::cyl_bessel_i(0.0, x); };
    return i0(beta * t) / i0(beta);
}

int main() {
    using namespace std::chrono_literals;
    // const auto f = [](const double x) { return std::cos(x); };
    const auto f = [](const double x) { return kaiser_bessel(x, 1.5, 2); };
    try {
        test<Cheb<decltype(f)>>(f);
        if (debug_print)
            std::cout << std::string(80, '-') << "\n\n\n";
        test<BarCheb<decltype(f)>>(f);
        if (debug_print)
            std::cout << std::string(80, '-') << "\n\n\n";
        test<BarChebSIMD1<decltype(f)>>(f);
        if (debug_print)
            std::cout << std::string(80, '-') << "\n\n\n";
        test<BarChebSIMD2<decltype(f)>>(f);
        if (debug_print)
            std::cout << std::string(80, '-') << "\n\n\n";
        test<BarChebSIMD3<decltype(f)>>(f);
        if (debug_print)
            std::cout << std::string(80, '-') << "\n\n\n";
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    ankerl::nanobench::Bench bench;
    bench.title("Interpolation Benchmark")
         .unit("evals")
         .relative(false)
         .minEpochTime(20ms);
    bench.run("Analytical", [&] {
        std::minstd_rand rng{42};
        std::uniform_real_distribution<double> dist{0, 1};
        ankerl::nanobench::doNotOptimizeAway(kaiser_bessel(dist(rng), 16, 2));
    });
    // print an horyzontal line
    {
        const auto n = 16;
        printf("\n%s\n\n", std::string(200, '-').c_str());
        bench_interpolation<Cheb<decltype(f)>>(bench, "Cheb", n, f);
        bench_interpolation<BarCheb<decltype(f)>>(bench, "BarCheb", n, f);
        bench_interpolation<BarChebSIMD1<decltype(f)>>(bench, "BarChebSIMD1", n, f);
        bench_interpolation<BarChebSIMD2<decltype(f)>>(bench, "BarChebSIMD2", n, f);
        bench_interpolation<BarChebSIMD3<decltype(f)>>(bench, "BarChebSIMD3", n, f);
    }
    for (size_t n = 32; n <= 128; n += 32) {
        // Analytical benchmark
        printf("\n%s\n\n", std::string(200, '-').c_str());
        bench_interpolation<Cheb<decltype(f)>>(bench, "Cheb", n, f);
        bench_interpolation<BarCheb<decltype(f)>>(bench, "BarCheb", n, f);
        bench_interpolation<BarChebSIMD1<decltype(f)>>(bench, "BarChebSIMD1", n, f);
        bench_interpolation<BarChebSIMD2<decltype(f)>>(bench, "BarChebSIMD2", n, f);
        bench_interpolation<BarChebSIMD3<decltype(f)>>(bench, "BarChebSIMD3", n, f);
    }
    return 0;
}