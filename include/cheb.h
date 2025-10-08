#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include <xsimd/xsimd.hpp>

constexpr double PI = 3.14159265358979323846;

template <class Func> class Cheb {
public:
    Cheb(Func F, const int n, const double a = -1, const double b = 1)
        : nodes(n), low(b - a), hi(b + a), inv_low(1.0 / (b - a)), coeffs(nodes) {

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

    // [[gnu::optimize("-ffast-math", "-funroll-loops", "-ftree-vectorize")]]
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

private:
    const int nodes;
    double low, hi, inv_low;
    std::vector<double> coeffs;

    template <class T> constexpr auto map_to_domain(const T x) const { return 0.5 * (low * x + hi); }

    [[gnu::always_inline]] constexpr double map_from_domain(double x) const { return (2.0 * x - hi) * inv_low; }
};

template <class Func> class BarCheb {
public:
    BarCheb(Func F, const int n, const double a = -1, const double b = 1)
        : N(n), a(a), b(b), x(N), w(N), fvals(N) {
        for (int i = N - 1; i >= 0; i--) {
            double theta = (2 * i + 1) * PI / (2 * N);
            x[i] = map_to_domain(std::cos(theta));
            w[i] = (1 - 2 * (i % 2)) * std::sin(theta);
            fvals[i] = F(x[i]);
        }
    }

    // [[gnu::optimize("-ffast-math", "-funroll-loops", "-ftree-vectorize")]]
    constexpr double operator()(const double pt) const {
        for (int i = 0; i < N; ++i) {
            if (pt == x[i]) { return fvals[i]; }
        }

        double num = 0, den = 0;
        for (int i = 0; i < N; ++i) {
            auto diff = pt - x[i];
            auto q = w[i] / diff;
            num += q * fvals[i];
            den += q;
        }

        return num / den;
    }

private:
    const int N;
    const double a, b;
    std::vector<double> x, w, fvals;

    template <class T> constexpr auto map_to_domain(const T x) const { return 0.5 * ((b - a) * x + (b + a)); }

    [[gnu::always_inline]]
    constexpr double map_from_domain(const double x) const { return (2.0 * x - (b + a)) / (b - a); }

};


template <class Func> class BarChebSIMD1 {
public:
    BarChebSIMD1(Func F, const int n, const double a = -1, const double b = 1)
        : N(n), b_plus_a(b + a), b_minus_a(b - a), inv_b_minus_a(1.0 / (b - a)),
          x(N), w(N), fvals(N) {
        for (int i = N - 1; i >= 0; i--) {
            double theta = (2 * i + 1) * PI / (2 * N);
            x[i] = map_to_domain(std::cos(theta));
            w[i] = (1 - 2 * (i % 2)) * std::sin(theta);
            fvals[i] = F(x[i]);
        }
        for (int i = N; i < N; ++i) {
            x[i] = b_plus_a * 0.5;
            w[i] = 0.0;
            fvals[i] = F(x[i]);
        }
    }

    // [[gnu::optimize("-ffast-math", "-funroll-loops", "-ftree-vectorize")]]
    double operator()(const double pt) const {
        for (int i = 0; i < N; ++i) {
            if (pt == x[i]) { return fvals[i]; }
        }
        //     // shorthand for the xsimd type
        using batch = xsimd::batch<double>;
        // simd width since it is architecture/compile flags dependent
        constexpr std::int64_t simd_width = batch::size;

        const auto trunc_n = N & (-simd_width); // round down to multiple of simd_width

        const batch bpt(pt);
        batch bnum(0);
        batch bden(0);

        std::size_t i = 0;
        for (; i < trunc_n; i += simd_width) {
            const auto bx = batch::load_unaligned(x.data() + i);
            const auto bw = batch::load_unaligned(w.data() + i);
            const auto bf = batch::load_unaligned(fvals.data() + i);

            const auto bdiff = bpt - bx;
            const auto bq = bw / bdiff;
            bnum += bq * bf;
            // bnum = xsimd::fma(bq, bf, bnum);
            bden += bq;
        }

        double num = xsimd::reduce_add(bnum);
        double den = xsimd::reduce_add(bden);

        for (; i < N; ++i) {
            double diff = pt - x[i];
            double q = w[i] / diff;
            num += q * fvals[i];
            den += q;
        }
        return num / den;
    }

private:
    const int N;
    const double b_plus_a, b_minus_a, inv_b_minus_a;
    std::vector<double> x, w, fvals;

    template <class T> constexpr auto map_to_domain(const T x) const {
        return 0.5 * (b_minus_a * x + b_plus_a);
    }

    [[gnu::always_inline]]
    constexpr double map_from_domain(double x) const {
        return (2.0 * x - b_plus_a) * inv_b_minus_a;
    }
};


template <class Func> class BarChebSIMD2 {
public:
    BarChebSIMD2(Func F, const int n, const double a = -1, const double b = 1)
        : N(n), b_plus_a(b + a), b_minus_a(b - a), inv_b_minus_a(1.0 / (b - a)),
          x(padded(N)), w(padded(N)), fvals(padded(N)) {
        for (int i = N - 1; i >= 0; i--) {
            double theta = (2 * i + 1) * PI / (2 * N);
            x[i] = map_to_domain(std::cos(theta));
            w[i] = (1 - 2 * (i % 2)) * std::sin(theta);
            fvals[i] = F(x[i]);
        }
        for (int i = N; i < padded(N); ++i) {
            x[i] = b_plus_a * 0.5;
            w[i] = 0.0;
            fvals[i] = F(x[i]);
        }
    }

    // [[gnu::optimize("-ffast-math", "-funroll-loops", "-ftree-vectorize")]]
    double operator()(const double pt) const {
        // shorthand for the xsimd type
        using batch = xsimd::batch<double>;
        // simd width since it is architecture/compile flags dependent
        constexpr std::size_t simd_width = batch::size;

        const batch bpt(pt);

        batch bnum(0);
        batch bden(0);

        for (std::size_t i = 0; i < N; i += simd_width) {
            const auto bx = batch::load_aligned(x.data() + i);
            const auto bw = batch::load_aligned(w.data() + i);
            const auto bf = batch::load_aligned(fvals.data() + i);

            if (const auto mask_eq = (bx == bpt); xsimd::any(mask_eq)) {
                // Return the corresponding fval for the first match
                for (std::size_t j = 0; j < simd_width; ++j) {
                    if (mask_eq.get(j)) {
                        return bf.get(j);
                    }
                }
            }

            const auto bdiff = bpt - bx;
            const auto bq = bw / bdiff;
            bnum = xsimd::fma(bq, bf, bnum);
            bden += bq;
        }

        // Reduce SIMD accumulators to scalars
        const auto num = xsimd::reduce_add(bnum);
        const auto den = xsimd::reduce_add(bden);

        return num / den;
    }

private:
    const int N;
    const double b_plus_a, b_minus_a, inv_b_minus_a;
    std::vector<double, xsimd::aligned_allocator<double>> x, w, fvals;

    template <class T> constexpr auto map_to_domain(const T x) const {
        return 0.5 * (b_minus_a * x + b_plus_a);
    }

    [[gnu::always_inline]]
    constexpr double map_from_domain(double x) const {
        return (2.0 * x - b_plus_a) * inv_b_minus_a;
    }

    // Round up to the next multiple of the SIMD width
    // works only for powers of 2
    static constexpr std::size_t padded(const int n) {
        using batch = xsimd::batch<double>;
        constexpr std::size_t simd_width = batch::size;
        return (n + simd_width - 1) & (-simd_width);
    }
};

template <class Func> class BarChebSIMD3 {
public:
    BarChebSIMD3(Func F, const int n, const double a = -1, const double b = 1)
        : N(n), b_plus_a(b + a), b_minus_a(b - a), inv_b_minus_a(1.0 / (b - a)),
          x(padded(N)), w(padded(N)), fvals(padded(N)) {
        for (int i = N - 1; i >= 0; i--) {
            double theta = (2 * i + 1) * PI / (2 * N);
            x[i] = map_to_domain(std::cos(theta));
            w[i] = (1 - 2 * (i % 2)) * std::sin(theta);
            fvals[i] = F(x[i]);
        }
        for (int i = N; i < padded(N); ++i) {
            x[i] = b_plus_a * 0.5;
            w[i] = 0.0;
            fvals[i] = F(x[i]);
        }
    }

    // [[gnu::optimize("-ffast-math", "-funroll-loops", "-ftree-vectorize")]]
    double operator()(const double pt) const {
        // shorthand for the xsimd type
        using batch = xsimd::batch<double>;
        // simd width since it is architecture/compile flags dependent
        constexpr std::size_t simd_width = batch::size;

        const batch bpt(pt);

        batch bnum(0);
        batch bden(0);

        for (std::size_t i = 0; i < N; i += simd_width) {
            const auto bx = batch::load_aligned(x.data() + i);
            const auto bw = batch::load_aligned(w.data() + i);
            const auto bf = batch::load_aligned(fvals.data() + i);

            if (const auto mask = (bx == bpt).mask()) [[unlikely]] {
                // Return the corresponding fval for the first match
                // use bitwise operations to convert mask to index
                const auto index = __builtin_ctzll(mask);
                return fvals[i + index];
            }

            const auto bdiff = bpt - bx;
            const auto bq = bw / bdiff;
            bnum = xsimd::fma(bq, bf, bnum);
            bden += bq;
        }

        // Reduce SIMD accumulators to scalars
        const auto num = xsimd::reduce_add(bnum);
        const auto den = xsimd::reduce_add(bden);

        return num / den;
    }

private:
    const int N;
    const double b_plus_a, b_minus_a, inv_b_minus_a;
    std::vector<double, xsimd::aligned_allocator<double>> x, w, fvals;

    template <class T> constexpr auto map_to_domain(const T x) const {
        return 0.5 * (b_minus_a * x + b_plus_a);
    }

    [[gnu::always_inline]]
    constexpr double map_from_domain(double x) const {
        return (2.0 * x - b_plus_a) * inv_b_minus_a;
    }

    // Round up to the next multiple of the SIMD width
    // works only for powers of 2
    static constexpr std::size_t padded(const int n) {
        using batch = xsimd::batch<double>;
        constexpr std::size_t simd_width = batch::size;
        return (n + simd_width - 1) & (-simd_width);
    }
};