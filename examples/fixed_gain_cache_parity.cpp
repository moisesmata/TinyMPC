#include <iomanip>
#include <iostream>

#include "tinympc/tiny_api.hpp"

static void print_flat(const tinyMatrix &matrix) {
    bool first = true;
    for (int row = 0; row < matrix.rows(); ++row) {
        for (int col = 0; col < matrix.cols(); ++col) {
            if (!first) {
                std::cout << ",";
            }
            first = false;
            std::cout << matrix(row, col);
        }
    }
}

int main() {
    const int nx = 2;
    const int nu = 1;
    const int horizon = 5;
    tinyMatrix A(nx, nx);
    A << 1.0, 0.1,
         0.0, 1.0;
    tinyMatrix B(nx, nu);
    B << 0.005,
         0.1;
    tinyMatrix f = tinyMatrix::Zero(nx, 1);
    tinyMatrix Q = tinyMatrix::Zero(nx, nx);
    Q.diagonal() << 2.0, 0.5;
    tinyMatrix R = tinyMatrix::Constant(nu, nu, 0.2);
    tinyMatrix K(nu, nx);
    K << 0.8, 1.4;
    const tinytype rho = 0.3;

    TinyCache cache;
    int status = tiny_precompute_and_set_fixed_gain_cache(
        &cache, A, B, f, Q, R, K, nx, nu, rho, 0);
    if (status != 0) {
        return status;
    }
    TinyCache dare_cache;
    status = tiny_precompute_and_set_cache(
        &dare_cache, A, B, f, Q, R, nx, nu, rho, 0);
    if (status != 0) {
        return status;
    }

    tinyMatrix q(nx, horizon);
    q << 0.1, -0.2, 0.3, -0.4, 0.5,
         -0.3, 0.4, -0.5, 0.6, -0.7;
    tinyMatrix r(nu, horizon - 1);
    r << 0.2, -0.1, 0.05, -0.3;
    tinyMatrix p = tinyMatrix::Zero(nx, horizon);
    tinyMatrix d = tinyMatrix::Zero(nu, horizon - 1);
    p.col(horizon - 1) = q.col(horizon - 1);
    for (int k = horizon - 2; k >= 0; --k) {
        d.col(k) = cache.Quu_inv * (B.transpose() * p.col(k + 1) + r.col(k));
        p.col(k) = q.col(k) + cache.AmBKt * p.col(k + 1)
            - K.transpose() * r.col(k) - cache.C_extra * d.col(k);
    }
    tinyMatrix x = tinyMatrix::Zero(nx, horizon);
    tinyMatrix u = tinyMatrix::Zero(nu, horizon - 1);
    x.col(0) << 0.25, -0.15;
    for (int k = 0; k < horizon - 1; ++k) {
        u.col(k) = -K * x.col(k) - d.col(k);
        x.col(k + 1) = A * x.col(k) + B * u.col(k);
    }

    std::cout << std::setprecision(17);
    std::cout << "{\"dare_C_extra_norm\":" << dare_cache.C_extra.norm();
    std::cout << ",\"P\":[";
    print_flat(cache.Pinf);
    std::cout << "],\"Quu_inv\":[";
    print_flat(cache.Quu_inv);
    std::cout << "],\"AmBKt\":[";
    print_flat(cache.AmBKt);
    std::cout << "],\"C_extra\":[";
    print_flat(cache.C_extra);
    std::cout << "],\"d\":[";
    print_flat(d);
    std::cout << "],\"p\":[";
    print_flat(p);
    std::cout << "],\"x\":[";
    print_flat(x);
    std::cout << "],\"u\":[";
    print_flat(u);
    std::cout << "]}" << std::endl;
    return 0;
}
