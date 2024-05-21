#include "runge-kutta-general.h"


// richardson 5-point rule
auto gderivative(auto &&f, double x, double h){
    return (- f(x + 2. * h) + 8. * f(x + h) - 8. * f(x - h) + f(x - 2. * h)) * (1. / (12 * h));
}

template <typename result_type, typename vec_type>
result_type jacobian(auto &&f, vec_type point) {
    result_type res {};

    const double step = 0.0001;

    for (int j = 0; j < res.length(); ++ j) {
        auto partial = [point, j, &f](double x) {
            auto current = point;

            current[j] = x;
            return f(current);
        };

        auto result = gderivative(partial, point[j], step);
        for (int i = 0; i < result.length(); ++ i)
            res[i][j] = result[i];
    }

    return res;
}


template <typename vec_type>
vec_type solve_newtons(auto &&f, vec_type init, double precision) {

    auto previous = init, current = init;

    const int max_iter = 10000;
    for (int i = 0; i < max_iter; ++ i) {
        using jacobin_result = glm::mat<init.length(), init.length(), double, glm::defaultp>;
        auto jac = ::jacobian<jacobin_result>(f, previous);

        auto inverse_jacobian = glm::inverse(jac);
        current = previous - inverse_jacobian * f(previous);

        if (glm::length(previous - current) < precision)
            break;

        previous = current;
    }

    return current;
}

template <typename vec_type>
point<vec_type> runge_kutta_step_implicit(auto &&f, point<vec_type> nth, double step_size, dmatrix &a, dvector &b, dvector &c) {

    assert(a.cols() == a.rows());
    const size_t num_stages = a.rows();

    assert(b.size() == num_stages);
    assert(c.size() == num_stages);

    vec_type ks[num_stages];

    vec_type sum = {};
    for (size_t i = 0; i < num_stages; ++ i) {

        auto k_eq = [&](auto k) -> vec_type {
            vec_type f_y {}; // TODO: is it 0?
            for (size_t j = 0; j < i; ++ j)
                f_y += a[i][j] * ks[j];

            f_y += a[i][i] * k; // TODO: ?

            f_y *= step_size;
            f_y += nth.y;

            vec_type k_i = ks[i] = f(nth.t + c[i] * step_size, f_y);
            return k_i - k;
        };

        vec_type solution = solve_newtons(k_eq, vec_type { 1. }, 0.00001);

        sum += b[i] * solution;
    }

    return { nth.t, sum * step_size + nth.y };
}







int main() {
    dmatrix a = {
        { 1  /   4.,   0   /   1.,   0   /  1.,   0  / 1.,   0 /1. },
        { 1  /   2.,   1   /   4.,   0   /  1.,   0  / 1.,   0 /1. },
        { 17 /  50.,   -1  /  25.,   1   /  4.,   0  / 1.,   0 /1. },
        { 371/1360.,   -137/2720.,   15  /544.,   -1 / 4.,   0 /1. },
        { 25 /  24.,   -49 /  48.,   -125/ 16.,   -85/12.,   -1/4. },
    };

    dvector b = { 25/244., -49/48., 125/16., -85/12., 1/4. };
    dvector c = { 1 /  4., 3  / 4., 11 /20., 1  / 2., 1/1. };

    point init = { 0, glm::dvec2 { 2, 0 } };
    auto f = []([[maybe_unused]] auto t, auto y) {
        return glm::dvec2 { y[1], 5*(1 - y[0]*y[0])*y[1] - y[0] };
    };


    const int points_to_plot = 100000;
    auto points = runge_kutta(WRAP_IN_LAMBDA(runge_kutta_step_implicit), f,
                              init, .0001, 1000,
                              a, b, c, points_to_plot);


    for (auto &&point: points) {
        std::string_view format = "{:>27.20f} ";

        fmt::vprint(format, fmt::make_format_args(point.t));
        for (int i = 0; i < point.y.length(); ++ i)
            fmt::vprint(format, fmt::make_format_args(point.y[i]));

        fmt::print("\n");
    }
}
