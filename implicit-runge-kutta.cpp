#include <glm/fwd.hpp>
#define GLM_ENABLE_EXPERIMENTAL

#include <initializer_list>
#include <vector>
#include <cassert>

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

#include <fmt/format.h>

template <typename underlying_linear_container>
class proxy2d {
public:
    constexpr proxy2d(underlying_linear_container &container, int cols, int i):
        container_(container), cols_(cols), i_(i) {}

    constexpr decltype(auto) operator[](int j) {
        assert(j < cols_);
        return container_[cols_ * i_ + j];
    }

private:
    underlying_linear_container &container_;
    int cols_, i_;
};


template <typename underlying_linear_container>
class wrap2d {
public:
    constexpr wrap2d(underlying_linear_container &&container, int rows, int cols):
        rows_(rows), cols_(cols), container_(std::move(container)) {}

    constexpr proxy2d<underlying_linear_container> operator[](int i) {
        assert(i < rows_);
        return { container_, cols_, i };
    }

    constexpr int cols() const { return cols_; }
    constexpr int rows() const { return rows_; }

private:
    int rows_, cols_;
    underlying_linear_container container_;
};


template <typename type>
class matrix: private wrap2d<std::vector<type>> {
public:
    using wrap2d<std::vector<type>>::operator[];

    using wrap2d<std::vector<type>>::cols;
    using wrap2d<std::vector<type>>::rows;

    constexpr matrix(std::initializer_list<std::initializer_list<type>> init):
        wrap2d<std::vector<type>>(std::vector<type>(init.size() * init.begin()->size()),
                                  /* rows = */ init.size(),
                                  /* cols = */ init.begin()->size()) {

        int i = 0;
        for (auto &&row: init) {
            int j = 0;
            for (auto &&element: row)
                (*this)[i][j ++] = element;

            ++ i;
        }
    }
};

using dmatrix = matrix<double>;
using dvector = std::vector<double>;


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
struct point {
    double t;
    vec_type y;
};

template <typename vec_type>
point(double t, vec_type y) -> point<vec_type>;

template <typename vec_type>
point<vec_type> runge_kutta_step(auto &&f, point<vec_type> nth, double step_size, dmatrix &a, dvector &b, dvector &c) {

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

template <typename vec_type>
std::vector<point<vec_type>> runge_kutta(auto &&f, point<vec_type> init, double step_size, double t_end,
                                         dmatrix &a, dvector &b, dvector &c, int num_points_to_save) {

    const int num_steps = (t_end - init.t) / step_size;
    assert(num_steps > 0);

    const int saved_point_frequency = num_steps / num_points_to_save;
    assert(saved_point_frequency > 0);

    std::vector<point<vec_type>> y;
    y.reserve(ceil(num_steps / (double) saved_point_frequency));

    point nth = y[0] = init;
    for (int i = 0; i < num_steps; ++ i, nth.t += step_size) {
        if (i % 1000 == 0)
            fmt::print(stderr, "Progress {:.2f}% \r", (i / (double) num_steps) * 100);

        nth = runge_kutta_step(f, nth, step_size, a, b, c);
        if (i % saved_point_frequency == 0)
            y.push_back(nth);
    }

    return y;
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
    auto points = runge_kutta(f, init, .0001, 1000, a, b, c, points_to_plot);


    for (auto &&point: points) {
        std::string_view format = "{:>27.20f} ";

        fmt::vprint(format, fmt::make_format_args(point.t));
        for (int i = 0; i < point.y.length(); ++ i)
            fmt::vprint(format, fmt::make_format_args(point.y[i]));

        fmt::print("\n");
    }
}
