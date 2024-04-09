#include <cmath>
#include <functional>
#include <initializer_list>
#include <vector>
#include <iostream>
#include <cassert>
#include <iomanip>

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>


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

// ------------------------------------------------------------------------------------------------------------------

auto gderivative(auto &&f, double x, double h){
    return (- f(x + 2. * h) + 8. * f(x + h) - 8. * f(x - h) + f(x - 2. * h)) * (1. / (12 * h));
}

template <int size>
glm::mat<size, size, double> jacobian(auto &&f, glm::vec<size, double> point) {
    glm::mat<size, size, double> res {};

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


template <int size>
glm::vec<size, double> solve_newtons(auto &&f, glm::vec<size, double> init, double precision) {
    auto previous = init, current = init;

    const int max_iter = 10000;
    for (int i = 0; i < max_iter; ++ i) {
        auto jac = ::jacobian(f, previous);
        auto inverse_jacobian = glm::inverse(jac);
        current = previous - inverse_jacobian * f(previous);

        if (glm::length(previous - current) < precision)
            break;

        previous = current;
    }

    return current;
}



template <int size>
struct point { double t; glm::vec<size, double> y; };

template <int size>
point(double t, glm::vec<size, double> y) -> point<size>;

template <int size>
point<size> runge_kutta_step(auto &&f, point<size> nth, double step_size, dmatrix &a, dvector &b, dvector &c) {

    assert(a.cols() == a.rows());
    const int num_stages = a.rows();

    assert(b.size() == num_stages);
    assert(c.size() == num_stages);

    glm::vec<size, double> ks[num_stages];

    glm::vec<size, double> sum = {};
    for (int i = 0; i < num_stages; ++ i) {

        std::cerr << "f(" << nth.t + c[i] * step_size << ", ";
        std::cerr << "(";
        for (int j = 0; j < i; ++ j) {
            if (j != 0)
                std::cerr << " + ";

            std::cerr << glm::to_string(a[i][j] * ks[j]);
        }
        std::cerr << "+ " << a[i][i] << "* k)";
        std::cerr << " * " << step_size;
        std::cerr << " + " << glm::to_string(nth.y);

        std::cerr << ") == k\n";

        auto k_eq = [&](auto k) -> glm::vec<size, double> {
            glm::vec<size, double> f_y {};
            for (int j = 0; j < i; ++ j)
                f_y += a[i][j] * ks[j];

            f_y += a[i][i] * k;

            f_y *= step_size;
            f_y += nth.y;

            auto k_i = ks[i] = f(nth.t + c[i] * step_size, f_y);
            return k_i - k;
        };

        auto solution = solve_newtons(k_eq, glm::vec<size, double> { 1. }, 0.000001);
        std::cerr << glm::to_string(solution) << "\n";

        sum += b[i] * solution;
    }

    return { nth.t, sum * step_size + nth.y };
}

template <int size>
std::vector<point<size>> runge_kutta(auto &&f, point<size> init, double step_size, double t_end,
                                     dmatrix &a, dvector &b, dvector &c, int num_points_to_save) {

    const int num_steps = (t_end - init.t) / step_size;
    assert(num_steps > 0);

    const int saved_point_frequency = num_steps / num_points_to_save;
    assert(saved_point_frequency > 0);

    std::vector<point<size>> y(num_steps / saved_point_frequency);
    int point_index = 0;

    point nth = y[0] = init;
    for (int i = 0; i < num_steps; ++ i, nth.t += step_size) {
        if (i % 1000 == 0)
            std::cerr << "Progress " << std::fixed << std::setprecision(2)
                      << (i / (double) num_steps) * 100 << "% " << point_index << "\r";

        nth = runge_kutta_step(f, nth, step_size, a, b, c);

        if (i % saved_point_frequency == 0)
            y[point_index ++] = nth;
    }

    return y;
}






int main() {
    dmatrix a = {
        {     1/   4.,           0,           0,          0,     0 },
        {     1/   2.,     1/   4.,           0,          0,     0 },
        {    17/  50.,    -1/  25.,        1/4.,          0,     0 },
        {   371/ 1360.,   -137/2720.,   15/544.,      -1/4.,     0 },
        {   25 / 24.,     -49/48.,     -125/16.,    -85/12., -1/4. },
    };

    dvector b = { 25/244., -49/48., 125/16., -85/12., 1/4.};
    dvector c = {    1/4.,    3/4.,  11/20.,    1/2.,   1 };


    point<3> init = { 150, { 1, 2, 3 } };
    auto f = [](auto t, auto y) {
        return glm::dvec3 {
            77.27 * (y[1] + y[0] * (1 - 8.375 * 10e-6 * y[0] - y[1])),
            1/77.27 * (y[2] - (1 + y[0]) * y[1]),
            0.16 * (y[0] - y[2])
        };
    };


    const int points_to_plot = 5000;
    auto points = runge_kutta(f, init, .00003, 1000, a, b, c, points_to_plot);


    for (auto &&point: points) {
        std::cout << point.t << " ";

        for (int i = 0; i < point.y.length(); ++ i)
            std::cout << point.y[i] << " ";

        std::cout << "\n";
    }
}
