#include <functional>
#include <initializer_list>
#include <vector>
#include <iostream>
#include <cassert>

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



template <typename vec_type>
struct point {
    double t;
    vec_type y;
};

template <typename vec_type>
point<vec_type> runge_kutta_step(auto &&f, point<vec_type> nth, double step_size, dmatrix &a, dvector &b, dvector &c) {

    assert(a.cols() == a.rows() - 1);
    const size_t num_stages = a.rows();

    assert(b.size() == num_stages);
    assert(c.size() == num_stages);

    assert(c[0]    == 0);
    assert(a[0][0] == 0);

    vec_type sum = {};
    for (size_t i = 0; i < num_stages; ++ i) {
        vec_type f_y {}; // TODO: is it 0?
        for (int j = 0; j < i; ++ j)
            f_y += a[i][j];

        f_y *= step_size;
        f_y += nth.y;

        vec_type k_i = f(nth.t + c[i] * step_size, f_y);
        sum += b[i] * k_i;
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

    std::vector<point<vec_type>> y(num_steps / saved_point_frequency);
    int point_index = 0;

    point nth = y[0] = init;
    for (int i = 1; i < num_steps; ++ i, nth.t += step_size) {
        if (i % 1000 == 0)
            std::cerr << "Progress " << round((i / (double) num_steps) * 100 * 100) / 100. << "%\r";

        nth = runge_kutta_step(f, nth, step_size, a, b, c);
        if (i % saved_point_frequency == 0)
            y[point_index ++] = nth;
    }

    return y;
}


int main() {
    dmatrix a = {
        {           0,           0,           0,          0,       0 },
        {     1/   4.,           0,           0,          0,       0 },
        {     3/  32.,     9/  32.,           0,          0,       0 },
        {  1932/2197., -7200/2197.,  7296/2197.,          0,       0 },
        {   439/ 216.,         -8.,  3680/ 513., -845/4104.,       0 },
        {   -8 / 216.,           2, -3544/2565., 1859/4104., -11/40. }
    };

    dvector b = { 16/135.,    0, 6656/12825., 28561/56430., -9/50., 2/55. };
    dvector c = {       0, 1/4.,        3/8.,       12/13.,      1, 1/ 2. };

    std::ios::sync_with_stdio(0);
    std::cout.tie(0); std::cin.tie(0);

    const int points_to_plot = 100;

    point<glm::dvec1> init = { 0, glm::dvec1 { -1 } };
    auto f = [](auto t, auto y) { return glm::dvec1 { t * t }; };

    auto points = runge_kutta(f, init, .00001, 100, a, b, c, points_to_plot);

    for (auto &&point: points) {
        std::cout << point.t << " ";

        for (int i = 0; i < point.y.length(); ++ i)
            std::cout << point.y[i] << " ";

        std::cout << "\n";
    }
}
