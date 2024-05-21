#include "runge-kutta-general.h"

#include <iostream>
#include <cassert>

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>


template <typename vec_type>
point<vec_type> runge_kutta_step_explicit(auto &&f, point<vec_type> nth, double step_size, dmatrix &a, dvector &b, dvector &c) {

    assert(a.cols() == a.rows() - 1);
    const size_t num_stages = a.rows();

    assert(b.size() == num_stages);
    assert(c.size() == num_stages);

    assert(c[0]    == 0);
    assert(a[0][0] == 0);

    vec_type ks[num_stages];

    vec_type sum = {};
    for (size_t i = 0; i < num_stages; ++ i) {
        vec_type f_y {}; // TODO: is it 0?
        for (int j = 0; j < i; ++ j)
            f_y += a[i][j] * ks[j];

        f_y *= step_size;
        f_y += nth.y;

        vec_type k_i = ks[i] = f(nth.t + c[i] * step_size, f_y);
        sum += b[i] * k_i;
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


    const int points_to_plot = 50000;
    auto points = runge_kutta(WRAP_IN_LAMBDA(runge_kutta_step_explicit),
                              f, init, .00001, 1000,
                              a, b, c, points_to_plot);


    for (auto &&point: points) {
        std::cout << point.t << " ";

        for (int i = 0; i < point.y.length(); ++ i)
            std::cout << point.y[i] << " ";

        std::cout << "\n";
    }
}
