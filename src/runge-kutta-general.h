#include "matrix.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

#include <fmt/format.h>

#include <cassert>

#define WRAP_IN_LAMBDA(f) [](auto&&... args) { return f(std::forward<decltype(args)>(args)...); }

template <typename vec_type>
struct point {
    double t;
    vec_type y;
};

template <typename vec_type>
point(double t, vec_type y) -> point<vec_type>;

template <typename vec_type>
std::vector<point<vec_type>> runge_kutta(auto &&runge_step,
                                         auto &&f, point<vec_type> init, double step_size, double t_end,
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

        nth = runge_step(f, nth, step_size, a, b, c);
        if (i % saved_point_frequency == 0)
            y.push_back(nth);
    }

    return y;
}


void print_points(const auto &points) {
    for (auto &&point: points) {
        std::string_view format = "{:>27.20f} ";

        fmt::vprint(format, fmt::make_format_args(point.t));
        for (int i = 0; i < point.y.length(); ++ i)
            fmt::vprint(format, fmt::make_format_args(point.y[i]));

        fmt::print("\n");
    }
}
