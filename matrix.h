#include <cassert>
#include <utility>
#include <vector>


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
