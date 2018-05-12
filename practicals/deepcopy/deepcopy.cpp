#include <cassert>
#include <iostream>
#include <openacc.h>


template<typename T>
class Field {
public:
    Field(size_t xdim, size_t ydim=1)
        : xdim_(xdim), ydim_(ydim) {

        data_ = new T[xdim_*ydim_];
        rows_ = new T*[xdim_];
        for (size_t i = 0; i < xdim_; ++i) {
            rows_[i] = &data_[i*ydim_];
        }

        // TODO: Copy this object to the GPU; take particular care of the rows_ pointers!
    }

    ~Field() {
        // TODO: Deallocate this object from the GPU
        delete[] data_;
        delete[] rows_;
    }


    size_t xdim() const { return xdim_; }
    size_t ydim() const { return ydim_; }
    size_t size() const { return xdim_*ydim_; }
    T& operator()(int i, int j) {
        // row-major ordering
        return data_[i*ydim_+j];
    }

    T& operator[](size_t i) const {
        return data_[i];
    }

    T* row(size_t i) { return rows_[i]; }
    T* data() { return data_; }

    void update_host() {
        // TODO: update the host copy of the object
    }

    void update_device() {
        // TODO: update the device copy of the object
    }

    template<typename S>
    friend std::ostream& operator<<(std::ostream& out, const Field<S>& f);

private:
    size_t xdim_, ydim_;
    T *data_;
    T **rows_;
};


template<typename T>
std::ostream& operator<<(std::ostream& out, const Field<T>& f)
{
    for (size_t i = 0; i < f.xdim(); ++i) {
        for (size_t j = 0; j < f.ydim(); ++j) {
            out << f.data_[i*f.ydim() + j] << " ";
        }
        out << "\n";
    }

    return out;
}

int main()
{
    Field<double> f(3, 4);

    auto X = f.xdim();
    auto Y = f.ydim();

    // TODO: offload this loop to the GPU
    for (size_t i = 0; i < X; ++i) {
        for (size_t j = 0; j < Y; ++j) {
            f(i, j) = 2;
        }
    }

    // TODO: offload this loop to the GPU
    for (size_t i = 0; i < Y; ++i) {
        f.row(1)[i] = 3;
    }

    f.update_host();
    std::cout << f;
    return 0;
}
