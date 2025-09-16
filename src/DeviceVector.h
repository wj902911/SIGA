#pragma once

#include "DeviceMatrix.h"
#include <Eigen/Core>

template <typename T>
class DeviceVector: public DeviceMatrix<T>
{
public:
    __host__ __device__
    DeviceVector()
        : DeviceMatrix<T>(0, 1) // Initialize with zero size
    {
    }

    __host__ __device__
    DeviceVector(int size)
        : DeviceMatrix<T>(size, 1)
    {
    }

    __host__ __device__
    DeviceVector(int size, const T* data)
        : DeviceMatrix<T>(size, 1, data)
    {
    }

    __host__ __device__
    DeviceVector(const Eigen::Vector<T, Eigen::Dynamic>& v)
        : DeviceMatrix<T>(v.size(), 1, v.data())
    {
    }

    __host__ __device__
    DeviceVector(int size, T* data, bool owns_data)
        : DeviceMatrix<T>(size, 1, data, owns_data)
    {
    }

    // copy constructor
    __host__ __device__
    DeviceVector(const DeviceVector& other)
        : DeviceMatrix<T>(other.rows(), 1, other.data())
    {
    }

    template <typename Derived>
    __host__ __device__
    DeviceVector(const DeviceMatrixBase<Derived, T>& m)
        : DeviceMatrix<T>(static_cast<const Derived&>(m).rows(), 1)
    {
        assert(static_cast<const Derived&>(m).cols() == 1 && "Conversion requires exactly one column");

        const Derived& derived = static_cast<const Derived&>(m);
    #if defined(__CUDA_ARCH__)
        for (int i = 0; i < derived.rows(); i++)
            (*this)(i) = derived(i, 0);
    #else
        cudaError_t err = cudaMemcpy(this->data(), derived.data(), derived.rows() * sizeof(T), cudaMemcpyDeviceToDevice);
        assert(err == cudaSuccess && "cudaMemcpy failed in DeviceVector constructor");
    #endif
    }


    __host__ __device__
    int size() const
    {
        return this->rows();
    }

    __device__
    T& operator()(int i)
    {
        return DeviceMatrix<T>::operator()(i, 0);
    }

    __device__
    const T& operator()(int i) const
    {
        return DeviceMatrix<T>::operator()(i, 0);
    }

    __host__
    typename DeviceMatrix<T>::Element operator[](int i) const
    {
        return DeviceMatrix<T>::at(i, 0);
    }

    __host__
    typename DeviceMatrix<T>::Element at(int i) const
    {
        return DeviceMatrix<T>::at(i, 0);
    }

    __device__
    T norm() const
    {
        T sum = 0;
        for (int i = 0; i < size(); i++)
        {
            sum += this->operator()(i) * this->operator()(i);
        }
        return sqrt(sum);
    }

    __device__
    DeviceVector normalize() const
    {
        T n = norm();
        DeviceVector result(size());
        for (int i = 0; i < size(); i++)
        {
            result(i) = this->operator()(i) / n;
        }
        return result;
    }

    __device__
    DeviceVector cross(const DeviceVector& other) const
    {
        assert(size() == 3 && other.size() == 3 && "Cross product is only defined for 3D vectors");
        DeviceVector result(3);
        result(0) = this->operator()(1) * other(2) - this->operator()(2) * other(1);
        result(1) = this->operator()(2) * other(0) - this->operator()(0) * other(2);
        result(2) = this->operator()(0) * other(1) - this->operator()(1) * other(0);
        return result;
    }

    __host__ __device__
    void resize(int newSize)
    {
        DeviceMatrix<T>::resize(newSize, 1);
    }

    __host__ __device__
    void setZero()
    {
        DeviceMatrix<T>::setZero();
    }

    __host__ __device__
    void setZero(int size)
    {
        DeviceMatrix<T>::setZero(size, 1);
    }

    __device__
    T prod() const
    {
        T prod = 1;
        for (int i = 0; i < size(); i++)
            prod *= this->operator()(i);
        return prod;
    }

    __device__
    T sum() const
    {
        T sum = 0;
        for (int i = 0; i < size(); i++)
            sum += this->operator()(i);
        return sum;
    }

    //Returns the vector as a resized to n x m matrix
    __device__
    DeviceMatrix<T> reshape(int n, int m) const
    {
        return this->reshapeCol(0, n, m);
    }

};

template <typename T>
class DeviceVectorView : public DeviceVector<T>
{
public:
    __device__
    DeviceVectorView() : DeviceVector<T>(0, nullptr, false) {} // Initialize with zero size

    __device__
    DeviceVectorView(int size) : DeviceVector<T>(size, nullptr, false) {} // Initialize with given size, no data
    
    __device__
    DeviceVectorView(int size, T* data) : DeviceVector<T>(size, data, false) {} // No ownership of data
    
};