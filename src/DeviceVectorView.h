#pragma once

#include <DeviceMatrixView.h>

template <typename T>
class DeviceVectorView : public DeviceMatrixView<T>
{
public:
    __host__ __device__
    DeviceVectorView(T* data, int size)
    : DeviceMatrixView<T>(data, size, 1) {}

    __device__ 
    T& operator()(int i)
    { return DeviceMatrixView<T>::operator()(i, 0); }

    __device__ 
    const T& operator()(int i) const
    { return DeviceMatrixView<T>::operator()(i, 0); }

    __device__
    T& operator[](int i)
    { 
        if (i < 0)
            return DeviceMatrixView<T>::operator()(this->size() + i, 0);
        else
            return DeviceMatrixView<T>::operator()(i, 0); 
    }

    __device__
    const T& operator[](int i) const
    { 
        if (i < 0)
            return DeviceMatrixView<T>::operator()(this->size() + i, 0);
        else
            return DeviceMatrixView<T>::operator()(i, 0); 
    }

    __device__
    T back() const
    { return DeviceMatrixView<T>::operator()(this->size() - 1, 0); }

    __device__
    T dot(const DeviceVectorView<T>& other) const
    {
        T result = T(0);
        for (int i = 0; i < this->size(); i++)
            result += (*this)(i) * other(i);
        return result;
    }

    __host__
    double norm() const;

    __device__
    double norm_device() const
    {
        double sum = 0.0;
        for (int i = 0; i < this->size(); i++)
        {
            double val = static_cast<double>((*this)(i));
            sum += val * val;
        }
        return sqrt(sum);
    }
};

template <typename T>
class OneElementDeviceVectorView : public DeviceVectorView<T>
{
public:
    __host__ __device__
    OneElementDeviceVectorView(T* data)
    : DeviceVectorView<T>(data, 1) {}
};


