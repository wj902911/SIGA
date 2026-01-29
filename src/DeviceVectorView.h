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
};


