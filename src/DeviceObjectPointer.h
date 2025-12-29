#pragma once

#include <cuda_runtime.h>

template <class T>
class DeviceObjectPointer
{
private:
    T* m_ptr;

public:
    __host__
    DeviceObjectPointer() : m_ptr(nullptr) {}

    __host__
    DeviceObjectPointer(const T &obj)
    {
        cudaError_t err = cudaMalloc((void**)&m_ptr, sizeof(T));
        assert(err == cudaSuccess && "cudaMalloc failed in DeviceObjectPointer constructor");

        err = cudaMemcpy(m_ptr, &obj, sizeof(T), cudaMemcpyHostToDevice);
        assert(err == cudaSuccess && "cudaMemcpy failed in DeviceObjectPointer constructor");
    }

    __host__
    ~DeviceObjectPointer()
    {
        cudaFree(m_ptr);
    }

    __host__
    T* pointer() const
    {
        return m_ptr;
    }

    __host__
    void get(T &obj) const
    {
        cudaError_t err = cudaMemcpy(&obj, m_ptr, sizeof(T), cudaMemcpyDeviceToHost);
        assert(err == cudaSuccess && "cudaMemcpy failed in DeviceObjectPointer get");
    }

};