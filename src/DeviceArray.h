#pragma once

#include <cuda_runtime.h>
#include <Eigen/Core>
#include <vector>
#include <DeviceVectorView.h>

template <typename T>
class DeviceArrayView
{
private:
    T* m_data = nullptr;
    int m_size = 0;
public:
    __host__
    DeviceArrayView(T* data, int size) : m_data(data), m_size(size) {}

    __host__ __device__
    T* data() const { return m_data; }

    __host__ __device__
    int size() const { return m_size; }
};

template <typename T>
class DeviceNestedArrayView : public DeviceArrayView<T>
{
private:
    DeviceArrayView<int> m_offsets;
public:
    __host__
    DeviceNestedArrayView(int* offsets, int numOffsets, T* data, int dataSize)
    : DeviceArrayView<T>(data, dataSize),
      m_offsets(offsets, numOffsets)
    {
    }

    __host__
    DeviceNestedArrayView(const DeviceArrayView<int>& offsets, DeviceArrayView<T> data)
    : DeviceArrayView<T>(data),
      m_offsets(offsets)
    {
    }

    __device__
    DeviceVectorView<T> operator[](int index) const
    {
        int start = (index == 0) ? 0 : m_offsets.data()[index - 1];
        int end = m_offsets.data()[index];
        return DeviceVectorView<T>(this->data() + start, end - start);
    }

    __device__
    void print(int index) const
    {
        DeviceVectorView<T> vec = (*this)[index];
        printf("Array %d:\n", index);
        vec.print();
    }

    __device__
    void print() const
    {
        printf("DeviceNestedArrayView contents:\n");
        int numArrays = m_offsets.size();
        for (int i = 0; i < numArrays; i++)
        {
            DeviceVectorView<T> vec = (*this)[i];
            printf("Array %d:\n", i);
            vec.print();
        }
    }
};

template <typename T>
class DeviceArray
{
private:
    T* m_data = nullptr;
    int m_size = 0;

public:
    __host__ __device__
    DeviceArray() = default;
    
    __host__
    DeviceArray(T* data, int size) : m_size(size)
    {
        if (size > 0)
        {
            cudaError_t err = cudaMalloc(&m_data, size * sizeof(T));
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceArray constructor");
            err = cudaMemcpy(m_data, data, size * sizeof(T), cudaMemcpyHostToDevice);
            assert(err == cudaSuccess && "cudaMemcpy failed in DeviceArray constructor");
        }
        else
            m_data = nullptr;
    }

    __host__
    DeviceArray(const std::vector<T>& vec)
    {
        m_size = static_cast<int>(vec.size());
        if (m_size > 0)
        {
            cudaError_t err = cudaMalloc(&m_data, m_size * sizeof(T));
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceArray constructor");
            err = cudaMemcpy(m_data, vec.data(), m_size * sizeof(T), cudaMemcpyHostToDevice);
            assert(err == cudaSuccess && "cudaMemcpy failed in DeviceArray constructor");
        }
        else
            m_data = nullptr;
    }

    template <typename Derived> __host__
    DeviceArray(const Eigen::MatrixBase<Derived> &vec)
    {
         m_size = static_cast<int>(vec.size());
        if (m_size > 0)
        {
            cudaError_t err = cudaMalloc(&m_data, m_size * sizeof(T));
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceArray constructor");
            err = cudaMemcpy(m_data, vec.derived().data(), m_size * sizeof(T), cudaMemcpyHostToDevice);
            assert(err == cudaSuccess && "cudaMemcpy failed in DeviceArray constructor");
        }
        else
            m_data = nullptr;
    }

    //copy constructor
    __host__
    DeviceArray(const DeviceArray<T>& other)
    {
        m_size = other.m_size;
        if (m_size > 0)
        {
            cudaError_t err = cudaMalloc(&m_data, m_size * sizeof(T));
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceArray copy constructor");
            err = cudaMemcpy(m_data, other.m_data, m_size * sizeof(T), cudaMemcpyDeviceToDevice);
            assert(err == cudaSuccess && "cudaMemcpy failed in DeviceArray copy constructor");
        }
        else
            m_data = nullptr;
    }

    __host__ 
    DeviceArray(DeviceArray&& other) noexcept
     : m_data(other.m_data), m_size(other.m_size)
    {
       other.m_data = nullptr;
       other.m_size = 0;
    }

    __host__ 
    DeviceArray& operator=(DeviceArray&& other) noexcept
    {
       if (this != &other) {
           if (m_data) cudaFree(m_data);
           m_data = other.m_data;
           m_size = other.m_size;
           other.m_data = nullptr;
           other.m_size = 0;
       }
       return *this;
    }

    //copy assignment operator
    __host__
    DeviceArray<T>& operator=(const DeviceArray<T>& other)
    {
        if (this != &other)
        {
            if (m_data)
            {
                cudaError_t err = cudaFree(m_data);
                assert(err == cudaSuccess && "cudaFree failed in DeviceArray copy assignment");
            }
            m_size = other.m_size;
            if (m_size > 0)
            {
                cudaError_t err = cudaMalloc(&m_data, m_size * sizeof(T));
                assert(err == cudaSuccess && "cudaMalloc failed in DeviceArray copy assignment");
                err = cudaMemcpy(m_data, other.m_data, m_size * sizeof(T), cudaMemcpyDeviceToDevice);
                assert(err == cudaSuccess && "cudaMemcpy failed in DeviceArray copy assignment");
            }
            else
                m_data = nullptr;
        }
        return *this;
    }

    //copy assignment operator from std::vector
    __host__
    DeviceArray<T>& operator=(const std::vector<T>& vec)
    {
        if (m_data)
        {
            cudaError_t err = cudaFree(m_data);
            assert(err == cudaSuccess && "cudaFree failed in DeviceArray copy assignment from std::vector");
        }
        m_size = static_cast<int>(vec.size());
        if (m_size > 0)
        {
            cudaError_t err = cudaMalloc(&m_data, m_size * sizeof(T));
            assert(err == cudaSuccess && "cudaMalloc failed in DeviceArray copy assignment from std::vector");
            err = cudaMemcpy(m_data, vec.data(), m_size * sizeof(T), cudaMemcpyHostToDevice);
            assert(err == cudaSuccess && "cudaMemcpy failed in DeviceArray copy assignment from std::vector");
        }
        else
            m_data = nullptr;
        return *this;
    }

    __host__
    ~DeviceArray()
    {
        if (m_data)
        {
            cudaError_t err = cudaFree(m_data);
            assert(err == cudaSuccess && "cudaFree failed in DeviceArray destructor");
        }
    }

    __host__
    T* data() const { return m_data; }

    __host__ __device__
    int size() const { return m_size; }

    __host__
    DeviceArrayView<T> view() const 
    { return DeviceArrayView<T>(m_data, m_size); }

    __host__
    DeviceVectorView<T> vectorView() const 
    { return DeviceVectorView<T>(m_data, m_size); }

    __host__
    void setZero()
    {
        if (m_data && m_size > 0) {
            cudaError_t err = cudaMemset(m_data, 0, m_size * sizeof(T));
            assert(err == cudaSuccess);
        }
    }
};

template <typename T>
class DeviceNestedArray : public DeviceArray<T>
{
private:
    DeviceArray<int> m_offsets;
public:
    __host__
    DeviceNestedArray() = default;

    __host__
    DeviceNestedArray(int* offsets, int numOffsets, T* data)
    : DeviceArray<T>(data, offsets[numOffsets - 1]),
      m_offsets(offsets, numOffsets)
    {
    }

    __host__
    DeviceNestedArray(const std::vector<int>& offsets, const std::vector<T>& data)
    : DeviceArray<T>(data),
      m_offsets(offsets)
    {
    }

    __host__
    DeviceNestedArray(const std::vector<std::vector<T>>& nestedVec)
    {
        std::vector<int> offsets;
        std::vector<T> data;
        offsets.reserve(nestedVec.size());
        int currentOffset = 0;
        for (const auto& vec : nestedVec)
        {
            currentOffset += static_cast<int>(vec.size());
            offsets.push_back(currentOffset);
            data.reserve(data.size() + vec.size());
            data.insert(data.end(), vec.begin(), vec.end());
        }
        m_offsets = offsets;
        DeviceArray<T>::operator=(data);
    }

    DeviceNestedArray(const DeviceNestedArray&) = default;
    DeviceNestedArray& operator=(const DeviceNestedArray&) = default;

    DeviceNestedArray(DeviceNestedArray&&) noexcept = default;
    DeviceNestedArray& operator=(DeviceNestedArray&&) noexcept = default;

    ~DeviceNestedArray() = default;

    __host__
    void setData(const std::vector<std::vector<T>>& nestedVec)
    {
        std::vector<int> offsets;
        std::vector<T> data;
        offsets.reserve(nestedVec.size());
        int currentOffset = 0;
        for (const auto& vec : nestedVec)
        {
            currentOffset += static_cast<int>(vec.size());
            offsets.push_back(currentOffset);
            data.reserve(data.size() + vec.size());
            data.insert(data.end(), vec.begin(), vec.end());
        }
        m_offsets = offsets;
        DeviceArray<T>::operator=(data);
    }

    template <typename VecType>
    __host__ void setData(const std::vector<VecType>& nestedVec)
    {
        static_assert(std::is_base_of_v<Eigen::EigenBase<VecType>, VecType>,
                      "VecType must be an Eigen type");

        std::vector<int> offsets;
        offsets.reserve(nestedVec.size());

        // 1) compute total size (for one reserve)
        size_t total = 0;
        for (const auto& v : nestedVec) total += static_cast<size_t>(v.size());

        std::vector<T> data;
        data.reserve(total);

        // 2) pack
        int currentOffset = 0;
        for (const auto& v : nestedVec)
        {
            const int n = static_cast<int>(v.size());
            currentOffset += n;
            offsets.push_back(currentOffset);

            // VectorXd / MatrixXd are contiguous, so this is OK
            const auto* p = v.data();
            data.insert(data.end(), p, p + n);
        }

        m_offsets = std::move(offsets);
        DeviceArray<T>::operator=(data);  // host->device copy (assuming your operator= does that)
    }

    __host__
    const DeviceArray<int>& offsets() const { return m_offsets; }

    __host__
    DeviceNestedArrayView<T> view() const 
    { 
        return DeviceNestedArrayView<T>(m_offsets.view(), 
                                        DeviceArray<T>::view()); 
    }
};


