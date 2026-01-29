#pragma once

#include <cuda_runtime.h>

template <typename T>
class DeviceMatrixView
{
private:
    T* m_data = nullptr;
    int m_rows = 0;
    int m_cols = 0;

public:
    __host__ __device__
    DeviceMatrixView(T* data, int rows, int cols)
    : m_data(data), m_rows(rows), m_cols(cols) { }

    __device__ 
    T& operator()(int row, int col) 
    { 
        if (row > m_rows || col > m_cols)
            assert("Index out of bounds in DeviceMatrixView");
            
        return m_data[col * m_rows + row]; 
    }

    __device__ 
    const T& operator()(int row, int col) const
    { return m_data[col * m_rows + row]; }

    __host__ __device__
    int size() const { return m_rows * m_cols; }

    __host__ __device__
    int rows() const { return m_rows; }

    __host__ __device__
    int cols() const { return m_cols; }

    __device__
    void print() const
    {
        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                if constexpr (std::is_same<T, int>::value)
                    printf("%d ", m_data[j * m_rows + i]);
                else if constexpr (std::is_same<T, float>::value)
                    printf("%f ", m_data[j * m_rows + i]);
                else if constexpr (std::is_same<T, double>::value)
                    printf("%f ", m_data[j * m_rows + i]);
                else if constexpr (std::is_same<T, bool>::value)
                    printf("%d ", m_data[j * m_rows + i]);
                else
                    printf("Unsupported type ");
            }
            printf("\n");
        }
    }

    __device__
    T* data() const { return m_data; }
};
