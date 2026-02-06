#pragma once

#include <cuda_runtime.h>
#include <cstdio>


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

    __device__
    void firstMinor(int row, int col, DeviceMatrixView<T> minor) const
    {
        for (int i = 0, i2 = 0; i < m_rows; i++)
        {
            if (i == row)
            {
                continue;
            }
            for (int j = 0, j2 = 0; j < m_cols; j++)
            {
                if (j == col)
                {
                    continue;
                }
                minor(i2, j2) = this->operator()(i, j);
                j2++;
            }
            i2++;
        }
    }

    __device__
    T cofactor(int row, int col) const
    {
        T tempData[4]; //max 2x2 minor
        DeviceMatrixView<T> minor(tempData, m_rows - 1, m_cols - 1);
        firstMinor(row, col, minor);
        T det = minor(0, 0) * minor(1, 1) - minor(0, 1) * minor(1, 0);
        if ((row + col) % 2 == 1)
            det = -det;
        return det;
    }

    __device__
    T determinant() const
    {
        //assert(m_rows == m_cols && "Matrix must be square for determinant");

        if (m_rows == 1)
        {
            return this->operator()(0, 0);
        }
        else if (m_rows == 2)
        {
            //printf("Calculating 2x2 determinant\n");
            return this->operator()(0, 0) * this->operator()(1, 1) 
                 - this->operator()(0, 1) * this->operator()(1, 0);
        }
        else if (m_rows == 3)
        {
            T det = 0;
            for (int j = 0; j < m_cols; j++)
            {
                det += this->operator()(0, j) * cofactor(0, j);
            }
            return det;
        }
        else
        {
            printf("Error: Determinant for matrices larger than 2x2 is not supported yet.\n");
            return T(0);
        }
    }

    __device__
    void inverse(DeviceMatrixView<T> inv) const
    {
        T det = determinant();
        assert(det != 0 && "Matrix is singular and cannot be inverted");
        if (m_rows == 1)
        {
            inv(0, 0) = 1 / (*this)(0, 0);
        }
        else if (m_rows == 2)
        {
            inv(0, 0) = (*this)(1, 1) / det;
            inv(0, 1) = -(*this)(0, 1) / det;
            inv(1, 0) = -(*this)(1, 0) / det;
            inv(1, 1) = (*this)(0, 0) / det;
        }
        else
        {
            for (int i = 0; i < m_rows; i++)
            {
                for (int j = 0; j < m_cols; j++)
                {
                    inv(j, i) = cofactor(i, j) / det; // Note the transpose here
                }
            }
        }
    }

    __device__
    void times(const DeviceMatrixView<T>& other, DeviceMatrixView<T> result) const
    {
        assert(m_cols == other.m_rows && "Inner dimensions must match for multiplication");
        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < other.m_cols; j++)
            {
                T sum = 0;
                for (int k = 0; k < m_cols; k++)
                {
                    sum += this->operator()(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
    }

    __device__
    void plusIdentity(DeviceMatrixView<T> result) const
    {
        assert(m_rows == m_cols && "Matrix must be square to add identity");
        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                result(i, j) = this->operator()(i, j) + (i == j ? T(1) : T(0));
            }
        }
    }
};
