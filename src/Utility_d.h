#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
//#include <DeviceVector.h>
//#include "Matrix.h"

__host__ __device__
inline void getTensorCoordinate(int dim, int* numVlues, int index, int* out)
{
    for (int i = 0; i < dim; i++)
    {
        out[i] = index % numVlues[i];
        index /= numVlues[i];
    }
}

__host__ __device__
inline int getTotalNumber(int dim, int* numVlues)
{
    int total = 1;
    for (int i = 0; i < dim; i++)
    {
        total *= numVlues[i];
    }
    return total;
}

__host__ __device__
inline void getTensorProduct(int dim, int* numVlues, double* in, double* out)
{
    int numVluesTotal = getTotalNumber(dim, numVlues);
    for (int i = 0; i < numVluesTotal; i++)
    {
        int index[3] = { 0 };
        getTensorCoordinate(dim, numVlues, i, index);
        out[i] = in[index[0]];
        for (int j = 1, valuesStart = numVlues[0]; j < dim; valuesStart += numVlues[j], j++)
        {
            out[i] *= in[valuesStart+index[j]];
        }
    }
}

__host__ __device__
inline bool nextLexicographic(int dim, int* cur, int* start, int* end)
{
    for (int i = 0; i < dim; i++)
    {
        if (++cur[i] == end[i])
        {
            if (i == dim - 1)
                return false;
            else
                cur[i] = start[i];
        }
        else
            return true;
    }
    printf("Error: nextLexicographic\n");
    return false;
}

template<class Vec>
__device__
bool nextLexicographic_d(Vec& cur, const Vec& size)
{
    const int d = cur.size();
    assert(d == size.size() && "cur and size must have the same size.");

    for (int i = 0; i < d; i++)
    {
        if (++cur(i) == size(i))
        {
            if (i == d - 1)
                return false;
            else
                cur(i) = 0;
        }
        else
            return true;
    }
    printf("Error: nextLexicographic\n");
}

// A templated helper function to print an element with the correct format.
template <typename U>
__host__ __device__ 
inline void printElement(U element)
{
    if constexpr (std::is_same<U, int>::value)
        printf("%d ", element);
    else if constexpr (std::is_same<U, float>::value)
        printf("%f ", element);
    else if constexpr (std::is_same<U, double>::value)
        printf("%f ", element);
    else if constexpr (std::is_same<U, bool>::value)
        printf("%d ", element);
    else
        printf("Unsupported type ");
}

template <typename U>
__host__ __device__
inline void printArray(const U* array, int size, const char* name = "Array")
{
    printf("%s: [", name);
    for (int i = 0; i < size; ++i)
    {
        printElement(array[i]);
        if (i < size - 1)
            printf(", ");
    }
    printf("]\n");
}

template <typename U>
__host__
inline void printDeviceArray_onHost(const U* d_array, int size, const char* name = "Array")
{
    U* array = new U[size];
    cudaMemcpy(array, d_array, size * sizeof(U), cudaMemcpyDeviceToHost);
    printf("%s: [", name);
    for (int i = 0; i < size; ++i)
    {
        printElement(array[i]);
        if (i < size - 1)
            printf(", ");
    }
    printf("]\n");
    delete[] array;
}

template<typename T>
inline void printThrustVector(const thrust::device_vector<T>& vec, const char* name = "Vector")
{
    std::cout << name << ": [";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        std::cout << vec[i];
        if (i < vec.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}