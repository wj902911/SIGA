#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
//#include <DeviceVector.h>
//#include "Matrix.h"
#include <DeviceVectorView.h>

template <typename T>
class DeviceMatrix;

template <typename T>
class DeviceVector;

__host__ __device__
inline void getTensorCoordinate(int dim, int* numVlues, int index, int* out)
{
    for (int i = 0; i < dim; i++)
    {
        out[i] = index % numVlues[i];
        index /= numVlues[i];
    }
}

__device__
inline void getTensorCoordinate(int dim, int numVlue, int index, 
                                int* out)
{
    for (int i = 0; i < dim; i++)
    {
        out[i] = index % numVlue;
        index /= numVlue;
    }
}

__host__ __device__
inline int getTensorCoordinate(int dir, int numVlue, int index)
{
    
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
__device__ inline
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

template<class Vec>
__device__ inline
bool nextLexicographic_d(Vec& cur, const Vec& start, const Vec& end)
{
    const int d = cur.size();
    assert(d == start.size() && d == end.size() && "cur, start and end must have the same size.");

    for (int i = 0; i < d; ++i)
    {
        if (++cur(i) == end(i))
        {
            if (i == d - 1)
                return false;
            else
                cur(i) = start(i);
        }
        else
            return true;
    }
    printf("Error: nextLexicographic\n");
    return false;
}

template <typename T>
__host__ __device__ 
inline T min_dev(T a, T b) 
{ return (a < b) ? a : b; }

template <typename Z>
__device__
inline Z binomial(const Z n, const Z r)
{
    assert(r>=0);
    const Z diff = min_dev(n - r, r);
    int result = 1;
    for (Z i=0;i < diff;)
    {
        result *= n-i;
        result /= ++i;
    }
    return result;
}

__device__
inline unsigned numCompositions(int sum, int dim)
{
    return binomial(sum+dim-1,dim-1);
}

template<class Vec>
__device__
void firstComposition(int sum, int dim, Vec& res)
{
    res.resize(dim);
    res.setZero();
    res(0) = sum;
}

template<class Vec>
__device__
inline bool nextComposition(Vec & v)
{
    const int k = v.size() - 1;

    if (v(k) != v.sum())
    {
        for (int i = 0; i <= k; ++i)
        {
            if ( v(i)!=0 )
            {
                const int t = v(i);
                v(i) = 0;
                v(0) = t - 1;
                v(i+1) += 1;
                return true;
            }
        }
    }
    return false;
}

// A templated helper function to print an element with the correct format.
template <typename U>
__host__ __device__ 
inline void printElement(U element)
{
    if constexpr (std::is_same<U, int>::value)
        printf("%d ", element);
    else if constexpr (std::is_same<U, float>::value)
        printf("%e ", element);
    else if constexpr (std::is_same<U, double>::value)
        printf("%e ", element);
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

// Indices of the Voigt notation
__host__ __device__
inline int voigt(int dim, int I, int J)
{
    if (dim == 2)
        switch(I)
        {
        case 0: return J == 0 ? 0 : 0;
        case 1: return J == 0 ? 1 : 1;
        case 2: return J == 0 ? 0 : 1;
        }
    else if (dim == 3)
        switch (I)
        {
        case 0: return J == 0 ? 0 : 0;
        case 1: return J == 0 ? 1 : 1;
        case 2: return J == 0 ? 2 : 2;
        case 3: return J == 0 ? 0 : 1;
        case 4: return J == 0 ? 1 : 2;
        case 5: return J == 0 ? 0 : 2;
        }
    return -1;
}

template <class T>
__device__
inline void matrixTraceTensor(DeviceMatrix<T> &C, const DeviceMatrix<T> &R, const DeviceMatrix<T> &S)
{
    int dim = R.cols();
    int dimTensor = (dim * (dim + 1)) / 2;
    C.setZero(dimTensor,dimTensor);
    for (int i = 0; i < dimTensor; i++)
        for (int j = 0; j < dimTensor; j++)
            C(i, j) = R(voigt(dim, i, 0), voigt(dim, i, 1)) * S(voigt(dim, j, 0), voigt(dim, j, 1));
}

template <class T>
__device__
inline void matrixViewTraceTensor(DeviceMatrixView<T> &C, const DeviceMatrixView<T> &R, const DeviceMatrixView<T> &S)
{
    int dim = R.cols();
    int dimTensor = (dim * (dim + 1)) / 2;
    for (int i = 0; i < dimTensor; i++)
        for (int j = 0; j < dimTensor; j++)
            C(i, j) = R(voigt(dim, i, 0), voigt(dim, i, 1)) * S(voigt(dim, j, 0), voigt(dim, j, 1));
}

template <class T>
__device__
inline void matrixViewTraceTensor_parallel(int tidx, int numThreadsx, int tidy, int numThreadsy,
    DeviceMatrixView<T> C, const DeviceMatrixView<T> &R, const DeviceMatrixView<T> &S)
{
    int dim = R.cols();
    int dimTensor = (dim * (dim + 1)) / 2;
    for (int i = tidx; i < dimTensor; i += numThreadsx)
        for (int j = tidy; j < dimTensor; j += numThreadsy)
            C(i, j) = R(voigt(dim, i, 0), voigt(dim, i, 1)) * S(voigt(dim, j, 0), voigt(dim, j, 1));
}

template <class T>
__device__
inline void symmetricIdentityTensor(DeviceMatrix<T> &C, const DeviceMatrix<T> &R)
{
    int dim = R.cols();
    int dimTensor = (dim * (dim + 1)) / 2;
    C.setZero(dimTensor,dimTensor);
    for (int i = 0; i < dimTensor; i++)
        for (int j = 0; j < dimTensor; j++)
            C(i, j) = (R(voigt(dim, i, 0), voigt(dim, j, 0)) * R(voigt(dim, i, 1), voigt(dim, j, 1))
                           + R(voigt(dim, i, 0), voigt(dim, j, 1)) * R(voigt(dim, i, 1), voigt(dim, j, 0)));
}

template <class T>
__device__
inline void symmetricIdentityViewTensor(DeviceMatrixView<T> &C, const DeviceMatrixView<T> &R)
{
    int dim = R.cols();
    int dimTensor = (dim * (dim + 1)) / 2;
    for (int i = 0; i < dimTensor; i++)
        for (int j = 0; j < dimTensor; j++)
            C(i, j) = (R(voigt(dim, i, 0), voigt(dim, j, 0)) * R(voigt(dim, i, 1), voigt(dim, j, 1))
                           + R(voigt(dim, i, 0), voigt(dim, j, 1)) * R(voigt(dim, i, 1), voigt(dim, j, 0)));
}

template <class T>
__device__
inline void electroelasticMechanicalTensor(DeviceMatrixView<T> &C, const DeviceMatrixView<T> &R, const DeviceMatrixView<T> &E)
{
    int dim = R.cols();
    int dimTensor = (dim * (dim + 1)) / 2;
    
    for (int i = 0; i < dimTensor; i++)
        for (int j = 0; j < dimTensor; j++)
        {
            T tempData[3*3];
            DeviceMatrixView<T> temp(tempData, dim, dim);
            for (int m = 0; m < dim; m++)
                for (int n = 0; n < dim; n++)
                {
                    temp(m, n) =       R(voigt(dim, j, 0), voigt(dim, j, 1))                                           //     C^(-1)_CD
			            		   * ( R(voigt(dim, i, 0),                m) * R(voigt(dim, i, 1),               n)    // * ( C^(-1)_AI * C^(-1)_BJ
			            			 + R(voigt(dim, i, 0),                n) * R(voigt(dim, i, 1),               m)    //   + C^(-1)_AJ * C^(-1)_BI
			            			 - R(voigt(dim, i, 0), voigt(dim, i, 1)) * R(               m,               n) )  //   - C^(-1)_AB * C^(-1)_IJ )
			                       -   R(voigt(dim, i, 1),                n)                                           // -   C^(-1)_BJ
			                       * ( R(voigt(dim, i, 0), voigt(dim, j, 0)) * R(voigt(dim, j, 1),               m)    // * ( C^(-1)_AC * C^(-1)_DI
			            			 + R(voigt(dim, i, 0), voigt(dim, j, 1)) * R(voigt(dim, j, 0),               m) )  //   + C^(-1)_AD * C^(-1)_CI )
			            		   -   R(voigt(dim, i, 0),                m)                                           // -   C^(-1)_AI
			                       * ( R(voigt(dim, i, 1), voigt(dim, j, 0)) * R(voigt(dim, j, 1),               n)    // * ( C^(-1)_BC * C^(-1)_DJ
			            		     + R(voigt(dim, i, 1), voigt(dim, j, 1)) * R(voigt(dim, j, 0),               n) )  //   + C^(-1)_BD * C^(-1)_CJ )
			               /* + */ -   R(voigt(dim, i, 1), m)                                                          // -   C^(-1)_BI
			                       * ( R(voigt(dim, i, 0), voigt(dim, j, 0)) * R(voigt(dim, j, 1),               n)    // * ( C^(-1)_AC * C^(-1)_DJ
			            	         + R(voigt(dim, i, 0), voigt(dim, j, 1)) * R(voigt(dim, j, 0),               n) )  //   + C^(-1)_AD * C^(-1)_CJ )
			                       -   R(voigt(dim, i, 0),                n)                                           // -   C^(-1)_AJ
			                       * ( R(voigt(dim, i, 1), voigt(dim, j, 0)) * R(voigt(dim, j, 1),               m)    // * ( C^(-1)_BC * C^(-1)_DI
			            			 + R(voigt(dim, i, 1), voigt(dim, j, 1)) * R(voigt(dim, j, 0),               m) )  //   + C^(-1)_BD * C^(-1)_CI )
			                       +   R(               m,                n)                                           // +   C^(-1)_IJ
			                       * ( R(voigt(dim, i, 0), voigt(dim, j, 0)) * R(voigt(dim, i, 1), voigt(dim, j, 1))   // * ( C^(-1)_AC * C^(-1)_BD
			            			 + R(voigt(dim, i, 0), voigt(dim, j, 1)) * R(voigt(dim, i, 1), voigt(dim, j, 0)))  //   + C^(-1)_AD * C^(-1)_BC
			                       +   R(voigt(dim, i, 0), voigt(dim, i, 1))                                           // +   C^(-1)_AB
			                       * ( R(voigt(dim, j, 0),                m) * R(voigt(dim, j, 1),               n)    // * ( C^(-1)_CI * C^(-1)_DJ
			            			 + R(voigt(dim, j, 0),                n) * R(voigt(dim, j, 1),               m) ); //   + C^(-1)_CJ * C^(-1)_DI )
                    //printf("temp(%d, %d) = %f\n", m, n, temp(m, n));
                }
            double temp2Data[3] = {0};
            DeviceMatrixView<T> temp2(temp2Data, 1, dim);
            E.transposeTime(temp, temp2);
            double temp3Data = 0;
            DeviceMatrixView<T> temp3(&temp3Data, 1, 1);
            temp2.times(E, temp3);
            C(i, j) = temp3Data;
        }
}

template <class T>
__device__
inline void electroelasticCouplingTensor(DeviceMatrixView<T> &C, const DeviceMatrixView<T> &R, const DeviceMatrixView<T> &E)
{
    int dim = R.cols();
    int dimTensor = (dim * (dim + 1)) / 2;
    for (int i = 0; i < dimTensor; i++)
        for (int j = 0; j < dim; j++)
        {
            T tempData[3];
            DeviceMatrixView<T> temp(tempData, dim, 1);
            for (int m = 0; m < dim; m++)
            {
                temp(m, 0) = R(voigt(dim, i, 0),                j) * R(voigt(dim, i, 1), m)  //   C^(-1)_AC * C^(-1)_BI
						   + R(voigt(dim, i, 0),                m) * R(voigt(dim, i, 1), j)  // + C^(-1)_AI * C^(-1)_BC
						   - R(voigt(dim, i, 0), voigt(dim, i, 1)) * R(               j, m); // - C^(-1)_AB * C^(-1)_CI
                //printf("temp(%d, %d) = %f\n", m, n, temp(m, n));
            }
            double temp2Data = 0;
            DeviceMatrixView<T> temp2(&temp2Data, 1, 1);
            E.transposeTime(temp, temp2);
            C(i,j) = temp2Data;
        }
}

template <class T>
__device__
inline void symmetricIdentityViewTensor_parallel(int tidx, int numThreadsx, int tidy, int numThreadsy,
    DeviceMatrixView<T> &C, const DeviceMatrixView<T> &R)
{
    int dim = R.cols();
    int dimTensor = (dim * (dim + 1)) / 2;
    for (int i = tidx; i < dimTensor; i += numThreadsx)
        for (int j = tidy; j < dimTensor; j += numThreadsy)
            C(i, j) = (R(voigt(dim, i, 0), voigt(dim, j, 0)) * R(voigt(dim, i, 1), voigt(dim, j, 1))
                           + R(voigt(dim, i, 0), voigt(dim, j, 1)) * R(voigt(dim, i, 1), voigt(dim, j, 0)));
}

template <class T>
__device__
inline void setB(DeviceMatrix<T>& B, const DeviceMatrix<T>& F, const DeviceVector<T>& bGrad)
{
    int dim = F.cols();
    int dimTensor = (dim * (dim + 1)) / 2;
    B.setZero(dimTensor, dim);

    for (int j = 0; j < dim; j++)
    {
        for (int i = 0; i < dim; i++)
            B(i,j) = F(j,i) * bGrad(i);
        if (dim == 2)
            B(2,j) = F(j,0) * bGrad(1) + F(j,1) * bGrad(0);
        if (dim == 3)
            for (int i = 0; i < dim; i++)
            {
                int k = (i+1)%3;
                B(i+dim,j) = F(j,i) * bGrad(k) + F(j,k) * bGrad(i);
            }
    }
}

template <class T>
__device__
inline void setBSingleDim(int dir, DeviceVectorView<T> B, DeviceMatrixView<T> F, DeviceVectorView<T> bGrad)
{
    int dim = F.cols();
    //int dimTensor = (dim * (dim + 1)) / 2;
    for (int i = 0; i < dim; i++)
        B(i) = F(dir,i) * bGrad(i);
    if (dim == 2)
        B(2) = F(dir,0) * bGrad(1) + F(dir,1) * bGrad(0);
    if (dim == 3)
        for (int i = 0; i < dim; i++)
        {
            int k = (i+1)%3;
            B(i+dim) = F(dir,i) * bGrad(k) + F(dir,k) * bGrad(i);
        }
}

template <class T>
__device__
inline void voigtStress(DeviceVector<T>& Svec, const DeviceMatrix<T>& S)
{
    int dim = S.cols();
    int dimTensor = (dim * (dim + 1)) / 2;
    Svec.setZero(dimTensor);
    for (int i = 0; i < dimTensor; i++)
        Svec(i) = S(voigt(dim, i, 0), voigt(dim, i, 1));
}

template <class T>
__device__
inline void voigtStressView(DeviceVectorView<T> Svec, DeviceMatrixView<T> S)
{
    int dim = S.cols();
    int dimTensor = (dim * (dim + 1)) / 2;
    for (int i = 0; i < dimTensor; i++)
        Svec(i) = S(voigt(dim, i, 0), voigt(dim, i, 1));
}

template <typename T>
__host__ __device__ __forceinline__
constexpr T dmin(const T& a, const T& b) noexcept
{
    return (b < a) ? b : a;
}

template <class T, class U>
__host__ __device__ __forceinline__
const T* upper_bound_ptr(const T* first, const T* last, const U& value)
{
    // returns first element > value
    auto count = last - first;
    while (count > 0) {
        auto step = count / 2;
        const T* mid = first + step;
        if (!(value < *mid)) {   // value >= *mid
            first = mid + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return first;
}

template <class It, class U>
__host__ __device__ __forceinline__
It upper_bound_it(It first, It last, const U& value)
{
    // returns first iterator it in [first,last) such that value < *it  (i.e., *it > value)
    auto count = last - first;              // requires random-access iterator
    while (count > 0) {
        auto step = count / 2;
        It mid = first + step;
        if (!(value < *mid)) {             // value >= *mid
            first = mid + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return first;
}

__device__ inline
void tensorBasisDerivative(int r, int P1, int dim, int numDerivatives,
    DeviceMatrixView<double> valuesAndDers, 
    DeviceVectorView<double> dN_r)
{
    int tensorCoordData[3]; //max 3D
    DeviceVectorView<int> tensorCoord(tensorCoordData, dim);
    getTensorCoordinate(dim, P1, r, tensorCoordData);
    for (int dir = 0; dir < dim; dir++)
    {
        double dN_rj = 1.0;
        for (int d = 0; d < dim; d++)
        {
            if (d == dir)
                dN_rj *= valuesAndDers(tensorCoord[d], (numDerivatives + 1) * d + 1);
            else
                dN_rj *= valuesAndDers(tensorCoord[d], (numDerivatives + 1) * d);
        }
        dN_r[dir] = dN_rj;
    }
}

__device__ inline
double tensorBasisValue(int r, int P1, int dim, int numDerivatives,
    DeviceMatrixView<double> valuesAndDers)
{
    int tensorCoordData[3]; //max 3D
    DeviceVectorView<int> tensorCoord(tensorCoordData, dim);
    getTensorCoordinate(dim, P1, r, tensorCoordData);
    double N_r = 1.0;
    for (int d = 0; d < dim; d++)
    {
        N_r *= valuesAndDers(tensorCoord[d], (numDerivatives + 1) * d);
    }
    return N_r;
}