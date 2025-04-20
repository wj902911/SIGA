#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <type_traits>
#include <cassert>
#include "Utility_d.h"
#include <Eigen/Core>

template <typename T>
class DeviceMatrix;


template <typename T>
__global__
void parallPlus(T* a, T* b, T* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) 
        c[i] = a[i] + b[i];
}
#if 1
template <typename T>
__global__
void parallPlus(T* a, T b, T* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) 
    {
        c[i] = a[i] + b;
        printf("c[%d] = %d + %d = %d\n", i, a[i], b, c[i]);
    }
}
#endif
template <typename T>
__global__
void parallMinus(T* a, T* b, T* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] - b[i];
}

template <typename T>
__global__
void parallMinus(T* a, T b, T* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] - b;
}

template <typename T>
__global__
void parallMult(T* a, T* b, T* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] * b[i];
}

template <typename T>
__global__
void parallMult(T* a, T b, T* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] * b;
}

template <typename T>
__global__
void parallDiv(T* a, T* b, T* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] / b[i];
}

template <typename T>
__global__
void parallDiv(T* a, T b, T* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] / b;
    }
}



template <typename Derived, typename T>
class DeviceMatrixBase 
{
public:
    __device__
    DeviceMatrix<T> transpose() const 
    {
        const Derived& derived = static_cast<const Derived&>(*this);
        DeviceMatrix<T> result(derived.cols(), derived.rows());
        for (int i = 0; i < derived.rows(); ++i) 
            for (int j = 0; j < derived.cols(); ++j) 
                result(j, i) = derived(i, j);
        return result;
    }

    __host__ __device__
#if 0
    void print(const char* chs = "") const 
    {
    #if defined(__CUDA_ARCH__)
        printf("%s", chs);
    #else
        printf("%s", chs);
    #endif
#else
    void print() const 
    {
#endif
        const Derived& derived = static_cast<const Derived&>(*this);
        for (int i = 0; i < derived.rows(); i++) 
        {
            for (int j = 0; j < derived.cols(); j++)
    #if defined(__CUDA_ARCH__)
                printElement(derived(i, j));
    #else
                printElement(T(derived.at(i, j)));
    #endif
            printf("\n");
        }
    }

#if 0
    __host__ __device__
    T* data() 
    {
        // Downcast to the actual derived type.
        Derived& derived = static_cast<Derived&>(*this);
        return &derived(0, 0); // Return pointer to the first element
    }
#endif

    __host__ __device__
    void setZero() 
    {
        // Downcast to the actual derived type.
        Derived& derived = static_cast<Derived&>(*this);
        for (int i = 0; i < derived.rows(); ++i) 
            for (int j = 0; j < derived.cols(); ++j) 
    #if defined(__CUDA_ARCH__)
                derived(i, j) = T(0);
    #else
                derived.at(i, j) = T(0);
    #endif
    }

#if 0
    template <typename OtherDerived>
    __device__
    Derived& operator=(const DeviceMatrixBase<OtherDerived, T>& other) {
        // Downcast 'other' to its actual derived type
        const OtherDerived& otherDerived = static_cast<const OtherDerived&>(other);
        // Downcast 'this' to Derived to access the correct rows(), cols(), etc.
        Derived& derived = static_cast<Derived&>(*this);

        // Check dimensions match
        assert(derived.rows() == otherDerived.rows() &&
               derived.cols() == otherDerived.cols() &&
               "Matrix dimensions mismatch for assignment");

        for (int i = 0; i < derived.rows(); ++i)
            for (int j = 0; j < derived.cols(); ++j)
                derived(i, j) = otherDerived(i, j);

        return derived;
    }
#endif
    // Derived classes must implement:
    // int rows() const;
    // int cols() const;
    // T operator()(int i, int j) const;
};


template <typename DerivedA, typename DerivedB, typename T>
__host__ __device__
DeviceMatrix<T> operator+(const DeviceMatrixBase<DerivedA, T>& A,
                          const DeviceMatrixBase<DerivedB, T>& B)
{
    const DerivedA& derivedA = static_cast<const DerivedA&>(A);
    const DerivedB& derivedB = static_cast<const DerivedB&>(B);
    assert(derivedA.rows() == derivedB.rows() &&
           derivedA.cols() == derivedB.cols() &&
           "Matrix dimensions mismatch for addition");

    DeviceMatrix<T> result(derivedA.rows(), derivedB.cols());
#if defined(__CUDA_ARCH__)
    for (int i = 0; i < derivedA.rows(); ++i)
        for (int j = 0; j < derivedA.cols(); ++j)
            result(i, j) = derivedA(i, j) + derivedB(i, j);
#else
    int size = derivedA.rows() * derivedA.cols(); 
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    T* d_A = derivedA.data();
    T* d_B = derivedB.data();
    parallPlus<T><<<numBlocks, blockSize>>>(d_A, d_B, result.data(), size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in operator+: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) 
        printf("Error in cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
#endif
    return result;
}

template <typename Derived, typename T>
__host__ __device__
Derived operator+(const DeviceMatrixBase<Derived, T>& matrix, T s) 
{
    // Downcast to the actual derived type.
    const Derived& derived = static_cast<const Derived&>(matrix);
    // Construct a result object with the same dimensions.
    Derived result(derived.rows(), derived.cols());

#if defined(__CUDA_ARCH__)
    for (int i = 0; i < derived.rows(); ++i) 
        for (int j = 0; j < derived.cols(); ++j) 
            result(i, j) = derived(i, j) + s;
#else
    int size = derived.rows() * derived.cols();
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    parallPlus<T><<<numBlocks, blockSize>>>(derived.data(), s, result.data(), size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in operator+: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) 
        printf("Error in cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
#endif
    return result;
}

template <typename Derived, typename T>
__device__
Derived operator+(T s, const DeviceMatrixBase<Derived, T>& matrix) 
{
    return matrix + s;
}

template <typename DerivedA, typename DerivedB, typename T>
__device__
DeviceMatrix<T> operator-(const DeviceMatrixBase<DerivedA, T>& A,
                          const DeviceMatrixBase<DerivedB, T>& B)
{
    assert(A.rows() == B.rows() && A.cols() == B.cols() 
           && "Matrix dimensions mismatch for addition");

    DeviceMatrix<T> result(A.rows(), A.cols());

    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            result(i, j) = A(i, j) - B(i, j);

    return result;
}

template <typename DerivedA, typename DerivedB, typename T>
__device__
DeviceMatrix<T> operator*(const DeviceMatrixBase<DerivedA, T>& A,
                          const DeviceMatrixBase<DerivedB, T>& B)
{
    assert(A.cols() == B.rows() && "Matrix dimensions mismatch for multiplication");

    DeviceMatrix<T> result(A.rows(), B.cols());

    for (int i = 0; i < A.rows(); ++i)
    {
        for (int j = 0; j < B.cols(); ++j)
        {
            T sum = 0;
            for (int k = 0; k < A.cols(); ++k)
            {
                sum += A(i, k) * B(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

template <typename Derived, typename T>
__device__
Derived operator*(const DeviceMatrixBase<Derived, T>& matrix, T s) 
{
    // Downcast to the actual derived type.
    const Derived& derived = static_cast<const Derived&>(matrix);
    // Construct a result object with the same dimensions.
    Derived result(derived.rows(), derived.cols());

    for (int i = 0; i < derived.rows(); ++i) 
        for (int j = 0; j < derived.cols(); ++j) 
            result(i, j) = derived(i, j) * s;
        
    return result;
}

template <typename Derived, typename T>
__device__
Derived operator*(T s, const DeviceMatrixBase<Derived, T>& matrix) 
{
    return matrix * s;
}

template <typename Derived, typename T>
__device__
Derived operator/(const DeviceMatrixBase<Derived, T>& matrix, T s) 
{
    // Downcast to the actual derived type.
    const Derived& derived = static_cast<const Derived&>(matrix);
    // Construct a result object with the same dimensions.
    Derived result(derived.rows(), derived.cols());

    for (int i = 0; i < derived.rows(); ++i) 
        for (int j = 0; j < derived.cols(); ++j) 
            result(i, j) = derived(i, j) / s;
        
    return result;
}


template <typename T>
class DeviceMatrix : public DeviceMatrixBase<DeviceMatrix<T>, T>
{
public:
    // Default constructor
    __host__ __device__ 
    DeviceMatrix() : m_rows(0), m_cols(0), m_data(nullptr), m_owns_data(true) {}

    __host__ __device__
    DeviceMatrix(int rows, int cols) : m_rows(rows), m_cols(cols), m_owns_data(true)
    {
    #if defined(__CUDA_ARCH__)
        m_data = new T[rows * cols]; 
    #else
        size_t size = static_cast<size_t>(rows * cols * sizeof(T));
        cudaError_t err = cudaMalloc((void**)&m_data, size);
        assert(err == cudaSuccess && "cudaMalloc failed");
    #endif
    }

    __host__ __device__
    DeviceMatrix(int rows, int cols, const T* data)
    : m_rows(rows), m_cols(cols), m_owns_data(true)
    {
    #if defined(__CUDA_ARCH__)
        m_data = new T[rows * cols];
        for (int i = 0; i < rows * cols; i++)
        {
            m_data[i] = data[i];
        }
    #else
        size_t size = static_cast<size_t>(rows * cols * sizeof(T));
        cudaError_t err = cudaMalloc((void**)&m_data, size);
        assert(err == cudaSuccess && "cudaMalloc failed");
        cudaMemcpy(m_data, data, size, cudaMemcpyHostToDevice);
        //printArray(data, m_temp.size());
        //printDeviceArray(m_data, m_rows * m_cols);
    #endif
    }

    __host__
    DeviceMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m)
        : m_rows(m.rows()), m_cols(m.cols()), m_owns_data(true)
    {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m_temp = m.transpose();
        size_t size = static_cast<size_t>(m_rows * m_cols * sizeof(T));
        cudaError_t err = cudaMalloc((void**)&m_data, size);
        assert(err == cudaSuccess && 
               "cudaMalloc failed in DeviceMatrix constructor from Eigen::Matrix");
        err = cudaMemcpy(m_data, m_temp.data(), size, cudaMemcpyHostToDevice);
        assert(err == cudaSuccess && 
               "cudaMemcpy failed in DeviceMatrix constructor from Eigen::Matrix");
        //printArray(m_temp.data(), m_temp.size());
        //printDeviceArray(m_data, m_rows * m_cols);
    }

    // Constructor that accepts an external data pointer and an ownership flag.
    __host__ __device__
    DeviceMatrix(int rows, int cols, T* data, bool owns_data)
        : m_rows(rows), m_cols(cols), m_data(data), m_owns_data(owns_data)
    {
    #if defined(__CUDA_ARCH__)
        if (owns_data && data == nullptr) 
        {
            m_data = new T[rows * cols];
        }
    #else
        if (owns_data && data == nullptr) 
        {
            size_t size = static_cast<size_t>(rows * cols * sizeof(T));
            cudaError_t err = cudaMalloc((void**)&m_data, size);
            assert(err == cudaSuccess && "cudaMalloc failed");
        }
    #endif
    }

    // Copy constructor
    __host__ __device__
    DeviceMatrix(const DeviceMatrix& other)
        : m_rows(other.m_rows), m_cols(other.m_cols)
    {
    #if defined(__CUDA_ARCH__)
        int size = m_rows * m_cols;
        m_data = new T[size];
        //printf("From DeviceMatrix copy constructor:\n");
        for (int i = 0; i < size; i++) 
        {
            m_data[i] = other.m_data[i];
            //printf("m_data[%d] = %f\n", i, m_data[i]);
        }
        //printf("\n");
    #else
        size_t size = static_cast<size_t>(m_rows * m_cols * sizeof(T));
        cudaError_t err = cudaMalloc((void**)&m_data, size);
        assert(err == cudaSuccess && "cudaMalloc failed in copy constructor");
        err = cudaMemcpy(m_data, other.m_data, size, cudaMemcpyDeviceToDevice);
        assert(err == cudaSuccess && "cudaMemcpy failed in copy constructor");
    #endif
    }

    // Copy assignment operator
    __host__ __device__
    DeviceMatrix& operator=(const DeviceMatrix& other)
    {
    #if defined(__CUDA_ARCH__)
        if (this != &other)
        {
            delete[] m_data;
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            int size = m_rows * m_cols;
            m_data = new T[size];
            for (int i = 0; i < size; i++) 
            {
                m_data[i] = other.m_data[i];
            }
        }
    #else
        if (this != &other)
        {
            cudaFree(m_data);
            m_data = nullptr;
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            size_t size = static_cast<size_t>(m_rows * m_cols * sizeof(T));
            cudaError_t err = cudaMalloc((void**)&m_data, size);
            assert(err == cudaSuccess && "cudaMalloc failed in copy assignment");
            err = cudaMemcpy(m_data, other.m_data, size, cudaMemcpyDeviceToDevice);
            assert(err == cudaSuccess && "cudaMemcpy failed in copy assignment");
        }
    #endif
        return *this;
    }

    // Move constructor
    __host__ __device__
    DeviceMatrix(DeviceMatrix&& other) noexcept
        : m_rows(other.m_rows), m_cols(other.m_cols), m_data(other.m_data)
    {
        other.m_data = nullptr;
        other.m_rows = 0;
        other.m_cols = 0;
    }

    // Move assignment operator
    __host__ __device__
    DeviceMatrix& operator=(DeviceMatrix&& other) noexcept
    {
        if (this != &other)
        {
        #if defined(__CUDA_ARCH__)
            delete[] m_data;
        #else
            cudaFree(m_data);
        #endif
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            m_data = other.m_data;
            other.m_data = nullptr;
            other.m_rows = 0;
            other.m_cols = 0;
        }
        return *this;
    }

    __host__ __device__
    ~DeviceMatrix()
    {
        // Only delete the data if we own it
        // and it was allocated (not null).
        if (m_owns_data && m_data != nullptr)
    #if defined(__CUDA_ARCH__)
            delete[] m_data;
    #else
        {
            cudaFree(m_data);
            m_data = nullptr;
        }
    #endif
    }


#if 0
    __device__
    void print() const
    {
        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
                printElement((*this)(i, j));
            printf("\n");
        }
    }
#endif

    __device__
    T& operator()(int row, int col)
    {
        return m_data[row * m_cols + col];
    }

    __device__
    const T& operator()(int row, int col) const
    {
        return m_data[row * m_cols + col];
    }
    
    __host__ __device__
    int rows() const
    {
        return m_rows;
    }

    __host__ __device__
    int cols() const
    {
        return m_cols;
    }

    __host__ __device__
    T* data() const
    {
        return m_data;
    }

    class Element;

// block proxy class ------------------------------------------------------------------------------------
    class Block : public DeviceMatrixBase<Block, T>
    {
    public:
        __host__ __device__
        Block(T* rowData, int startCol, int totalCols, int rows, int cols) 
            : m_data(rowData), m_startCol(startCol), m_totalCols(totalCols),
              m_rows(rows), m_cols(cols) {}

        __device__
        T& operator()(int i, int j) { return m_data[i * m_totalCols + m_startCol + j]; }

        __device__
        const T& operator()(int i, int j) const { return m_data[i * m_totalCols + m_startCol + j]; }

        __host__ __device__
        int rows() const { return m_rows; }

        __host__ __device__
        int cols() const { return m_cols; }

        __host__ __device__
        Block& operator=(const Block& other)
        {
            assert(m_rows == other.rows() && m_cols == other.cols() 
                   && "Matrix dimensions mismatch for assignment");
            for (int i = 0; i < m_rows; i++)
                for (int j = 0; j < m_cols; j++)
        #if defined(__CUDA_ARCH__)
                    m_data[i * m_cols + j] = other(i, j);
        #else
                {
                    cudaError_t err = cudaMemcpy(m_data + i * m_totalCols + m_startCol + j, 
                                                 other.m_data + i * m_totalCols + m_startCol + j, 
                                                 sizeof(T), cudaMemcpyDeviceToDevice);
                    assert(err == cudaSuccess && "cudaMemcpy failed in Block assignment");
                }
        #endif
            return *this;
        }

        template <typename Derived>
        __host__ __device__
        Block& operator=(const DeviceMatrixBase<Derived, T>& other)
        {
            const Derived& derived = static_cast<const Derived&>(other);

            assert(m_rows == derived.rows() && m_cols == derived.cols() 
                   && "Matrix dimensions mismatch for assignment");

            for (int i = 0; i < m_rows; i++)
                for (int j = 0; j < m_cols; j++)
        #if defined(__CUDA_ARCH__)
                    m_data[i * m_cols + j] = derived(i, j);
        #else
                {
                    cudaError_t err = cudaMemcpy(m_data + i * m_totalCols + m_startCol + j, 
                                                 other.m_data + i * m_totalCols + m_startCol + j, 
                                                 sizeof(T), cudaMemcpyDeviceToDevice);
                    assert(err == cudaSuccess && "cudaMemcpy failed in Block assignment");
                }
        #endif
            return *this;
        }

        __host__ __device__
        T* data() const
        { 
            return m_data; 
        }

        __host__ __device__
        int startCol() const
        { 
            return m_startCol; 
        }

        __host__
        Element at(int i, int j) const 
        { 
            assert(i >= 0 && i < m_rows && j >= 0 && j < m_cols && "Index out of bounds");
            return Element(m_data + i * m_totalCols, j, m_totalCols); 
        }

    private:
        T* m_data;
        //int m_startRow;
        int m_startCol;
        //int m_totalRows;
        int m_totalCols;
        int m_rows;
        int m_cols;
    };
// end of Block class -----------------------------------------------------------------------------------
    __host__ __device__
    Block block(int row, int col, int rows, int cols) 
    { 
        assert(row + rows <= m_rows && col + cols <= m_cols && "Block dimensions exceed matrix dimensions");
        assert(row >= 0 && col >= 0 && "Block starting index must be non-negative");
        assert(row < m_rows && col < m_cols && "Block starting index must be within matrix dimensions");
        return Block(m_data + row * m_cols, col, rows, cols); 
    }

    __host__ __device__
    Block middleRows(int startRow, int rows) 
    { 
        assert(startRow + rows <= m_rows && "Block dimensions exceed matrix dimensions");
        assert(startRow >= 0 && "Block starting index must be non-negative");
        assert(startRow < m_rows && "Block starting index must be within matrix dimensions");
        return Block(m_data + startRow * m_cols, 0, m_cols, rows, m_cols); 
    }

    __host__ __device__
    Block middleCols(int startCol, int cols) 
    { 
        assert(startCol + cols <= m_cols && "Block dimensions exceed matrix dimensions");
        assert(startCol >= 0 && "Block starting index must be non-negative");
        assert(startCol < m_cols && "Block starting index must be within matrix dimensions");
        return Block(m_data, startCol, m_cols, m_rows, cols);
    }

// element proxy class ----------------------------------------------------------------------------------
    class Element : public Block
    {
    public:
        __host__
        Element(T* data, int startCol, int totalCols) : Block(data, startCol, totalCols, 1, 1) {}

        __device__
        T& operator()(int /*ignored*/, int /*ignored*/) { return Block::operator()(0, 0); }

        __device__
        const T& operator()(int /*ignored*/, int /*ignored*/) const { return Block::operator()(0, 0); }

        __host__
        operator T() const
        {
            T value;
            cudaError_t err = cudaMemcpy(&value, this->data() + Block::startCol(), sizeof(T), cudaMemcpyDeviceToHost);
            assert(err == cudaSuccess && "cudaMemcpy failed in Element::operator T()");
            return value;
        }

        __host__
        T operator=(const T& value) 
        {
            cudaError_t err = cudaMemcpy(this->data() + Block::startCol(), &value, sizeof(T), cudaMemcpyHostToDevice);
            assert(err == cudaSuccess && "cudaMemcpy failed in Element::operator=");
            return value;
        }

        __host__
        Element at(int /*ignored*/, int /*ignored*/) const 
        { 
            return Block::at(0, 0); 
        }
    };
// end of Element class --------------------------------------------------------------------------------

    __host__
    Element at(int i, int j) const
    {
        //printf("at(%d, %d)\n", i, j);
        //printf("m_rows = %d, m_cols = %d\n", m_rows, m_cols);
        assert(i >= 0 && i < m_rows && j >= 0 && j < m_cols && "Index out of bounds");
        return Element(m_data + i * m_cols, j, m_cols); 
    }

// row proxy class --------------------------------------------------------------------------------------
#if 1
class Row : public Block
    {
    public:
        __host__ __device__
        Row(T* rowData, int cols) : Block(rowData, 0, cols, 1, cols) {}

        __device__
        T& operator()(int /*ignored*/, int j) { return Block::operator()(0, j); }

        __device__
        const T& operator()(int /*ignored*/, int j) const { return Block::operator()(0, j); }

#if 1
        __host__ __device__
        Element at(int j) const { return Block::at(0, j); }
#endif 
    };
#else
    class Row : public DeviceMatrixBase<Row, T>
    {
    public:
        __device__
        Row(T* rowData, int cols) : m_data(rowData), m_cols(cols) {}

        __device__
        T& operator()(int /*ignored*/, int j) { return m_data[j]; }

        __device__
        const T& operator()(int /*ignored*/, int j) const { return m_data[j]; }

        __device__
        int rows() const { return 1; }

        __device__
        int cols() const { return m_cols; }

        __device__
        Row& operator=(const Row& other)
        {
            for (int j = 0; j < m_cols; j++)
                m_data[j] = other(0, j);
            return *this;
        }

    private:
        T* m_data;
        int m_cols;
    }; 
#endif
// end of Row class --------------------------------------------------------------------------------------

    __host__ __device__
    Row row(int i) { return Row(m_data + i * m_cols, m_cols); }

// column proxy class -------------------------------------------------------------------------------------
#if 1
    class Col : public Block
    {
    public:
        __host__ __device__
        Col(T* data, int rows, int cols, int colIndex) : Block(data, colIndex, cols, rows, 1) {}

        __device__
        T& operator()(int i, int /*ignored*/) { return Block::operator()(i, 0); }

        __device__
        const T& operator()(int i, int /*ignored*/) const { return Block::operator()(i, 0); }

#if 1
        __host__ __device__
        Element at(int i) const { return Block::at(i, 0); }
#endif
    private:
        int m_colIndex;
    };
#else
    class Col : public DeviceMatrixBase<Col, T>
    {
    public:
        __device__
        Col(T* data, int rows, int cols, int colIndex) 
            : m_data(data), m_rows(rows), m_cols(cols), m_colIndex(colIndex) {}

        __device__
        T& operator()(int i, int /*ignored*/) { return m_data[i * m_cols + m_colIndex]; }

        __device__
        const T& operator()(int i, int /*ignored*/) const { return m_data[i * m_cols + m_colIndex]; }

        __device__
        int rows() const { return m_rows; }

        __device__
        int cols() const { return 1; }

        __device__
        Col& operator=(const Col& other)
        {
            for (int i = 0; i < m_rows; i++)
                (*this)(i, 0) = other(i, 0);
            return *this;
        }
    private:
        T* m_data;
        int m_rows;
        int m_cols;
        int m_colIndex;
    }; 
#endif
// end of Col class -------------------------------------------------------------------------------------

    __host__ __device__
    Col col(int j) { return Col(m_data, m_rows, m_cols, j); }

    __device__
    DeviceMatrix rowMinor(int row) const
    {
        DeviceMatrix minor(m_rows - 1, m_cols);
        for (int i = 0, i2 = 0; i < m_rows; i++)
        {
            if (i == row)
            {
                continue;
            }
            for (int j = 0; j < m_cols; j++)
            {
                minor(i2, j) = (*this)(i, j);
            }
            i2++;
        }
        return minor;
    }

    __device__
    DeviceMatrix colMinor(int col) const
    {
        DeviceMatrix minor(m_rows, m_cols - 1);
        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0, j2 = 0; j < m_cols; j++)
            {
                if (j == col)
                {
                    continue;
                }
                minor(i, j2) = (*this)(i, j);
                j2++;
            }
        }
        return minor;
    }

    __device__
    DeviceMatrix firstMinor(int row, int col) const
    {
        DeviceMatrix minor(m_rows - 1, m_cols - 1);
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
                minor(i2, j2) = (*this)(i, j);
                j2++;
            }
            i2++;
        }
        return minor;
    }

    __device__
    T cofactor(int row, int col) const
    {
        assert(m_rows == m_cols && "Matrix must be square for cofactor");

        return firstMinor(row, col).determinant() * ((row + col) % 2 == 0 ? 1 : -1);
    }

    __device__
    T determinant() const
    {
        assert(m_rows == m_cols && "Matrix must be square for determinant");

        if (m_rows == 1)
        {
            return (*this)(0, 0);
        }
        else if (m_rows == 2)
        {
            return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
        }
        else
        {
            T det = 0;
            for (int j = 0; j < m_cols; j++)
            {
                det += (*this)(0, j) * cofactor(0, j);
            }
            return det;
        }
    }

    __host__ __device__
    void resize(int rows, int cols)
    {
    #if defined(__CUDA_ARCH__)
        delete[] m_data;
        m_rows = rows;
        m_cols = cols;
        m_data = new T[rows * cols];
    #else
        if (m_data) 
        {
            cudaFree(m_data);
            m_data = nullptr;
        }
        size_t size = static_cast<size_t>(rows * cols * sizeof(T));
        cudaError_t err = cudaMalloc((void**)&m_data, size);
        assert(err == cudaSuccess && "cudaMalloc failed in resize");
        m_rows = rows;
        m_cols = cols;
    #endif
    }

    __host__ __device__
    void setZero()
    {
        DeviceMatrixBase<DeviceMatrix<T>, T>::setZero();
    }

    __host__ __device__
    void setZero(int rows, int cols)
    {
        resize(rows, cols);
        DeviceMatrixBase<DeviceMatrix<T>, T>::setZero();
    }

    __device__
    T prod() const
    {
        T prod = 1;
        for (int i = 0; i < m_rows; i++)
            for (int j = 0; j < m_cols; j++)
                prod *= (*this)(i, j);
        return prod;
    }

    __device__
    T sum() const
    {
        T sum = 0;
        for (int i = 0; i < m_rows; i++)
            for (int j = 0; j < m_cols; j++)
                sum += (*this)(i, j);
        return sum;
    }

private:
    int m_rows = 0;
    int m_cols = 0;
    T* m_data = nullptr;
    bool m_owns_data;
};

template <typename T>
class DeviceMatrixView : public DeviceMatrix<T>
{
public:
    __host__ __device__
    DeviceMatrixView() : DeviceMatrix<T>(0, 0, nullptr, false) {}

    __host__ __device__
    DeviceMatrixView(int rows, int cols)
    : DeviceMatrix<T>(rows, cols, nullptr, false) {}

    __host__ __device__
    DeviceMatrixView(int rows, int cols, T* data)
    : DeviceMatrix<T>(rows, cols, data, false) {}

    __host__ __device__
    ~DeviceMatrixView() {}
};