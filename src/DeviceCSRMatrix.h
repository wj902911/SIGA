#pragma once

#include <cuda_runtime.h>
#include <DeviceArray.h>
#include <DeviceVectorView.h>
#include <Eigen/Sparse>
#include <DeviceMatrixView.h>

class DeviceCSRMatrixView
{
private:
    int m_numCols = 0;
    DeviceVectorView<int> m_rowPtr;
    DeviceVectorView<int> m_colInd;
    DeviceVectorView<double> m_values;

public:
    __host__
    DeviceCSRMatrixView() = default;

    __host__
    DeviceCSRMatrixView(int numCols,
                        DeviceVectorView<int> rowPtr, 
                        DeviceVectorView<int> colInd, 
                        DeviceVectorView<double> values)
    : m_numCols(numCols), m_rowPtr(rowPtr), m_colInd(colInd), m_values(values) 
    {}

    __device__
    double& operator()(int row, int col)
    {
        const int start = m_rowPtr[row];
        const int end   = m_rowPtr[row + 1];
        
        int lo = start;
        int hi = end - 1;
        while (lo <= hi)
        {
            const int mid = (lo + hi) >> 1;
            const int cc  = m_colInd[mid];
            if (cc < col) lo = mid + 1;
            else if (cc > col) hi = mid - 1;
            else return m_values[mid];
        }
        // If we reach here, the element is zero and not stored in the CSR format
        // We can either return a reference to a temporary variable or handle this case differently
        // For simplicity, we'll return a reference to a static variable (not thread-safe)
        static double zero = 0.0;
        return zero; // Note: Modifying this will not change the actual matrix
    }

    __device__
    double operator()(int row, int col) const
    {
        const int start = m_rowPtr[row];
        const int end   = m_rowPtr[row + 1];
        
        int lo = start;
        int hi = end - 1;
        while (lo <= hi)
        {
            const int mid = (lo + hi) >> 1;
            const int cc  = m_colInd[mid];
            if (cc < col) lo = mid + 1;
            else if (cc > col) hi = mid - 1;
            else return m_values[mid];
        }
        return 0.0; // Element is zero and not stored in the CSR format
    }
};


class DeviceCSRMatrix
{
private:
    int m_numCols = 0;
    DeviceArray<int> m_rowPtr;
    DeviceArray<int> m_colInd;
    DeviceArray<double> m_values;
public:
    __host__
    DeviceCSRMatrix() = default;

    __host__
    DeviceCSRMatrix(int numRows, int numCols,
                    DeviceVectorView<int> cooR, 
                    DeviceVectorView<int> cooC, 
                    DeviceVectorView<double> cooV);

    __host__
    void setFromCOO(int numRows, int numCols,
                    DeviceVectorView<int> cooR, 
                    DeviceVectorView<int> cooC, 
                    DeviceVectorView<double> cooV);

    __host__
    void setFromCOO(int numRows, int numCols,
                    DeviceVectorView<int> cooR, 
                    DeviceVectorView<int> cooC);

    __host__
    DeviceCSRMatrixView view() const
    {
        return DeviceCSRMatrixView(m_numCols, 
            m_rowPtr.vectorView(), 
            m_colInd.vectorView(), 
            m_values.vectorView());
    }

    __host__
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> toEigenCSR() const;

    __host__
    int numCols() const { return m_numCols; }

    __host__
    int numRows() const { return m_rowPtr.size() > 0 ? m_rowPtr.size() - 1 : 0; }

    __host__
    int numNonZeros() const { return m_colInd.size(); }

    __host__
    DeviceVectorView<int> rowPtr() const { return m_rowPtr.vectorView(); }

    __host__
    DeviceVectorView<int> colInd() const { return m_colInd.vectorView(); }

    __host__
    DeviceVectorView<double> values() const { return m_values.vectorView(); }

    __host__
    void print_host() const;

    __host__
    void sparsePrint_host() const;

    __host__
    void setZero()
    {
        cudaError_t err = cudaMemset(m_values.data(), 0, m_values.size() * sizeof(double));
        assert(err == cudaSuccess && "cudaMemset failed in DeviceCSRMatrix::setZero");
    }
    
    __host__
    void toDense(DeviceMatrixView<double> denseMat) const;
};