#include "DeviceCSRMatrix.h"

#include <thrust/device_vector.h>
#include <cusparse.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/iterator/zip_iterator.h>
#include <stdexcept>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        std::abort();                                                          \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        std::abort();                                                          \
    }                                                                          \
}

DeviceCSRMatrix::DeviceCSRMatrix(int numRows, int numCols,
                                 DeviceVectorView<int> cooR, 
                                 DeviceVectorView<int> cooC, 
                                 DeviceVectorView<double> cooV)

{
    m_numCols = numCols;
    const int nnz_coo = static_cast<int>(cooV.size());

    thrust::device_vector<int>    R(cooR.data(), cooR.data() + nnz_coo);
    thrust::device_vector<int>    C(cooC.data(), cooC.data() + nnz_coo);
    thrust::device_vector<double> V(cooV.data(), cooV.data() + nnz_coo);

    auto keys_begin = thrust::make_zip_iterator(thrust::make_tuple(R.begin(), C.begin()));
    auto keys_end   = thrust::make_zip_iterator(thrust::make_tuple(R.end(),   C.end()));

    thrust::sort_by_key(keys_begin, keys_end, V.begin());

    thrust::device_vector<int>    R2(nnz_coo);
    thrust::device_vector<int>    C2(nnz_coo);
    thrust::device_vector<double> V2(nnz_coo);

    auto out_keys_begin = thrust::make_zip_iterator(thrust::make_tuple(R2.begin(), C2.begin()));

    auto new_ends = thrust::reduce_by_key(keys_begin, keys_end, V.begin(), out_keys_begin, V2.begin());
    const int nnz = static_cast<int>(new_ends.second - V2.begin());

    m_rowPtr.resize(numRows + 1);
    m_colInd.resize(nnz);
    m_values.resize(nnz);

    CHECK_CUDA(cudaMemcpy(m_colInd.data(), thrust::raw_pointer_cast(C2.data()), 
                              nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(m_values.data(), thrust::raw_pointer_cast(V2.data()), 
                              nnz * sizeof(double), cudaMemcpyDeviceToDevice));

    cusparseHandle_t cH = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&cH));

    CHECK_CUSPARSE(cusparseXcoo2csr(
        cH,
        thrust::raw_pointer_cast(R2.data()),
        nnz,
        numRows,
        m_rowPtr.data(),
        CUSPARSE_INDEX_BASE_ZERO));

    CHECK_CUSPARSE(cusparseDestroy(cH));
}

void DeviceCSRMatrix::setFromCOO(int numRows, int numCols,
                                 DeviceVectorView<int> cooR, 
                                 DeviceVectorView<int> cooC, 
                                 DeviceVectorView<double> cooV)
{
    m_numCols = numCols;
    const int nnz_coo = static_cast<int>(cooV.size());

    thrust::device_vector<int>    R(cooR.data(), cooR.data() + nnz_coo);
    thrust::device_vector<int>    C(cooC.data(), cooC.data() + nnz_coo);
    thrust::device_vector<double> V(cooV.data(), cooV.data() + nnz_coo);

    auto keys_begin = thrust::make_zip_iterator(thrust::make_tuple(R.begin(), C.begin()));
    auto keys_end   = thrust::make_zip_iterator(thrust::make_tuple(R.end(),   C.end()));

    thrust::sort_by_key(keys_begin, keys_end, V.begin());

    thrust::device_vector<int>    R2(nnz_coo);
    thrust::device_vector<int>    C2(nnz_coo);
    thrust::device_vector<double> V2(nnz_coo);

    auto out_keys_begin = thrust::make_zip_iterator(thrust::make_tuple(R2.begin(), C2.begin()));

    auto new_ends = thrust::reduce_by_key(keys_begin, keys_end, V.begin(), out_keys_begin, V2.begin());
    const int nnz = static_cast<int>(new_ends.second - V2.begin());

    m_rowPtr.resize(numRows + 1);
    m_colInd.resize(nnz);
    m_values.resize(nnz);

    CHECK_CUDA(cudaMemcpy(m_colInd.data(), thrust::raw_pointer_cast(C2.data()), 
                              nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(m_values.data(), thrust::raw_pointer_cast(V2.data()), 
                              nnz * sizeof(double), cudaMemcpyDeviceToDevice));

    cusparseHandle_t cH = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&cH));

    CHECK_CUSPARSE(cusparseXcoo2csr(
        cH,
        thrust::raw_pointer_cast(R2.data()),
        nnz,
        numRows,
        m_rowPtr.data(),
        CUSPARSE_INDEX_BASE_ZERO));

    CHECK_CUSPARSE(cusparseDestroy(cH));
}

void DeviceCSRMatrix::setFromCOO(int numRows, int numCols, 
                                 DeviceVectorView<int> cooR, 
                                 DeviceVectorView<int> cooC)
{
    m_numCols = numCols;
    const int nnz_coo = static_cast<int>(cooR.size());
    std::cout << "Setting CSR matrix from COO with " << nnz_coo << " entries." << std::endl;
    thrust::device_vector<int> R(cooR.data(), cooR.data() + nnz_coo);
    thrust::device_vector<int> C(cooC.data(), cooC.data() + nnz_coo);

    auto keys_begin = thrust::make_zip_iterator(thrust::make_tuple(R.begin(), C.begin()));
    auto keys_end   = thrust::make_zip_iterator(thrust::make_tuple(R.end(),   C.end()));

    // 1. Sort by (row, col)
	auto start = std::chrono::high_resolution_clock::now();
    thrust::sort(keys_begin, keys_end);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Sorted COO entries in " << elapsed.count() << " s." << std::endl;

    // 2. Remove duplicate (row, col) pairs
    auto new_end = thrust::unique(keys_begin, keys_end);
    const int nnz = static_cast<int>(new_end - keys_begin);
    std::cout << "Reduced COO entries to " << nnz << " unique entries." << std::endl;

    m_rowPtr.resize(numRows + 1);
    m_colInd.resize(nnz);
    m_values.resize(nnz);
    
    // copy unique column indices
    CHECK_CUDA(cudaMemcpy(m_colInd.data(),
                          thrust::raw_pointer_cast(C.data()),
                          nnz * sizeof(int),
                          cudaMemcpyDeviceToDevice));

    cusparseHandle_t cH = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&cH));

    // build rowPtr from unique row array
    CHECK_CUSPARSE(cusparseXcoo2csr(
        cH,
        thrust::raw_pointer_cast(R.data()),
        nnz,
        numRows,
        m_rowPtr.data(),
        CUSPARSE_INDEX_BASE_ZERO));

    CHECK_CUSPARSE(cusparseDestroy(cH));
}

Eigen::SparseMatrix<double, Eigen::RowMajor, int>
DeviceCSRMatrix::toEigenCSR() const
{
    const int numRows = static_cast<int>(m_rowPtr.size()) - 1;
    const int numCols = m_numCols;
    const int nnz     = static_cast<int>(m_values.size());

    if (numRows < 0)
        throw std::runtime_error("DeviceCSRMatrix::toEigenCSR: invalid rowPtr size.");

    if (m_colInd.size() != m_values.size())
        throw std::runtime_error("DeviceCSRMatrix::toEigenCSR: colInd and values size mismatch.");

    // ---- Allocate host buffers ----
    std::vector<int>    h_rowPtr(m_rowPtr.size());
    std::vector<int>    h_colInd(m_colInd.size());
    std::vector<double> h_values(m_values.size());

    // ---- Copy device → host ----
    cudaMemcpy(h_rowPtr.data(), m_rowPtr.data(),
               h_rowPtr.size() * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(h_colInd.data(), m_colInd.data(),
               h_colInd.size() * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(h_values.data(), m_values.data(),
               h_values.size() * sizeof(double),
               cudaMemcpyDeviceToHost);

    // ---- Wrap CSR arrays using Eigen Map ----
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;

    Eigen::Map<const SpMat> mapped(
        numRows,
        numCols,
        nnz,
        h_rowPtr.data(),
        h_colInd.data(),
        h_values.data()
    );

    // ---- Return owning copy ----
    return SpMat(mapped);
}

void DeviceCSRMatrix::print_host() const
{
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> hostCSRMat = toEigenCSR();
    Eigen::MatrixXd denseMat(hostCSRMat);
    std::cout << denseMat << std::endl;
}

void DeviceCSRMatrix::sparsePrint_host() const
{
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> hostCSRMat = toEigenCSR();
    std::cout << hostCSRMat << std::endl;
}

void DeviceCSRMatrix::toDense(DeviceMatrixView<double> denseMat) const
{
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB;

    CHECK_CUSPARSE(
        cusparseCreateCsr(
            &matA,
            numRows(),
            numCols(),
            numNonZeros(),
            m_rowPtr.data(),
            m_colInd.data(),
            m_values.data(),
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F
        ));

    CHECK_CUSPARSE(
        cusparseCreateDnMat(
            &matB,
            denseMat.rows(),
            denseMat.cols(),
            denseMat.rows(),
            denseMat.data(),
            CUDA_R_64F,
            CUSPARSE_ORDER_COL
        ));

    size_t bufferSize = 0;
    CHECK_CUSPARSE(
        cusparseSparseToDense_bufferSize(
            handle,
            matA,
            matB,
            CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
            &bufferSize
        ));

    void* dBuffer = nullptr;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(
        cusparseSparseToDense(
            handle,
            matA,
            matB,
            CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
            dBuffer
        ));

    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroy(handle));
}
