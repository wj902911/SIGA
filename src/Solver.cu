#include "Solver.h"
#include <cusparse.h>
#include <cusolverSp.h>

#if 0
#define CUDA_CHECK(err) do { \
    cudaError_t _e = (err); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        std::abort(); \
    } \
} while(0)

#define CUSPARSE_CHECK(stat) do { \
    cusparseStatus_t _s = (stat); \
    if (_s != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error %s:%d: %d\n", __FILE__, __LINE__, int(_s)); \
        std::abort(); \
    } \
} while(0)
#endif

#define CUSOLVER_CHECK(stat) do { \
    cusolverStatus_t _s = (stat); \
    if (_s != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSOLVER error %s:%d: %d\n", __FILE__, __LINE__, int(_s)); \
        std::abort(); \
    } \
} while(0)

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



Solver::Solver(Assembler &assembler): m_assembler(assembler)
{
    m_solVector.setZero(assembler.numDofs());
}

Solver::~Solver()
{
}

bool Solver::solveSingleIteration()
{
#if 0
    const DeviceMatrix<double>& Adev = m_assembler.matrix();
    const int m = Adev.rows();
    const int n = Adev.cols();
    assert(m == n && "This example expects a square system.");

    void* dA = const_cast<void*>(static_cast<const void*>(Adev.data()));

    double* d_csr_offsets = nullptr;
    CUDA_CHECK( cudaMalloc((void**) &d_csr_offsets,
                           (m + 1) * sizeof(double)) );

    cusparseHandle_t cusparseH = nullptr;
    CUSPARSE_CHECK(cusparseCreate(&cusparseH));
    
    cusparseDnMatDescr_t dnA = nullptr;
    const int64_t rows = m, cols = n, ld = n;
    CUSPARSE_CHECK(cusparseCreateDnMat(&dnA, 
                                       rows, cols, ld,
                                       dA,
                                       CUDA_R_64F,
                                       CUSPARSE_ORDER_ROW));

    cusparseSpMatDescr_t spA = nullptr;
    int64_t csrRows = rows, csrCols = cols, csrNNZ = 0;
    CUSPARSE_CHECK(cusparseCreateCsr(&spA, csrRows, csrCols, csrNNZ,
                                     d_csr_offsets, nullptr, nullptr,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_64F));

    size_t bufferSize = 0;
    void*  dBuffer    = nullptr;

    CUSPARSE_CHECK(cusparseDenseToSparse_bufferSize(
        cusparseH, dnA, spA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize));

    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    CUSPARSE_CHECK(cusparseDenseToSparse_analysis(
        cusparseH, dnA, spA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));

    CUDA_CHECK(cudaFree(dBuffer));
    CUDA_CHECK(cudaFree(d_csr_offsets));
    CUSPARSE_CHECK(cusparseDestroySpMat(spA));
    CUSPARSE_CHECK(cusparseDestroyDnMat(dnA));
    CUSPARSE_CHECK(cusparseDestroy(cusparseH));
#endif
    m_assembler.assemble(m_solVector);

    const DeviceMatrix<double>& Adev = m_assembler.matrix();
    const DeviceVector<double>& bdev = m_assembler.rhs();
    int num_rows = Adev.rows();
    int num_cols = Adev.cols();
    int ld       = num_cols;
    DeviceVector<double> solutionVector(num_rows);

    //std::cout << "m_solVector:\n";
    //m_solVector.print();
    //std::cout << "bdev:\n";
    //bdev.print();
    //std::cout << "Adev:\n";
    //Adev.print();
    //--------------------------------------------------------------------------
    // Device memory management
    int   *d_csr_offsets, *d_csr_columns;
    double *d_csr_values;
    CHECK_CUDA( cudaMalloc((void**) &d_csr_offsets, (num_rows + 1) * sizeof(int)) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, num_rows, num_cols, ld, Adev.data(),
                                        CUDA_R_64F, CUSPARSE_ORDER_ROW) )
    // Create sparse matrix B in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, num_rows, num_cols, 0,
                                      d_csr_offsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize(
                                        handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_analysis(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp,
                                         &nnz) )
    // allocate CSR column indices and values
    CHECK_CUDA( cudaMalloc((void**) &d_csr_columns, nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &d_csr_values,  nnz * sizeof(float)) )
    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE( cusparseCsrSetPointers(matB, d_csr_offsets, d_csr_columns,
                                           d_csr_values) )
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_convert(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )

    cusolverSpHandle_t solverH = nullptr;
    CUSOLVER_CHECK(cusolverSpCreate(&solverH));
    cusparseMatDescr_t descrA_legacy = nullptr;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA_legacy))
    CHECK_CUSPARSE(cusparseSetMatType(descrA_legacy, CUSPARSE_MATRIX_TYPE_GENERAL))
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA_legacy, CUSPARSE_INDEX_BASE_ZERO))
    double tol = 1e-12;
    int    reorder = 1;
    int singularity = -1;
    CUSOLVER_CHECK(cusolverSpDcsrlsvchol(solverH,
                                        num_rows,
                                        nnz,
                                        descrA_legacy,
                                        d_csr_values,
                                        d_csr_offsets,
                                        d_csr_columns,
                                        bdev.data(),
                                        tol,
                                        reorder,
                                        solutionVector.data(),
                                        &singularity));
    if (singularity >= 0)
        printf("WARNING: The matrix is singular at row %d under tol %E\n", singularity, tol);

    double* host_sol = new double[num_rows];
    double* host_residual = new double[num_rows];
    CHECK_CUDA( cudaMemcpy(host_sol, solutionVector.data(), sizeof(double)*num_rows, cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(host_residual, bdev.data(), sizeof(double)*num_rows, cudaMemcpyDeviceToHost) )
    Eigen::Map<Eigen::VectorXd> solvec(host_sol, num_rows);
    Eigen::Map<Eigen::VectorXd> resvec(host_residual, num_rows);

    m_updateNorm = solvec.norm();
    m_residualNorm = resvec.norm();
    m_solVector = m_solVector + solutionVector;

    if (m_numIterations == 0)
    {
        m_initResidualNorm = m_residualNorm;
        m_initUpdateNorm = m_updateNorm;
    }

    m_numIterations++;

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    CUSOLVER_CHECK(cusolverSpDestroy(solverH));
    CHECK_CUSPARSE( cusparseDestroyMatDescr(descrA_legacy) )
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(d_csr_offsets) )
    CHECK_CUDA( cudaFree(d_csr_columns) )
    CHECK_CUDA( cudaFree(d_csr_values) )
    delete[] host_sol;
    delete[] host_residual;


    return true;
}

void Solver::solve()
{
    double absTol = 1e-10;
    double relTol = 1e-9;
    int maxIterations = 50;
    m_numIterations = 0;
    m_status = working;
    while (m_status == working)
    {
        if(!solveSingleIteration())
        {
            m_status = bad_solution;
            break;
        }
        std::cout << status() << std::endl;
        if (m_residualNorm < absTol || 
            m_updateNorm < absTol || 
            m_residualNorm/m_initResidualNorm < relTol ||
            m_updateNorm/m_initUpdateNorm < relTol)
            m_status = converged;
        else if (m_numIterations >= maxIterations)
            m_status = interrupted;
    }

    std::cout << status() << std::endl;
}

std::string Solver::status()
{
    std::string statusString;
if (m_status == converged)
    statusString = "Iterative solver converged after " +
             std::to_string(m_numIterations) + " iteration(s).";
else if (m_status == interrupted)
    statusString = "Iterative solver was interrupted after " +
            std::to_string(m_numIterations) + " iteration(s).";
else if (m_status == bad_solution)
    statusString = "Iterative solver was interrupted after " +
            std::to_string(m_numIterations) + " iteration(s) due to an invalid solution";
else if (m_status == working)
    statusString = "It: " + std::to_string(m_numIterations) +
             ", updAbs: " + std::to_string(m_updateNorm) +
             ", updRel: " + std::to_string(m_updateNorm/m_initUpdateNorm) +
             ", resAbs: " + std::to_string(m_residualNorm) +
             ", resRel: " + std::to_string(m_residualNorm/m_initResidualNorm);
return statusString;
}
