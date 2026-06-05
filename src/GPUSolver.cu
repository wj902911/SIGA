#include <GPUSolver.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
//#include <cudss.h>
#include <Utility_h.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/Eigenvalues>
#if ENABLE_PARDISO
#include <Eigen/PardisoSupport>
#endif

#if ENABLE_SPECTRA
#include "SpectraEigenSolver.h"
#endif

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>

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

#if ENABLE_AMGX
#define AMGX_CHECK(call) AMGX_SAFE_CALL(call)

namespace
{
const char* amgxSolveStatusName(AMGX_SOLVE_STATUS status)
{
    switch (status)
    {
    case AMGX_SOLVE_SUCCESS:
        return "success";
    case AMGX_SOLVE_FAILED:
        return "failed";
    case AMGX_SOLVE_DIVERGED:
        return "diverged";
    case AMGX_SOLVE_NOT_CONVERGED:
        return "not converged";
    default:
        return "unknown";
    }
}
} // namespace
#endif

namespace
{
const char* cpuDirectSolverName()
{
#if ENABLE_PARDISO
    return "Eigen PARDISO LDLT";
#else
    return "Eigen SparseLU";
#endif
}
} // namespace

#if ENABLE_PARDISO
bool GPUSolver::solveWithPardiso(const PardisoSpMat& A,
                                 const Eigen::VectorXd& b,
                                 Eigen::VectorXd& x,
                                 const char* context)
{
    const bool matrixShapeChanged =
        A.rows() != m_pardisoRows ||
        A.cols() != m_pardisoCols ||
        A.nonZeros() != m_pardisoNonZeros;

    if (!m_pardisoPatternAnalyzed || matrixShapeChanged)
    {
        m_pardisoSolver.analyzePattern(A);
        if (m_pardisoSolver.info() != Eigen::Success)
        {
            printf("Eigen PARDISO LDLT analyzePattern failed in %s\n", context);
            m_pardisoPatternAnalyzed = false;
            return false;
        }

        m_pardisoRows = static_cast<int>(A.rows());
        m_pardisoCols = static_cast<int>(A.cols());
        m_pardisoNonZeros = static_cast<int>(A.nonZeros());
        m_pardisoPatternAnalyzed = true;
    }

    m_pardisoSolver.factorize(A);
    if (m_pardisoSolver.info() != Eigen::Success)
    {
        printf("Eigen PARDISO LDLT factorization failed in %s\n", context);
        return false;
    }

    x = m_pardisoSolver.solve(b);
    if (m_pardisoSolver.info() != Eigen::Success)
    {
        printf("Eigen PARDISO LDLT solve failed in %s\n", context);
        return false;
    }

    return true;
}
#endif

struct BlockedToInterleavedND
{
    int N; // n / d
    int d; // dimension

    __host__ __device__ int operator()(int i) const
    {
        // i = comp*N + node
        int comp = i / N;
        int node = i - comp * N; // i % N
        // j = node*d + comp
        return node * d + comp;
    }
};

struct InterleavedToBlockedND
{
    int N; // n / d
    int d;

    __host__ __device__ int operator()(int j) const
    {
        // j = node*d + comp
        int node = j / d;
        int comp = j - node * d; // j % d
        // i = comp*N + node
        return comp * N + node;
    }
};

__global__
void printKernel(DeviceVectorView<double> solVector,
                 DeviceNestedArrayView<double> fixedDoFs)
{
    printf("Solution Vector:\n");
    solVector.print();
    printf("Fixed DoFs:\n");
    fixedDoFs.print();
}

__host__
GPUSolver::GPUSolver(GPUAssembler &assembler): m_assembler(assembler)
{
    Eigen::VectorXd solVector(assembler.numDofs());
    solVector.setZero();
    m_solVector = solVector;
    m_deltaSolVector = solVector;
    m_fixedDoFs = assembler.allFixedDofs();
    m_fixedDoFs.setZero();

#if ENABLE_AMGX
    initAMGXOnce();
#endif
}

__host__
void GPUSolver::print() const
{
    m_assembler.print();
    printKernel<<<1,1>>>(m_solVector.vectorView(), 
                         m_fixedDoFs.view());
    cudaError_t err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUSolver::print");
}

#if 0
bool GPUSolver::solveSingleIteration()
{
	auto start = std::chrono::high_resolution_clock::now();
    m_assembler.assemble(m_solVector.vectorView(), m_numIterations, m_fixedDoFs.view());
    auto end = std::chrono::high_resolution_clock::now();
    printf("Assemble time: %f seconds\n", std::chrono::duration<double>(end - start).count());

    //DeviceMatrixView<double> Adev = m_assembler.matrix();
    DeviceVectorView<double> bdev = m_assembler.rhs();
    const int n = static_cast<int>(bdev.size());

	start = std::chrono::high_resolution_clock::now();
    // --- COO from assembler (device pointers) ---
    auto cooR = m_assembler.rows();    // int, length nnz_coo
    auto cooC = m_assembler.cols();    // int, length nnz_coo
    auto cooV = m_assembler.values();  // double, length nnz_coo
    const int nnz_coo = static_cast<int>(cooV.size());

    // --- Copy to thrust vectors so we can sort/reduce (do not mutate assembler arrays) ---
    thrust::device_vector<int>    R(cooR.data(), cooR.data() + nnz_coo);
    thrust::device_vector<int>    C(cooC.data(), cooC.data() + nnz_coo);
    thrust::device_vector<double> V(cooV.data(), cooV.data() + nnz_coo);

    // key = (row,col)
    auto keys_begin = thrust::make_zip_iterator(thrust::make_tuple(R.begin(), C.begin()));
    auto keys_end   = thrust::make_zip_iterator(thrust::make_tuple(R.end(),   C.end()));

    // 1) Sort by (row,col)
    thrust::sort_by_key(keys_begin, keys_end, V.begin());

    // 2) Reduce duplicates: (row,col) identical -> sum values
    thrust::device_vector<int>    R2(nnz_coo);
    thrust::device_vector<int>    C2(nnz_coo);
    thrust::device_vector<double> V2(nnz_coo);

    auto out_keys_begin = thrust::make_zip_iterator(thrust::make_tuple(R2.begin(), C2.begin()));

    auto new_ends = thrust::reduce_by_key(
        keys_begin, keys_end,
        V.begin(),
        out_keys_begin,
        V2.begin());

    // new nnz after merging duplicates
    const int nnz = static_cast<int>(new_ends.second - V2.begin());

    // --- Build CSR arrays (device malloc) ---
    int*    d_csr_offsets = nullptr;
    int*    d_csr_cols    = nullptr;
    double* d_csr_vals    = nullptr;

    CHECK_CUDA(cudaMalloc(&d_csr_offsets, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_cols,    nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_vals,    nnz * sizeof(double)));

    // Copy reduced (col,val) into CSR col/val buffers
    CHECK_CUDA(cudaMemcpy(d_csr_cols,
                          thrust::raw_pointer_cast(C2.data()),
                          nnz * sizeof(int),
                          cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_csr_vals,
                          thrust::raw_pointer_cast(V2.data()),
                          nnz * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    // COO row-indices (reduced) -> CSR row offsets
    cusparseHandle_t cH = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&cH));
    CHECK_CUSPARSE(cusparseXcoo2csr(
        cH,
        thrust::raw_pointer_cast(R2.data()),
        nnz,
        n,
        d_csr_offsets,
        CUSPARSE_INDEX_BASE_ZERO));
    auto mid = std::chrono::high_resolution_clock::now();
    printf("COO->CSR conversion time: %f seconds\n", std::chrono::duration<double>(mid - start).count());
    
    // --- Solve Ax=b from CSR ---
    cusolverSpHandle_t sH = nullptr;
    CUSOLVER_CHECK(cusolverSpCreate(&sH));

    cusparseMatDescr_t descrA = nullptr;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));

    // IMPORTANT for csrlsvchol:
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)); // <- change this
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // Optional (usually fine to set; solver uses one triangle)
    CHECK_CUSPARSE(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT));

    const double tol = 1e-12;
    const int reorder = 1;
    int singularity = -1;

    CUSOLVER_CHECK(cusolverSpDcsrlsvchol(
        sH, n, nnz,
        descrA,
        d_csr_vals,
        d_csr_offsets,
        d_csr_cols,
        bdev.data(),
        tol,
        reorder,
        m_deltaSolVector.data(),
        &singularity
    ));

    if (singularity >= 0)
        printf("WARNING: matrix is singular at row %d under tol %E\n", singularity, tol);
    end = std::chrono::high_resolution_clock::now();
    printf("Linear solve time: %f seconds\n", std::chrono::duration<double>(end - mid).count());
    m_updateNorm = m_deltaSolVector.vectorView().norm();
    m_residualNorm = bdev.norm();
    m_solVector.vectorView() += m_deltaSolVector.vectorView();
    CHECK_CUDA (cudaMemset(m_deltaSolVector.data(), 0, m_deltaSolVector.size() * sizeof(double)) );
    //std::cout << "Solution after iteration " << m_numIterations << ":\n";
    //m_solVector.vectorView().print();
    if (m_numIterations == 0)
    {
        m_initResidualNorm = m_residualNorm;
        m_initUpdateNorm = m_updateNorm;
    }

    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CUSOLVER_CHECK(cusolverSpDestroy(sH));
    CHECK_CUSPARSE(cusparseDestroy(cH));

    CHECK_CUDA(cudaFree(d_csr_offsets));
    CHECK_CUDA(cudaFree(d_csr_cols));
    CHECK_CUDA(cudaFree(d_csr_vals));

    return true;    
}
#endif

bool GPUSolver::solveSingleIteration_Eigen()
{
    //m_solVector.vectorView().print();
    //auto start = std::chrono::high_resolution_clock::now();
    m_assembler.assemble(m_solVector.vectorView(), m_numIterations, m_fixedDoFs.view());
    //auto end = std::chrono::high_resolution_clock::now();
    //printf("Assemble time: %f seconds\n", std::chrono::duration<double>(end - start).count());
#if 0
    if(m_numIterations == 0)
    {
        start = std::chrono::high_resolution_clock::now();
        DeviceArray<double> denseMatData(m_assembler.numDofs() * m_assembler.numDofs());
        m_assembler.denseMatrix(denseMatData.matrixView(m_assembler.numDofs(), m_assembler.numDofs()));
    #if 1
        DeviceArray<double> Eigenvalues(m_assembler.numDofs());
        eigenvalues_symm_dense(denseMatData.matrixView(m_assembler.numDofs(), m_assembler.numDofs()), 
                               Eigenvalues.vectorView());
        end = std::chrono::high_resolution_clock::now();
        printf("Eigenvalue computation time: %f seconds\n", std::chrono::duration<double>(end - start).count());
        std::cout << "Smallest eigenvalue: " << Eigenvalues[0] << std::endl;
    #else
        Eigen::MatrixXd A(m_assembler.numDofs(), m_assembler.numDofs());
        denseMatData.copyToHost(A.data());
        double lambda_min = smallestEigenvalue_symm_dense_Eigen(A);
        end = std::chrono::high_resolution_clock::now();
        printf("Eigenvalue computation time: %f seconds\n", std::chrono::duration<double>(end - start).count());
        std::cout << "Smallest eigenvalue: " << lambda_min << std::endl;
    #endif
    }
#endif

    //start = std::chrono::high_resolution_clock::now();
    DeviceVectorView<double> bdev = m_assembler.rhs();
    const int n = static_cast<int>(bdev.size());
#if 0

    // --- COO from assembler (device pointers) ---
    auto cooR = m_assembler.rows();
    auto cooC = m_assembler.cols();
    auto cooV = m_assembler.values();
    const int nnz_coo = static_cast<int>(cooV.size());

    // --- Copy to thrust vectors so we can sort/reduce (do not mutate assembler arrays) ---
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

    auto new_ends = thrust::reduce_by_key(
        keys_begin, keys_end,
        V.begin(),
        out_keys_begin,
        V2.begin());

    const int nnz = static_cast<int>(new_ends.second - V2.begin());

    auto mid = std::chrono::high_resolution_clock::now();
    printf("COO sort+reduce time: %f seconds\n", std::chrono::duration<double>(mid - start).count());

    // ============================================================
    //  Move reduced COO + RHS to host (CPU) for Eigen
    // ============================================================

    // Copy reduced COO to host
    std::vector<int>    hR(nnz);
    std::vector<int>    hC(nnz);
    std::vector<double> hV(nnz);

    CHECK_CUDA(cudaMemcpy(hR.data(),
                          thrust::raw_pointer_cast(R2.data()),
                          nnz * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC.data(),
                          thrust::raw_pointer_cast(C2.data()),
                          nnz * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hV.data(),
                          thrust::raw_pointer_cast(V2.data()),
                          nnz * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // Copy RHS to host
    std::vector<double> hb(n);
    CHECK_CUDA(cudaMemcpy(hb.data(), bdev.data(), n * sizeof(double), cudaMemcpyDeviceToHost));

    auto t_cpu_start = std::chrono::high_resolution_clock::now();

    // ============================================================
    //  Build Eigen sparse matrix and solve with SimplicialLDLT
    // ============================================================

    using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
    using Triplet = Eigen::Triplet<double, int>;

    std::vector<Triplet> triplets;
    triplets.reserve(nnz);

    // IMPORTANT:
    // If your matrix is symmetric and you store both triangles, this is fine.
    // If it’s symmetric but you only want one triangle, you could filter here (r>=c).
    for (int k = 0; k < nnz; ++k)
        triplets.emplace_back(hR[k], hC[k], hV[k]);

    SpMat A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end()); // duplicates already merged on GPU
    A.makeCompressed();

    Eigen::Map<const Eigen::VectorXd> b(hb.data(), n);
#endif
    //std::cout << Eigen::MatrixXd(A) << std::endl;
    //std::cout << std::endl;
    //m_assembler.csrMatrix().print_host();


    //auto t_cpu_start = std::chrono::high_resolution_clock::now();
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
    SpMat A = m_assembler.csrMatrix().toEigenCSR();
    Eigen::VectorXd b = m_assembler.hostRHS();

    //std::cout << std::scientific << std::setprecision(12);
    //std::cout << "Matrix:\n" << Eigen::MatrixXd(A) << std::endl;
    //std::cout << "RHS:\n" << b << std::endl;
    //std::cout << std::fixed << std::setprecision(6);

#if 0
    if (m_numIterations == 0)
    {
        // Compute the smallest eigenvalue of the system matrix
        start = std::chrono::high_resolution_clock::now();
        double lambda_min = smallestEigenvalue_symm_sparse_Eigen(A);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Smallest eigenvalue: " << lambda_min << std::endl;
        printf("Eigenvalue computation time: %f seconds\n", std::chrono::duration<double>(end - start).count());
    }
#endif

    //std::cout << "Host matrix:\n" << Eigen::MatrixXd(A) << std::endl;
    //std::cout << "RHS:\n" << b << std::endl;

    Eigen::VectorXd x;
#if ENABLE_PARDISO
    if (!solveWithPardiso(A, b, x, "Newton correction"))
        return false;
#else
    Eigen::SparseLU<SpMat> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success)
    {
        printf("Eigen solver failed\n");
        return false;
    }
    x = solver.solve(b);
    //std::cout << "test result:\n" << b - Eigen::MatrixXd(A) * x << std::endl;

    if (solver.info() != Eigen::Success)
    {
        printf("Eigen solver failed\n");
        return false;
    }
#endif
    //std::cout << std::endl;
    //std::cout << "delta solution:\n" << x << std::endl;
    //auto t_cpu_end = std::chrono::high_resolution_clock::now();
    //printf("Eigen factor+solve time (CPU): %f seconds\n",
    //       std::chrono::duration<double>(t_cpu_end - t_cpu_start).count());

    // ============================================================
    //  Copy x back to device (m_deltaSolVector)
    // ============================================================
    CHECK_CUDA(cudaMemcpy(m_deltaSolVector.data(), x.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    //end = std::chrono::high_resolution_clock::now();
    //printf("Total linear solve time (incl H<->D): %f seconds\n",
    //       std::chrono::duration<double>(end - start).count());

    // Update (same as your code)
    m_updateNorm   = m_deltaSolVector.vectorView().norm();
    m_residualNorm = bdev.norm();
    m_solVector.vectorView() += m_deltaSolVector.vectorView();
    CHECK_CUDA(cudaMemset(m_deltaSolVector.data(), 0, m_deltaSolVector.size() * sizeof(double)));

    if (m_numIterations == 0)
    {
        m_initResidualNorm = m_residualNorm;
        m_initUpdateNorm   = m_updateNorm;
    }

    return true;
}

#if ENABLE_AMGX
bool GPUSolver::solveSingleIteration_AMGX()
{
    auto start = std::chrono::high_resolution_clock::now();
    m_assembler.assemble(m_solVector.vectorView(), m_numIterations, m_fixedDoFs.view());
    auto end = std::chrono::high_resolution_clock::now();
    printf("Assemble time: %f seconds\n", std::chrono::duration<double>(end - start).count());

    //DeviceMatrixView<double> Adev = m_assembler.matrix();
    DeviceVectorView<double> bdev = m_assembler.rhs();
    const int n = static_cast<int>(bdev.size());

	start = std::chrono::high_resolution_clock::now();
#if 0
    // --- COO from assembler (device pointers) ---
    auto cooR = m_assembler.rows();    // int, length nnz_coo
    auto cooC = m_assembler.cols();    // int, length nnz_coo
    auto cooV = m_assembler.values();  // double, length nnz_coo
    const int nnz_coo = static_cast<int>(cooV.size());

    // --- Copy to thrust vectors so we can sort/reduce (do not mutate assembler arrays) ---
    thrust::device_vector<int>    R(cooR.data(), cooR.data() + nnz_coo);
    thrust::device_vector<int>    C(cooC.data(), cooC.data() + nnz_coo);
    thrust::device_vector<double> V(cooV.data(), cooV.data() + nnz_coo);

    const int d = m_assembler.dim();
    if (n % d != 0) { printf("Permutation requires n divisible by d\n"); return false; }
    const int N = n / d;
    BlockedToInterleavedND P{N, d};
    thrust::transform(R.begin(), R.end(), R.begin(), P);
    thrust::transform(C.begin(), C.end(), C.begin(), P);

    // key = (row,col)
    auto keys_begin = thrust::make_zip_iterator(thrust::make_tuple(R.begin(), C.begin()));
    auto keys_end   = thrust::make_zip_iterator(thrust::make_tuple(R.end(),   C.end()));

    // 1) Sort by (row,col)
    thrust::sort_by_key(keys_begin, keys_end, V.begin());

    // 2) Reduce duplicates: (row,col) identical -> sum values
    thrust::device_vector<int>    R2(nnz_coo);
    thrust::device_vector<int>    C2(nnz_coo);
    thrust::device_vector<double> V2(nnz_coo);

    auto out_keys_begin = thrust::make_zip_iterator(thrust::make_tuple(R2.begin(), C2.begin()));

    auto new_ends = thrust::reduce_by_key(
        keys_begin, keys_end,
        V.begin(),
        out_keys_begin,
        V2.begin());

    // new nnz after merging duplicates
    const int nnz = static_cast<int>(new_ends.second - V2.begin());

    // --- Build CSR arrays (device malloc) ---
    int*    d_csr_offsets = nullptr;
    int*    d_csr_cols    = nullptr;
    double* d_csr_vals    = nullptr;

    CHECK_CUDA(cudaMalloc(&d_csr_offsets, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_cols,    nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_vals,    nnz * sizeof(double)));

    // Copy reduced (col,val) into CSR col/val buffers
    CHECK_CUDA(cudaMemcpy(d_csr_cols,
                          thrust::raw_pointer_cast(C2.data()),
                          nnz * sizeof(int),
                          cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_csr_vals,
                          thrust::raw_pointer_cast(V2.data()),
                          nnz * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    // COO row-indices (reduced) -> CSR row offsets
    cusparseHandle_t cH = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&cH));
    CHECK_CUSPARSE(cusparseXcoo2csr(
        cH,
        thrust::raw_pointer_cast(R2.data()),
        nnz,
        n,
        d_csr_offsets,
        CUSPARSE_INDEX_BASE_ZERO));
    auto mid = std::chrono::high_resolution_clock::now();
    printf("COO->CSR conversion time: %f seconds\n", std::chrono::duration<double>(mid - start).count());
#endif
    const int nnz = m_assembler.csrMatrix().numNonZeros();
    if (m_amgx_n != n || m_amgx_nnz != nnz)
    {
        // New pattern/size: upload full matrix & setup
        AMGX_CHECK(AMGX_matrix_upload_all(
            m_amgx_A,
            n, nnz,
            1, 1,
            m_assembler.csrMatrix().rowPtr().data(),
            m_assembler.csrMatrix().colInd().data(),
            m_assembler.csrMatrix().values().data(),
            nullptr)); // diag_data = nullptr

        AMGX_CHECK(AMGX_solver_setup(m_amgx_solver, m_amgx_A));

        m_amgx_n = n;
        m_amgx_nnz = nnz;
    }
    else
    {
        // Same pattern: update coefficients and rebuild the AMG hierarchy.
        // The tangent matrix changes every Newton iteration, so reusing an old
        // hierarchy can make AMGX stall on larger strain-gradient systems.
        AMGX_CHECK(AMGX_matrix_replace_coefficients(
            m_amgx_A,
            n, nnz,
            m_assembler.csrMatrix().values().data(),
            nullptr));

        AMGX_CHECK(AMGX_solver_setup(m_amgx_solver, m_amgx_A));
    }

    //auto idx0   = thrust::make_counting_iterator<int>(0);
    //auto map_it = thrust::make_transform_iterator(idx0, P);

    //thrust::device_vector<double> b_perm(n);
    //thrust::device_vector<double> x0_perm(n);

    //thrust::device_ptr<const double> bptr(bdev.data());
    //thrust::device_ptr<const double> x0ptr(m_deltaSolVector.data());

    //thrust::scatter(bptr,  bptr  + n, map_it, b_perm.begin());
    //thrust::scatter(x0ptr, x0ptr + n, map_it, x0_perm.begin());

    // Upload RHS and initial guess (device pointers because mode = dDDI)
    AMGX_CHECK(AMGX_vector_upload(m_amgx_b, n, 1, bdev.data()));
    AMGX_CHECK(AMGX_vector_upload(m_amgx_x, n, 1, m_deltaSolVector.data())); // initial guess

    // Solve
    AMGX_CHECK(AMGX_solver_solve(m_amgx_solver, m_amgx_b, m_amgx_x));

    AMGX_SOLVE_STATUS st;
    AMGX_CHECK(AMGX_solver_get_status(m_amgx_solver, &st));

#if 0
    int nits = 0;
    AMGX_CHECK(AMGX_solver_get_iterations_number(m_amgx_solver, &nits));

    for (int it = 0; it < nits; ++it)
    {
        double res = 0.0;
        AMGX_CHECK(AMGX_solver_get_iteration_residual(m_amgx_solver, it, /*idx=*/0, &res));
        printf("it %d residual %e\n", it, res);
    }
#endif

    if (st != AMGX_SOLVE_SUCCESS)
    {
        int iters = 0;
        AMGX_CHECK(AMGX_solver_get_iterations_number(m_amgx_solver, &iters));
        printf("AMGX solve status=%d (%s) after %d iters\n",
               (int)st, amgxSolveStatusName(st), iters);

        using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
        SpMat A = m_assembler.csrMatrix().toEigenCSR();
        Eigen::VectorXd b = m_assembler.hostRHS();

        Eigen::VectorXd x;
#if ENABLE_PARDISO
        if (!solveWithPardiso(A, b, x, "AMGX fallback Newton correction"))
            return false;
#else
        Eigen::SparseLU<SpMat> fallbackSolver;
        fallbackSolver.compute(A);
        if (fallbackSolver.info() != Eigen::Success)
        {
            printf("%s fallback solver factorization failed after AMGX non-convergence\n",
                   cpuDirectSolverName());
            return false;
        }

        x = fallbackSolver.solve(b);
        if (fallbackSolver.info() != Eigen::Success)
        {
            printf("%s fallback solver solve failed after AMGX non-convergence\n",
                   cpuDirectSolverName());
            return false;
        }
#endif

        CHECK_CUDA(cudaMemcpy(m_deltaSolVector.data(), x.data(),
                              n * sizeof(double), cudaMemcpyHostToDevice));
        printf("Used %s fallback for this Newton correction\n", cpuDirectSolverName());
    }
    else
    {
        Eigen::VectorXd xprime_host(n);
        AMGX_CHECK(AMGX_vector_download(m_amgx_x, xprime_host.data()));
    #if 0
        Eigen::VectorXd x_old(n);
        for (int i = 0; i < n; ++i)
        {
            int comp = i / N;
            int node = i - comp * N;
            int j = node * d + comp;   // P(i)
            x_old[i] = xprime_host[j];
        }
    #endif
        m_deltaSolVector.updateFromHost(xprime_host.data());
    }

    //Eigen::VectorXd x_host(m_deltaSolVector.size());
    //m_deltaSolVector.copyToHost(x_host.data());
    //std::cout << "Solution vector (host) before download: " << x_host.transpose() << std::endl;
    //AMGX_CHECK(AMGX_vector_download(m_amgx_x, x_host.data()));
    //m_deltaSolVector.updateFromHost(x_host.data());
    //std::cout << "Solution vector (host): " << x_host.transpose() << std::endl;
    end = std::chrono::high_resolution_clock::now();
    printf("Linear solve time: %f seconds\n", std::chrono::duration<double>(end - start).count());

    m_updateNorm   = m_deltaSolVector.vectorView().norm();
    m_residualNorm = bdev.norm();
    m_solVector.vectorView() += m_deltaSolVector.vectorView();
    CHECK_CUDA(cudaMemset(m_deltaSolVector.data(), 0, m_deltaSolVector.size() * sizeof(double)));

    if (m_numIterations == 0)
    {
        m_initResidualNorm = m_residualNorm;
        m_initUpdateNorm   = m_updateNorm;
    }

    //CHECK_CUSPARSE(cusparseDestroy(cH));
    //CHECK_CUDA(cudaFree(d_csr_offsets));
    //CHECK_CUDA(cudaFree(d_csr_cols));
    //CHECK_CUDA(cudaFree(d_csr_vals));

    return true;
}
#endif

void GPUSolver::solve()
{
    //double absTol = 1e-10;
    //double relTol = 1e-10;
    //int maxIterations = 100;
    m_numIterations = 0;
    m_status = working;
    //std::cout << std::scientific;
    while (m_status == working)
    {
#if ENABLE_AMGX
        if(!solveSingleIteration_AMGX())
        //if(!solveSingleIteration_Eigen())
#else
        if(!solveSingleIteration_Eigen())
        //if(!solveSingleIteration())
#endif
        {
            m_status = bad_solution;
            break;
        }
        std::cout << status() << std::endl;
        if (m_residualNorm < m_absTol || 
            m_updateNorm < m_absTol || 
            m_residualNorm/m_initResidualNorm < m_relTol ||
            m_updateNorm/m_initUpdateNorm < m_relTol)
            m_status = converged;
        else if (m_numIterations >= m_maxIter)
            m_status = interrupted;
        if (m_numIterations == 0)
        {
            //std::cout << "Old fixed DoFs:\n";
            //m_fixedDoFs.view().wholeView().print();

            m_fixedDoFs.view().wholeView() += m_assembler.allFixedDofs().view().wholeView();

            //std::cout << "New fixed DoFs:\n";
            //m_fixedDoFs.view().wholeView().print();
        }
            
        m_numIterations++;
    }

    std::cout << status() << std::endl;
}

double GPUSolver::smallestEigenValue()
{
    Eigen::VectorXd eigenvector;
    return smallestEigenValue(eigenvector);
}

double GPUSolver::smallestEigenValue(Eigen::VectorXd& eigenvector)
{
    //m_assembler.assemble(m_solVector.vectorView(), m_numIterations, m_fixedDoFs.view());

    double smallestEigenvalue = 0.0;
    eigenvector.resize(0);
#if ENABLE_AMGX
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;

    const int n = m_assembler.csrMatrix().numRows();
    const int nnz = m_assembler.csrMatrix().numNonZeros();
    if (n <= 0 || n != m_assembler.csrMatrix().numCols())
        return smallestEigenvalue;

    if (m_amgx_n != n || m_amgx_nnz != nnz)
    {
        AMGX_CHECK(AMGX_matrix_upload_all(
            m_amgx_A,
            n, nnz,
            1, 1,
            m_assembler.csrMatrix().rowPtr().data(),
            m_assembler.csrMatrix().colInd().data(),
            m_assembler.csrMatrix().values().data(),
            nullptr));
        AMGX_CHECK(AMGX_solver_setup(m_amgx_solver, m_amgx_A));
        m_amgx_n = n;
        m_amgx_nnz = nnz;
        m_cachedSmallestEigenvector.resize(0);
    }
    else
    {
        AMGX_CHECK(AMGX_matrix_replace_coefficients(
            m_amgx_A,
            n, nnz,
            m_assembler.csrMatrix().values().data(),
            nullptr));
        AMGX_CHECK(AMGX_solver_setup(m_amgx_solver, m_amgx_A));
    }

    Eigen::VectorXd x;
    const bool hasCachedEigenvector = (m_cachedSmallestEigenvector.size() == n);
    if (hasCachedEigenvector)
        x = m_cachedSmallestEigenvector;
    else
    {
        x.resize(n);
        for (int i = 0; i < n; ++i)
            x[i] = std::sin(0.37 * static_cast<double>(i + 1)) +
                   0.5 * std::cos(0.11 * static_cast<double>(i + 1));
    }
    x.normalize();

    DeviceArray<double> rhsDev(n);
    DeviceArray<double> solDev(n);
    Eigen::VectorXd y(n);
    SpMat fallbackMatrix;
#if ENABLE_PARDISO
    bool fallbackPatternReady = false;
#else
    Eigen::SparseLU<SpMat> fallbackSolver;
#endif
    bool useFallbackSolver = false;

    const int maxOuterIterations = hasCachedEigenvector ? 8 : 20;
    constexpr double tolerance = 1e-6;
    double previousEigenvalue = hasCachedEigenvector
        ? m_cachedSmallestEigenvalue
        : std::numeric_limits<double>::infinity();

    for (int outer = 0; outer < maxOuterIterations; ++outer)
    {
        if (!useFallbackSolver)
        {
            rhsDev.updateFromHost(x.data());
            solDev.setZero();

            AMGX_CHECK(AMGX_vector_upload(m_amgx_b, n, 1, rhsDev.data()));
            AMGX_CHECK(AMGX_vector_upload(m_amgx_x, n, 1, solDev.data()));
            AMGX_CHECK(AMGX_solver_solve(m_amgx_solver, m_amgx_b, m_amgx_x));

            AMGX_SOLVE_STATUS status = AMGX_SOLVE_FAILED;
            AMGX_CHECK(AMGX_solver_get_status(m_amgx_solver, &status));
            if (status == AMGX_SOLVE_SUCCESS)
            {
                AMGX_CHECK(AMGX_vector_download(m_amgx_x, y.data()));
            }
            else
            {
                int iters = 0;
                AMGX_CHECK(AMGX_solver_get_iterations_number(m_amgx_solver, &iters));
                printf("AMGX eigen inverse iteration solve status=%d (%s) after %d iters\n",
                       (int)status, amgxSolveStatusName(status), iters);

                fallbackMatrix = m_assembler.csrMatrix().toEigenCSR();
#if ENABLE_PARDISO
                fallbackPatternReady = true;
                useFallbackSolver = true;
                printf("Using %s fallback for inverse iteration\n", cpuDirectSolverName());
#else
                fallbackSolver.compute(fallbackMatrix);
                if (fallbackSolver.info() != Eigen::Success)
                {
                    printf("%s fallback factorization failed in inverse iteration\n",
                           cpuDirectSolverName());
                    break;
                }
                useFallbackSolver = true;
                printf("Using %s fallback for inverse iteration\n", cpuDirectSolverName());
#endif
            }
        }

        if (useFallbackSolver)
        {
#if ENABLE_PARDISO
            if (!fallbackPatternReady ||
                !solveWithPardiso(fallbackMatrix, x, y, "AMGX fallback inverse iteration"))
                break;
#else
            y = fallbackSolver.solve(x);
            if (fallbackSolver.info() != Eigen::Success)
            {
                printf("%s fallback solve failed in inverse iteration\n", cpuDirectSolverName());
                break;
            }
#endif
        }

        const double yNorm = y.norm();
        if (yNorm <= 0.0 || !std::isfinite(yNorm))
            break;

        const double inverseEigenvalue = x.dot(y);
        if (std::abs(inverseEigenvalue) > 0.0)
            smallestEigenvalue = 1.0 / inverseEigenvalue;

        x = y / yNorm;

        const double eigenvalueDelta = std::abs(smallestEigenvalue - previousEigenvalue);
        if (eigenvalueDelta < tolerance * (std::max)(1.0, std::abs(smallestEigenvalue)))
            break;

        previousEigenvalue = smallestEigenvalue;
    }

    const SpMat A = useFallbackSolver ? fallbackMatrix : m_assembler.csrMatrix().toEigenCSR();
    smallestEigenvalue = x.dot(A * x);
    m_cachedSmallestEigenvector = x;
    m_cachedSmallestEigenvalue = smallestEigenvalue;
    eigenvector = x;
#else
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;

    const SpMat A = m_assembler.csrMatrix().toEigenCSR();
    const int n = A.rows();
    if (n <= 0 || n != A.cols())
        return smallestEigenvalue;

#if !ENABLE_PARDISO
    Eigen::SparseLU<SpMat> inverseIterationSolver;
    inverseIterationSolver.compute(A);
    if (inverseIterationSolver.info() != Eigen::Success)
    {
        printf("%s factorization failed in inverse iteration\n", cpuDirectSolverName());
        return smallestEigenvalue;
    }
#endif

    Eigen::VectorXd x(n);
    for (int i = 0; i < n; ++i)
        x[i] = std::sin(0.37 * static_cast<double>(i + 1)) +
               0.5 * std::cos(0.11 * static_cast<double>(i + 1));
    x.normalize();

    constexpr int maxOuterIterations = 20;
    constexpr double tolerance = 1e-6;
    double previousEigenvalue = std::numeric_limits<double>::infinity();

    for (int outer = 0; outer < maxOuterIterations; ++outer)
    {
#if ENABLE_PARDISO
        Eigen::VectorXd y;
        if (!solveWithPardiso(A, x, y, "inverse iteration"))
            break;
#else
        Eigen::VectorXd y = inverseIterationSolver.solve(x);
        if (inverseIterationSolver.info() != Eigen::Success)
        {
            printf("%s solve failed in inverse iteration\n", cpuDirectSolverName());
            break;
        }
#endif

        const double yNorm = y.norm();
        if (yNorm <= 0.0 || !std::isfinite(yNorm))
            break;

        const double inverseEigenvalue = x.dot(y);
        if (std::abs(inverseEigenvalue) > 0.0)
            smallestEigenvalue = 1.0 / inverseEigenvalue;

        x = y / yNorm;

        const double eigenvalueDelta = std::abs(smallestEigenvalue - previousEigenvalue);
        if (eigenvalueDelta < tolerance * (std::max)(1.0, std::abs(smallestEigenvalue)))
            break;

        previousEigenvalue = smallestEigenvalue;
    }

    smallestEigenvalue = x.dot(A * x);
    eigenvector = x;
#endif
    return smallestEigenvalue;
}

double GPUSolver::stability()
{
    //m_assembler.assemble(m_solVector.vectorView(), m_numIterations, m_fixedDoFs.view());

    using RowMajorSpMat = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
    using ColMajorSpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

    RowMajorSpMat hostCSR = m_assembler.csrMatrix().toEigenCSR();
    ColMajorSpMat hostCSC = hostCSR;

    Eigen::SimplicialLDLT<ColMajorSpMat> solver;
    solver.compute(hostCSC);

    if (solver.info() != Eigen::Success)
    {
        printf("Eigen SimplicialLDLT failed while computing stability\n");
        return -std::numeric_limits<double>::infinity();
    }

    return solver.vectorD().minCoeff();
}

void GPUSolver::eigenvalues_symm_dense(DeviceMatrixView<double> matrix, 
                                       DeviceVectorView<double> eigenvalues,
                                       bool computeEigenvectors)
{
    int n = matrix.rows();

    cusolverDnHandle_t handle = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    cusolverDnParams_t params = nullptr;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    size_t work_dev_bytes = 0;
    size_t work_host_bytes = 0;

    const cusolverEigMode_t jobz = computeEigenvectors
        ? CUSOLVER_EIG_MODE_VECTOR
        : CUSOLVER_EIG_MODE_NOVECTOR;
    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
        handle,
        params,
        jobz,
        uplo,
        n,
        CUDA_R_64F, matrix.data(), n,
        CUDA_R_64F, eigenvalues.data(),
        CUDA_R_64F,
        &work_dev_bytes,
        &work_host_bytes));
    
    std::cout << "Workspace sizes - Device: " << work_dev_bytes << " bytes, Host: " << work_host_bytes << " bytes\n";

    // Allocate workspace
    void* work_dev = nullptr;
    void* work_host = nullptr;
    if (work_dev_bytes > 0) 
    {
        CHECK_CUDA(cudaMalloc(&work_dev, work_dev_bytes));
    }
    if (work_host_bytes > 0) {
        work_host = malloc(work_host_bytes);
        if (!work_host) throw std::bad_alloc();
    }

    int* d_info = nullptr;
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

    // Compute eigenvalues
    CUSOLVER_CHECK(cusolverDnXsyevd(
        handle,
        params,
        jobz,
        uplo,
        n,
        CUDA_R_64F, matrix.data(), n,
        CUDA_R_64F, eigenvalues.data(),
        CUDA_R_64F,
        work_dev, work_dev_bytes,
        work_host, work_host_bytes,
        d_info));

    int h_info = 0;
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
    if (h_info != 0) {
        // h_info < 0: bad parameter; h_info > 0: algorithm did not converge
        throw std::runtime_error("cusolverDnXsyevd failed, info=" + std::to_string(h_info));
    }
    CHECK_CUDA(cudaFree(d_info));

    // Free workspace
    if (work_dev) {
        CHECK_CUDA(cudaFree(work_dev));
    }
    if (work_host) {
        free(work_host);
    }

    // Destroy handles
    CUSOLVER_CHECK(cusolverDnDestroyParams(params));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));
}

double GPUSolver::smallestEigenvalue_symm_dense_Eigen(Eigen::MatrixXd mat)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(mat);
    if (eigensolver.info() != Eigen::Success) 
    {
        throw std::runtime_error("Eigen decomposition failed");
    }
    return eigensolver.eigenvalues()(0);
}

double GPUSolver::smallestEigenvalue_symm_sparse_Eigen(Eigen::SparseMatrix<double> mat)
{
    Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>> eigensolver(mat);
    if (eigensolver.info() != Eigen::Success) 
    {
        throw std::runtime_error("Eigen decomposition failed");
    }
    return eigensolver.eigenvalues()(0);
}

std::string GPUSolver::status()
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
        statusString = "It: " + std::to_string(m_numIterations + 1) +
                 ", updAbs: " + to_string_sientific(m_updateNorm) +
                 ", updRel: " + to_string_sientific(m_updateNorm/m_initUpdateNorm) +
                 ", resAbs: " + to_string_sientific(m_residualNorm) +
                 ", resRel: " + to_string_sientific(m_residualNorm/m_initResidualNorm);
    return statusString;
}

#if ENABLE_AMGX
void GPUSolver::initAMGXOnce()
{
    if (m_amgx_initialized) return;

    AMGX_CHECK(AMGX_initialize());
    AMGX_CHECK(AMGX_initialize_plugins());

    // A decent starting config for SPD problems:
    // - solver=CG
    // - AMG preconditioner
    // You can tune later.
    const char* cfg_str =
        "config_version=2, "
        "solver(main)=CG, "
        "main:preconditioner(amg)=AMG, "
        "main:max_iters=2000, "
        "main:tolerance=1e-6, "
        "main:norm=L2, "
        "main:monitor_residual=1, "
        "main:store_res_history=0, "
        "main:print_solve_stats=0, "
#if 0
        // --- AMG scope ---
        "amg:algorithm=AGGREGATION, "
        "amg:selector=SIZE_2, "
        "amg:interpolator=D2, "
        "amg:strength_threshold=0.5, "
        "amg:cycle=V, "
        "amg:max_iters=1, "
        "amg:max_levels=50, "
        "amg:presweeps=2, "
        "amg:postsweeps=2, "
        "amg:coarsest_sweeps=2, "
        "amg:smoother=JACOBI_L1, "
        "amg:coarse_solver=DENSE_LU_SOLVER"
#endif
        ;
    //AMGX_config_handle config;
    const std::filesystem::path configPaths[] = {
        "./SOLVER_CONFIG_INUSE.json",
        "./AMGX_config/SOLVER_CONFIG_INUSE.json",
        "../AMGX_config/SOLVER_CONFIG_INUSE.json",
        "../../AMGX_config/SOLVER_CONFIG_INUSE.json"
    };
    for (const auto& path : configPaths)
    {
        if (std::filesystem::exists(path))
        {
            AMGX_CHECK(AMGX_config_create_from_file(&m_amgx_cfg, path.string().c_str()));
            break;
        }
    }
    if (!m_amgx_cfg)
        AMGX_CHECK(AMGX_config_create(&m_amgx_cfg, cfg_str));
    AMGX_CHECK(AMGX_resources_create_simple(&m_amgx_rsrc, m_amgx_cfg))

    AMGX_CHECK(AMGX_matrix_create(&m_amgx_A, m_amgx_rsrc, m_amgx_mode));
    AMGX_CHECK(AMGX_vector_create(&m_amgx_b, m_amgx_rsrc, m_amgx_mode));
    AMGX_CHECK(AMGX_vector_create(&m_amgx_x, m_amgx_rsrc, m_amgx_mode));
    AMGX_CHECK(AMGX_solver_create(&m_amgx_solver, m_amgx_rsrc, m_amgx_mode, m_amgx_cfg));

    m_amgx_initialized = true;
}

void GPUSolver::finalizeAMGX()
{
    if (!m_amgx_initialized) return;

    AMGX_CHECK(AMGX_solver_destroy(m_amgx_solver));
    AMGX_CHECK(AMGX_vector_destroy(m_amgx_x));
    AMGX_CHECK(AMGX_vector_destroy(m_amgx_b));
    AMGX_CHECK(AMGX_matrix_destroy(m_amgx_A));
    AMGX_CHECK(AMGX_resources_destroy(m_amgx_rsrc));
    AMGX_CHECK(AMGX_config_destroy(m_amgx_cfg));
    AMGX_CHECK(AMGX_finalize_plugins());
    AMGX_CHECK(AMGX_finalize());

    m_amgx_solver = nullptr;
    m_amgx_x = m_amgx_b = nullptr;
    m_amgx_A = nullptr;
    m_amgx_rsrc = nullptr;
    m_amgx_cfg = nullptr;
    m_amgx_initialized = false;
    m_amgx_n = m_amgx_nnz = -1;
}
#endif
