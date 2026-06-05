#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "AMGXEigenSolver.h"

#if ENABLE_AMGX

#include <DeviceCSRMatrix.h>
#include <GPUAssembler.h>

#include <amgx_c.h>
#include <cuda_runtime.h>

#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace
{
void checkCuda(cudaError_t status, const char* call)
{
    if (status == cudaSuccess)
        return;
    throw std::runtime_error(std::string(call) + " failed: " + cudaGetErrorString(status));
}

void checkAMGX(AMGX_RC status, const char* call)
{
    if (status == AMGX_RC_OK)
        return;

    char msg[4096] = {};
    AMGX_get_error_string(status, msg, sizeof(msg));
    throw std::runtime_error(std::string(call) + " failed: " + msg);
}

std::string makeConfigString(const AMGXEigenSolver::Options& options)
{
    return
        "config_version=2, "
        "solver(main)=PCG, "
        "main:preconditioner(amg)=AMG, "
        "main:max_iters=" + std::to_string(options.amgxMaxIterations) + ", "
        "main:tolerance=" + std::to_string(options.amgxTolerance) + ", "
        "main:norm=L2, "
        "main:monitor_residual=0, "
        "main:store_res_history=0, "
        "main:print_solve_stats=0, "
        "amg:algorithm=AGGREGATION, "
        "amg:selector=SIZE_2, "
        "amg:interpolator=D2, "
        "amg:smoother=JACOBI_L1, "
        "amg:coarse_solver=DENSE_LU_SOLVER, "
        "amg:cycle=V, "
        "amg:max_iters=1, "
        "amg:max_levels=50, "
        "amg:presweeps=1, "
        "amg:postsweeps=1";
}

Eigen::MatrixXd orthonormalized(const Eigen::MatrixXd& X)
{
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(X);
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(X.rows(), X.cols());
    return Q.leftCols(X.cols());
}

Eigen::MatrixXd randomOrthonormalMatrix(int rows, int cols)
{
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    Eigen::MatrixXd X(rows, cols);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            X(i, j) = dist(rng);

    return orthonormalized(X);
}

double maxResidual(const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& A,
                   const Eigen::MatrixXd& vectors,
                   const Eigen::VectorXd& values)
{
    double maxRes = 0.0;
    for (int i = 0; i < values.size(); ++i)
    {
        const Eigen::VectorXd residual = A * vectors.col(i) - values[i] * vectors.col(i);
        const double denom = (std::max)(1.0, std::abs(values[i]));
        maxRes = (std::max)(maxRes, residual.norm() / denom);
    }
    return maxRes;
}

class AMGXLinearSolver
{
public:
    AMGXLinearSolver(const DeviceCSRMatrix& matrix, const AMGXEigenSolver::Options& options)
        : m_n(matrix.numRows())
    {
        if (matrix.numRows() != matrix.numCols())
            throw std::invalid_argument("AMGXEigenSolver requires a square CSR matrix.");
        if (m_n <= 0)
            throw std::invalid_argument("AMGXEigenSolver requires a non-empty CSR matrix.");

        const std::string config = makeConfigString(options);
        checkAMGX(AMGX_config_create(&m_config, config.c_str()), "AMGX_config_create");
        checkAMGX(AMGX_resources_create_simple(&m_resources, m_config), "AMGX_resources_create_simple");
        checkAMGX(AMGX_matrix_create(&m_A, m_resources, m_mode), "AMGX_matrix_create");
        checkAMGX(AMGX_vector_create(&m_rhs, m_resources, m_mode), "AMGX_vector_create(rhs)");
        checkAMGX(AMGX_vector_create(&m_sol, m_resources, m_mode), "AMGX_vector_create(sol)");

        checkAMGX(AMGX_matrix_upload_all(m_A,
                                         matrix.numRows(),
                                         matrix.numNonZeros(),
                                         1,
                                         1,
                                         matrix.rowPtr().data(),
                                         matrix.colInd().data(),
                                         matrix.values().data(),
                                         nullptr),
                  "AMGX_matrix_upload_all");

        checkAMGX(AMGX_solver_create(&m_solver, m_resources, m_mode, m_config), "AMGX_solver_create");
        checkAMGX(AMGX_solver_setup(m_solver, m_A), "AMGX_solver_setup");

        checkCuda(cudaMalloc(&m_dRhs, static_cast<size_t>(m_n) * sizeof(double)), "cudaMalloc(rhs)");
        checkCuda(cudaMalloc(&m_dSol, static_cast<size_t>(m_n) * sizeof(double)), "cudaMalloc(sol)");
    }

    ~AMGXLinearSolver()
    {
        cudaDeviceSynchronize();
        if (m_solver) AMGX_solver_destroy(m_solver);
        if (m_sol) AMGX_vector_destroy(m_sol);
        if (m_rhs) AMGX_vector_destroy(m_rhs);
        if (m_A) AMGX_matrix_destroy(m_A);
        if (m_resources) AMGX_resources_destroy(m_resources);
        if (m_config) AMGX_config_destroy(m_config);
        if (m_dSol) cudaFree(m_dSol);
        if (m_dRhs) cudaFree(m_dRhs);
        cudaGetLastError();
    }

    Eigen::VectorXd solve(const Eigen::VectorXd& rhs)
    {
        if (rhs.size() != m_n)
            throw std::invalid_argument("AMGXLinearSolver::solve rhs size mismatch.");

        checkCuda(cudaMemcpy(m_dRhs, rhs.data(), static_cast<size_t>(m_n) * sizeof(double), cudaMemcpyHostToDevice),
                  "cudaMemcpy(rhs H2D)");
        checkCuda(cudaMemset(m_dSol, 0, static_cast<size_t>(m_n) * sizeof(double)), "cudaMemset(sol)");

        checkAMGX(AMGX_vector_upload(m_rhs, m_n, 1, m_dRhs), "AMGX_vector_upload(rhs)");
        checkAMGX(AMGX_vector_upload(m_sol, m_n, 1, m_dSol), "AMGX_vector_upload(sol)");
        checkAMGX(AMGX_solver_solve(m_solver, m_rhs, m_sol), "AMGX_solver_solve");

        AMGX_SOLVE_STATUS status = AMGX_SOLVE_FAILED;
        checkAMGX(AMGX_solver_get_status(m_solver, &status), "AMGX_solver_get_status");
        if (status != AMGX_SOLVE_SUCCESS)
            throw std::runtime_error("AMGX linear solve did not converge while computing eigenvalues.");

        Eigen::VectorXd sol(m_n);
        checkAMGX(AMGX_vector_download(m_sol, sol.data()), "AMGX_vector_download(sol)");
        return sol;
    }

private:
    const AMGX_Mode m_mode = AMGX_mode_dDDI;
    int m_n = 0;
    AMGX_config_handle m_config = nullptr;
    AMGX_resources_handle m_resources = nullptr;
    AMGX_matrix_handle m_A = nullptr;
    AMGX_vector_handle m_rhs = nullptr;
    AMGX_vector_handle m_sol = nullptr;
    AMGX_solver_handle m_solver = nullptr;
    double* m_dRhs = nullptr;
    double* m_dSol = nullptr;
};
}

AMGXEigenSolver::AMGXEigenSolver()
{
}

AMGXEigenSolver::~AMGXEigenSolver()
{
}

Eigen::VectorXd AMGXEigenSolver::computeSmallestEigenvalues(const GPUAssembler& assembler,
                                                            int numEigenvalues,
                                                            const Options& options)
{
    return computeSmallestEigenvalues(assembler.csrMatrix(), numEigenvalues, options);
}

Eigen::VectorXd AMGXEigenSolver::computeSmallestEigenvalues(const DeviceCSRMatrix& matrix,
                                                            int numEigenvalues,
                                                            const Options& options)
{
    if (numEigenvalues <= 0)
        return Eigen::VectorXd();
    if (matrix.numRows() != matrix.numCols())
        throw std::invalid_argument("AMGXEigenSolver requires a square CSR matrix.");

    const int n = matrix.numRows();
    const int blockSize = (std::min)(n, numEigenvalues + (std::max)(0, options.extraSearchVectors));
    if (numEigenvalues > blockSize)
        throw std::invalid_argument("Requested more eigenvalues than the search space can contain.");

    const auto hostA = matrix.toEigenCSR();
    AMGXLinearSolver linearSolver(matrix, options);

    Eigen::MatrixXd X = randomOrthonormalMatrix(n, blockSize);
    Eigen::VectorXd eigenvalues = Eigen::VectorXd::Constant(numEigenvalues, std::numeric_limits<double>::infinity());
    Eigen::MatrixXd eigenvectors(n, numEigenvalues);

    for (int outer = 0; outer < options.maxOuterIterations; ++outer)
    {
        Eigen::MatrixXd Y(n, blockSize);
        for (int j = 0; j < blockSize; ++j)
            Y.col(j) = linearSolver.solve(X.col(j));

        Eigen::MatrixXd Q = orthonormalized(Y);
        Eigen::MatrixXd AQ = hostA * Q;
        Eigen::MatrixXd projected = Q.transpose() * AQ;
        projected = 0.5 * (projected + projected.transpose());

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> smallSolver(projected);
        if (smallSolver.info() != Eigen::Success)
            throw std::runtime_error("Dense projected eigensolve failed in AMGXEigenSolver.");

        const Eigen::VectorXd previous = eigenvalues;
        eigenvalues = smallSolver.eigenvalues().head(numEigenvalues);
        eigenvectors = Q * smallSolver.eigenvectors().leftCols(numEigenvalues);
        const double residual = maxResidual(hostA, eigenvectors, eigenvalues);
        const double delta = (previous.array().isFinite().all())
            ? (eigenvalues - previous).cwiseAbs().maxCoeff()
            : std::numeric_limits<double>::infinity();

        if (options.verbose)
        {
            std::cout << "AMGX eigensolver outer " << outer + 1
                      << ", max residual " << residual
                      << ", eigenvalue delta " << delta << std::endl;
        }

        if (residual < options.tolerance || delta < options.tolerance)
            break;

        X = Q * smallSolver.eigenvectors().leftCols(blockSize);
    }

    return eigenvalues;
}

#endif
