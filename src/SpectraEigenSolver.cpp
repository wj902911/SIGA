#include "SpectraEigenSolver.h"

#include <algorithm>
#include <cmath>
#include <Eigen/SparseCholesky>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#if ENABLE_SPECTRA
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#if ENABLE_PARDISO
#include <Eigen/PardisoSupport>
#endif

namespace
{
#if ENABLE_PARDISO
class PardisoSymShiftSolve
{
public:
    using Scalar = double;
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
    using Solver = Eigen::PardisoLDLT<SpMat, Eigen::Upper>;

    explicit PardisoSymShiftSolve(const SpMat& mat)
        : m_mat(mat), m_n(static_cast<int>(mat.rows()))
    {
        if (mat.rows() != mat.cols())
            throw std::invalid_argument("PardisoSymShiftSolve: matrix must be square");
    }

    int rows() const { return m_n; }
    int cols() const { return m_n; }

    void set_shift(double sigma)
    {
        SpMat shifted = m_mat.template selfadjointView<Eigen::Upper>();
        SpMat identity(m_n, m_n);
        identity.setIdentity();
        shifted = shifted - sigma * identity;
        shifted.makeCompressed();

        m_solver.compute(shifted);
        if (m_solver.info() != Eigen::Success)
            throw std::invalid_argument("PardisoSymShiftSolve: factorization failed");
    }

    void perform_op(const double* x_in, double* y_out) const
    {
        Eigen::Map<const Eigen::VectorXd> x(x_in, m_n);
        Eigen::Map<Eigen::VectorXd> y(y_out, m_n);
        y.noalias() = m_solver.solve(x);
    }

private:
    const SpMat& m_mat;
    int m_n;
    Solver m_solver;
};
#endif

Eigen::VectorXd sortByDistance(const Eigen::VectorXd& values,
                               double center,
                               Eigen::MatrixXd* vectors)
{
    std::vector<int> order(static_cast<std::size_t>(values.size()));
    for (int i = 0; i < values.size(); ++i)
        order[static_cast<std::size_t>(i)] = i;

    std::sort(order.begin(), order.end(),
              [&values, center](int lhs, int rhs)
              {
                  return std::abs(values[lhs] - center) <
                         std::abs(values[rhs] - center);
              });

    Eigen::VectorXd sorted(values.size());
    Eigen::MatrixXd sortedVectors;
    if (vectors && vectors->cols() == values.size())
        sortedVectors.resize(vectors->rows(), vectors->cols());

    for (int i = 0; i < values.size(); ++i)
    {
        const int src = order[static_cast<std::size_t>(i)];
        sorted[i] = values[src];
        if (sortedVectors.size() > 0)
            sortedVectors.col(i) = vectors->col(src);
    }

    if (vectors && sortedVectors.size() > 0)
        *vectors = sortedVectors;

    return sorted;
}

} // namespace

double smallestEigenvalue_SPD_spectra(const Eigen::SparseMatrix<double> &K)
{
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt;
    llt.compute(K);
    if (llt.info() != Eigen::Success)
        return std::numeric_limits<double>::quiet_NaN();

    const double diag_max = K.diagonal().cwiseAbs().maxCoeff();
    const double sigma = (diag_max > 0 ? 1e-10 * diag_max : 1e-10);

    using OpType = Spectra::SparseSymShiftSolve<double>;
    OpType op(K);

    const int k = 1;
    const int ncv = 30; // >= 2*k+1; you can increase to 10~20 for robustness

    Spectra::SymEigsShiftSolver<OpType> eigs(op, k, ncv, sigma);

    eigs.init();
    const int nconv = eigs.compute(Spectra::SortRule::SmallestAlge, 5000, 1e-8);

    if (eigs.info() != Spectra::CompInfo::Successful || nconv < 1)
        return std::numeric_limits<double>::quiet_NaN();

    return eigs.eigenvalues()(0);
}

Eigen::VectorXd smallestEigenvalues_spectra(
    const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& K,
    int numEigenvalues,
    Eigen::MatrixXd* eigenvectors,
    int maxIterations,
    double tolerance)
{
    if (eigenvectors)
        eigenvectors->resize(K.rows(), 0);

    const int n = static_cast<int>(K.rows());
    if (n <= 1 || K.cols() != n || numEigenvalues <= 0)
        return Eigen::VectorXd();

    const int nev = (std::min)(numEigenvalues, n - 1);
    const int ncvTarget = (std::max)(2 * nev + 1, nev + 20);
    const int ncv = (std::min)(n, ncvTarget);
    if (ncv <= nev)
        return Eigen::VectorXd();

    const double diagMax = K.diagonal().cwiseAbs().maxCoeff();
    const double sigma = (diagMax > 0.0 ? 1e-10 * diagMax : 1e-10);

    return eigenvaluesNearShift_spectra(K, numEigenvalues, sigma,
                                        eigenvectors, maxIterations,
                                        tolerance);
}

Eigen::VectorXd eigenvaluesNearShift_spectra(
    const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& K,
    int numEigenvalues,
    double shift,
    Eigen::MatrixXd* eigenvectors,
    int maxIterations,
    double tolerance)
{
    if (eigenvectors)
        eigenvectors->resize(K.rows(), 0);

    const int n = static_cast<int>(K.rows());
    if (n <= 1 || K.cols() != n || numEigenvalues <= 0)
        return Eigen::VectorXd();

    const int nev = (std::min)(numEigenvalues, n - 1);
    const int ncvTarget = (std::max)(2 * nev + 1, nev + 20);
    const int ncv = (std::min)(n, ncvTarget);
    if (ncv <= nev)
        return Eigen::VectorXd();

    try
    {
#if ENABLE_PARDISO
        PardisoSymShiftSolve op(K);
#else
        using ColSpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
        ColSpMat Kcol = K.template selfadjointView<Eigen::Upper>();
        using OpType = Spectra::SparseSymShiftSolve<double, Eigen::Upper,
                                                    Eigen::ColMajor, int>;
        OpType op(Kcol);
#endif
        Spectra::SymEigsShiftSolver<decltype(op)> eigs(op, nev, ncv, shift);
        eigs.init();
        const int nconv = eigs.compute(Spectra::SortRule::LargestMagn,
                                       maxIterations,
                                       tolerance,
                                       Spectra::SortRule::LargestMagn);

        if (eigs.info() != Spectra::CompInfo::Successful || nconv < 1)
            return Eigen::VectorXd();

        Eigen::VectorXd values = eigs.eigenvalues();
        Eigen::MatrixXd vectors;
        if (eigenvectors)
            vectors = eigs.eigenvectors();

        values = sortByDistance(values, shift,
                                eigenvectors ? &vectors : nullptr);
        if (eigenvectors)
            *eigenvectors = vectors;
        return values;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Spectra smallest eigenvalues failed: "
                  << e.what() << std::endl;
        return Eigen::VectorXd();
    }
}
#endif
