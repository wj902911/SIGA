#include "SpectraEigenSolver.h"

#include <Eigen/SparseCholesky>
#include <iostream>
#include <limits>
#if ENABLE_SPECTRA
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>

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
#endif