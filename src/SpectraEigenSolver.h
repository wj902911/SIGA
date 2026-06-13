#pragma once
#include <Eigen/Sparse>

#if ENABLE_SPECTRA
double smallestEigenvalue_SPD_spectra(const Eigen::SparseMatrix<double>& K);
Eigen::VectorXd smallestEigenvalues_spectra(
    const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& K,
    int numEigenvalues,
    Eigen::MatrixXd* eigenvectors = nullptr,
    int maxIterations = 5000,
    double tolerance = 1e-8);
Eigen::VectorXd eigenvaluesNearShift_spectra(
    const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& K,
    int numEigenvalues,
    double shift,
    Eigen::MatrixXd* eigenvectors = nullptr,
    int maxIterations = 5000,
    double tolerance = 1e-8);
#endif
