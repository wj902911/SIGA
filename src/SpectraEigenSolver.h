#pragma once
#include <Eigen/Sparse>

#if ENABLE_SPECTRA
double smallestEigenvalue_SPD_spectra(const Eigen::SparseMatrix<double>& K);
#endif