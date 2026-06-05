#pragma once

#include <Eigen/Core>

#if ENABLE_AMGX

class DeviceCSRMatrix;
class GPUAssembler;

class AMGXEigenSolver
{
public:
    struct Options
    {
        int maxOuterIterations = 40;
        int extraSearchVectors = 4;
        double tolerance = 1e-8;
        double amgxTolerance = 1e-8;
        int amgxMaxIterations = 2000;
        bool verbose = false;
    };

    AMGXEigenSolver();
    ~AMGXEigenSolver();

    AMGXEigenSolver(const AMGXEigenSolver&) = delete;
    AMGXEigenSolver& operator=(const AMGXEigenSolver&) = delete;

    Eigen::VectorXd computeSmallestEigenvalues(const DeviceCSRMatrix& matrix,
                                               int numEigenvalues,
                                               const Options& options = Options{});

    Eigen::VectorXd computeSmallestEigenvalues(const GPUAssembler& assembler,
                                               int numEigenvalues,
                                               const Options& options = Options{});
};

#endif
