#pragma once

#include <GPUAssembler.h>
#include <Eigen/Core>
#include <Eigen/Sparse>

#if ENABLE_PARDISO
#include <Eigen/PardisoSupport>
#endif

#if ENABLE_AMGX
#include <amgx_c.h>
#endif

enum solver_status { converged,      /// method successfully converged
                     interrupted,    /// solver was interrupted after exceeding the limit of iterations
                     working,        /// solver working
                     bad_solution }; /// method was interrupted because the current solution is invalid

class GPUSolver
{
private:
    GPUAssembler& m_assembler;
    DeviceArray<double> m_solVector;
    DeviceArray<double> m_deltaSolVector;
    solver_status m_status;
    double m_residualNorm = 0.0;
    double m_initResidualNorm = 0.0;
    double m_updateNorm = 0.0;
    double m_initUpdateNorm = 0.0;
    int m_numIterations = 0;
    DeviceNestedArray<double> m_fixedDoFs;

    double m_absTol = 1e-10;
    double m_relTol = 1e-10;
    int m_maxIter = 100;
    bool m_useSchurSolve = false;
    bool m_useExactSchurReduction = false;
    double m_schurGammaMax = 1.0;
    double m_schurDisplacementScale = 1.0;
    double m_schurPotentialScale = 1.0;

#if ENABLE_PARDISO
    using PardisoSpMat = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
    using PardisoSolver = Eigen::PardisoLDLT<PardisoSpMat, Eigen::Upper>;
    using PardisoLUSolver = Eigen::PardisoLU<PardisoSpMat>;

    PardisoSolver m_pardisoSolver;
    bool m_pardisoPatternAnalyzed = false;
    int m_pardisoRows = -1;
    int m_pardisoCols = -1;
    int m_pardisoNonZeros = -1;

    PardisoSolver m_schurPhiPardisoSolver;
    bool m_schurPhiPatternAnalyzed = false;
    int m_schurPhiRows = -1;
    int m_schurPhiCols = -1;
    int m_schurPhiNonZeros = -1;

    PardisoSolver m_schurUPardisoSolver;
    bool m_schurUPatternAnalyzed = false;
    int m_schurURows = -1;
    int m_schurUCols = -1;
    int m_schurUNonZeros = -1;

    bool factorWithPardiso(PardisoSolver& solver,
                           bool& patternAnalyzed,
                           int& rows,
                           int& cols,
                           int& nonZeros,
                           const PardisoSpMat& A,
                           const char* context);

    bool solveWithPardiso(const PardisoSpMat& A,
                          const Eigen::VectorXd& b,
                          Eigen::VectorXd& x,
                          const char* context);

    bool solveWithPardiso(PardisoSolver& solver,
                          bool& patternAnalyzed,
                          int& rows,
                          int& cols,
                          int& nonZeros,
                          const PardisoSpMat& A,
                          const Eigen::VectorXd& b,
                          Eigen::VectorXd& x,
                          const char* context);

    bool solveWithPardisoLU(const PardisoSpMat& A,
                            const Eigen::VectorXd& b,
                            Eigen::VectorXd& x,
                            const char* context);
#endif

#if ENABLE_AMGX
    bool m_amgx_initialized = false;

    AMGX_Mode             m_amgx_mode = AMGX_mode_dDDI;   // device data, double, int
    AMGX_config_handle    m_amgx_cfg   = nullptr;
    AMGX_resources_handle m_amgx_rsrc  = nullptr;
    AMGX_solver_handle    m_amgx_solver= nullptr;
    AMGX_matrix_handle    m_amgx_A     = nullptr;
    AMGX_vector_handle    m_amgx_b     = nullptr;
    AMGX_vector_handle    m_amgx_x     = nullptr;

    int m_amgx_n   = -1;
    int m_amgx_nnz = -1;
    Eigen::VectorXd m_cachedSmallestEigenvector;
    double m_cachedSmallestEigenvalue = 0.0;

    void initAMGXOnce();
    void finalizeAMGX();
#endif

public:
    __host__
    GPUSolver(GPUAssembler &assembler);

#if ENABLE_AMGX
    __host__
    ~GPUSolver() { finalizeAMGX(); }
#endif

    __host__
    void print() const;

    __host__
    DeviceNestedArrayView<double> allFixedDofsView() const
    { return m_fixedDoFs.view(); }

    __host__
    const DeviceNestedArray<double>& allFixedDofs() const { return m_fixedDoFs; }

    __host__
    DeviceVectorView<double> solutionView() const
    { return DeviceVectorView<double>(m_solVector.data(), m_solVector.size()); }

    //bool solveSingleIteration();
    bool solveSingleIteration_Eigen();
    bool solveSingleIteration_Schur();
#if ENABLE_AMGX
    bool solveSingleIteration_AMGX();
#endif
    void solve();
    double smallestEigenValue();
    double smallestEigenValue(Eigen::VectorXd& eigenvector);
    Eigen::VectorXd smallestEigenValue(int numEigenvalues,
                                       Eigen::MatrixXd* eigenvectors = nullptr,
                                       int maxIterations = 5000,
                                       double tolerance = 1e-8);
    Eigen::VectorXd smallestEigenValuesDenseEigen(int numEigenvalues);
    Eigen::VectorXd eigenValuesNearShift(int numEigenvalues,
                                         double shift,
                                         Eigen::MatrixXd* eigenvectors = nullptr,
                                         int maxIterations = 5000,
                                         double tolerance = 1e-8);
    double smallestCondensedMechanicalEigenpair(Eigen::VectorXd& displacementEigenvector,
                                                Eigen::VectorXd* electricEigenvector = nullptr,
                                                int maxIterations = 60,
                                                double tolerance = 1e-8);
    double condensedMechanicalStabilityLDLT();
    bool perturbWithCondensedEigenvectors(const Eigen::VectorXd& displacementEigenvector,
                                          const Eigen::VectorXd& electricEigenvector,
                                          double amplitude,
                                          int sign = 1);
    bool perturbWithCondensedMechanicalEigenvector(double amplitude,
                                                   int sign = 1,
                                                   double* eigenvalue = nullptr,
                                                   int maxIterations = 60,
                                                   double tolerance = 1e-8);
    double stability();
    void eigenvalues_symm_dense(DeviceMatrixView<double> matrix, 
                                DeviceVectorView<double> eigenvalues,
                                bool computeEigenvectors = false);

    double smallestEigenvalue_symm_dense_Eigen(Eigen::MatrixXd mat);
    double smallestEigenvalue_symm_sparse_Eigen(Eigen::SparseMatrix<double> mat);
    //double smallestEigenvalue_SPD_Dense_Spectra(const Eigen::SparseMatrix<double>& K);
    std::string status();

    int numIterations() const { return m_numIterations; }

    void solutionToHost(Eigen::VectorXd& hostSol) const
    {
        hostSol.resize(m_solVector.size());
        m_solVector.copyToHost(hostSol.data());
    }

    void setSolutionFromHost(const Eigen::VectorXd& hostSol)
    {
        assert(hostSol.size() == m_solVector.size() && "Host solution size must match device solution size");
        m_solVector.updateFromHost(hostSol.data());
    }

    void fixedDofsToHost(Eigen::VectorXd& hostFixedDofs) const
    {
        hostFixedDofs.resize(m_fixedDoFs.size());
        m_fixedDoFs.copyToHost(hostFixedDofs.data());
    }

    void setFixedDofsFromHost(const Eigen::VectorXd& hostFixedDofs)
    {
        assert(hostFixedDofs.size() == m_fixedDoFs.size() && "Host fixed DoFs size must match device fixed DoFs size");
        m_fixedDoFs.updateFromHost(hostFixedDofs.data());
    }

    bool isConverged() const { return m_status == converged; }  

    void setTolerance(double absTol, double relTol)
    {
        m_absTol = absTol;
        m_relTol = relTol;
    }

    void setMaxIterations(int maxIter) { m_maxIter = maxIter; }

    void setUseSchurSolve(bool useSchurSolve)
    {
        m_useSchurSolve = useSchurSolve;
    }

    void setModifiedStep(double gammaMax,
                               double displacementScale = 1.0,
                               double potentialScale = 1.0)
    {
        m_schurGammaMax = gammaMax;
        m_schurDisplacementScale = displacementScale;
        m_schurPotentialScale = potentialScale;
    }

    void setUseExactSchurReduction(bool useExactSchurReduction)
    {
        m_useExactSchurReduction = useExactSchurReduction;
    }
};
