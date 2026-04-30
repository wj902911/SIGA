#pragma once

#include <GPUAssembler.h>
#include <Eigen/Sparse>

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
#if ENABLE_AMGX
    bool solveSingleIteration_AMGX();
#endif
    void solve();
    void eigenvalues_symm_dense(DeviceMatrixView<double> matrix, 
                                DeviceVectorView<double> eigenvalues);

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
};