#pragma once

#include <GPUAssembler.h>

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

public:
    __host__
    GPUSolver(GPUAssembler &assembler);

    __host__
    void print() const;

    __host__
    DeviceNestedArrayView<double> allFixedDofsView() const
    { return m_fixedDoFs.view(); }

    __host__
    DeviceVectorView<double> solutionView() const
    { return DeviceVectorView<double>(m_solVector.data(), m_solVector.size()); }

    bool solveSingleIteration();
    bool solveSingleIteration_Eigen();
    void solve();
    std::string status();
};