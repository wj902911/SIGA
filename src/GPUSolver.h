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
    solver_status m_status;
    double m_residualNorm = 0.0;
    double m_initResidualNorm = 0.0;
    double m_updateNorm = 0.0;
    double m_initUpdateNorm = 0.0;
    int m_numIterations = 0;
    DeviceNestedArray<double> fixedDoFs;

public:
    __host__
    GPUSolver(GPUAssembler &assembler);



};