#pragma once
#include <Assembler.h>

enum solver_status { converged,      /// method successfully converged
                     interrupted,    /// solver was interrupted after exceeding the limit of iterations
                     working,        /// solver working
                     bad_solution }; /// method was interrupted because the current solution is invalid

class Solver
{
public:
    Solver(Assembler& assembler);
    ~Solver();

    bool solveSingleIteration();
    void solve();
    std::string status();
private:
    Assembler& m_assembler;
    DeviceVector<double> m_solVector;
    solver_status m_status;
    double m_residualNorm;
    double m_initResidualNorm;
    double m_updateNorm;
    double m_initUpdateNorm;
    int m_numIterations;
};