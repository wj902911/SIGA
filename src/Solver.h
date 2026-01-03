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
    DeviceVector<double> solution() const;
    void constructSolution(MultiPatch& displacement) const;
private:
    Assembler& m_assembler;
    DeviceVector<double> m_solVector;
    solver_status m_status;
    double m_residualNorm = 0.0;
    double m_initResidualNorm = 0.0;
    double m_updateNorm = 0.0;
    double m_initUpdateNorm = 0.0;
    int m_numIterations = 0;
    std::vector<Eigen::VectorXd> fixedDoFs;
};