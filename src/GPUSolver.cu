#include <GPUSolver.h>

__host__
GPUSolver::GPUSolver(GPUAssembler &assembler): m_assembler(assembler)
{
    Eigen::VectorXd solVector(assembler.numDofs());
    m_solVector = solVector;
    fixedDoFs = assembler.allFixedDofs();
    fixedDoFs.setZero();
}