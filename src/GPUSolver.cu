#include <GPUSolver.h>

__global__
void printKernel(DeviceVectorView<double> solVector,
                 DeviceNestedArrayView<double> fixedDoFs)
{
    printf("Solution Vector:\n");
    solVector.print();
    printf("Fixed DoFs:\n");
    fixedDoFs.print();
}

__host__
GPUSolver::GPUSolver(GPUAssembler &assembler): m_assembler(assembler)
{
    Eigen::VectorXd solVector(assembler.numDofs());
    solVector.setZero();
    m_solVector = solVector;
    fixedDoFs = assembler.allFixedDofs();
    fixedDoFs.setZero();
}

__host__
void GPUSolver::print() const
{
    m_assembler.print();
    printKernel<<<1,1>>>(m_solVector.vectorView(), 
                         fixedDoFs.view());
    cudaError_t err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUSolver::print");
}
