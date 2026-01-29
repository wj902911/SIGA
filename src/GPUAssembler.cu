#include <GPUAssembler.h>

__global__
void printKernel(MultiPatchDeviceView multiPatch,
                 MultiBasisDeviceView multiBasis,
                 SparseSystemDeviceView sparseSystem,
                 DeviceNestedArrayView<double> ddof,
                 DeviceVectorView<double> bodyForce)
{
    multiPatch.print();
    printf("\n");
    multiBasis.print();
    printf("\n");
    sparseSystem.print();
    printf("\n");
    ddof.print();
    printf("\n");
    bodyForce.print();
}

GPUAssembler::GPUAssembler(const MultiPatch &multiPatch, 
                           const MultiBasis &multiBasis, 
                           const BoundaryConditions &bc, 
                           const Eigen::VectorXd &bodyForce)
: m_multiPatch(multiPatch), m_multiBasis(multiBasis), 
  m_boundaryConditions(bc), m_bodyForce(bodyForce)
{
    int targetDim = multiPatch.getCPDim();
    std::vector<DofMapper> dofMappers_stdVec(targetDim);
    multiBasis.getMappers(true, m_boundaryConditions, 
                          dofMappers_stdVec, true);
    SparseSystem sparseSystem(dofMappers_stdVec, 
                              Eigen::VectorXi::Ones(targetDim));
    m_sparseSystem.setMatrixRows(sparseSystem.matrix().rows());
    m_sparseSystem.setMatrixCols(sparseSystem.matrix().cols());
    std::vector<int> intDataOffsets;
    std::vector<int> intData;
    std::vector<double> doubleData;
    sparseSystem.getDataVector(intDataOffsets, intData, doubleData);
    m_sparseSystem.setIntDataOffsets(intDataOffsets);
    m_sparseSystem.setIntData(intData);
    m_sparseSystem.setDoubleData(doubleData);

    std::vector<Eigen::VectorXd> ddof(targetDim);
    for (int unk = 0; unk < targetDim; ++unk)
        computeDirichletDofs(unk, dofMappers_stdVec, ddof, multiBasis);
    m_ddof.setData(ddof);
}

void GPUAssembler::
computeDirichletDofs(int unk_, 
                     const std::vector<DofMapper> &mappers,
                     std::vector<Eigen::VectorXd> &ddof,
                     const MultiBasis &multiBasis)
{
    DofMapper dofMapper = mappers[unk_]; 
    ddof[unk_].resize(dofMapper.boundarySize());
    const TensorBsplineBasis &basis = multiBasis.basis(unk_);

    for (std::deque<boundary_condition>::const_iterator 
         it = m_boundaryConditions.dirichletBegin();
         it != m_boundaryConditions.dirichletEnd(); ++it)
    {
        const int k = it->patchIndex();
        if (it -> unknown() != unk_)
            continue;
        const Eigen::VectorXi boundary = basis.boundary(it -> side());
        //std::cout << boundary << std::endl;
        for (int i = 0; i != boundary.size(); ++i)
        {
            const int ii = dofMapper.bindex(boundary[i], k);
            ddof[unk_][ii] = it->value(unk_);
        }
    }

    for (std::deque<corner_condition>::const_iterator 
         it = m_boundaryConditions.dirichletCornerBegin();
         it != m_boundaryConditions.dirichletCornerEnd(); ++it)
    {
        const int k = it->patchIndex();
        if (it -> unknown() != unk_)
            continue;
        const int i = basis.corner(it -> corner());
        const int ii = dofMapper.bindex(i, k);
        ddof[unk_][ii] = it->value(unk_);
    }
}

void GPUAssembler::print() const
{
    printKernel<<<1,1>>>(m_multiPatch.deviceView(),
                         m_multiBasis.deviceView(),
                         m_sparseSystem.deviceView(),
                         m_ddof.view(),
                         m_bodyForce.vectorView());
    cudaError_t err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::print");
}

int GPUAssembler::numDofs() const
{
    return m_sparseSystem.numDofs();
}
