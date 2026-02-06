#include <GPUAssembler.h>

__global__
void constructSolutionKernel(DeviceVectorView<double> solVector, 
                             DeviceNestedArrayView<double> fixedDoFs, 
                             MultiBasisDeviceView multiBasis,
                             SparseSystemDeviceView sparseSystem,
                             MultiPatchDeviceView result, int CPSize)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < CPSize; idx += blockDim.x * gridDim.x)
    {
        int patch(0);
        int unk(0);
        int point_idx = result.threadPatchAndDof(idx, patch, unk);
        int index(0);
        //printf("patch %d, point_idx %d, unknown:%d\n", patch, point_idx, unk);
        if (sparseSystem.mapper(unk).is_free(point_idx, patch))
        {
            //printf("free dof\n");
            index = sparseSystem.mapToGlobalColIndex(point_idx, patch, unk);
            //printf("global index: %d\n", index);
            result.setCoefficients(patch, point_idx, unk, solVector[index]);
        }
        else
        {
            //printf("fixed dof\n");
            index = sparseSystem.mapper(unk).bindex(point_idx, patch);
            //printf("global index: %d\n", index);
            result.setCoefficients(patch, point_idx, unk, fixedDoFs[unk][index]);
        }
    }
    //result.print();
}

__global__
void assembleDomainKernel(int totalGPs,
                          MultiPatchDeviceView displacement,
                          MultiPatchDeviceView multiPatch,
                          MultiGaussPointsDeviceView multiGaussPoints,
                          DeviceVectorView<double> bodyForce,
                          SparseSystemDeviceView sparseSystem,
                          DeviceNestedArrayView<double> ddof)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < totalGPs; idx += blockDim.x * gridDim.x)
    {
        double YM = 1.0;
        double PR = 0.3;
        double lambda = YM * PR / ( ( 1. + PR ) * ( 1. - 2. * PR ) );
        double mu = YM / ( 2. * ( 1. + PR ) );

        int numDerivatives = 1;
        int CPdim = multiPatch.targetDim();
        int dim = multiPatch.domainDim();
        int patch_idx(0);
        int point_idx = displacement.threadPatch(idx, patch_idx);
        double ptData[3]; //max 3D
        DeviceVectorView<double> pt(ptData, multiGaussPoints.dim());
        double wt = displacement.gsPoint(point_idx, patch_idx, 
                                         multiGaussPoints[patch_idx], pt);
        printf("Patch %d, GP %d\n Weight %f\n Point:\n", patch_idx, point_idx, wt);
        pt.print();
        double geoJacobianData[3*3] = {0.0}; //max 3D
        DeviceMatrixView<double> geoJacobian(geoJacobianData, dim, dim);
        PatchDeviceView geoPatch = multiPatch.patch(patch_idx);
        geoPatch.jacobian(pt, geoJacobian);
        //printf("Geometry Jacobian:\n");
        //geoJacobian.print();
        double measure = geoJacobian.determinant();
        //printf("measure %f\n", measure);
        double weightForce = wt * measure;
        double weightBody = wt * measure;

        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
        PatchDeviceView dispPatch = displacement.patch(patch_idx);
        int P1 = dispBasis.knotsOrder(0) + 1;
        double dispValuesAndDersData[5*2*3]; //max 4th order basis, max 2 derivatives, max 3D
        DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDersData, P1, 
                                                   (numDerivatives+1)*dim);
        dispBasis.evalAllDers_into(pt, numDerivatives, dispValuesAndDers);
        //printf("Displacement basis values and derivatives:\n");
        //dispValuesAndDers.print();
        double FData[3*3] = {0.0}; //max 3D
        DeviceMatrixView<double> F(FData, dim, dim);
        {
            double dispJacobianData[3*3] = {0.0}; //max 3D
            double geoJacobianInvData[3*3] = {0.0}; //max 3D
            double physDispJacData[3*3] = {0.0}; //max 3D
            DeviceMatrixView<double> dispJacobian(dispJacobianData, dim, dim);
            DeviceMatrixView<double> geoJacobianInv(geoJacobianInvData, dim, dim);
            DeviceMatrixView<double> physDispJac(physDispJacData, dim, dim);
            geoJacobian.inverse(geoJacobianInv);
            int numActiveCPs = dispBasis.numActiveControlPoints();
            for (int r = 0; r < numActiveCPs; r++)
            {
                int tensorCoordData[3]; //max 3D
                DeviceVectorView<int> tensorCoord(tensorCoordData, dim);
                getTensorCoordinate(dim, P1, r, tensorCoordData);
		    	for (int j = 0; j < dim; j++)
                {
                    double dN_rj = 1.0;
                    for (int d = 0; d < dim; d++)
                    {
                        if (d == j)
                            dN_rj *= dispValuesAndDers(tensorCoord[d], 
                                (1 + numDerivatives) * d);
                        else
                            dN_rj *= dispValuesAndDers(tensorCoord[d], 
                                (1 + numDerivatives) * d - 1);
                    }
                    for (int i = 0; i < dim; i++)
                        dispJacobian(i, j) += dN_rj * 
                            dispPatch.activeControlPointComponent(pt, r, i);
                }
            }
            dispJacobian.times(geoJacobianInv, physDispJac);
            physDispJac.plusIdentity(F);
        }
        printf("Deformation gradient F:\n");
        F.print();

#if 0
        for (int r = 0; r < numActiveCPs; r++)
        {
            for (int c = 0; c < CPdim; c++)
            {
                double coeff = geoPatch.activeControlPointComponent(pt, r, c);
                
                
                int localIndexDisp = displacement.basis(patch_idx)
                                        .activeIndex(pt, r);
                //printf("Active CP %d, component %d: %f, local index: %d\n", 
                //        r, c, coeff, localIndexDisp);
            }
        }    
#endif
        
    }
}

__global__
void printKernel(MultiPatchDeviceView multiPatch,
                 MultiPatchDeviceView displacement,
                 MultiBasisDeviceView multiBasis,
                 SparseSystemDeviceView sparseSystem,
                 DeviceNestedArrayView<double> ddof,
                 DeviceNestedArrayView<double> ddof_zero,
                 DeviceVectorView<double> bodyForce,
                 MultiGaussPointsDeviceView multiGaussPoints)
{
    printf("multiPatch data:\n");
    multiPatch.print();
    printf("\n");
    printf("displacement data:\n");
    displacement.print();
    printf("\n");
    printf("multiBasis data:\n");
    multiBasis.print();
    printf("\n");
    printf("sparseSystem data:\n");
    sparseSystem.print();
    printf("\n");
    printf("ddof data:\n");
    ddof.print();
    printf("\n");
    printf("ddof_zero data:\n");
    ddof_zero.print();
    printf("\n");
    printf("bodyForce data:\n");
    bodyForce.print();
    printf("\n");
    printf("multiGaussPoints data:\n");
    multiGaussPoints.print();
}

GPUAssembler::GPUAssembler(const MultiPatch &multiPatch, 
                           const MultiBasis &multiBasis, 
                           const BoundaryConditions &bc, 
                           const Eigen::VectorXd &bodyForce)
: m_multiPatch(multiPatch), m_multiBasis(multiBasis), 
  m_boundaryConditions(bc), m_bodyForce(bodyForce),
  m_multiGaussPoints(multiBasis),
  m_multiPatchHost(multiPatch), m_multiBasisHost(multiBasis)
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
    std::vector<Eigen::VectorXd> ddof_zero(targetDim);
    for (int unk = 0; unk < targetDim; ++unk)
    {
        computeDirichletDofs(unk, dofMappers_stdVec, ddof, multiBasis);
        ddof_zero[unk] = Eigen::VectorXd::Zero(ddof[unk].size());
    }    
    m_ddof.setData(ddof);
    m_ddof_zero.setData(ddof_zero);

    MultiPatch displacement;
    m_multiBasisHost.giveBasis(displacement, targetDim);
    m_displacement = MultiPatchDeviceData(displacement);
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
                         m_displacement.deviceView(),
                         m_multiBasis.deviceView(),
                         m_sparseSystem.deviceView(),
                         m_ddof.view(),
                         m_ddof_zero.view(),
                         m_bodyForce.vectorView(),
                         m_multiGaussPoints.view());
    cudaError_t err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::print");
}

int GPUAssembler::numDofs() const
{
    return m_sparseSystem.numDofs();
}

void GPUAssembler::constructSolution(const DeviceVectorView<double>& solVector, 
                                     const DeviceNestedArrayView<double>& fixedDoFs, 
                                     MultiPatchDeviceView& displacementDeviceView) const
{
    constructSolutionKernel<<<1,1>>>(solVector, fixedDoFs,
                                     m_multiBasis.deviceView(),
                                     m_sparseSystem.deviceView(),
                                     displacementDeviceView,
                                     m_multiPatchHost.CPSize());
    cudaError_t err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructSolution");
}

void GPUAssembler::assemble(const DeviceVectorView<double> &solVector, 
                            int numIter, 
                            const DeviceNestedArrayView<double> &fixedDoFs)
{
    int minGrid, blockSize;
    int CPSize = m_multiPatchHost.CPSize();

    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
        constructSolutionKernel, 0, CPSize);

    int gridSize = (CPSize + blockSize - 1) / blockSize;
    constructSolutionKernel<<<gridSize, blockSize>>>(solVector, fixedDoFs,
                                                     m_multiBasis.deviceView(),
                                                     m_sparseSystem.deviceView(),
                                                     m_displacement.deviceView(),
                                                     CPSize);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error after constructSolutionKernel in GPUAssembler::assemble: "
                  << cudaGetErrorString(err) << std::endl;

    DeviceNestedArrayView<double> fixedDofs_assemble;
    if (numIter != 0)
        fixedDofs_assemble = m_ddof_zero.view();
    else
        fixedDofs_assemble = m_ddof.view();

    int totalGPs = m_multiBasisHost.totalNumGPs();
    int domainDim = m_multiPatchHost.getBasisDim();
    int basisOrder = m_multiBasisHost.basis(0).getOrder(0); //assume same order in all directions
#if 0    
    if (domainDim == 2)
    {
        switch (basisOrder)
        {
            case 1:
                cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                    assembleDomainKernel<1, 2>, 0, totalGPs);
                gridSize = (totalGPs + blockSize - 1) / blockSize;
#if 1
                assembleDomainKernel<1, 2><<<1, 1>>>(
                    totalGPs,
                    m_displacement.deviceView(),
                    m_multiPatch.deviceView(),
                    m_multiGaussPoints.view(),
                    m_bodyForce.vectorView(),
                    m_sparseSystem.deviceView(),
                    fixedDofs_assemble);
#else
                assembleDomainKernel<1, 2><<<gridSize, blockSize>>>(
                    totalGPs,
                    m_displacement.deviceView(),
                    m_multiPatch.deviceView(),
                    m_multiGaussPoints.view(),
                    m_bodyForce.vectorView(),
                    m_sparseSystem.deviceView(),
                    fixedDofs_assemble);
                
#endif
                break;
            case 2:
                cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                    assembleDomainKernel<2, 2>, 0, totalGPs);
                gridSize = (totalGPs + blockSize - 1) / blockSize;
                assembleDomainKernel<2, 2><<<gridSize, blockSize>>>(
                    totalGPs,
                    m_displacement.deviceView(),
                    m_multiPatch.deviceView(),
                    m_multiGaussPoints.view(),
                    m_bodyForce.vectorView(),
                    m_sparseSystem.deviceView(),
                    fixedDofs_assemble);
                break;
            case 3:
                cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                    assembleDomainKernel<3, 2>, 0, totalGPs);
                gridSize = (totalGPs + blockSize - 1) / blockSize;
                assembleDomainKernel<3, 2><<<gridSize, blockSize>>>(
                    totalGPs,
                    m_displacement.deviceView(),
                    m_multiPatch.deviceView(),
                    m_multiGaussPoints.view(),
                    m_bodyForce.vectorView(),
                    m_sparseSystem.deviceView(),
                    fixedDofs_assemble);
                break;
            case 4:
                cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                    assembleDomainKernel<4, 2>, 0, totalGPs);
                gridSize = (totalGPs + blockSize - 1) / blockSize;
                assembleDomainKernel<4, 2><<<gridSize, blockSize>>>(
                    totalGPs,
                    m_displacement.deviceView(),
                    m_multiPatch.deviceView(),
                    m_multiGaussPoints.view(),
                    m_bodyForce.vectorView(),
                    m_sparseSystem.deviceView(),
                    fixedDofs_assemble);
                break;
            default:
                assert(false && "Unsupported basis order in GPUAssembler::assemble");
        }
    }
    else if (domainDim == 3)
    {
        switch (basisOrder)
        {
            case 1:
                cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                    assembleDomainKernel<1, 3>, 0, totalGPs);
                gridSize = (totalGPs + blockSize - 1) / blockSize;
                assembleDomainKernel<1, 3><<<gridSize, blockSize>>>(
                    totalGPs,
                    m_displacement.deviceView(),
                    m_multiPatch.deviceView(),
                    m_multiGaussPoints.view(),
                    m_bodyForce.vectorView(),
                    m_sparseSystem.deviceView(),
                    fixedDofs_assemble);
                break;
            case 2:
                cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                    assembleDomainKernel<2, 3>, 0, totalGPs);
                gridSize = (totalGPs + blockSize - 1) / blockSize;
                assembleDomainKernel<2, 3><<<gridSize, blockSize>>>(
                    totalGPs,
                    m_displacement.deviceView(),
                    m_multiPatch.deviceView(),
                    m_multiGaussPoints.view(),
                    m_bodyForce.vectorView(),
                    m_sparseSystem.deviceView(),
                    fixedDofs_assemble);
                break;
            case 3:
                cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                    assembleDomainKernel<3, 3>, 0, totalGPs);
                gridSize = (totalGPs + blockSize - 1) / blockSize;
                assembleDomainKernel<3, 3><<<gridSize, blockSize>>>(
                    totalGPs,
                    m_displacement.deviceView(),
                    m_multiPatch.deviceView(),
                    m_multiGaussPoints.view(),
                    m_bodyForce.vectorView(),
                    m_sparseSystem.deviceView(),
                    fixedDofs_assemble);
                break;
            case 4:
                cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                    assembleDomainKernel<4, 3>, 0, totalGPs);
                gridSize = (totalGPs + blockSize - 1) / blockSize;
                assembleDomainKernel<4, 3><<<gridSize, blockSize>>>(
                    totalGPs,
                    m_displacement.deviceView(),
                    m_multiPatch.deviceView(),
                    m_multiGaussPoints.view(),
                    m_bodyForce.vectorView(),
                    m_sparseSystem.deviceView(),
                    fixedDofs_assemble);
                break;
            default:
                assert(false && "Unsupported basis order in GPUAssembler::assemble");
        }
    }
    else
    {
        assert(false && "Unsupported domain dimension in GPUAssembler::assemble");
    }
#else
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                         assembleDomainKernel, 0, totalGPs);
    gridSize = (totalGPs + blockSize - 1) / blockSize;
#if 0
    assembleDomainKernel<<<gridSize, blockSize>>>(
                    totalGPs,
                    m_displacement.deviceView(),
                    m_multiPatch.deviceView(),
                    m_multiGaussPoints.view(),
                    m_bodyForce.vectorView(),
                    m_sparseSystem.deviceView(),
                    fixedDofs_assemble);
#else
    assembleDomainKernel<<<1, 1>>>(
                    totalGPs,
                    m_displacement.deviceView(),
                    m_multiPatch.deviceView(),
                    m_multiGaussPoints.view(),
                    m_bodyForce.vectorView(),
                    m_sparseSystem.deviceView(),
                    fixedDofs_assemble);
#endif
#endif
    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after kernel assembleDomain launch: " 
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (assembleDomain): " 
                  << cudaGetErrorString(err) << std::endl;
}
