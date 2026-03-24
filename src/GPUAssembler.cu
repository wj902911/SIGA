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

__device__
void tensorBasisDerivative(int r, int P1, int dim, int numDerivatives,
    DeviceMatrixView<double> valuesAndDers, 
    DeviceVectorView<double> dN_r)
{
    int tensorCoordData[3]; //max 3D
    DeviceVectorView<int> tensorCoord(tensorCoordData, dim);
    getTensorCoordinate(dim, P1, r, tensorCoordData);
    for (int dir = 0; dir < dim; dir++)
    {
        double dN_rj = 1.0;
        for (int d = 0; d < dim; d++)
        {
            if (d == dir)
                dN_rj *= valuesAndDers(tensorCoord[d], (numDerivatives + 1) * d + 1);
            else
                dN_rj *= valuesAndDers(tensorCoord[d], (numDerivatives + 1) * d);
        }
        dN_r[dir] = dN_rj;
    }
}

__device__
double tensorBasisValue(int r, int P1, int dim, int numDerivatives,
    DeviceMatrixView<double> valuesAndDers)
{
    int tensorCoordData[3]; //max 3D
    DeviceVectorView<int> tensorCoord(tensorCoordData, dim);
    getTensorCoordinate(dim, P1, r, tensorCoordData);
    double N_r = 1.0;
    for (int d = 0; d < dim; d++)
    {
        N_r *= valuesAndDers(tensorCoord[d], (numDerivatives + 1) * d);
    }
    return N_r;
}

#if 1
__global__
void countEntrysKernel(int numElements, int numBlocksPerElement, int numActivePerBlock,
                       MultiPatchDeviceView displacement,
                       MultiPatchDeviceView multiPatch,
                       MultiGaussPointsDeviceView multiGaussPoints,
                       SparseSystemDeviceView system,
                       DeviceNestedArrayView<double> eliminatedDofs,
                       int* entryCount)
{
    __shared__ int sharedEntryCount;
    int totalNumBlocks = numElements * numBlocksPerElement * numBlocksPerElement;
    for (int idx = blockIdx.x; idx < totalNumBlocks; idx += gridDim.x)
    {
        __shared__ int patch_idx, N_D, dim, blockCoord[2];
        __shared__ double ptData[3]; //max 3D
        DeviceVectorView<double> pt(ptData, multiGaussPoints.dim());
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            for (int d = 0; d < 2; d++)
            {
                blockCoord[d] = idx % numBlocksPerElement;
                idx /= numBlocksPerElement;
            }
            sharedEntryCount = 0;
            int element_idx = displacement.threadPatch_element(idx, patch_idx);
            int numGPsInElement = displacement.basis(patch_idx).numGPsInElement();
            int point_idx = element_idx * numGPsInElement;
            displacement.gsPoint(point_idx, patch_idx, 
                                 multiGaussPoints[patch_idx], pt);
            TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
            N_D = dispBasis.numActiveControlPoints();
            dim = multiPatch.targetDim();
        }
        __syncthreads();
        
        for (int i = threadIdx.x + blockCoord[0] * numActivePerBlock, ii = threadIdx.x; 
             ii < numActivePerBlock && i < N_D; i += blockDim.x, ii += blockDim.x)
        {
            for (int di = 0; di < dim; di++)
            {
                int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx)
                    .activeIndex(pt, i), patch_idx, di);
                for (int j = threadIdx.y + blockCoord[1] * numActivePerBlock, jj = threadIdx.y; 
                     jj < numActivePerBlock && j < N_D; j += blockDim.y, jj += blockDim.y)
                {
                    for (int dj = 0; dj < dim; dj++)
                    {
                        int globalIndex_j = system.mapColIndex(displacement.basis(patch_idx)
                            .activeIndex(pt, j), patch_idx, dj);
                        if (system.isEntry(globalIndex_i, globalIndex_j, di, dj))
                        {
                            //printf("Counting entry for i=%d, di=%d, j=%d, dj=%d, globalIndex_i=%d, globalIndex_j=%d\n", 
                            //       i, di, j, dj, globalIndex_i, globalIndex_j);
                            atomicAdd(&sharedEntryCount, 1);
                        }
                        else
                        {
                            //printf("No entry for i=%d, di=%d, j=%d, dj=%d, globalIndex_i=%d, globalIndex_j=%d\n", 
                            //       i, di, j, dj, globalIndex_i, globalIndex_j);
                        }
                        //atomicAdd(entryCount, 1);
                        //printf("idx=%d, i=%d, di=%d, j=%d, dj=%d, globalIndex_i=%d, globalIndex_j=%d, count=%d\n", 
                               //idx, i, di, j, dj, globalIndex_i, globalIndex_j, *entryCount);
                    }
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0)
        atomicAdd(entryCount, sharedEntryCount);
}
#else
__global__
void countEntrysKernel(int totalNumElements,
                       MultiPatchDeviceView displacement,
                       MultiPatchDeviceView multiPatch,
                       MultiGaussPointsDeviceView multiGaussPoints,
                       SparseSystemDeviceView system,
                       DeviceNestedArrayView<double> eliminatedDofs,
                       int* entryCount)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < totalNumElements; idx += blockDim.x * gridDim.x)
    {
        int patch_idx(0);
        int element_idx = displacement.threadPatch_element(idx, patch_idx);
        int numGPsInElement = displacement.basis(patch_idx).numGPsInElement();
        int point_idx = element_idx * numGPsInElement;
        double ptData[3]; //max 3D
        DeviceVectorView<double> pt(ptData, multiGaussPoints.dim());
        double wt = displacement.gsPoint(point_idx, patch_idx, 
                                         multiGaussPoints[patch_idx], pt);
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
        int N_D = dispBasis.numActiveControlPoints();
        int dim = multiPatch.targetDim();
        for (int i = 0; i < N_D; i++)
        {
            for (int di = 0; di < dim; di++)
            {
                int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx)
                    .activeIndex(pt, i), patch_idx, di);
                for (int j = 0; j < N_D; j++)
                {
                    for (int dj = 0; dj < dim; dj++)
                    {
                        int globalIndex_j = system.mapColIndex(displacement.basis(patch_idx)
                            .activeIndex(pt, j), patch_idx, dj);
                        if (system.isEntry(globalIndex_i, globalIndex_j, di, dj))
                        {
                            //printf("Counting entry for i=%d, di=%d, j=%d, dj=%d, globalIndex_i=%d, globalIndex_j=%d\n", 
                            //       i, di, j, dj, globalIndex_i, globalIndex_j);
                            atomicAdd(entryCount, 1);
                        }
                        else
                        {
                            //printf("No entry for i=%d, di=%d, j=%d, dj=%d, globalIndex_i=%d, globalIndex_j=%d\n", 
                            //       i, di, j, dj, globalIndex_i, globalIndex_j);
                        }
                        //atomicAdd(entryCount, 1);
                        //printf("idx=%d, i=%d, di=%d, j=%d, dj=%d, globalIndex_i=%d, globalIndex_j=%d, count=%d\n", 
                               //idx, i, di, j, dj, globalIndex_i, globalIndex_j, *entryCount);
                    }
                }
            }
        }
    }
}
#endif

#if 1
__global__
void computeCOOKernel(int totalNumElements, int* counter,
                      int numBlocksPerElement, int numActivePerBlock,
                      MultiPatchDeviceView displacement,
                      MultiPatchDeviceView multiPatch,
                      MultiGaussPointsDeviceView multiGaussPoints,
                      SparseSystemDeviceView system,
                      DeviceNestedArrayView<double> eliminatedDofs,
                      DeviceVectorView<int> rows,
                      DeviceVectorView<int> cols)
{
    int totalNumBlocks = totalNumElements * numBlocksPerElement * numBlocksPerElement;
    for (int idx = blockIdx.x; idx < totalNumBlocks; idx += gridDim.x)
    {
        __shared__ int patch_idx, N_D, dim, blockCoord[2];
        __shared__ double ptData[3]; //max 3D
        DeviceVectorView<double> pt(ptData, multiGaussPoints.dim());
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            for (int d = 0; d < 2; d++)
            {
                blockCoord[d] = idx % numBlocksPerElement;
                idx /= numBlocksPerElement;
            }
            int element_idx = displacement.threadPatch_element(idx, patch_idx);
            int numGPsInElement = displacement.basis(patch_idx).numGPsInElement();
            int point_idx = element_idx * numGPsInElement;
            displacement.gsPoint(point_idx, patch_idx, 
                                         multiGaussPoints[patch_idx], pt);
            TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
            N_D = dispBasis.numActiveControlPoints();
            dim = multiPatch.targetDim();
        }
        __syncthreads();
        for (int i = threadIdx.x + blockCoord[0] * numActivePerBlock, ii = threadIdx.x; 
             ii < numActivePerBlock && i < N_D; i += blockDim.x, ii += blockDim.x)
        {
            for (int di = 0; di < dim; di++)
            {
                int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx)
                    .activeIndex(pt, i), patch_idx, di);
                for (int j = threadIdx.y + blockCoord[1] * numActivePerBlock, jj = threadIdx.y; 
                     jj < numActivePerBlock && j < N_D; j += blockDim.y, jj += blockDim.y)
                {
                    for (int dj = 0; dj < dim; dj++)
                    {
                        int globalIndex_j = system.mapColIndex(displacement.basis(patch_idx)
                            .activeIndex(pt, j), patch_idx, dj);
                        if (system.isEntry(globalIndex_i, globalIndex_j, di, dj))
                            system.pushToEntryIndex(globalIndex_i, globalIndex_j, di, dj, 
                                                    counter, rows, cols);
                    }
                }
            }
        }
    }
}
#else
__global__
void computeCOOKernel(int totalNumElements, int* counter,
                      MultiPatchDeviceView displacement,
                      MultiPatchDeviceView multiPatch,
                      MultiGaussPointsDeviceView multiGaussPoints,
                      SparseSystemDeviceView system,
                      DeviceNestedArrayView<double> eliminatedDofs,
                      DeviceVectorView<int> rows,
                      DeviceVectorView<int> cols)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < totalNumElements; idx += blockDim.x * gridDim.x)
    {
        int patch_idx(0);
        int element_idx = displacement.threadPatch_element(idx, patch_idx);
        int numGPsInElement = displacement.basis(patch_idx).numGPsInElement();
        int point_idx = element_idx * numGPsInElement;
        double ptData[3]; //max 3D
        DeviceVectorView<double> pt(ptData, multiGaussPoints.dim());
        double wt = displacement.gsPoint(point_idx, patch_idx, 
                                         multiGaussPoints[patch_idx], pt);
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
        int N_D = dispBasis.numActiveControlPoints();
        int dim = multiPatch.targetDim();
        for (int i = 0; i < N_D; i++)
        {
            for (int di = 0; di < dim; di++)
            {
                int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx)
                    .activeIndex(pt, i), patch_idx, di);
                for (int j = 0; j < N_D; j++)
                {
                    for (int dj = 0; dj < dim; dj++)
                    {
                        int globalIndex_j = system.mapColIndex(displacement.basis(patch_idx)
                            .activeIndex(pt, j), patch_idx, dj);
                        if (system.isEntry(globalIndex_i, globalIndex_j, di, dj))
                            system.pushToEntryIndex(globalIndex_i, globalIndex_j, di, dj, 
                                                    counter, rows, cols);
                    }
                }
            }
        }
    }
}
#endif

__global__
void assembleDomainKernel_perTileBlock_loopOverGps(int numDerivatives,
    int totalEles, int numBlocksPerEle, int numActivePerBlock,
    DeviceVectorView<double> parameters,
    MultiPatchDeviceView displacement,
    MultiPatchDeviceView multiPatch,
    MultiGaussPointsDeviceView multiGaussPoints,
    DeviceVectorView<double> bodyForce,
    SparseSystemDeviceView system,
    DeviceNestedArrayView<double> eliminatedDofs)
{
    extern __shared__ double shmem[];
    int totalNumBlocks = totalEles * numBlocksPerEle * numBlocksPerEle;
    for (int bidx = blockIdx.x; bidx < totalNumBlocks; bidx += gridDim.x)
    {
        __shared__ int patch_idx, CPdim, dim, blockCoord[2], numThreadsPerBlock;
        __shared__ int N_D_geo, N_D, dimTensor, ele_idx;
        __shared__ double wt, ptData[3], measure, weightForce, weightBody;
        __shared__ double lambda, mu, J;
        __shared__ TensorBsplineBasisDeviceView dispBasis;
        __shared__ PatchDeviceView geoPatch, dispPatch;

        DeviceVectorView<double> pt(ptData, multiGaussPoints.dim());
        int threadId = threadIdx.y * blockDim.x + threadIdx.x;
        if (threadId == 0)
        {
            numThreadsPerBlock = blockDim.x * blockDim.y;
            int idx = bidx;
            for (int d = 0; d < 2; d++)
            {
                blockCoord[d] = idx % numBlocksPerEle;
                idx /= numBlocksPerEle;
            }
            CPdim = multiPatch.targetDim();
            dim = multiPatch.domainDim();
            dimTensor = (dim * (dim + 1)) / 2;
            ele_idx = displacement.threadPatch_element(idx, patch_idx);
            
            dispBasis = displacement.basis(patch_idx);
            geoPatch = multiPatch.patch(patch_idx);
            dispPatch = displacement.patch(patch_idx);
            N_D = dispBasis.numActiveControlPoints();
            N_D_geo = geoPatch.basis().numActiveControlPoints();
            double YM = parameters[1];
            double PR = parameters[0];
            lambda = YM * PR / ( ( 1. + PR ) * ( 1. - 2. * PR ) );
            mu = YM / ( 2. * ( 1. + PR ) );
            //printf("blockIdx.x=%d, patch_idx=%d, blockCoord=(%d, %d), wt=%f, pt=(%f, %f)\n", 
            //       blockIdx.x, patch_idx, blockCoord[0], blockCoord[1], wt, pt[0], pt[1]);
        }
        __syncthreads();

        __shared__ double geoJacobianData[3*3], dispJacobianData[3*3], geoJacobianInvData[3*3]; //max 3D
        __shared__ double FData[3*3], SData[3*3], CData[6*6], RCGinvData[3*3], CtempData[6*6];
        //__shared__ double localMatData[3*3], localRHSData[3];
        DeviceMatrixView<double> geoJacobian(geoJacobianData, dim, dim);
        DeviceMatrixView<double> dispJacobian(dispJacobianData, dim, dim);    
        DeviceMatrixView<double> geoJacobianInv(geoJacobianInvData, dim, dim);
        DeviceMatrixView<double> F(FData, dim, dim);
        DeviceMatrixView<double> S(SData, dim, dim);
        DeviceMatrixView<double> C(CData, dimTensor, dimTensor);
        DeviceMatrixView<double> RCGinv(RCGinvData, dim, dim);
        DeviceMatrixView<double> Ctemp(CtempData, dimTensor, dimTensor);
        

        int geoP1 = geoPatch.basis().knotsOrder(0) + 1;
        int dataStart = 0;
        DeviceMatrixView<double> geoValuesAndDers(shmem + dataStart, geoP1, (numDerivatives + 1) * dim);
        dataStart += geoP1 * (numDerivatives + 1) * dim;
        int P1 = dispBasis.knotsOrder(0) + 1;
        DeviceMatrixView<double> dispValuesAndDers(shmem + dataStart, 
                                                   P1, (numDerivatives + 1) * dim);
        dataStart += P1 * (numDerivatives + 1) * dim;
        DeviceMatrixView<double> localMat(shmem + dataStart, numActivePerBlock * dim, numActivePerBlock * dim);
        dataStart += numActivePerBlock * dim * numActivePerBlock * dim;
        DeviceVectorView<double> localRHS(shmem + dataStart, numActivePerBlock * dim);
        dataStart += numActivePerBlock * dim;

        for (int i = threadIdx.x; i < numActivePerBlock; i += blockDim.x)
            for (int di = 0; di < dim; di++)
            {
                for (int j = threadIdx.y; j < numActivePerBlock; j += blockDim.y)
                    for (int dj = 0; dj < dim; dj++)
                        localMat(i * dim + di, j * dim + dj) = 0.0;
                if (threadIdx.y == 0 && blockCoord[1] == 0)
                    localRHS[i * dim + di] = 0.0;
            }

        for (int gpid = 0; gpid < N_D; gpid++)
        {
            if (threadId == 0)
                wt = displacement.gsPoint(gpid, ele_idx, patch_idx, 
                    multiGaussPoints[patch_idx], pt);
            __syncthreads();
#if 1
            if (numThreadsPerBlock == 1) {
                geoPatch.basis().evalAllDers_into(pt, numDerivatives, shmem + dataStart, geoValuesAndDers);
                dispBasis.evalAllDers_into(pt, numDerivatives, shmem + dataStart, dispValuesAndDers);
            }
            else {
                int workingSpaceStart = dataStart;
                if(threadId < dim)
                    geoPatch.basis().evalAllDers_into(threadId, min(dim, numThreadsPerBlock), pt, numDerivatives, shmem + workingSpaceStart, geoValuesAndDers);
                workingSpaceStart += (geoP1 * geoP1 + 4 * geoP1) * dim;
                if(threadId >= dim && threadId < 2 * dim)
                    dispBasis.evalAllDers_into(threadId - dim, min(dim, numThreadsPerBlock), pt, numDerivatives, shmem + workingSpaceStart, dispValuesAndDers);
                workingSpaceStart += (P1 * P1 + 4 * P1) * dim;
                //if (threadId == 0)
                //    printf("dataStart=%d\n", workingSpaceStart);
                __syncthreads();
            }
#else
            if(threadId < dim)
            {
                geoPatch.basis().evalAllDers_into(threadId, min(dim, numThreadsPerBlock), pt, numDerivatives, shmem + dataStart, geoValuesAndDers);
                dispBasis.evalAllDers_into(threadId, min(dim, numThreadsPerBlock), pt, numDerivatives, shmem + dataStart, dispValuesAndDers);
            }
            __syncthreads();
#endif
            for (int i = threadIdx.x; i < dim; i += blockDim.x)
                for (int j = threadIdx.y; j < dim; j += blockDim.y) {
                    geoJacobian(i, j) = 0.0;
                    dispJacobian(i, j) = 0.0;
                    geoJacobianInv(i, j) = 0.0;
                    F(i, j) = 0.0;
                    S(i, j) = 0.0;
                    RCGinv(i, j) = 0.0;
                }
            for (int i = threadIdx.x; i < dimTensor; i += blockDim.x)
                for (int j = threadIdx.y; j < dimTensor; j += blockDim.y) {
                    C(i, j) = 0.0;
                    Ctemp(i, j) = 0.0;
                }
            __syncthreads();

            if(threadId < N_D_geo)
                geoPatch.jacobian(threadId, min(N_D_geo, numThreadsPerBlock), 
                    pt, geoValuesAndDers, numDerivatives, geoJacobian);
            if(threadId < N_D)
                dispPatch.jacobian(threadId, min(N_D, numThreadsPerBlock), 
                    pt, dispValuesAndDers, numDerivatives, dispJacobian);
            __syncthreads();

            if(threadId == 0)
            {
                geoJacobian.inverse(geoJacobianInv);
                double physDispJacData[3*3] = {0.0};
                DeviceMatrixView<double> physDispJac(physDispJacData, dim, dim);
                dispJacobian.times(geoJacobianInv, physDispJac);
                physDispJac.plusIdentity(F);
#if 0   
                printf("bidx=%d\n", bidx);
                printf("dispJacobian = \n");
                dispJacobian.print();
                printf("F = \n");
                F.print();
#endif  
                J = F.determinant();
                double RCGData[3*3] = {0.0}; //max 3D
                DeviceMatrixView<double> RCG(RCGData, dim, dim);
                double F_transposeData[3*3] = {0.0}; //max 3D
                DeviceMatrixView<double> F_transpose(F_transposeData, dim, dim);
                F.transpose(F_transpose);
                F_transpose.times(F, RCG);
                RCG.inverse(RCGinv);
                RCGinv.times((lambda*(J*J-1)/2-mu), S);
                S.tracePlus(mu);
            }
            if(threadId == (numThreadsPerBlock - 1))
            {
                measure = geoJacobian.determinant();
                weightForce = wt * measure;
                weightBody = wt * measure;
            }
            __syncthreads();

            if (threadIdx.x < dimTensor && threadIdx.y < dimTensor)
                matrixViewTraceTensor_parallel(threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, C, RCGinv, RCGinv);
            __syncthreads();
        
            if(threadId == 0)
                C.times(lambda*J*J);
            __syncthreads();

            if(threadIdx.x < dimTensor && threadIdx.y < dimTensor)
                symmetricIdentityViewTensor_parallel(threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, Ctemp, RCGinv);
            __syncthreads();

            if(threadId == 0) {
                Ctemp.times(mu-lambda*(J*J-1)/2);
                C.plus(Ctemp);
            }
            __syncthreads();

            for (int i = blockCoord[0] * numActivePerBlock + threadIdx.x; 
                 i < (blockCoord[0] + 1) * numActivePerBlock && i < N_D; 
                 i += blockDim.x)
            {
                double dN_iData[3] = {0.0}; //max 3D
                DeviceVectorView<double> dN_i(dN_iData, dim);
                tensorBasisDerivative(i, P1, dim, numDerivatives, dispValuesAndDers, dN_i);
                double physGrad_iData[3] = {0.0}; //max 3D
                DeviceVectorView<double> physGrad_i(physGrad_iData, dim);
                geoJacobianInv.transposeTime(dN_i, physGrad_i);
                double geometricTangentTempData[3] = {0.0}; //max 3D
                DeviceVectorView<double> geometricTangentTemp(geometricTangentTempData, dim);
                S.times(physGrad_i, geometricTangentTemp);
                for (int di = 0; di < dim; di++)
                {
                    int localRow = (i - blockCoord[0] * numActivePerBlock) * dim + di;
                    double B_i_diTransData[6] = {0.0}; //max 3D
                    DeviceMatrixView<double> B_i_diTrans(B_i_diTransData, 1, dimTensor);
                    {
                        double B_i_diData[6] = {0.0}; //max 3D
                        DeviceVectorView<double> B_i_di(B_i_diData, dimTensor);
                        setBSingleDim<double>(di, B_i_di, F, physGrad_i);
                        B_i_di.transpose(B_i_diTrans);
                    }
                    double materialTangentTempData[6] = {0.0}; //max 3D
                    DeviceMatrixView<double> materialTangentTemp
                        (materialTangentTempData, 1,dimTensor);
                    B_i_diTrans.times(C, materialTangentTemp);
                    //int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx)
                    //    .activeIndex(pt, i), patch_idx, di);
                    for (int j = blockCoord[1] * numActivePerBlock + threadIdx.y; 
                         j < (blockCoord[1] + 1) * numActivePerBlock && j < N_D; 
                         j += blockDim.y)
                    {
                        double dN_jData[3] = {0.0}; //max 3D
                        DeviceVectorView<double> dN_j(dN_jData, dim);
                        tensorBasisDerivative(j, P1, dim, numDerivatives, dispValuesAndDers, dN_j);
                        double physGrad_jData[3] = {0.0}; //max 3D
                        DeviceVectorView<double> physGrad_j(physGrad_jData, dim);
                        geoJacobianInv.transposeTime(dN_j, physGrad_j);
                        double geometricTangent = geometricTangentTemp.dot(physGrad_j);
                        for (int dj = 0; dj < dim; dj++)
                        {
                            int localCol = (j - blockCoord[1] * numActivePerBlock) * dim + dj;
                            double B_j_djData[6] = {0.0}; //max 3D
                            DeviceVectorView<double> B_j_dj(B_j_djData, dimTensor);
                            setBSingleDim<double>(dj, B_j_dj, F, physGrad_j);
                            double materialTangent = 0;
                            DeviceMatrixView<double> materialTangentMat(&materialTangent, 1, 1);
                            materialTangentTemp.times(B_j_dj, materialTangentMat);
                            if (di == dj)
                                materialTangent += geometricTangent;
                            double stiffnessEntry = weightForce * materialTangent;
                            //int globalIndex_j = system.mapColIndex(displacement.basis(patch_idx)
                            //    .activeIndex(pt, j), patch_idx, dj);
                            //printf("i: %d, di: %d, j: %d, dj: %d, globalIndex_i: %d, globalIndex_j: %d, stiffnessEntry: %f\n", i, di, j, dj, globalIndex_i, globalIndex_j, stiffnessEntry);
                            //system.pushToMatrix(stiffnessEntry, globalIndex_i, globalIndex_j,
                            //                    eliminatedDofs, di, dj);
                            localMat(localRow, localCol) += stiffnessEntry;
                        }
                    }
                    if (threadIdx.y == 0 && blockCoord[1] == 0)
                    {
                        double SvecData[6] = {0.0}; //max 3D
                        DeviceVectorView<double> Svec(SvecData, dimTensor);
                        voigtStressView(Svec, S);
                        double residualEntry = 0.0;
                        DeviceMatrixView<double> residualEntryMat(&residualEntry, 1, 1);
                        B_i_diTrans.times(Svec, residualEntryMat);
                        residualEntry = -residualEntry * weightBody + weightForce * bodyForce[di] * 
                            tensorBasisValue(i, P1, dim, numDerivatives, dispValuesAndDers);
                        //printf("i: %d, di: %d, globalIndex_i: %d, residualEntry: %f\n", i, di, globalIndex_i, residualEntry);
                        //system.pushToRhs(residualEntry, globalIndex_i, di);
                        localRHS[localRow] += residualEntry;
                    }
                }
            }
        }
        __syncthreads();
#if 0
        if(threadId == 0)
        {
            printf("Local matrix:\n");
            localMat.print();
            printf("Local RHS:\n");
            localRHS.print();
        }
#endif

        for (int i = blockCoord[0] * numActivePerBlock + threadIdx.x; 
             i < (blockCoord[0] + 1) * numActivePerBlock && i < N_D; 
             i += blockDim.x)
        {
            for (int di = 0; di < dim; di++)
            {
                int localRow = (i - blockCoord[0] * numActivePerBlock) * dim + di;
                int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx)
                    .activeIndex(pt, i), patch_idx, di);
                for (int j = blockCoord[1] * numActivePerBlock + threadIdx.y; 
                     j < (blockCoord[1] + 1) * numActivePerBlock && j < N_D; 
                     j += blockDim.y)
                {
                    for (int dj = 0; dj < dim; dj++)
                    {
                        int localCol = (j - blockCoord[1] * numActivePerBlock) * dim + dj;
                        int globalIndex_j = system.mapColIndex(displacement.basis(patch_idx)
                            .activeIndex(pt, j), patch_idx, dj);
                        //printf("patch_idx: %d, i: %d, di: %d, j: %d, dj: %d, globalIndex_i: %d, globalIndex_j: %d, stiffnessEntry: %f\n", patch_idx, i, di, j, dj, globalIndex_i, globalIndex_j, localMat(localRow, localCol));
                        system.pushToMatrix(localMat(localRow, localCol), globalIndex_i, globalIndex_j,
                                            eliminatedDofs, di, dj);

                    }
                }
                if (threadIdx.y == 0 && blockCoord[1] == 0)
                    system.pushToRhs(localRHS(localRow), globalIndex_i, di);
            }
        }
    }
}

__global__
void assembleDomainKernel_perTileBlock(int numDerivatives,
    int totalGPs, int numBlocksPerGP, int numActivePerBlock,
    DeviceVectorView<double> parameters,
    MultiPatchDeviceView displacement,
    MultiPatchDeviceView multiPatch,
    MultiGaussPointsDeviceView multiGaussPoints,
    DeviceVectorView<double> bodyForce,
    SparseSystemDeviceView system,
    DeviceNestedArrayView<double> eliminatedDofs)
{
    extern __shared__ double shmem[];
    int totalNumBlocks = totalGPs * numBlocksPerGP * numBlocksPerGP;
    for (int bidx = blockIdx.x; bidx < totalNumBlocks; bidx += gridDim.x)
    {
        __shared__ int patch_idx, CPdim, dim, blockCoord[2], numThreadsPerBlock;
        __shared__ int N_D_geo, N_D, dimTensor;
        __shared__ double wt, ptData[3], measure, weightForce, weightBody;
        __shared__ double lambda, mu, J;
        __shared__ TensorBsplineBasisDeviceView dispBasis;
        __shared__ PatchDeviceView geoPatch, dispPatch;
        DeviceVectorView<double> pt(ptData, multiGaussPoints.dim());
        int threadId = threadIdx.y * blockDim.x + threadIdx.x;
        if (threadId == 0)
        {
            numThreadsPerBlock = blockDim.x * blockDim.y;
            int idx = bidx;
            for (int d = 0; d < 2; d++)
            {
                blockCoord[d] = idx % numBlocksPerGP;
                idx /= numBlocksPerGP;
            }
            CPdim = multiPatch.targetDim();
            dim = multiPatch.domainDim();
            dimTensor = (dim * (dim + 1)) / 2;
            int point_idx = displacement.threadPatch(idx, patch_idx);
            wt = displacement.gsPoint(point_idx, patch_idx, 
                                         multiGaussPoints[patch_idx], pt);
            dispBasis = displacement.basis(patch_idx);
            geoPatch = multiPatch.patch(patch_idx);
            dispPatch = displacement.patch(patch_idx);
            N_D = dispBasis.numActiveControlPoints();
            N_D_geo = geoPatch.basis().numActiveControlPoints();
            double YM = parameters[1];
            double PR = parameters[0];
            lambda = YM * PR / ( ( 1. + PR ) * ( 1. - 2. * PR ) );
            mu = YM / ( 2. * ( 1. + PR ) );
            //printf("blockIdx.x=%d, patch_idx=%d, blockCoord=(%d, %d), wt=%f, pt=(%f, %f)\n", 
            //       blockIdx.x, patch_idx, blockCoord[0], blockCoord[1], wt, pt[0], pt[1]);
        }
        __syncthreads();
        int geoP1 = geoPatch.basis().knotsOrder(0) + 1;
        int dataStart = 0;
        DeviceMatrixView<double> geoValuesAndDers(shmem + dataStart, geoP1, (numDerivatives + 1) * dim);
        dataStart += geoP1 * (numDerivatives + 1) * dim;
        int P1 = dispBasis.knotsOrder(0) + 1;
        DeviceMatrixView<double> dispValuesAndDers(shmem + dataStart, 
                                                   P1, (numDerivatives + 1) * dim);
        dataStart += P1 * (numDerivatives + 1) * dim;
#if 1
        if (numThreadsPerBlock == 1) {
            geoPatch.basis().evalAllDers_into(pt, numDerivatives, shmem + dataStart, geoValuesAndDers);
            dispBasis.evalAllDers_into(pt, numDerivatives, shmem + dataStart, dispValuesAndDers);
#if 0
            printf("blockId = %d\n", bidx);
            geoValuesAndDers.print();
            printf("\n");
            dispValuesAndDers.print();
            printf("\n");
#endif
        }
        else {
            if(threadId < dim)
                geoPatch.basis().evalAllDers_into(threadId, min(dim, numThreadsPerBlock), pt, numDerivatives, shmem + dataStart, geoValuesAndDers);
            dataStart += (geoP1 * geoP1 + 4 * geoP1) * dim;
            if(threadId >= dim && threadId < 2 * dim)
                dispBasis.evalAllDers_into(threadId - dim, min(dim, numThreadsPerBlock), pt, numDerivatives, shmem + dataStart, dispValuesAndDers);
            __syncthreads();
#if 0
            if (blockIdx.x == 8 && threadIdx.x == 0 && threadIdx.y == 0)
            {
                geoValuesAndDers.print();
                printf("\n");
                dispValuesAndDers.print();
                printf("\n");
            }
#endif
        }
#else
        if(threadId < dim)
        {
            geoPatch.basis().evalAllDers_into(threadId, min(dim, numThreadsPerBlock), pt, numDerivatives, shmem + dataStart, geoValuesAndDers);
            dispBasis.evalAllDers_into(threadId, min(dim, numThreadsPerBlock), pt, numDerivatives, shmem + dataStart, dispValuesAndDers);
        }
#endif

        __shared__ double geoJacobianData[3*3], dispJacobianData[3*3], geoJacobianInvData[3*3]; //max 3D
        __shared__ double FData[3*3], SData[3*3], CData[6*6], RCGinvData[3*3], CtempData[6*6];
        DeviceMatrixView<double> geoJacobian(geoJacobianData, dim, dim);
        DeviceMatrixView<double> dispJacobian(dispJacobianData, dim, dim);    
        DeviceMatrixView<double> geoJacobianInv(geoJacobianInvData, dim, dim);
        DeviceMatrixView<double> F(FData, dim, dim);
        DeviceMatrixView<double> S(SData, dim, dim);
        DeviceMatrixView<double> C(CData, dimTensor, dimTensor);
        DeviceMatrixView<double> RCGinv(RCGinvData, dim, dim);
        DeviceMatrixView<double> Ctemp(CtempData, dimTensor, dimTensor);

        for (int i = threadIdx.x; i < dim; i += blockDim.x)
            for (int j = threadIdx.y; j < dim; j += blockDim.y) {
                geoJacobian(i, j) = 0.0;
                dispJacobian(i, j) = 0.0;
                geoJacobianInv(i, j) = 0.0;
                F(i, j) = 0.0;
                S(i, j) = 0.0;
                RCGinv(i, j) = 0.0;
            }
        for (int i = threadIdx.x; i < dimTensor; i += blockDim.x)
            for (int j = threadIdx.y; j < dimTensor; j += blockDim.y) {
                C(i, j) = 0.0;
                Ctemp(i, j) = 0.0;
            }
        __syncthreads();
        
        if(threadId < N_D_geo)
            geoPatch.jacobian(threadId, min(N_D_geo, numThreadsPerBlock), 
                pt, geoValuesAndDers, numDerivatives, geoJacobian);
        if(threadId < N_D)
            dispPatch.jacobian(threadId, min(N_D, numThreadsPerBlock), 
                pt, dispValuesAndDers, numDerivatives, dispJacobian);
        __syncthreads();
#if 0
        if (numThreadsPerBlock == 1) {
            printf("blockId = %d\n", bidx);
            printf("geoJacobian = \n");
            geoJacobian.print();
            printf("dispJacobian = \n");
            dispJacobian.print();
        }
        else 
            if(threadId == 0 && blockIdx.x == 0) {
                printf("blockId = %d\n", blockIdx.x);
                printf("geoJacobian = \n");
                geoJacobian.print();
                printf("dispJacobian = \n");
                dispJacobian.print();
            }
#endif
        if(threadId == 0)
        {
            geoJacobian.inverse(geoJacobianInv);
            double physDispJacData[3*3] = {0.0};
            DeviceMatrixView<double> physDispJac(physDispJacData, dim, dim);
            dispJacobian.times(geoJacobianInv, physDispJac);
            physDispJac.plusIdentity(F);
#if 0
            printf("bidx=%d\n", bidx);
            printf("dispJacobian = \n");
            dispJacobian.print();
            printf("F = \n");
            F.print();
#endif
            J = F.determinant();
            double RCGData[3*3] = {0.0}; //max 3D
            DeviceMatrixView<double> RCG(RCGData, dim, dim);
            double F_transposeData[3*3] = {0.0}; //max 3D
            DeviceMatrixView<double> F_transpose(F_transposeData, dim, dim);
            F.transpose(F_transpose);
            F_transpose.times(F, RCG);
            RCG.inverse(RCGinv);
            RCGinv.times((lambda*(J*J-1)/2-mu), S);
            S.tracePlus(mu);
        }
        if(threadId == (numThreadsPerBlock - 1))
        {
            measure = geoJacobian.determinant();
            weightForce = wt * measure;
            weightBody = wt * measure;
        }
        __syncthreads();

        if (threadIdx.x < dimTensor && threadIdx.y < dimTensor)
            matrixViewTraceTensor_parallel(threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, C, RCGinv, RCGinv);
        __syncthreads();
       
        if(threadId == 0)
            C.times(lambda*J*J);
        __syncthreads();

        if(threadIdx.x < dimTensor && threadIdx.y < dimTensor)
            symmetricIdentityViewTensor_parallel(threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, Ctemp, RCGinv);
        __syncthreads();

        if(threadId == 0) {
            Ctemp.times(mu-lambda*(J*J-1)/2);
            C.plus(Ctemp);
        }
        __syncthreads();

        for (int i = blockCoord[0] * numActivePerBlock + threadIdx.x; 
             i < (blockCoord[0] + 1) * numActivePerBlock && i < N_D; 
             i += blockDim.x)
        {
            double dN_iData[3] = {0.0}; //max 3D
            DeviceVectorView<double> dN_i(dN_iData, dim);
            tensorBasisDerivative(i, P1, dim, numDerivatives, dispValuesAndDers, dN_i);
            double physGrad_iData[3] = {0.0}; //max 3D
            DeviceVectorView<double> physGrad_i(physGrad_iData, dim);
            geoJacobianInv.transposeTime(dN_i, physGrad_i);
            double geometricTangentTempData[3] = {0.0}; //max 3D
            DeviceVectorView<double> geometricTangentTemp(geometricTangentTempData, dim);
            S.times(physGrad_i, geometricTangentTemp);
            for (int di = 0; di < dim; di++)
            {
                double B_i_diTransData[6] = {0.0}; //max 3D
                DeviceMatrixView<double> B_i_diTrans(B_i_diTransData, 1, dimTensor);
                {
                    double B_i_diData[6] = {0.0}; //max 3D
                    DeviceVectorView<double> B_i_di(B_i_diData, dimTensor);
                    setBSingleDim<double>(di, B_i_di, F, physGrad_i);
                    B_i_di.transpose(B_i_diTrans);
                }
                double materialTangentTempData[6] = {0.0}; //max 3D
                DeviceMatrixView<double> materialTangentTemp
                    (materialTangentTempData, 1,dimTensor);
                B_i_diTrans.times(C, materialTangentTemp);
                int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx)
                    .activeIndex(pt, i), patch_idx, di);
                for (int j = blockCoord[1] * numActivePerBlock + threadIdx.y; 
                     j < (blockCoord[1] + 1) * numActivePerBlock && j < N_D; 
                     j += blockDim.y)
                {
                    double dN_jData[3] = {0.0}; //max 3D
                    DeviceVectorView<double> dN_j(dN_jData, dim);
                    tensorBasisDerivative(j, P1, dim, numDerivatives, dispValuesAndDers, dN_j);
                    double physGrad_jData[3] = {0.0}; //max 3D
                    DeviceVectorView<double> physGrad_j(physGrad_jData, dim);
                    geoJacobianInv.transposeTime(dN_j, physGrad_j);
                    double geometricTangent = geometricTangentTemp.dot(physGrad_j);
                    for (int dj = 0; dj < dim; dj++)
                    {
                        double B_j_djData[6] = {0.0}; //max 3D
                        DeviceVectorView<double> B_j_dj(B_j_djData, dimTensor);
                        setBSingleDim<double>(dj, B_j_dj, F, physGrad_j);
                        double materialTangent = 0;
                        DeviceMatrixView<double> materialTangentMat(&materialTangent, 1, 1);
                        materialTangentTemp.times(B_j_dj, materialTangentMat);
                        if (di == dj)
                            materialTangent += geometricTangent;
                        double stiffnessEntry = weightForce * materialTangent;
                        int globalIndex_j = system.mapColIndex(displacement.basis(patch_idx)
                            .activeIndex(pt, j), patch_idx, dj);
                        //printf("patch_idx: %d, i: %d, di: %d, j: %d, dj: %d, globalIndex_i: %d, globalIndex_j: %d, stiffnessEntry: %f\n", patch_idx, i, di, j, dj, globalIndex_i, globalIndex_j, stiffnessEntry);
                        system.pushToMatrix(stiffnessEntry, globalIndex_i, globalIndex_j,
                                            eliminatedDofs, di, dj);
                    }
                }
                if (threadIdx.y == 0 && blockCoord[1] == 0)
                {
                    double SvecData[6] = {0.0}; //max 3D
                    DeviceVectorView<double> Svec(SvecData, dimTensor);
                    voigtStressView(Svec, S);
                    double residualEntry = 0.0;
                    DeviceMatrixView<double> residualEntryMat(&residualEntry, 1, 1);
                    B_i_diTrans.times(Svec, residualEntryMat);
                    residualEntry = -residualEntry * weightBody + weightForce * bodyForce[di] * 
                        tensorBasisValue(i, P1, dim, numDerivatives, dispValuesAndDers);
                    //printf("i: %d, di: %d, globalIndex_i: %d, residualEntry: %f\n", i, di, globalIndex_i, residualEntry);
                    system.pushToRhs(residualEntry, globalIndex_i, di);
                }
            }
        }
    }
}

__global__
void assembleDomainKernel(
    int totalGPs,
    //int* counter,
    DeviceVectorView<double> parameters,
    MultiPatchDeviceView displacement,
    MultiPatchDeviceView multiPatch,
    MultiGaussPointsDeviceView multiGaussPoints,
    DeviceVectorView<double> bodyForce,
    SparseSystemDeviceView system,
    DeviceNestedArrayView<double> eliminatedDofs)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalGPs)
    {
        //extern __shared__ double shmem[];
        
        double YM = parameters[1];
        double PR = parameters[0];
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
        double geoJacobianData[3*3] = {0.0}; //max 3D
        DeviceMatrixView<double> geoJacobian(geoJacobianData, dim, dim);
        PatchDeviceView geoPatch = multiPatch.patch(patch_idx);
        geoPatch.jacobian(pt, numDerivatives, geoJacobian);
        double measure = geoJacobian.determinant();
        double weightForce = wt * measure;
        double weightBody = wt * measure;
        
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
        PatchDeviceView dispPatch = displacement.patch(patch_idx);
        int P1 = dispBasis.knotsOrder(0) + 1;
        double dispValuesAndDersData[5*3*3]; //max 4th order basis, max 2 derivatives, max 3D
        DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDersData, P1, 
                                                   (numDerivatives+1)*dim);
        dispBasis.evalAllDers_into(pt, numDerivatives, dispValuesAndDers);
        
        double geoJacobianInvData[3*3] = {0.0}; //max 3D
        DeviceMatrixView<double> geoJacobianInv(geoJacobianInvData, dim, dim);
        geoJacobian.inverse(geoJacobianInv);
        
        
        double FData[3*3] = {0.0}; //max 3D
        DeviceMatrixView<double> F(FData, dim, dim);
        {
            double dispJacobianData[3*3] = {0.0}; //max 3D
            double physDispJacData[3*3] = {0.0}; //max 3D
            DeviceMatrixView<double> dispJacobian(dispJacobianData, dim, dim);
            DeviceMatrixView<double> physDispJac(physDispJacData, dim, dim);
            dispPatch.jacobian(pt, dispValuesAndDers, numDerivatives, dispJacobian);
            dispJacobian.times(geoJacobianInv, physDispJac);
            physDispJac.plusIdentity(F);
        }
    
        double SData[3*3] = {0.0}; //max 3D
        DeviceMatrixView<double> S(SData, dim, dim);
        double CData[6*6] = {0.0}; //max 3D
        int dimTensor = (dim * (dim + 1)) / 2;
        DeviceMatrixView<double> C(CData, dimTensor, dimTensor);
        {
            double J = F.determinant();
            double RCGData[3*3] = {0.0}; //max 3D
            DeviceMatrixView<double> RCG(RCGData, dim, dim);
            double F_transposeData[3*3] = {0.0}; //max 3D
            DeviceMatrixView<double> F_transpose(F_transposeData, dim, dim);
            F.transpose(F_transpose);
            F_transpose.times(F, RCG);
            double RCGinvData[3*3] = {0.0}; //max 3D
            DeviceMatrixView<double> RCGinv(RCGinvData, dim, dim);
            RCG.inverse(RCGinv);
            RCGinv.times((lambda*(J*J-1)/2-mu), S);
            S.tracePlus(mu);
            matrixViewTraceTensor(C, RCGinv, RCGinv);
            C.times(lambda*J*J);
            double CtempData[6*6] = {0.0}; //max 3D
            DeviceMatrixView<double> Ctemp(CtempData, dimTensor, dimTensor);
            symmetricIdentityViewTensor(Ctemp, RCGinv);
            Ctemp.times(mu-lambda*(J*J-1)/2);
            C.plus(Ctemp);
        }
    
        int N_D = dispBasis.numActiveControlPoints();
        for (int ti = 0; ti < N_D; ti += 3)
        {
            for (int tj = 0; tj < N_D; tj += 3)
            {   
                __shared__ double localEntryValues[81]; //max 3x3 tile, max 3D
                DeviceMatrixView<double> localEntries(localEntryValues, 3*dim, 3*dim);
                for (int t = tid; t < 3*dim*3*dim; t += blockDim.x)
                    localEntryValues[t] = 0.0;
                __syncthreads();
                for (int ii = 0; ii < 3 && (ti + ii) < N_D; ii++)
                {
                    int i = ti + ii;
                    double dN_iData[3] = {0.0}; //max 3D
                    DeviceVectorView<double> dN_i(dN_iData, dim);
                    tensorBasisDerivative(i, P1, dim, numDerivatives, dispValuesAndDers, dN_i);
                    //double geoJacobianInvTransData[3*3] = {0.0}; //max 3D
                    //DeviceMatrixView<double> geoJacobianInvTrans
                    //    (geoJacobianInvTransData, dim, dim);
                    //geoJacobianInv.transpose(geoJacobianInvTrans);
                    double physGrad_iData[3] = {0.0}; //max 3D
                    DeviceVectorView<double> physGrad_i(physGrad_iData, dim);
                    geoJacobianInv.transposeTime(dN_i, physGrad_i);
                    double geometricTangentTempData[3] = {0.0}; //max 3D
                    DeviceVectorView<double> geometricTangentTemp
                        (geometricTangentTempData, dim);
                    S.times(physGrad_i, geometricTangentTemp);
                    for (int di = 0; di < dim; di++)
                    {
                        int local_i = ii * dim + di;
                        double B_i_diTransData[6] = {0.0}; //max 3D
                        DeviceMatrixView<double> B_i_diTrans(B_i_diTransData, 1, dimTensor);
                        {
                            double B_i_diData[6] = {0.0}; //max 3D
                            DeviceVectorView<double> B_i_di(B_i_diData, dimTensor);
                            setBSingleDim<double>(di, B_i_di, F, physGrad_i);
                            B_i_di.transpose(B_i_diTrans);
                        }
                        double materialTangentTempData[6] = {0.0}; //max 3D
                        DeviceMatrixView<double> materialTangentTemp
                            (materialTangentTempData, 1,dimTensor);
                        B_i_diTrans.times(C, materialTangentTemp);
                        //int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx)
                        //    .activeIndex(pt, i), patch_idx, di);
                        for (int jj = 0; jj < 3 && (tj + jj) < N_D; jj++)
                        {
                            int j = tj + jj;
                            double dN_jData[3] = {0.0}; //max 3D
                            DeviceVectorView<double> dN_j(dN_jData, dim);
                            tensorBasisDerivative(j, P1, dim, numDerivatives, dispValuesAndDers, dN_j);
                            double physGrad_jData[3] = {0.0}; //max 3D
                            DeviceVectorView<double> physGrad_j(physGrad_jData, dim);
                            geoJacobianInv.transposeTime(dN_j, physGrad_j);
                            double geometricTangent = geometricTangentTemp.dot(physGrad_j);
                            for (int dj = 0; dj < dim; dj++)
                            {
                                int local_j = jj * dim + dj;
                                double B_j_djData[6] = {0.0}; //max 3D
                                DeviceVectorView<double> B_j_dj(B_j_djData, dimTensor);
                                setBSingleDim<double>(dj, B_j_dj, F, physGrad_j);
                                double materialTangent = 0;
                                DeviceMatrixView<double> materialTangentMat(&materialTangent, 1, 1);
                                materialTangentTemp.times(B_j_dj, materialTangentMat);
                                if (di == dj)
                                    materialTangent += geometricTangent;
                                double stiffnessEntry = weightForce * materialTangent;
                                //int globalIndex_j = system.mapColIndex(displacement.basis(patch_idx)
                                //    .activeIndex(pt, j), patch_idx, dj);
                                atomicAdd(&localEntries(local_i, local_j), stiffnessEntry);
                                //printf("local_i=%d, local_j=%d, stiffnessEntry=%f\n", local_i, local_j, stiffnessEntry);
                            }
                        }
                        
                    }
                }
                __syncthreads();
                for (int t = tid; t < 3 * dim * 3 * dim; t += blockDim.x)
                {
                    int local_i = t / (3 * dim);
                    int local_j = t % (3 * dim);
                    int ii = local_i / dim;
                    int jj = local_j / dim;
                    int i = ti + ii;
                    int j = tj + jj;
                    if (i >= N_D || j >= N_D)
                        continue;
                    int di = local_i % dim;
                    int dj = local_j % dim;
                    int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx)
                                        .activeIndex(pt, i), patch_idx, di);
                    int globalIndex_j = system.mapColIndex(displacement.basis(patch_idx)
                                        .activeIndex(pt, j), patch_idx, dj);
                    //int out = atomicAdd(counter, 1);
                    system.pushToMatrix(localEntries(local_i, local_j), globalIndex_i, globalIndex_j,
                                            eliminatedDofs, di, dj/*, counter*/);
                    //system.pushToMatrix(localEntries(local_i, local_j), globalIndex_i, globalIndex_j,
                    //                    eliminatedDofs, di, dj);
                    //printf("idx=%d, tid=%d, ti=%d, tj=%d, local_i=%d, local_j=%d, i=%d, di=%d, j=%d, dj=%d, globalIndex_i=%d, globalIndex_j=%d\n",
                    //        idx, tid, ti, tj, local_i, local_j, i, di, j, dj, globalIndex_i, globalIndex_j);
                }
                __syncthreads();
            }
            __shared__ double localRHSValues[9]; //max 3 unknowns, max 3D
            DeviceMatrixView<double> localRHS(localRHSValues, 3*dim, 1);
            for (int t = tid; t < 3*dim; t += blockDim.x)
                localRHSValues[t] = 0.0;
            __syncthreads();
            for (int ii = 0; ii < 3 && (ti + ii) < N_D; ii++)
            {
                int i = ti + ii;
                double dN_iData[3] = {0.0}; //max 3D
                DeviceVectorView<double> dN_i(dN_iData, dim);
                tensorBasisDerivative(i, P1, dim, numDerivatives, dispValuesAndDers, dN_i);
                //double geoJacobianInvTransData[3*3] = {0.0};
                //DeviceMatrixView<double> geoJacobianInvTrans
                //    (geoJacobianInvTransData, dim, dim);
                //geoJacobian.inverse(geoJacobianInvTrans);
                //geoJacobianInvTrans.transpose();
                double physGrad_iData[3] = {0.0}; //max 3D
                DeviceVectorView<double> physGrad_i(physGrad_iData, dim);
                geoJacobianInv.transposeTime(dN_i, physGrad_i);
                for (int di = 0; di < dim; di++)
                {
                    int local_i = ii * dim + di;
                    double B_i_diTransData[6] = {0.0}; //max 3D
                    DeviceMatrixView<double> B_i_diTrans(B_i_diTransData, 1, dimTensor);
                    {
                        double B_i_diData[6] = {0.0}; //max 3D
                        DeviceVectorView<double> B_i_di(B_i_diData, dimTensor);
                        setBSingleDim<double>(di, B_i_di, F, physGrad_i);
                        B_i_di.transpose(B_i_diTrans);
                    }
                    double SvecData[6] = {0.0}; //max 3D
                    DeviceVectorView<double> Svec(SvecData, dimTensor);
                    voigtStressView(Svec, S);
                    double residualEntry = 0.0;
                    DeviceMatrixView<double> residualEntryMat(&residualEntry, 1, 1);
                    //printf("B_i_diTrans*Svec:\n");
                    B_i_diTrans.times(Svec, residualEntryMat);
                    residualEntry = -residualEntry * weightBody + weightForce * bodyForce[di] * 
                        tensorBasisValue(i, P1, dim, numDerivatives, dispValuesAndDers);
                    atomicAdd(&localRHS(local_i, 0), residualEntry);
                }
                
            }
            __syncthreads();
            for (int t = tid; t < 3 * dim; t += blockDim.x)
            {
                int ii = t / dim;
                int i = ti + ii;
                int di = t % dim;
                if (i >= N_D)
                    continue;
                int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx)
                                    .activeIndex(pt, i), patch_idx, di);
                //int out = atomicAdd(counter, 1);
                system.pushToRhs(localRHS(t, 0), globalIndex_i, di);
                //system.pushToRhs(localRHS(t, 0), globalIndex_i, di);
                //printf("idx=%d, ti=%d, ii=%d, i=%d, di=%d, globalIndex_i=%d, localRHS=%f\n", 
                //       idx, ti, ii, i, di, globalIndex_i, localRHS(t, 0));
            }
            __syncthreads();
        }   
    }
}

#if 0
__global__
void assembleDomainKernel(int totalGPs,
                          MultiPatchDeviceView displacement,
                          MultiPatchDeviceView multiPatch,
                          MultiGaussPointsDeviceView multiGaussPoints,
                          DeviceVectorView<double> bodyForce,
                          SparseSystemDeviceView system,
                          DeviceNestedArrayView<double> eliminatedDofs)
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
        //printf("Patch %d, GP %d\nWeight %f\nPoint:\n", patch_idx, point_idx, wt);
        //pt.print();
        double geoJacobianData[3*3] = {0.0}; //max 3D
        DeviceMatrixView<double> geoJacobian(geoJacobianData, dim, dim);
        PatchDeviceView geoPatch = multiPatch.patch(patch_idx);
        geoPatch.jacobian(pt, numDerivatives, geoJacobian);
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
            dispPatch.jacobian(pt, dispValuesAndDers, numDerivatives, dispJacobian);
            dispJacobian.times(geoJacobianInv, physDispJac);
            physDispJac.plusIdentity(F);
        }
        //printf("Deformation gradient F:\n");
        //F.print();
        
        double SData[3*3] = {0.0}; //max 3D
        DeviceMatrixView<double> S(SData, dim, dim);
        double CData[6*6] = {0.0}; //max 3D
        int dimTensor = (dim * (dim + 1)) / 2;
        DeviceMatrixView<double> C(CData, dimTensor, dimTensor);
        {
            double J = F.determinant();
            double RCGData[3*3] = {0.0}; //max 3D
            DeviceMatrixView<double> RCG(RCGData, dim, dim);
            double F_transposeData[3*3] = {0.0}; //max 3D
            DeviceMatrixView<double> F_transpose(F_transposeData, dim, dim);
            F.transpose(F_transpose);
            F_transpose.times(F, RCG);
            double RCGinvData[3*3] = {0.0}; //max 3D
            DeviceMatrixView<double> RCGinv(RCGinvData, dim, dim);
            RCG.inverse(RCGinv);
            RCGinv.times((lambda*(J*J-1)/2-mu), S);
            S.tracePlus(mu);
            matrixViewTraceTensor(C, RCGinv, RCGinv);
            C.times(lambda*J*J);
            double CtempData[6*6] = {0.0}; //max 3D
            DeviceMatrixView<double> Ctemp(CtempData, dimTensor, dimTensor);
            symmetricIdentityViewTensor(Ctemp, RCGinv);
            Ctemp.times(mu-lambda*(J*J-1)/2);
            C.plus(Ctemp);
        }
        //printf("Second Piola-Kirchhoff stress S:\n");
        //S.print();
        //printf("Material stiffness tensor C:\n");
        //C.print();
        int N_D = dispBasis.numActiveControlPoints();
        for (int i = 0; i < N_D; i++)
        {
            double dN_iData[3] = {0.0}; //max 3D
            DeviceVectorView<double> dN_i(dN_iData, dim);
            tensorBasisDerivative(i, P1, dim, numDerivatives, dispValuesAndDers, dN_i);
            double geoJacobianInvTransData[3*3] = {0.0}; //max 3D
            DeviceMatrixView<double> geoJacobianInvTrans
                (geoJacobianInvTransData, dim, dim);
            geoJacobian.inverse(geoJacobianInvTrans);
            geoJacobianInvTrans.transpose();
            double physGrad_iData[3] = {0.0}; //max 3D
            DeviceVectorView<double> physGrad_i(physGrad_iData, dim);
            geoJacobianInvTrans.times(dN_i, physGrad_i);
            double geometricTangentTempData[3] = {0.0}; //max 3D
            DeviceVectorView<double> geometricTangentTemp
                (geometricTangentTempData, dim);
            S.times(physGrad_i, geometricTangentTemp);
            //printf("Geometric tangent:\n");
            //geometricTangentTemp.print();
            //printf("physGrad of basis function %d:\n", i);
            //physGrad.print();
            
            for (int di = 0; di < dim; di++)
            {
                double B_i_diTransData[6] = {0.0}; //max 3D
                DeviceMatrixView<double> B_i_diTrans(B_i_diTransData, 1, dimTensor);
                {
                    double B_i_diData[6] = {0.0}; //max 3D
                    DeviceVectorView<double> B_i_di(B_i_diData, dimTensor);
                    setBSingleDim<double>(di, B_i_di, F, physGrad_i);
                    B_i_di.transpose(B_i_diTrans);
                }
                double materialTangentTempData[6] = {0.0}; //max 3D
                DeviceMatrixView<double> materialTangentTemp
                    (materialTangentTempData, 1,dimTensor);
                //printf("B_i_diTrans*C:\n");
                B_i_diTrans.times(C, materialTangentTemp);
                //printf("materialTangentTemp:\n");
                //B_i_diTrans.print();
                //materialTangentTemp.print();
                int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx)
                    .activeIndex(pt, i), patch_idx, di);
                //printf("Global index i: %d\n", globalIndex_i);
                for (int j = 0; j < N_D; j++)
                {
                    double dN_jData[3] = {0.0}; //max 3D
                    DeviceVectorView<double> dN_j(dN_jData, dim);
                    tensorBasisDerivative(j, P1, dim, numDerivatives, dispValuesAndDers, dN_j);
                    double physGrad_jData[3] = {0.0}; //max 3D
                    DeviceVectorView<double> physGrad_j(physGrad_jData, dim);
                    geoJacobianInvTrans.times(dN_j, physGrad_j);
                    double geometricTangent = geometricTangentTemp.dot(physGrad_j);
                    for (int dj = 0; dj < dim; dj++)
                    {
                        double B_j_djData[6] = {0.0}; //max 3D
                        DeviceVectorView<double> B_j_dj(B_j_djData, dimTensor);
                        setBSingleDim<double>(dj, B_j_dj, F, physGrad_j);
                        //printf("B_j_dj for j=%d, dj=%d:\n", j, dj);
                        //B_j_dj.print();
                        double materialTangent = 0;
                        DeviceMatrixView<double> materialTangentMat(&materialTangent, 1, 1);
                        materialTangentTemp.times(B_j_dj, materialTangentMat);
                        //printf("Material tangent for i=%d, di=%d, j=%d, dj=%d: %f\n", 
                        //    i, di, j, dj, materialTangent);
                        if (di == dj)
                            materialTangent += geometricTangent;
                        double stiffnessEntry = weightForce * materialTangent;
                        int globalIndex_j = system.mapColIndex(displacement.basis(patch_idx)
                            .activeIndex(pt, j), patch_idx, dj);
                        //printf("Global index j: %d\n", globalIndex_j);
                        system.pushToMatrix(stiffnessEntry, globalIndex_i, globalIndex_j,
                                            eliminatedDofs, di, dj);
                    }

                }
                double SvecData[6] = {0.0}; //max 3D
                DeviceVectorView<double> Svec(SvecData, dimTensor);
                voigtStressView(Svec, S);
                double residualEntry = 0.0;
                DeviceMatrixView<double> residualEntryMat(&residualEntry, 1, 1);
                //printf("B_i_diTrans*Svec:\n");
                B_i_diTrans.times(Svec, residualEntryMat);
                residualEntry = -residualEntry * weightBody + weightForce * bodyForce[di] * 
                    tensorBasisValue(i, P1, dim, numDerivatives, dispValuesAndDers);
                //printf("Residual entry for dof %d: %f\n", globalIndex_i, residualEntry);
                system.pushToRhs(residualEntry, globalIndex_i, di);
            }
            
        }    
    }
    //printf("Matrix after assembly:\n");
    //system.matrix().print();
    //printf("RHS after assembly:\n");
    //system.rhs().print();
}
#endif

__global__
void cooToDenseKernel(DeviceVectorView<int> rows,
                      DeviceVectorView<int> cols,
                      DeviceVectorView<double> values,
                      DeviceMatrixView<double> denseMatrix)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < rows.size(); idx += blockDim.x * gridDim.x)
        atomicAdd(&denseMatrix(rows[idx], cols[idx]), values[idx]);
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

__global__
void printMultiPatchKernel(MultiPatchDeviceView multiPatch)
{
    printf("MultiPatch:\n");
    multiPatch.print();
}

__global__
void printMultiBasisKernel(MultiBasisDeviceView multiBasis)
{
    printf("MultiBasis:\n");
    multiBasis.print();
}

GPUAssembler::GPUAssembler(const MultiPatch &multiPatch, 
                           const MultiBasis &multiBasis, 
                           const BoundaryConditions &bc, 
                           const Eigen::VectorXd &bodyForce)
:   m_multiPatch(multiPatch), m_multiBasis(multiBasis), 
    m_boundaryConditions(bc), m_bodyForce(bodyForce),
    m_multiGaussPoints(multiBasis), 
    m_multiPatchHost(multiPatch), m_multiBasisHost(multiBasis)
{
    int targetDim = multiPatch.getCPDim();
    m_problemDim = targetDim;
    std::vector<DofMapper> dofMappers_stdVec(targetDim);
    multiBasis.getMappers(true, m_boundaryConditions, 
                          dofMappers_stdVec, true);
#if 0
    m_sparseSystemHost = SparseSystem(dofMappers_stdVec, 
                              Eigen::VectorXi::Ones(targetDim));
#else
    SparseSystem sparseSystem(dofMappers_stdVec, 
                              Eigen::VectorXi::Ones(targetDim));
#endif
#ifdef STORE_MATRIX
    m_sparseSystem.setMatrixRows(sparseSystem.matrix().rows());
    m_sparseSystem.setMatrixCols(sparseSystem.matrix().cols());
#else
    m_sparseSystem.setMatrixRows(sparseSystem.matrixRows());
    m_sparseSystem.setMatrixCols(sparseSystem.matrixCols());
#endif
    std::vector<int> intDataOffsets;
    std::vector<int> intData;
#ifdef STORE_MATRIX
    std::vector<double> doubleData;
    sparseSystem.getDataVector(intDataOffsets, intData, doubleData);
#else
    sparseSystem.getDataVector(intDataOffsets, intData);
#endif
    m_sparseSystem.setIntDataOffsets(intDataOffsets);
    m_sparseSystem.setIntData(intData);
#ifdef STORE_MATRIX
    m_sparseSystem.setDoubleData(doubleData);
#else
    //m_sparseSystem.resizeDoubleData(sparseSystem.matrixRows() * 
    //                                sparseSystem.matrixCols() + 
    //                                sparseSystem.matrixRows());
    m_sparseSystem.resizeRHS(sparseSystem.matrixRows());
#endif
    m_sparseSystem.setPermVectors(sparseSystem.permOld2New(), 
                                  sparseSystem.permNew2Old());

    std::vector<Eigen::VectorXd> ddof(targetDim);
    std::vector<Eigen::VectorXd> ddof_zero(targetDim);
    for (int unk = 0; unk < targetDim; ++unk)
    {
        computeDirichletDofs(unk, dofMappers_stdVec, ddof, multiBasis);
        ddof_zero[unk] = Eigen::VectorXd::Zero(ddof[unk].size());
    }    
    m_ddof.setData(ddof);
    m_ddof_zero.setData(ddof_zero);

    m_multiBasisHost.giveBasis(m_displacementHost, targetDim);
    m_displacement = MultiPatchDeviceData(m_displacementHost);

    int* entryCountDevicePtr;
    cudaError_t err = cudaMalloc((void**)&entryCountDevicePtr, sizeof(int));
    assert(err == cudaSuccess && "cudaMalloc failed in GPUAssembler constructor during counting matrix entries");
    err = cudaMemset(entryCountDevicePtr, 0, sizeof(int));
    assert(err == cudaSuccess && "cudaMemset failed in GPUAssembler constructor during counting matrix entries");
#if 0
    int totalGPs = m_multiBasisHost.totalNumGPs();
    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                         countEntrysKernel, 0, totalGPs);
    int gridSize = (totalGPs + blockSize - 1) / blockSize;
    countEntrysKernel<<<gridSize, blockSize>>>(totalGPs,
                                               m_displacement.deviceView(),
                                               m_multiPatch.deviceView(),
                                               m_multiGaussPoints.view(),
                                               m_sparseSystem.deviceView(),
                                               m_ddof.view(),
                                               entryCountDevicePtr);
#else
    int numElements = m_multiBasisHost.totalNumElements();
#if 1
    int N_D = m_multiBasisHost.numActive();
    int numActivePerBlock = std::min(16, N_D);
    int numBlocksPerElement = (N_D + numActivePerBlock - 1) / numActivePerBlock;
    dim3 blockSize(numActivePerBlock, numActivePerBlock);
    int gridSize = numElements * numBlocksPerElement * numBlocksPerElement;
#else
    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                         countEntrysKernel, 0, numElements);
    int gridSize = (numElements + blockSize - 1) / blockSize;
#endif
    countEntrysKernel<<<gridSize, blockSize>>>(numElements, numBlocksPerElement, 
                                               numActivePerBlock,
                                               m_displacement.deviceView(),
                                               m_multiPatch.deviceView(),
                                               m_multiGaussPoints.view(),
                                               m_sparseSystem.deviceView(),
                                               m_ddof.view(),
                                               entryCountDevicePtr);
#endif
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler constructor during counting matrix entries");
    int entryCountHost;
    err = cudaMemcpy(&entryCountHost, entryCountDevicePtr, sizeof(int), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess && "cudaMemcpy failed in GPUAssembler constructor during counting matrix entries");
    
    //m_sparseSystem.setNumMatrixEntries(entryCountHost);
    //m_sparseSystem.resizeMatrixData(entryCountHost);

    DeviceArray<int> cooRows(entryCountHost);
    DeviceArray<int> cooCols(entryCountHost);
    DeviceArray<double> cooValues(entryCountHost);

    err = cudaMemset(entryCountDevicePtr, 0, sizeof(int));
    assert(err == cudaSuccess && "cudaMemset failed in GPUAssembler constructor during counting matrix entries");
    computeCOOKernel<<<gridSize, blockSize>>>(numElements, entryCountDevicePtr,
                                numBlocksPerElement, numActivePerBlock,
                                m_displacement.deviceView(),
                                m_multiPatch.deviceView(),
                                m_multiGaussPoints.view(),
                                m_sparseSystem.deviceView(),
                                m_ddof.view(),
                                cooRows.vectorView(),
                                cooCols.vectorView());

    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler constructor during COO construction");

    m_sparseSystem.setCSRMatrixFromCOO(sparseSystem.matrixRows(), 
                                       sparseSystem.matrixCols(),
                                       cooRows.vectorView(), 
                                       cooCols.vectorView());
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler constructor during COO to CSR conversion");

    //m_sparseSystem.csrMatrix().sparsePrint_host();

    err = cudaFree(entryCountDevicePtr);
    assert(err == cudaSuccess && "cudaFree failed in GPUAssembler constructor during counting matrix entries");

    m_options = defaultOptions();
}

OptionList GPUAssembler::defaultOptions()
{
    OptionList opt;
    opt.addReal("youngs_modulus", "Young's modulus", 1.0);
    opt.addReal("poissons_ratio", "Poisson's ratio", 0.3);
    return opt;
}

void GPUAssembler::
computeDirichletDofs(int unk_, 
                     const std::vector<DofMapper> &mappers,
                     std::vector<Eigen::VectorXd> &ddof,
                     const MultiBasis &multiBasis)
{
    DofMapper dofMapper = mappers[unk_]; 
    ddof[unk_].resize(dofMapper.boundarySize());

    for (std::deque<boundary_condition>::const_iterator 
         it = m_boundaryConditions.dirichletBegin();
         it != m_boundaryConditions.dirichletEnd(); ++it)
    {
        const int k = it->patchIndex();
        if (it -> unknown() != unk_)
            continue;
        const Eigen::VectorXi boundary = multiBasis.basis(k).boundary(it -> side());
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
        const int i = multiBasis.basis(k).corner(it -> corner());
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

void GPUAssembler::printMultiPatch() const
{
    printMultiPatchKernel<<<1,1>>>(m_multiPatch.deviceView());
    cudaError_t err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::printMultiPatch");
}

void GPUAssembler::printMultiBasis() const
{
    printMultiBasisKernel<<<1,1>>>(m_multiBasis.deviceView());
    cudaError_t err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::printMultiBasis");
}

int GPUAssembler::numDofs() const
{
    return m_sparseSystem.numDofs();
}

void GPUAssembler::constructSolution(const DeviceVectorView<double>& solVector, 
                                     const DeviceNestedArrayView<double>& fixedDoFs, 
                                     MultiPatchDeviceView& displacementDeviceView) const
{
    int minGrid, blockSize;
    int CPSize = m_displacementHost.CPSize();
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
        constructSolutionKernel, 0, CPSize);

    int gridSize = (CPSize + blockSize - 1) / blockSize;
    constructSolutionKernel<<<gridSize, blockSize>>>(solVector, fixedDoFs,
                                                     m_multiBasis.deviceView(),
                                                     m_sparseSystem.deviceView(),
                                                     displacementDeviceView,
                                                     CPSize);
    cudaError_t err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructSolution");
}

void GPUAssembler::assemble(const DeviceVectorView<double> &solVector, 
                            int numIter, 
                            const DeviceNestedArrayView<double> &fixedDoFs)
{
    m_sparseSystem.matrixSetZero();
    m_sparseSystem.RHSSetZero();
    //cudaError_t err = cudaMemset(m_sparseSystem.deviceView().matrix().data(), 0, 
    //                             m_sparseSystem.deviceView().matrix().size() * sizeof(double));
    //if (err != cudaSuccess)
    //    std::cerr << "CUDA error during memset of matrix in GPUAssembler::assemble: "
    //              << cudaGetErrorString(err) << std::endl;
    //cudaError_t err = cudaMemset(m_sparseSystem.deviceView().rhs().data(), 0, 
    //                 m_sparseSystem.deviceView().rhs().size() * sizeof(double));
    //if (err != cudaSuccess)
    //    std::cerr << "CUDA error during memset of rhs in GPUAssembler::assemble: "
    //              << cudaGetErrorString(err) << std::endl;
    int minGrid, blockSize;
    int CPSize = m_displacementHost.CPSize();

    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
        constructSolutionKernel, 0, CPSize);

    int gridSize = (CPSize + blockSize - 1) / blockSize;
#if 1
    constructSolutionKernel<<<gridSize, blockSize>>>(solVector, fixedDoFs,
                                                     m_multiBasis.deviceView(),
                                                     m_sparseSystem.deviceView(),
                                                     m_displacement.deviceView(),
                                                     CPSize);
#else
    constructSolutionKernel<<<1, 1>>>(solVector, fixedDoFs,
                                      m_multiBasis.deviceView(),
                                      m_sparseSystem.deviceView(),
                                      m_displacement.deviceView(),
                                      CPSize);
#endif
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error after constructSolutionKernel in GPUAssembler::assemble: "
                  << cudaGetErrorString(err) << std::endl;

    DeviceNestedArrayView<double> fixedDofs_assemble;
    if (numIter != 0)
        fixedDofs_assemble = m_ddof_zero.view();
    else
        fixedDofs_assemble = m_ddof.view();

    int domainDim = m_multiPatchHost.getBasisDim();
    //int basisOrder = m_multiBasisHost.basis(0).getOrder(0); //assume same order in all directions

    
#if 0
#if 1
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
    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after kernel assembleDomain launch: " 
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (assembleDomain): " 
                  << cudaGetErrorString(err) << std::endl;

    printf("Dense Matrix:\n");
    m_sparseSystem.deviceView().matrix().print();
    printf("RHS Vector:\n");
    m_sparseSystem.deviceView().rhs().print();
#endif
    DeviceArray<double> parameterValues(m_options.realValues());
    //int* entryCountDevicePtr;
    //err = cudaMalloc((void**)&entryCountDevicePtr, sizeof(int));
    //assert(err == cudaSuccess && "cudaMalloc failed in GPUAssembler constructor during counting matrix entries");
    //err = cudaMemset(entryCountDevicePtr, 0, sizeof(int));
    //assert(err == cudaSuccess && "cudaMemset failed in GPUAssembler constructor during counting matrix entries");

#if 1
    int numElements = m_multiBasisHost.totalNumElements();
    int N_D = m_multiBasisHost.numActive();
    int numActivePerBlock = std::min(16, N_D);
    dim3 blockSize2D(numActivePerBlock, numActivePerBlock);
    int numBlocksPerEle = (N_D + numActivePerBlock - 1) / numActivePerBlock;
    gridSize = numBlocksPerEle * numBlocksPerEle * numElements;
    int numDerivatives = 1;
    int geoP1 = m_multiPatchHost.knotOrder() + 1;
    int dispP1 = m_multiBasisHost.knotOrder() + 1;
    int numDouble = geoP1 * (numDerivatives + 1) * domainDim + (geoP1 * geoP1 + 4 * geoP1) * domainDim + //geoValuesAndDers + workingSpace
                    dispP1 * (numDerivatives + 1) * domainDim + (dispP1 * dispP1 + 4 * dispP1) * domainDim + //dispValuesAndDers + workingSpace
                    numActivePerBlock * domainDim * numActivePerBlock * domainDim + //localMat
                    numActivePerBlock * domainDim; // localRHS
    size_t shmemBytes = numDouble * sizeof(double);
    assembleDomainKernel_perTileBlock_loopOverGps<<<gridSize, blockSize2D, shmemBytes>>>(
                    numDerivatives, numElements, 
                    numBlocksPerEle, numActivePerBlock,
                    parameterValues.vectorView(),
                    m_displacement.deviceView(),
                    m_multiPatch.deviceView(),
                    m_multiGaussPoints.view(),
                    m_bodyForce.vectorView(),
                    m_sparseSystem.deviceView(),
                    fixedDofs_assemble);

    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after kernel assembleDomain_perTileBlock_loopOverGps launch: " 
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)        
        std::cerr << "CUDA error during device synchronization (assembleDomain_perTileBlock_loopOverGps): " 
                  << cudaGetErrorString(err) << std::endl;
#endif
#if 0    
    m_sparseSystem.csrMatrix().print_host();
    std::cout << std::endl;
    std::cout << m_sparseSystem.hostRHS();
    std::cout << std::endl << std::endl;
#endif
#if 0 
    m_sparseSystem.matrixSetZero();
    m_sparseSystem.RHSSetZero();
#endif    

#if 0    
    int totalGPs = m_multiBasisHost.totalNumGPs();
    int N_D2 = m_multiBasisHost.numActive();
    int numActivePerBlock2 = std::min(16, N_D2);
    dim3 blockSize2D2(numActivePerBlock2, numActivePerBlock2);
    int numBlocksPerGP2 = (N_D2 + numActivePerBlock2 - 1) / numActivePerBlock2;
    gridSize = numBlocksPerGP2 * numBlocksPerGP2 * totalGPs;
    int numDerivatives2 = 1;
    int geoP12 = m_multiPatchHost.knotOrder() + 1;
    int dispP12 = m_multiBasisHost.knotOrder() + 1;
    int numDouble2 = geoP12 * (numDerivatives2 + 1) * domainDim + (geoP12 * geoP12 + 4 * geoP12) * domainDim + //geoValuesAndDers + workingSpace
                    dispP12 * (numDerivatives2 + 1) * domainDim + (dispP12 * dispP12 + 4 * dispP12) * domainDim; //dispValuesAndDers + workingSpace
    size_t shmemBytes2 = numDouble2 * sizeof(double);
    assembleDomainKernel_perTileBlock<<<gridSize, blockSize2D2, shmemBytes2>>>(
                    numDerivatives2, totalGPs, 
                    numBlocksPerGP2, numActivePerBlock2,
                    parameterValues.vectorView(),
                    m_displacement.deviceView(),
                    m_multiPatch.deviceView(),
                    m_multiGaussPoints.view(),
                    m_bodyForce.vectorView(),
                    m_sparseSystem.deviceView(),
                    fixedDofs_assemble);

    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after kernel assembleDomain_perTileBlock launch: " 
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)        
        std::cerr << "CUDA error during device synchronization (assembleDomain_perTileBlock): " 
                  << cudaGetErrorString(err) << std::endl;
#endif
#if 0    
    m_sparseSystem.csrMatrix().print_host();
    std::cout << std::endl;
    std::cout << m_sparseSystem.hostRHS();
    std::cout << std::endl << std::endl;
#endif
#if 0 
    m_sparseSystem.matrixSetZero();
    m_sparseSystem.RHSSetZero();
#endif
#if 0
    int totalGPs = m_multiBasisHost.totalNumGPs();
    blockSize = m_multiBasisHost.numGPs();
    gridSize = (totalGPs + blockSize - 1) / blockSize;
    assembleDomainKernel<<<gridSize, blockSize>>>(
                    totalGPs,
                    //entryCountDevicePtr,
                    parameterValues.vectorView(),
                    m_displacement.deviceView(),
                    m_multiPatch.deviceView(),
                    m_multiGaussPoints.view(),
                    m_bodyForce.vectorView(),
                    m_sparseSystem.deviceView(),
                    fixedDofs_assemble);

    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after kernel assembleDomain launch: " 
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (assembleDomain): " 
                  << cudaGetErrorString(err) << std::endl;
#endif
#if 0 
    m_sparseSystem.csrMatrix().print_host();
    std::cout << std::endl;
    std::cout << m_sparseSystem.hostRHS();
    std::cout << std::endl << std::endl;
#endif
    //int entryCountHost;
    //err = cudaMemcpy(&entryCountHost, entryCountDevicePtr, sizeof(int), cudaMemcpyDeviceToHost);
    //assert(err == cudaSuccess && "cudaMemcpy failed in GPUAssembler constructor during counting matrix entries");
#if 0
    printf("COO Matrix:\n");
    printf("rows:");
    m_sparseSystem.deviceView().rows().print();
    printf("cols:");
    m_sparseSystem.deviceView().cols().print();
    printf("values:");
    m_sparseSystem.deviceView().values().print();

    printf("COO RHS:\n");
    m_sparseSystem.deviceView().rhs().print();
#endif
}

void GPUAssembler::denseMatrix(DeviceMatrixView<double> denseMat) const
{
    m_sparseSystem.csrMatrix().toDense(denseMat);
}
