#include "GPUElectroelasticityAssembler.h"

__global__
void countWholeMatrixEntrysKernel(int numElements, int numBlocksPerElement, int numActivePerBlock,
                       MultiPatchDeviceView displacement,
                       MultiPatchDeviceView electricPotential,
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
            dim = displacement.targetDim();
        }
        __syncthreads();
        
        for (int i = threadIdx.x + blockCoord[0] * numActivePerBlock, ii = threadIdx.x; 
             ii < numActivePerBlock && i < N_D; i += blockDim.x, ii += blockDim.x)
        {
            for (int di = 0; di < dim + 1; di++)
            {
                int activeIndex_i = 0;
                if (di < dim)
                    activeIndex_i = displacement.basis(patch_idx).activeIndex(pt, i);
                else
                    activeIndex_i = electricPotential.basis(patch_idx).activeIndex(pt, i);
                int globalIndex_i = system.mapColIndex(activeIndex_i, patch_idx, di);
                for (int j = threadIdx.y + blockCoord[1] * numActivePerBlock, jj = threadIdx.y; 
                     jj < numActivePerBlock && j < N_D; j += blockDim.y, jj += blockDim.y)
                {
                    for (int dj = 0; dj < dim + 1; dj++)
                    {
                        int activeIndex_j = 0;
                        if (dj < dim)
                            activeIndex_j = displacement.basis(patch_idx).activeIndex(pt, j);
                        else
                            activeIndex_j = electricPotential.basis(patch_idx).activeIndex(pt, j);
                        int globalIndex_j = system.mapColIndex(activeIndex_j, patch_idx, dj);
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
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
        atomicAdd(entryCount, sharedEntryCount);
}

__global__
void countMixMatrixEntrysKernel(int numElements, 
                       DeviceVectorView<int> numBlocksPerElement, 
                       DeviceVectorView<int> numActivePerBlock,
                       MultiPatchDeviceView displacement,
                       MultiPatchDeviceView electricPotential,
                       MultiGaussPointsDeviceView multiGaussPoints,
                       SparseSystemDeviceView system,
                       DeviceNestedArrayView<double> eliminatedDofs,
                       int* entryCount)
{
    __shared__ int sharedEntryCount;
    int totalNumBlocks = numElements * numBlocksPerElement[0] * numBlocksPerElement[1];
    for (int idx = blockIdx.x; idx < totalNumBlocks; idx += gridDim.x)
    {
        __shared__ int patch_idx, N_D, N_P, dim, blockCoord[2];
        __shared__ double ptData[3]; //max 3D
        DeviceVectorView<double> pt(ptData, multiGaussPoints.dim());
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            for (int d = 0; d < 2; d++)
            {
                blockCoord[d] = idx % numBlocksPerElement[d];
                idx /= numBlocksPerElement[d];
            }
            sharedEntryCount = 0;
            int element_idx = displacement.threadPatch_element(idx, patch_idx);
            int numGPsInElement = displacement.basis(patch_idx).numGPsInElement();
            int point_idx = element_idx * numGPsInElement;
            displacement.gsPoint(point_idx, patch_idx, 
                                 multiGaussPoints[patch_idx], pt);
            TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
            TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patch_idx);
            N_D = dispBasis.numActiveControlPoints();
            N_P = elecBasis.numActiveControlPoints();
            dim = displacement.targetDim();
        }
        __syncthreads();
        
        for (int i = threadIdx.x + blockCoord[0] * numActivePerBlock[0], ii = threadIdx.x; 
             ii < numActivePerBlock[0] && i < N_D; i += blockDim.x, ii += blockDim.x)
        {
            for (int di = 0; di < dim; di++)
            {
                int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx)
                    .activeIndex(pt, i), patch_idx, di);
                for (int j = threadIdx.y + blockCoord[1] * numActivePerBlock[1], jj = threadIdx.y; 
                     jj < numActivePerBlock[1] && j < N_P; j += blockDim.y, jj += blockDim.y)
                {
                    int globalIndex_j = system.mapColIndex(electricPotential.basis(patch_idx)
                        .activeIndex(pt, j), patch_idx, dim);
                    if (system.isEntry(globalIndex_i, globalIndex_j, di, dim))
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
    if (threadIdx.x == 0 && threadIdx.y == 0)
        atomicAdd(entryCount, sharedEntryCount);
}

__global__
void countEleMatrixEntrysKernel(int numElements, int numBlocksPerElement, int numActivePerBlock,
                       MultiPatchDeviceView displacement,
                       MultiPatchDeviceView electricPotential,
                       MultiGaussPointsDeviceView multiGaussPoints,
                       SparseSystemDeviceView system,
                       DeviceNestedArrayView<double> eliminatedDofs,
                       int* entryCount)
{
    __shared__ int sharedEntryCount;
    int totalNumBlocks = numElements * numBlocksPerElement * numBlocksPerElement;
    for (int idx = blockIdx.x; idx < totalNumBlocks; idx += gridDim.x)
    {
        __shared__ int patch_idx, N_P, dim, blockCoord[2];
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
            TensorBsplineBasisDeviceView eleBasis = electricPotential.basis(patch_idx);
            N_P = eleBasis.numActiveControlPoints();
            dim = displacement.targetDim();
        }
        __syncthreads();
        
        for (int i = threadIdx.x + blockCoord[0] * numActivePerBlock, ii = threadIdx.x; 
             ii < numActivePerBlock && i < N_P; i += blockDim.x, ii += blockDim.x)
        {
            int globalIndex_i = system.mapColIndex(electricPotential.basis(patch_idx)
                .activeIndex(pt, i), patch_idx, dim);
            for (int j = threadIdx.y + blockCoord[1] * numActivePerBlock, jj = threadIdx.y; 
                 jj < numActivePerBlock && j < N_P; j += blockDim.y, jj += blockDim.y)
            {
                int globalIndex_j = system.mapColIndex(electricPotential.basis(patch_idx)
                    .activeIndex(pt, j), patch_idx, dim);
                if (system.isEntry(globalIndex_i, globalIndex_j, dim, dim))
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
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
        atomicAdd(entryCount, sharedEntryCount);
}

__global__
void computeWholeMatrixCOOKernel(int totalNumElements, int* counter,
                      int numBlocksPerElement, int numActivePerBlock,
                      MultiPatchDeviceView displacement,
                      MultiPatchDeviceView electricPotential,
                      MultiGaussPointsDeviceView multiGaussPoints,
                      SparseSystemDeviceView system,
                      //DeviceNestedArrayView<double> eliminatedDofs,
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
            dim = displacement.targetDim();
        }
        __syncthreads();
        for (int i = threadIdx.x + blockCoord[0] * numActivePerBlock, ii = threadIdx.x; 
             ii < numActivePerBlock && i < N_D; i += blockDim.x, ii += blockDim.x)
        {
            for (int di = 0; di < dim + 1; di++)
            {
                int activeIndex_i = 0;
                if (di < dim)
                    activeIndex_i = displacement.basis(patch_idx).activeIndex(pt, i);
                else
                    activeIndex_i = electricPotential.basis(patch_idx).activeIndex(pt, i);
                int globalIndex_i = system.mapColIndex(activeIndex_i, patch_idx, di);
                for (int j = threadIdx.y + blockCoord[1] * numActivePerBlock, jj = threadIdx.y; 
                     jj < numActivePerBlock && j < N_D; j += blockDim.y, jj += blockDim.y)
                {
                    for (int dj = 0; dj < dim + 1; dj++)
                    {
                        int activeIndex_j = 0;
                        if (dj < dim)
                            activeIndex_j = displacement.basis(patch_idx).activeIndex(pt, j);
                        else
                            activeIndex_j = electricPotential.basis(patch_idx).activeIndex(pt, j);
                        int globalIndex_j = system.mapColIndex(activeIndex_j, patch_idx, dj);
                        if (system.isEntry(globalIndex_i, globalIndex_j, di, dj))
                            system.pushToEntryIndex(globalIndex_i, globalIndex_j, di, dj, 
                                                    counter, rows, cols);
                    }
                }
            }
        }
    }
}

__global__
void computeMixMatrixCOOKernel(int numElements, int* counter,
                       DeviceVectorView<int> numBlocksPerElement, 
                       DeviceVectorView<int> numActivePerBlock,
                       MultiPatchDeviceView displacement,
                       MultiPatchDeviceView electricPotential,
                       MultiGaussPointsDeviceView multiGaussPoints,
                       SparseSystemDeviceView system,
                       DeviceVectorView<int> cooRows,
                       DeviceVectorView<int> cooCols)
{
    // Similar structure to countMixMatrixEntrysKernel, but instead of counting, it writes the row and column indices to cooRows and cooCols
    // This kernel would need to use atomic operations to write to cooRows and cooCols at the correct index, which would require an additional atomic counter similar to entryCount
    int totalNumBlocks = numElements * numBlocksPerElement[0] * numBlocksPerElement[1];
    for (int idx = blockIdx.x; idx < totalNumBlocks; idx += gridDim.x)
    {
        __shared__ int patch_idx, N_D, N_P, dim, blockCoord[2];
        __shared__ double ptData[3]; //max 3D
        DeviceVectorView<double> pt(ptData, multiGaussPoints.dim());
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            for (int d = 0; d < 2; d++)
            {
                blockCoord[d] = idx % numBlocksPerElement[d];
                idx /= numBlocksPerElement[d];
            }
            int element_idx = displacement.threadPatch_element(idx, patch_idx);
            int numGPsInElement = displacement.basis(patch_idx).numGPsInElement();
            int point_idx = element_idx * numGPsInElement;
            displacement.gsPoint(point_idx, patch_idx, 
                                 multiGaussPoints[patch_idx], pt);
            TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
            TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patch_idx);
            N_D = dispBasis.numActiveControlPoints();
            N_P = elecBasis.numActiveControlPoints();
            dim = displacement.targetDim();
        }
        __syncthreads();
        for (int i = threadIdx.x + blockCoord[0] * numActivePerBlock[0], ii = threadIdx.x; 
             ii < numActivePerBlock[0] && i < N_D; i += blockDim.x, ii += blockDim.x)
        {
            for (int di = 0; di < dim; di++)
            {
                int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx)
                    .activeIndex(pt, i), patch_idx, di);
                for (int j = threadIdx.y + blockCoord[1] * numActivePerBlock[1], jj = threadIdx.y; 
                     jj < numActivePerBlock[1] && j < N_D; j += blockDim.y, jj += blockDim.y)
                {
                    int globalIndex_j = system.mapColIndex(displacement.basis(patch_idx)
                        .activeIndex(pt, j), patch_idx, dim);
                    if (system.isEntry(globalIndex_i, globalIndex_j, di, dim))
                        system.pushToEntryIndex(globalIndex_i, globalIndex_j, di, dim, 
                                                counter, cooRows, cooCols);
                }
            }
        }
    }
}

__global__
void computeEleMatrixCOOKernel(int totalNumElements, int* counter,
                      int numBlocksPerElement, int numActivePerBlock,
                      MultiPatchDeviceView displacement,
                      MultiPatchDeviceView electricPotential,
                      MultiGaussPointsDeviceView multiGaussPoints,
                      SparseSystemDeviceView system,
                      DeviceVectorView<int> rows,
                      DeviceVectorView<int> cols)
{
    int totalNumBlocks = totalNumElements * numBlocksPerElement * numBlocksPerElement;
    for (int idx = blockIdx.x; idx < totalNumBlocks; idx += gridDim.x)
    {
        __shared__ int patch_idx, N_P, dim, blockCoord[2];
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
            TensorBsplineBasisDeviceView eleBasis = electricPotential.basis(patch_idx);
            N_P = eleBasis.numActiveControlPoints();
            dim = displacement.targetDim();
        }
        __syncthreads();
        for (int i = threadIdx.x + blockCoord[0] * numActivePerBlock, ii = threadIdx.x; 
             ii < numActivePerBlock && i < N_P; i += blockDim.x, ii += blockDim.x)
        {
            int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx)
                .activeIndex(pt, i), patch_idx, dim);
            for (int j = threadIdx.y + blockCoord[1] * numActivePerBlock, jj = threadIdx.y; 
                 jj < numActivePerBlock && j < N_P; j += blockDim.y, jj += blockDim.y)
            {
                int globalIndex_j = system.mapColIndex(displacement.basis(patch_idx)
                    .activeIndex(pt, j), patch_idx, dim);
                if (system.isEntry(globalIndex_i, globalIndex_j, dim, dim))
                    system.pushToEntryIndex(globalIndex_i, globalIndex_j, dim, dim, 
                                            counter, rows, cols);
            }
        }
    }
}

__global__
void evaluateEleBasisValuesAndDerivativesAtGPsKernel(int numDerivatives, 
                            int totalNumGPs, int dim,
                            MultiPatchDeviceView displacement,
                            MultiPatchDeviceView electricPotential,
                            DeviceMatrixView<double> pts,
                            DeviceVectorView<double> elecWorkingSpaces,
                            DeviceMatrixView<double> elecValuesAndDerss)
{
    for (int tidx = blockIdx.x * blockDim.x + threadIdx.x; tidx < totalNumGPs * dim; tidx += blockDim.x * gridDim.x) {
        int GPIdx = tidx / dim;
        int d = tidx % dim;
        int patch_idx(0);
        displacement.threadPatch(GPIdx, patch_idx);
        DeviceVectorView<double> pt(pts.data() + GPIdx * dim, dim);
        TensorBsplineBasisDeviceView eleBasis = electricPotential.basis(patch_idx);
        int P1 = eleBasis.knotsOrder(0) + 1;
        double* elecWorkingSpace = elecWorkingSpaces.data() + GPIdx * P1 * (P1 + 4) * dim;
        DeviceMatrixView<double> elecValuesAndDers(elecValuesAndDerss.data() + GPIdx * P1 * (numDerivatives+1) * dim, P1, (numDerivatives+1)*dim);
        eleBasis.evalAllDers_into(d, dim, pt, numDerivatives, elecWorkingSpace, elecValuesAndDers);
    }
}

__global__
void constructElecSolutionKernel(DeviceVectorView<double> solVector, 
                             DeviceNestedArrayView<double> fixedDoFs, 
                             MultiBasisDeviceView multiBasis,
                             SparseSystemDeviceView sparseSystem,
                             MultiPatchDeviceView result, int CPSize)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < CPSize; idx += blockDim.x * gridDim.x)
    {
        int patch(0);
        int dim = multiBasis.domainDim();
        int point_idx = result.threadPatch_CPBase(idx, patch);
        int index(0);
        //printf("patch %d, point_idx %d\n", patch, point_idx);
        if (sparseSystem.mapper(dim).is_free(point_idx, patch))
        {
            //printf("free dof\n");
            index = sparseSystem.mapToGlobalColIndex(point_idx, patch, dim);
            //printf("global index: %d\n", index);
            result.setCoefficients(patch, point_idx, 0, solVector[index]);
        }
        else
        {
            //printf("fixed dof\n");
            index = sparseSystem.mapper(dim).bindex(point_idx, patch);
            //printf("global index: %d\n", index);
            result.setCoefficients(patch, point_idx, 0, fixedDoFs[dim][index]);
        }
    }
    //result.printControlPoints();
}

__global__
void evaluateGPKernel(int numDerivatives, int totalNumGPs,
                      DeviceVectorView<double> parameters,
                      MultiPatchDeviceView displacement,
                      MultiPatchDeviceView electricPotential,
                      MultiPatchDeviceView multiPatch,
                      DeviceMatrixView<double> pts,
                      DeviceVectorView<double> wts,
                      DeviceMatrixView<double> geoValuesAndDerss,
                      DeviceMatrixView<double> dispValuesAndDerss,
                      DeviceMatrixView<double> elecValuesAndDerss,
                      DeviceMatrixView<double> geoJacobianInvs,
                      DeviceVectorView<double> measures,
                      DeviceVectorView<double> weightForces,
                      DeviceVectorView<double> weightBodys,
                      DeviceVectorView<double> Js,
                      DeviceMatrixView<double> Fs, 
                      DeviceMatrixView<double> Ss, 
                      DeviceMatrixView<double> Cs, 
                      DeviceMatrixView<double> RCGinvs,
                      DeviceMatrixView<double> elecDisps,
                      DeviceMatrixView<double> As)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < totalNumGPs; idx += blockDim.x * gridDim.x) 
    {
        int dim = multiPatch.domainDim();
        int dimTensor = (dim * (dim + 1)) / 2;

        //int GPIdx = idx;
        DeviceVectorView<double> pt(pts.data() + idx * dim, dim);
        double wt = wts[idx];
        
        int patch_idx(0);
        int point_idx = displacement.threadPatch(idx, patch_idx);
        PatchDeviceView geoPatch = multiPatch.patch(patch_idx);
        int geoP1 = geoPatch.basis().knotsOrder(0) + 1;
        DeviceMatrixView<double> geoValuesAndDers(geoValuesAndDerss.data() + idx * geoP1 * (numDerivatives + 1) * dim, geoP1, (numDerivatives + 1) * dim);

        PatchDeviceView dispPatch = displacement.patch(patch_idx);
        PatchDeviceView elecPatch = electricPotential.patch(patch_idx);
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
        int P1 = dispBasis.knotsOrder(0) + 1;
        DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDerss.data() + idx * P1 * (numDerivatives + 1) * dim, P1, (numDerivatives + 1) * dim);
        TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patch_idx);
        int eleP1 = elecBasis.knotsOrder(0) + 1;
        DeviceMatrixView<double> elecValuesAndDers(elecValuesAndDerss.data() + idx * eleP1 * (numDerivatives + 1) * dim, eleP1, (numDerivatives + 1) * dim);

        DeviceMatrixView<double> geoJacobianInv(geoJacobianInvs.data() + idx * dim * dim, dim, dim);
        {
            double geoJacobianData[3*3] = {0.0}; //max 3D
            DeviceMatrixView<double> geoJacobian(geoJacobianData, dim, dim);
            geoPatch.jacobian(pt, geoValuesAndDers, numDerivatives, geoJacobian);
            geoJacobian.inverse(geoJacobianInv);
            measures[idx] = geoJacobian.determinant();
        }
        weightForces[idx] = wt * measures[idx];
        weightBodys[idx] = wt * measures[idx];

        double electricFieldData[3] = {0.0}; //max 3D
        DeviceMatrixView<double> electricField(electricFieldData, dim, 1);
        double electricFieldTransData[3] = {0.0}; //max 3D
        DeviceMatrixView<double> electricFieldTrans(electricFieldTransData, 1, dim);
        {
            double elecJacobianData[3] = {0.0}; //max 3D
            DeviceMatrixView<double> elecJacobian(elecJacobianData, 1, dim);
            elecPatch.jacobian(pt, elecValuesAndDers, numDerivatives, elecJacobian);
            elecJacobian.times(geoJacobianInv, electricFieldTrans);
            electricFieldTrans.transpose(electricField);
            electricField.times(-1.0);
        }
        //printf("Electric field at GP %d in patch %d: ", idx, patch_idx);
        //electricField.print();
        DeviceMatrixView<double> F(Fs.data() + idx * dim * dim, dim, dim);
        {
            double dispJacobianData[3*3] = {0.0}; //max 3D
            double physDispJacData[3*3] = {0.0}; //max 3D
            DeviceMatrixView<double> dispJacobian(dispJacobianData, dim, dim);
            DeviceMatrixView<double> physDispJac(physDispJacData, dim, dim);
            dispPatch.jacobian(pt, dispValuesAndDers, numDerivatives, dispJacobian);
            dispJacobian.times(geoJacobianInv, physDispJac);
            physDispJac.plusIdentity(F);
        }
        
        DeviceMatrixView<double> S(Ss.data() + idx * dim * dim, dim, dim);
        DeviceMatrixView<double> elecDisp(elecDisps.data() + idx * dim, dim, 1);
        DeviceMatrixView<double> C(Cs.data() + idx * dimTensor * dimTensor, dimTensor, dimTensor);
        DeviceMatrixView<double> RCGinv(RCGinvs.data() + idx * dim * dim, dim, dim);
        DeviceMatrixView<double> A(As.data() + idx * dimTensor * dim, dimTensor, dim);
        {
            double YM = parameters[2];
            double PR = parameters[1];
            double epsilon = parameters[0];
            double lambda = YM * PR / ( ( 1. + PR ) * ( 1. - 2. * PR ) );
            double mu = YM / ( 2. * ( 1. + PR ) );

            double J = F.determinant();
            Js[idx] = J;
            double RCGData[3*3] = {0.0}; //max 3D
            DeviceMatrixView<double> RCG(RCGData, dim, dim);
            double F_transposeData[3*3] = {0.0}; //max 3D
            {
                DeviceMatrixView<double> F_transpose(F_transposeData, dim, dim);
                F.transpose(F_transpose);
                F_transpose.times(F, RCG);
            }
            //double RCGinvData[3*3] = {0.0}; //max 3D
            //DeviceMatrixView<double> RCGinv(RCGinvData, dim, dim);
            RCG.inverse(RCGinv);
            //printf("RCGinv at GP %d:\n", idx);
            //RCGinv.print();
            RCGinv.times((lambda*(J*J-1)/2-mu), S);
            S.tracePlus(mu);
            {
                double RCGinvTimesElectricFieldData[3] = {0.0}; //max 3D
                //double RCGinvTimesElectricFieldTransData[3] = {0.0}; //max 3D
                DeviceMatrixView<double> RCGinvTimesElectricField(RCGinvTimesElectricFieldData, dim, 1);
                //DeviceMatrixView<double> RCGinvTimesElectricFieldTrans(RCGinvTimesElectricFieldTransData, 1, dim);
                RCGinv.times(electricField, RCGinvTimesElectricField);
                //RCGinvTimesElectricField.transpose(RCGinvTimesElectricFieldTrans);
                double temp = 0.0;
                DeviceMatrixView<double> tempMat(&temp, 1, 1);
                electricField.transposeTime(RCGinvTimesElectricField, tempMat);
                double temp1Data[3*3] = {0.0}; //max 3D 
                DeviceMatrixView<double> temp1(temp1Data, dim, dim);
                //RCGinvTimesElectricField.times(RCGinvTimesElectricFieldTrans, temp1);
                RCGinv.times(temp, temp1);
                //temp1.times(epsilon * J);
                temp1.times(- epsilon * J / 2.0);
                S.plus(temp1);
                //double temp2Data = 0.0;
                double temp2Data[3*3] = {0.0}; //max 3D
                //DeviceMatrixView<double> temp2(&temp2Data, 1, 1);
                DeviceMatrixView<double> temp2(temp2Data, dim, dim);
                //electricFieldTrans.times(RCGinvTimesElectricField, temp2);
                //RCGinv.times(temp2Data, temp1);
                //temp1.times(- epsilon * J / 2.0);
                //S.plus(temp1);
                RCGinvTimesElectricField.timeTranspose(RCGinvTimesElectricField, temp2);
                temp2.times(epsilon * J);
                S.plus(temp2);
                RCGinvTimesElectricField.times(epsilon * J, elecDisp);
            }
            //printf("S at GP %d in patch %d:\n", idx, patch_idx);
            //S.print();
            matrixViewTraceTensor(C, RCGinv, RCGinv);
            C.times(lambda*J*J);
            double CtempData[6*6] = {0.0}; //max 3D
            DeviceMatrixView<double> Ctemp(CtempData, dimTensor, dimTensor);
            symmetricIdentityViewTensor(Ctemp, RCGinv);
            Ctemp.times(mu-lambda*(J*J-1)/2.0);
            C.plus(Ctemp);
            electroelasticMechanicalTensor(Ctemp, RCGinv, electricField);
            Ctemp.times(epsilon * J / 2.0);
            C.plus(Ctemp);
            //printf("C at GP %d in patch %d:\n", idx, patch_idx);
            //C.print();
            electroelasticCouplingTensor(A, RCGinv, electricField);
            A.times(epsilon * J);
            //printf("A at GP %d in patch %d:\n", idx, patch_idx);
            //A.print();
        }
    }
}

#if 0
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
#endif

__global__
void assembleMatrixKernel(int numDerivatives, int EleStartId,
                          int numElementsBatched, int N_D,
                          DeviceVectorView<double> parameters,
                          MultiPatchDeviceView displacement,
                          MultiPatchDeviceView electricPotential,
                          SparseSystemDeviceView system,
                          DeviceNestedArrayView<double> eliminatedDofs,
                          DeviceMatrixView<double> pts,
                          //DeviceVectorView<double> wts,
                          DeviceMatrixView<double> geoJacobianInvs,
                          DeviceVectorView<double> measures,
                          DeviceVectorView<double> weightForces,
                          DeviceVectorView<double> weightBodys,
                          DeviceVectorView<double> Js,
                          DeviceMatrixView<double> dispValuesAndDerss,
                          DeviceMatrixView<double> elecValuesAndDerss,
                          DeviceMatrixView<double> Fs, 
                          DeviceMatrixView<double> Ss, 
                          DeviceMatrixView<double> Cs, 
                          DeviceMatrixView<double> RCGinvs,
                          DeviceMatrixView<double> As)
{
    int totalNumShapeFunctions = numElementsBatched * N_D * N_D;
    //printf("totalNumShapeFunctions: %d\n", totalNumShapeFunctions);
    for (int bidx = blockIdx.x; bidx < totalNumShapeFunctions; bidx += gridDim.x)
    {
        __shared__ int patch_idx, ele_idx, shapeFuncCoord[2], idx;
        __shared__ double localMatrxData[4*4]; //max 3D

        const int dim = pts.rows();
        const int dimTensor = dim * (dim + 1) / 2;

        int threadId = threadIdx.y * blockDim.x + threadIdx.x;
        if (threadId == 0) {
            idx = bidx;
            for (int d = 0; d < 2; d++) {
                shapeFuncCoord[d] = idx % N_D;
                idx /= N_D;
            }
            ele_idx = displacement.threadPatch_element(EleStartId + idx, patch_idx);
            //if(blockIdx.x == 0)
                //printf("bidx:%d, idx:%d, ele_idx:%d, patch_idx:%d, shapeFuncCoord[0]:%d, shapeFuncCoord[1]:%d\n", bidx, idx, ele_idx, patch_idx, shapeFuncCoord[0], shapeFuncCoord[1]);
        }

        for (int i = threadId; i < (dim + 1) * (dim + 1); i += blockDim.x * blockDim.y)
            localMatrxData[i] = 0.0;
        __syncthreads();
        //localMatrix.print();

        DeviceMatrixView<double> localMatrix(localMatrxData, dim + 1, dim + 1);
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
        TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patch_idx);
        int P1 = dispBasis.knotsOrder(0) + 1;
        int elecP1 = elecBasis.knotsOrder(0) + 1;
        int i = shapeFuncCoord[0];
        int j = shapeFuncCoord[1];
        double epsilon = parameters[0];
        for (int i_GP = threadIdx.x; i_GP < N_D; i_GP += blockDim.x) {
            //int GPIdx = ele_idx * N_D + i_GP;
            //if(blockIdx.x == 334 && threadId == 32)
            //    printf("i_GP: %d, GPIdx: %d\n", i_GP, idx * N_D + i_GP);
            int GPIdx = idx * N_D + i_GP;
            //DeviceVectorView<double> pt(pts.data() + GPIdx * dim, dim);
            //if(blockIdx.x == 334 && threadId == 32)
            //    pt.print();
            //double wt = wts[GPIdx];
            //printf("wt: %f\n", wt);
            DeviceMatrixView<double> geoJacobianInv(geoJacobianInvs.data() + GPIdx * dim * dim, dim, dim);
            //geoJacobianInv.print();
            double measure = measures[GPIdx];
            //printf("measure: %f\n", measure);
            double weightForce = weightForces[GPIdx];
            //printf("weightForce: %f\n", weightForce);
            double weightBody = weightBodys[GPIdx];
            //printf("weightBody: %f\n", weightBody);
            double J = Js[GPIdx];
            DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDerss.data() + GPIdx * P1 * (numDerivatives + 1) * dim, P1, (numDerivatives + 1) * dim);
            DeviceMatrixView<double> elecValuesAndDers(elecValuesAndDerss.data() + GPIdx * elecP1 * (numDerivatives + 1) * dim, elecP1, (numDerivatives + 1) * dim);
            //dispValuesAndDers.print();
            DeviceMatrixView<double> F(Fs.data() + GPIdx * dim * dim, dim, dim);
            //F.print();
            DeviceMatrixView<double> S(Ss.data() + GPIdx * dim * dim, dim, dim);
            //S.print();
            DeviceMatrixView<double> C(Cs.data() + GPIdx * dimTensor * dimTensor, dimTensor, dimTensor);
            //if(blockIdx.x == 334 && threadId == 31)
            //    C.print();
            DeviceMatrixView<double> RCGinv(RCGinvs.data() + GPIdx * dim * dim, dim, dim);
            DeviceMatrixView<double> A(As.data() + GPIdx * dimTensor * dim, dimTensor, dim);

            double dN_iData[3] = {0.0}; //max 3D
            double dN_jData[3] = {0.0}; //max 3D
            double dNP_iData[3] = {0.0}; //max 3D
            double dNP_jData[3] = {0.0}; //max 3D
            DeviceVectorView<double> dN_i(dN_iData, dim);
            DeviceVectorView<double> dN_j(dN_jData, dim);
            DeviceVectorView<double> dNP_i(dNP_iData, dim);
            DeviceVectorView<double> dNP_j(dNP_jData, dim);
            tensorBasisDerivative(i, P1, dim, numDerivatives, dispValuesAndDers, dN_i);
            //dN_i.print();
            tensorBasisDerivative(j, P1, dim, numDerivatives, dispValuesAndDers, dN_j);
            //dN_j.print();
            tensorBasisDerivative(i, elecP1, dim, numDerivatives, elecValuesAndDers, dNP_i);
            tensorBasisDerivative(j, elecP1, dim, numDerivatives, elecValuesAndDers, dNP_j);
            //dNP_j.print();
            double physGrad_jData[3] = {0.0}; //max 3D
            double physGrad_iData[3] = {0.0}; //max 3D
            double physPGrad_iData[3] = {0.0}; //max 3D
            double physPGrad_jData[3] = {0.0}; //max 3D
            DeviceVectorView<double> physGrad_i(physGrad_iData, dim);
            DeviceVectorView<double> physGrad_j(physGrad_jData, dim);
            DeviceVectorView<double> physPGrad_i(physPGrad_iData, dim);
            DeviceVectorView<double> physPGrad_j(physPGrad_jData, dim);
            geoJacobianInv.transposeTime(dN_i, physGrad_i);
            //if(blockIdx.x == 334 && threadId == 32)
            //    physGrad_i.print();
            geoJacobianInv.transposeTime(dN_j, physGrad_j);
            //physGrad_j.print();
            geoJacobianInv.transposeTime(dNP_i, physPGrad_i);
            geoJacobianInv.transposeTime(dNP_j, physPGrad_j);
            //printf("physPGrad_i at GP %d in patch %d:\n", GPIdx, patch_idx);
            //physPGrad_i.print();
            double geometricTangentTempData[3] = {0.0}; //max 3D
            DeviceVectorView<double> geometricTangentTemp(geometricTangentTempData, dim);
            S.times(physGrad_i, geometricTangentTemp);
            double geometricTangent = geometricTangentTemp.dot(physGrad_j);
            //if(blockIdx.x == 334 && threadId == 32)
            //    printf("geometricTangent: %f\n", geometricTangent);
            for (int di = 0; di < dim; di++) {
                double materialTangentTempData[6] = {0.0}; //max 3D
                DeviceMatrixView<double> materialTangentTemp(materialTangentTempData, 1, dimTensor);
                {
                    double B_i_diTransData[6] = {0.0}; //max 3D
                    DeviceMatrixView<double> B_i_diTrans(B_i_diTransData, 1, dimTensor);
                    {
                        double B_i_diData[6] = {0.0}; //max 3D
                        DeviceVectorView<double> B_i_di(B_i_diData,     dimTensor);
                        setBSingleDim<double>(di, B_i_di, F, physGrad_i);
                        //if(blockIdx.x == 334 && threadId == 32)
                        //    B_i_di.print();
                        B_i_di.transpose(B_i_diTrans);
                    }
                    B_i_diTrans.times(C, materialTangentTemp);
                    double tempData[3] = {0.0}; //max 3D
                    DeviceMatrixView<double> temp(tempData, 1, dim);
                    B_i_diTrans.times(A, temp);
                    double stiffnessEntryData = 0.0;
                    DeviceMatrixView<double> stiffnessEntry(&stiffnessEntryData, 1, 1);
                    temp.times(physPGrad_j, stiffnessEntry);
                    atomicAdd(&localMatrix(di, dim), - stiffnessEntryData);
                    atomicAdd(&localMatrix(dim, di), - stiffnessEntryData);
                }
                //materialTangentTemp.print();
                for (int dj = 0; dj < dim; dj++){
                    double materialTangent = 0;
                    //if(blockIdx.x == 334 && threadId == 32)
                    //    printf("materialTangent: %f\n", materialTangent);
                    DeviceMatrixView<double> materialTangentMat(&materialTangent, 1, 1);
                    {
                        double B_j_djData[6] = {0.0}; //max 3D
                        DeviceVectorView<double> B_j_dj(B_j_djData, dimTensor);
                        setBSingleDim<double>(dj, B_j_dj, F, physGrad_j);
                        materialTangentTemp.times(B_j_dj, materialTangentMat);
                    }
                    if (di == dj)
                        materialTangent += geometricTangent;
                    double stiffnessEntry = weightBody *materialTangent;
                    atomicAdd(&localMatrix(di, dj), stiffnessEntry);
                    //if(blockIdx.x == 334 && threadIdx.x == 32) {
                    //    localMatrix.print();
                    //    printf("di:%d, dj:%d, stiffnessEntry: %f\n", di, dj, stiffnessEntry);
                    //}
                }
            }
            double tempData[3] = {0.0}; //max 3D
            DeviceMatrixView<double> temp(tempData, 1, dim);
            physPGrad_i.transposeTime(RCGinv, temp);
            double stiffnessEntryData = 0.0;
            DeviceMatrixView<double> stiffnessEntry(&stiffnessEntryData, 1, 1);
            temp.times(physPGrad_j, stiffnessEntry);
            //localMatrix(dim, dim) -= weightBody * epsilon * J * stiffnessEntryData;
            atomicAdd(&localMatrix(dim, dim), -weightBody * epsilon * J * stiffnessEntryData);
            //printf("localMatrix(%d, %d): %f\n", dim, dim, localMatrix(dim, dim));
        }
        __syncthreads();
        //printf("bidx:%d, localMatrix:\n", bidx);
        //localMatrix.print();
        //int GPIdx = ele_idx * N_D + threadIdx.x;
        //int GPIdx = idx * N_D + threadIdx.x;
        int GPIdx = idx * N_D;
        DeviceVectorView<double> pt(pts.data() + GPIdx * dim, dim);
        for (int tid = threadId; tid < (dim + 1) * (dim + 1); tid += blockDim.x * blockDim.y) {
            int di = tid % (dim + 1);
            int dj = tid / (dim + 1);
            int activeIndex_i = 0;
            int activeIndex_j = 0;
            if (di < dim)
                activeIndex_i = displacement.basis(patch_idx).activeIndex(pt, i);
            else
                activeIndex_i = electricPotential.basis(patch_idx).activeIndex(pt, i);
            if (dj < dim)
                activeIndex_j = displacement.basis(patch_idx).activeIndex(pt, j);
            else
                activeIndex_j = electricPotential.basis(patch_idx).activeIndex(pt, j);
            int globalIndex_i = system.mapColIndex(activeIndex_i, patch_idx, di);
            int globalIndex_j = system.mapColIndex(activeIndex_j, patch_idx, dj);
            system.pushToMatrix(localMatrix(di, dj), globalIndex_i, globalIndex_j, eliminatedDofs, di, dj);
            //printf("tid:%d, i:%d, j:%d, di:%d, dj:%d, globalIndex_i:%d, globalIndex_j:%d, value:%f\n", tid, i, j, di, dj, globalIndex_i, globalIndex_j, localMatrix(di, dj));
        }
    }

}

__global__
void assembleMixMatrixKernel(int numDerivatives, int EleStartId,
                          int numElementsBatched, int N_D, int N_P,
                          MultiPatchDeviceView displacement,
                          MultiPatchDeviceView electricPotential,
                          SparseSystemDeviceView system,
                          DeviceNestedArrayView<double> eliminatedDofs,
                          DeviceMatrixView<double> pts,
                          DeviceMatrixView<double> geoJacobianInvs,
                          DeviceMatrixView<double> dispValuesAndDerss,
                          DeviceMatrixView<double> elecValuesAndDerss,
                          DeviceMatrixView<double> Fs, 
                          DeviceMatrixView<double> As)
{
    int totalNumShapeFunctions = numElementsBatched * N_D * N_P;
    //printf("totalNumShapeFunctions: %d\n", totalNumShapeFunctions);
    for (int bidx = blockIdx.x; bidx < totalNumShapeFunctions; bidx += gridDim.x)
    {
        __shared__ int patch_idx, ele_idx, shapeFuncCoord[2], idx;
        __shared__ double localMatrxData[3]; //max 3D

        const int dim = pts.rows();
        const int dimTensor = dim * (dim + 1) / 2;

        int threadId = threadIdx.y * blockDim.x + threadIdx.x;
        if (threadId == 0) {
            idx = bidx;
            shapeFuncCoord[0] = idx % N_D;
            idx /= N_D;
            shapeFuncCoord[1] = idx % N_P;
            ele_idx = displacement.threadPatch_element(EleStartId + idx, patch_idx);
            //if(blockIdx.x == 0)
                //printf("bidx:%d, idx:%d, ele_idx:%d, patch_idx:%d, shapeFuncCoord[0]:%d, shapeFuncCoord[1]:%d\n", bidx, idx, ele_idx, patch_idx, shapeFuncCoord[0], shapeFuncCoord[1]);
        }

        for (int i = threadId; i < dim; i += blockDim.x * blockDim.y)
            localMatrxData[i] = 0.0;
        __syncthreads();
        //localMatrix.print();
        DeviceMatrixView<double> localMatrix(localMatrxData, dim, 1);
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
        TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patch_idx);
        int P1 = dispBasis.knotsOrder(0) + 1;
        int elecP1 = elecBasis.knotsOrder(0) + 1;
        int i = shapeFuncCoord[0];
        int j = shapeFuncCoord[1];
        for (int i_GP = threadIdx.x; i_GP < N_D; i_GP += blockDim.x) {
            //int GPIdx = ele_idx * N_D + i_GP;
            //if(blockIdx.x == 334 && threadId == 32)
                //printf("i_GP: %d, GPIdx: %d\n", i_GP, idx * N_D + i_GP);
            int GPIdx = idx * N_D + i_GP;
            //DeviceVectorView<double> pt(pts.data() + GPIdx * dim, dim);
            //if(blockIdx.x == 334 && threadId == 32)
            //    pt.print();
            //double wt = wts[GPIdx];
            //printf("wt: %f\n", wt);
            DeviceMatrixView<double> geoJacobianInv(geoJacobianInvs.data() + GPIdx * dim * dim, dim, dim);
            //printf("geoJacobianInv:\n");
            //geoJacobianInv.print();
            DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDerss.data() + GPIdx * P1 * (numDerivatives + 1) * dim, P1, (numDerivatives + 1) * dim);
            //printf("dispValuesAndDers:\n");
            //dispValuesAndDers.print();
            DeviceMatrixView<double> elecValuesAndDers(elecValuesAndDerss.data() + GPIdx * elecP1 * (numDerivatives + 1) * dim, elecP1, (numDerivatives + 1) * dim);
            //printf("elecValuesAndDers:\n");
            //elecValuesAndDers.print();
            DeviceMatrixView<double> F(Fs.data() + GPIdx * dim * dim, dim, dim);
            //printf("F:\n");
            //F.print();
            DeviceMatrixView<double> A(As.data() + GPIdx * dimTensor * dim, dimTensor, dim);
            //printf("A:\n");
            //A.print();
            double dN_iData[3] = {0.0}; //max 3D
            double dN_jData[3] = {0.0}; //max 3D
            DeviceVectorView<double> dN_i(dN_iData, dim);
            DeviceVectorView<double> dN_j(dN_jData, dim);
            tensorBasisDerivative(i, P1, dim, numDerivatives, dispValuesAndDers, dN_i);
            //printf("dN_i:\n");
            //dN_i.print();
            tensorBasisDerivative(j, elecP1, dim, numDerivatives, elecValuesAndDers, dN_j);
            //printf("dN_j:\n");
            //dN_j.print();
            double physGrad_jData[3] = {0.0}; //max 3D
            double physGrad_iData[3] = {0.0}; //max 3D
            DeviceVectorView<double> physGrad_i(physGrad_iData, dim);
            DeviceVectorView<double> physGrad_j(physGrad_jData, dim);
            geoJacobianInv.transposeTime(dN_i, physGrad_i);
            geoJacobianInv.transposeTime(dN_j, physGrad_j);
            //printf("physGrad_i:\n");            
            //physGrad_i.print();
            //printf("physGrad_j:\n");
            //physGrad_j.print();
            for (int di = 0; di < dim; di++) {
                double B_i_diTransData[6] = {0.0}; //max 3D
                DeviceMatrixView<double> B_i_diTrans(B_i_diTransData, 1, dimTensor);
                {
                    double B_i_diData[6] = {0.0}; //max 3D
                    DeviceVectorView<double> B_i_di(B_i_diData, dimTensor);
                    setBSingleDim<double>(di, B_i_di, F, physGrad_i);
                    B_i_di.transpose(B_i_diTrans);
                }
                //printf("B_i_diTrans for dim %d:\n", di);
                //B_i_diTrans.print();
                double tempData[3] = {0.0}; //max 3D
                DeviceMatrixView<double> temp(tempData, 1, dim);
                B_i_diTrans.times(A, temp);
                //printf("temp after B_i_diTrans times A for dim %d:\n", di);
                //temp.print();
                double stiffnessEntryData = 0.0;
                DeviceMatrixView<double> stiffnessEntry(&stiffnessEntryData, 1, 1);
                temp.times(physGrad_j, stiffnessEntry);
                atomicAdd(&localMatrix(di, 0), stiffnessEntryData);
            }
        }
        __syncthreads();
        //printf("bidx:%d, localMatrix:\n", bidx);
        //localMatrix.print();
        //int GPIdx = ele_idx * N_D + threadIdx.x;
#if 1
        //int GPIdx = idx * N_D + threadIdx.x;
        int GPIdx = idx * N_D;
        DeviceVectorView<double> pt(pts.data() + GPIdx * dim, dim);
        for (int tid = threadId; tid < dim; tid += blockDim.x * blockDim.y) {
            int di = tid;
            int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx).activeIndex(pt, i), patch_idx, di);
            int globalIndex_j = system.mapColIndex(electricPotential.basis(patch_idx).activeIndex(pt, j), patch_idx, dim);
            system.pushToMatrix(localMatrix(di, 0), globalIndex_i, globalIndex_j, eliminatedDofs, di, dim);
            //printf("tid:%d, i:%d, j:%d, di:%d, dj:%d, globalIndex_i:%d, globalIndex_j:%d, value:%f\n", tid, i, j, di, dj, globalIndex_i, globalIndex_j, localMatrix(di, dj));
        }
#endif
    }
}

__global__
void assembleRHSKernel(int numDerivatives, int EleStartId,
                       int numElementsBatched, int N_D,
                       MultiPatchDeviceView displacement,
                       MultiPatchDeviceView electricPotential,
                       SparseSystemDeviceView system,
                       DeviceMatrixView<double> pts,
                       DeviceMatrixView<double> geoJacobianInvs,
                       DeviceVectorView<double> weightForces,
                       DeviceVectorView<double> weightBodys,
                       DeviceMatrixView<double> dispValuesAndDerss,
                       DeviceMatrixView<double> elecValuesAndDerss,
                       DeviceMatrixView<double> elecDisps,
                       DeviceVectorView<double> bodyForce,
                       DeviceMatrixView<double> Fs, 
                       DeviceMatrixView<double> Ss)
{
    int totalNumShapeFunctions = numElementsBatched * N_D;
    for (int bidx = blockIdx.x; bidx < totalNumShapeFunctions; bidx += gridDim.x) {
        __shared__ int shapeFuncIdx, ele_idx, patch_idx, idx;
        __shared__ double localRHSData[4]; //max 3D

        const int dim = pts.rows();
        const int dimTensor = dim * (dim + 1) / 2;

        int threadId = threadIdx.y * blockDim.x + threadIdx.x;
        if (threadId == 0) {
            idx = bidx;
            shapeFuncIdx = idx % N_D;
            idx /= N_D;
            ele_idx = displacement.threadPatch_element(EleStartId + idx, patch_idx);
            //printf("bidx:%d, idx:%d, ele_idx:%d, patch_idx:%d, shapeFuncIdx:%d\n", bidx, idx, ele_idx, patch_idx, shapeFuncIdx);
        }
        DeviceVectorView<double> localRHS(localRHSData, dim + 1);
        for (int i = threadId; i < dim + 1; i += blockDim.x * blockDim.y)
            localRHSData[i] = 0.0;
        __syncthreads();

        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
        TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patch_idx);
        int P1 = dispBasis.knotsOrder(0) + 1;
        int elecP1 = elecBasis.knotsOrder(0) + 1;
        for (int i_GP = threadIdx.x; i_GP < N_D; i_GP += blockDim.x) {
            //int GPIdx = ele_idx * N_D + i_GP;
            int GPIdx = idx * N_D + i_GP;
            DeviceMatrixView<double> geoJacobianInv(geoJacobianInvs.data() + GPIdx * dim * dim, dim, dim);
            double weightForce = weightForces[GPIdx];
            double weightBody = weightBodys[GPIdx];
            //printf("weightBody = %f\n", weightBody);
            DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDerss.data() + GPIdx * P1 * (numDerivatives + 1) * dim, P1, (numDerivatives + 1) * dim);
            DeviceMatrixView<double> elecValuesAndDers(elecValuesAndDerss.data() + GPIdx * elecP1 * (numDerivatives + 1) * dim, elecP1, (numDerivatives + 1) * dim);
            DeviceVectorView<double> elecDisp(elecDisps.data() + GPIdx * dim, dim);
            DeviceMatrixView<double> F(Fs.data() + GPIdx * dim * dim, dim, dim);
            DeviceMatrixView<double> S(Ss.data() + GPIdx * dim * dim, dim, dim);
            double dN_iData[3] = {0.0}; //max 3D
            double dN_elec_iData[3] = {0.0}; //max 3D
            DeviceVectorView<double> dN_i(dN_iData, dim);
            DeviceVectorView<double> dN_elec_i(dN_elec_iData, dim);
            tensorBasisDerivative(shapeFuncIdx, P1, dim, numDerivatives, dispValuesAndDers, dN_i);
            tensorBasisDerivative(shapeFuncIdx, elecP1, dim, numDerivatives, elecValuesAndDers, dN_elec_i);
#if 1
            double physGrad_iData[3] = {0.0}; //max 3D
            double physGrad_elec_iData[3] = {0.0}; //max 3D
            DeviceVectorView<double> physGrad_i(physGrad_iData, dim);
            DeviceVectorView<double> physGrad_elec_i(physGrad_elec_iData, dim);
            geoJacobianInv.transposeTime(dN_i, physGrad_i);
            geoJacobianInv.transposeTime(dN_elec_i, physGrad_elec_i);
            double SvecData[6] = {0.0}; //max 3D
            DeviceVectorView<double> Svec(SvecData, dimTensor);
            voigtStressView(Svec, S);
            //printf("S at GP %d:\n", GPIdx);
            //S.print();
            for (int di = 0; di < dim; di++) {
                double B_i_diData[6] = {0.0}; //max 3D
                DeviceVectorView<double> B_i_di(B_i_diData, dimTensor);
                setBSingleDim<double>(di, B_i_di, F, physGrad_i);
                //printf("B_i_di for dim %d at GP %d:\n", di, GPIdx);
                //B_i_di.print();
                double residualEntry = 0.0;
                DeviceMatrixView<double> residualEntryMat(&residualEntry, 1, 1);
                B_i_di.transposeTime(Svec, residualEntryMat);
                //if(di == 0|| di == 1)
                //    printf("residualEntry before for di=%d at GP %d: %f\n", di, GPIdx, residualEntry);
                residualEntry = -residualEntry * weightBody + weightForce * bodyForce[di] * tensorBasisValue(shapeFuncIdx, P1, dim, numDerivatives, dispValuesAndDers);
                //if(di == 0|| di == 1)
                //    printf("residualEntry for di=%d at GP %d: %f\n", di, GPIdx, residualEntry);
                atomicAdd(&localRHS[di], residualEntry);
            }
            double residualEntryPotential = 0.0;
            DeviceMatrixView<double> residualEntryPotentialMat(&residualEntryPotential, 1, 1);
            physGrad_elec_i.transposeTime(elecDisp, residualEntryPotentialMat);
            atomicAdd(&localRHS[dim], - residualEntryPotential * weightBody);
#endif
        }
        __syncthreads();
        //printf("bidx:%d, localRHS: \n", bidx);
        //localRHS.print();
        //int GPIdx = ele_idx * N_D + threadIdx.x;
#if 1
        //printf("localRHS:\n");
        //localRHS.print();
        //int GPIdx = idx * N_D + threadIdx.x;
        int GPIdx = idx * N_D;
        DeviceVectorView<double> pt(pts.data() + GPIdx * dim, dim);
        for (int di = threadId; di < dim + 1; di += blockDim.x * blockDim.y) {
            int activeIndex_i = 0;
            if (di < dim)
                activeIndex_i = displacement.basis(patch_idx).activeIndex(pt, shapeFuncIdx);
            else
                activeIndex_i = electricPotential.basis(patch_idx).activeIndex(pt, shapeFuncIdx);
            int globalIndex_i = system.mapColIndex(activeIndex_i, patch_idx, di);
            //printf("bidx:%d, di:%d, globalIndex_i:%d, localRHS(di):%f\n", bidx, di, globalIndex_i, localRHS(di));
            system.pushToRhs(localRHS(di), globalIndex_i, di);
        }
#endif
    }

}

GPUElectroelasticityAssembler::GPUElectroelasticityAssembler(const MultiPatch& multiPatch,
                                const MultiBasis& displacementBasis,
                                const MultiBasis& electricPotentialBasis,
                                const BoundaryConditions& bc,
                                const Eigen::VectorXd& bodyForce)
    : GPUAssembler(multiPatch, displacementBasis, bc, bodyForce, true), m_electricPotentialBasisHost(electricPotentialBasis)
{
    m_N_P = electricPotentialBasis.numActive();
    m_elePotentialP1 = electricPotentialBasis.knotOrder() + 1;
    setDisplacementPatches(displacementBasis);
	MultiPatch hostPatch;
	electricPotentialBasis.giveBasis(hostPatch, m_electricPotentialTargetDim);
	m_electricPotentialPatch = MultiPatchDeviceData(hostPatch);

    std::vector<DofMapper> dofMappers_stdVec(targetDim() + m_electricPotentialTargetDim);
    displacementBasis.getMappers(true, boundaryConditions(), dofMappers_stdVec, true);
    electricPotentialBasis.getMapper(true, boundaryConditions(), targetDim(), dofMappers_stdVec.back(), true);
    SparseSystem sparseSystem(dofMappers_stdVec, Eigen::VectorXi::Ones(targetDim() + m_electricPotentialTargetDim));
    setupSparseSystem(sparseSystem);
    std::vector<Eigen::VectorXd> ddof(targetDim() + m_electricPotentialTargetDim);
    std::vector<Eigen::VectorXd> ddof_zero(targetDim() + m_electricPotentialTargetDim);
    for (int unk = 0; unk < targetDim(); ++unk)
    {
        computeDirichletDofs(unk, dofMappers_stdVec, ddof, displacementBasis);
        //std::cout << ddof[unk] << std::endl << std::endl;
        ddof_zero[unk] = Eigen::VectorXd::Zero(ddof[unk].size());
    } 
    for (int unk = targetDim(); unk < targetDim() + m_electricPotentialTargetDim; ++unk)
    {
        computeDirichletDofs(unk, dofMappers_stdVec, ddof, electricPotentialBasis);
        //std::cout << ddof[unk] << std::endl << std::endl;
        ddof_zero[unk] = Eigen::VectorXd::Zero(ddof[unk].size());
    }
    setDdof(ddof);
    setDdofZero(ddof_zero);

    setBasisPatches();
    setElecBasisPatches();
#if 0
    int* entryCountDevicePtr;
    cudaError_t err = cudaMalloc((void**)&entryCountDevicePtr, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA malloc failed");
    }
        err = cudaMemset(entryCountDevicePtr, 0, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memset failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(entryCountDevicePtr);
        throw std::runtime_error("CUDA memset failed");
    }
#endif
    int numActivePerBlock = std::min(16, N_D());
    int numBlocksPerElement = (N_D() + numActivePerBlock - 1) / numActivePerBlock;
    dim3 blockSize(numActivePerBlock, numActivePerBlock);
    int gridSize = numElements() * numBlocksPerElement * numBlocksPerElement;

    //int entryCountHost = numDispMatrixEntries();
    //entryCountHost += numMixMatrixEntries() * 2; 
    //entryCountHost += numEleMatrixEntries();
    int entryCountHost = numMatrixEntries();

    DeviceArray<int> cooRows(entryCountHost);
    DeviceArray<int> cooCols(entryCountHost);
    //cudaFree(entryCountDevicePtr);
    
    //computeCOO(cooRows.vectorView(), cooCols.vectorView());
    //computeMixMatrixCOO(cooRows.vectorView(), cooCols.vectorView());
    //computeEleMatrixCOO(cooRows.vectorView(), cooCols.vectorView());
    computeWholeMatrixCOO(cooRows.vectorView(), cooCols.vectorView());

    setCSRMatrixFromCOO(sparseSystem.matrixRows(), 
                        sparseSystem.matrixCols(),
                        cooRows.vectorView(), 
                        cooCols.vectorView());

    setDefaultOptions();

    computeGPTable();
    evaluateBasisValuesAndDerivativesAtGPs();
    evaluateElecBasisValuesAndDerivativesAtGPs();

    //m_elecValuesAndDerss.matrixView(m_elePotentialP1, numGPs() * 2 * domainDim()).print();
    //printDispValuesAndDerss();

    allocateGPData();
    allocateGPElecData();
}

int GPUElectroelasticityAssembler::numMatrixEntries() const
{
    int numActivePerBlock = std::min(16, N_D());
    int numBlocksPerElement = (N_D() + numActivePerBlock - 1) / numActivePerBlock;
    dim3 blockSize(numActivePerBlock, numActivePerBlock);
    int gridSize = numElements() * numBlocksPerElement * numBlocksPerElement;

    int* entryCountDevicePtr;
    cudaError_t err = cudaMalloc((void**)&entryCountDevicePtr, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA malloc failed");
    }
    err = cudaMemset(entryCountDevicePtr, 0, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memset failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(entryCountDevicePtr);
        throw std::runtime_error("CUDA memset failed");
    }

    countWholeMatrixEntrysKernel<<<gridSize, blockSize>>>(numElements(), numBlocksPerElement, numActivePerBlock,
                                                        displacementView(), 
                                                        m_electricPotentialPatch.deviceView(), 
                                                        gaussPointsView(), 
                                                        sparseSystemDeviceView(), 
                                                        allFixedDofs().view(),
                                                        entryCountDevicePtr);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(entryCountDevicePtr);
        throw std::runtime_error("CUDA kernel launch failed");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(entryCountDevicePtr);
        throw std::runtime_error("CUDA device synchronize failed");
    }
    int entryCountHost;
    err = cudaMemcpy(&entryCountHost, entryCountDevicePtr, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(entryCountDevicePtr);
        throw std::runtime_error("CUDA memcpy failed");
    }
    cudaFree(entryCountDevicePtr);
    return entryCountHost;
}

int GPUElectroelasticityAssembler::numMixMatrixEntries() const
{
    std::vector<int> numActivePerBlock = { std::min(16, N_D()), std::min(16, m_N_P) };
    std::vector<int> numBlocksPerElement = { (N_D() + numActivePerBlock[0] - 1) / numActivePerBlock[0],
                                             (m_N_P + numActivePerBlock[1] - 1) / numActivePerBlock[1] };
    dim3 blockSize(numActivePerBlock[0], numActivePerBlock[1]);
    int gridSize = numElements() * numBlocksPerElement[0] * numBlocksPerElement[1];
    DeviceArray<int> numBlocksPerElementDevice(numBlocksPerElement);
    DeviceArray<int> numActivePerBlockDevice(numActivePerBlock);

    int* entryCountDevicePtr;
    cudaError_t err = cudaMalloc((void**)&entryCountDevicePtr, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA malloc failed");
    }
    err = cudaMemset(entryCountDevicePtr, 0, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memset failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(entryCountDevicePtr);
        throw std::runtime_error("CUDA memset failed");
    }

    countMixMatrixEntrysKernel<<<gridSize, blockSize>>>(numElements(), 
                                                        numBlocksPerElementDevice.vectorView(), 
                                                        numActivePerBlockDevice.vectorView(),
                                                        displacementView(), 
                                                        m_electricPotentialPatch.deviceView(), 
                                                        gaussPointsView(), 
                                                        sparseSystemDeviceView(), 
                                                        allFixedDofs().view(),
                                                        entryCountDevicePtr);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(entryCountDevicePtr);
        throw std::runtime_error("CUDA kernel launch failed");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(entryCountDevicePtr);
        throw std::runtime_error("CUDA device synchronize failed");
    }
    int entryCountHost;
    err = cudaMemcpy(&entryCountHost, entryCountDevicePtr, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(entryCountDevicePtr);
        throw std::runtime_error("CUDA memcpy failed");
    }
    cudaFree(entryCountDevicePtr);
    return entryCountHost;
}

int GPUElectroelasticityAssembler::numEleMatrixEntries() const
{
    int numActivePerBlock = std::min(16, m_N_P);
    int numBlocksPerElement = (m_N_P + numActivePerBlock - 1) / numActivePerBlock;
    dim3 blockSize(numActivePerBlock, numActivePerBlock);
    int gridSize = numElements() * numBlocksPerElement * numBlocksPerElement;

    int* entryCountDevicePtr;
    cudaError_t err = cudaMalloc((void**)&entryCountDevicePtr, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA malloc failed");
    }
    err = cudaMemset(entryCountDevicePtr, 0, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memset failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(entryCountDevicePtr);
        throw std::runtime_error("CUDA memset failed");
    }

    countEleMatrixEntrysKernel<<<gridSize, blockSize>>>(numElements(), numBlocksPerElement, numActivePerBlock,
                                                        displacementView(), 
                                                        m_electricPotentialPatch.deviceView(), 
                                                        gaussPointsView(), 
                                                        sparseSystemDeviceView(), 
                                                        allFixedDofs().view(),
                                                        entryCountDevicePtr);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(entryCountDevicePtr);
        throw std::runtime_error("CUDA kernel launch failed");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(entryCountDevicePtr);
        throw std::runtime_error("CUDA device synchronize failed");
    }
    int entryCountHost;
    err = cudaMemcpy(&entryCountHost, entryCountDevicePtr, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(entryCountDevicePtr);
        throw std::runtime_error("CUDA memcpy failed");
    }
    cudaFree(entryCountDevicePtr);
    return entryCountHost;

}

void GPUElectroelasticityAssembler::computeWholeMatrixCOO(DeviceVectorView<int> cooRows, DeviceVectorView<int> cooCols) const
{
    int numActivePerBlock = std::min(16, N_D());
    int numBlocksPerElement = (N_D() + numActivePerBlock - 1) / numActivePerBlock;
    dim3 blockSize(numActivePerBlock, numActivePerBlock);
    int gridSize = numElements() * numBlocksPerElement * numBlocksPerElement;

    int* entryCountDevicePtr;
    cudaError_t err = cudaMalloc((void**)&entryCountDevicePtr, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA malloc failed");
    }
    err = cudaMemset(entryCountDevicePtr, 0, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memset failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(entryCountDevicePtr);
        throw std::runtime_error("CUDA memset failed");
    }

    computeWholeMatrixCOOKernel<<<gridSize, blockSize>>>(numElements(), entryCountDevicePtr, 
                                                        numBlocksPerElement, numActivePerBlock,
                                                        displacementView(), 
                                                        m_electricPotentialPatch.deviceView(), 
                                                        gaussPointsView(), 
                                                        sparseSystemDeviceView(), 
                                                        cooRows, cooCols);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(entryCountDevicePtr);
        throw std::runtime_error("CUDA kernel launch failed");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(entryCountDevicePtr);
        throw std::runtime_error("CUDA device synchronize failed");
    }
    cudaFree(entryCountDevicePtr);
}

void GPUElectroelasticityAssembler::computeMixMatrixCOO(
    DeviceVectorView<int> cooRows, DeviceVectorView<int> cooCols) const
{
    std::vector<int> numActivePerBlock = { std::min(16, N_D()), std::min(16, m_N_P) };
    std::vector<int> numBlocksPerElement = { (N_D() + numActivePerBlock[0] - 1) / numActivePerBlock[0],
                                             (m_N_P + numActivePerBlock[1] - 1) / numActivePerBlock[1] };
    dim3 blockSize(numActivePerBlock[0], numActivePerBlock[1]);
    int gridSize = numElements() * numBlocksPerElement[0] * numBlocksPerElement[1];
    DeviceArray<int> numBlocksPerElementDevice(numBlocksPerElement);
    DeviceArray<int> numActivePerBlockDevice(numActivePerBlock);

    int* counterDevicePtr;
    cudaError_t err = cudaMalloc((void**)&counterDevicePtr, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA malloc failed");
    }
    err = cudaMemset(counterDevicePtr, 0, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memset failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(counterDevicePtr);
        throw std::runtime_error("CUDA memset failed");
    }

    computeMixMatrixCOOKernel<<<gridSize, blockSize>>>(numElements(), counterDevicePtr, 
                                                        numBlocksPerElementDevice.vectorView(), 
                                                        numActivePerBlockDevice.vectorView(),
                                                        displacementView(), 
                                                        m_electricPotentialPatch.deviceView(), 
                                                        gaussPointsView(), 
                                                        sparseSystemDeviceView(), 
                                                        cooRows, cooCols);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(counterDevicePtr);
        throw std::runtime_error("CUDA kernel launch failed");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(counterDevicePtr);
        throw std::runtime_error("CUDA device synchronize failed");
    }
    cudaFree(counterDevicePtr);
}

void GPUElectroelasticityAssembler::computeEleMatrixCOO(
    DeviceVectorView<int> cooRows, DeviceVectorView<int> cooCols) const
{
    int numActivePerBlock = std::min(16, m_N_P);
    int numBlocksPerElement = (m_N_P + numActivePerBlock - 1) / numActivePerBlock;
    dim3 blockSize(numActivePerBlock, numActivePerBlock);
    int gridSize = numElements() * numBlocksPerElement * numBlocksPerElement;

    int* counterDevicePtr;
    cudaError_t err = cudaMalloc((void**)&counterDevicePtr, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA malloc failed");
    }
    err = cudaMemset(counterDevicePtr, 0, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memset failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(counterDevicePtr);
        throw std::runtime_error("CUDA memset failed");
    }

    computeEleMatrixCOOKernel<<<gridSize, blockSize>>>(numElements(), counterDevicePtr, 
                                                       numBlocksPerElement, numActivePerBlock,
                                                       displacementView(), 
                                                       m_electricPotentialPatch.deviceView(), 
                                                       gaussPointsView(), 
                                                       sparseSystemDeviceView(), 
                                                       cooRows, cooCols);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(counterDevicePtr);
        throw std::runtime_error("CUDA kernel launch failed");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(counterDevicePtr);
        throw std::runtime_error("CUDA device synchronize failed");
    }
    cudaFree(counterDevicePtr);
}

void GPUElectroelasticityAssembler::setDefaultOptions()
{
    OptionList opt;
    opt.addReal("youngs_modulus", "Young's modulus", 1.0);
    opt.addReal("poissons_ratio", "Poisson's ratio", 0.3);
    opt.addReal("dielectric_permittivity", "Dielectric permittivity of the material", 1.0);
    GPUAssembler::setDefaultOptions(opt);
}

void GPUElectroelasticityAssembler::evaluateElecBasisValuesAndDerivativesAtGPs()
{ 
    m_elecValuesAndDerss.resize(m_elePotentialP1 * numGPs() * 2 * domainDim()); 
    DeviceArray<double> elecWorkingSpaces(numGPs() * m_elePotentialP1 * (m_elePotentialP1 + 4) * domainDim());
    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, evaluateEleBasisValuesAndDerivativesAtGPsKernel, 0, numGPs() * domainDim());
    int gridSize = (numGPs() * domainDim() + blockSize - 1) / blockSize;
    evaluateEleBasisValuesAndDerivativesAtGPsKernel<<<gridSize, blockSize>>>(
        1, numGPs(), domainDim(), 
        displacementView(), 
        m_electricPotentialPatch.deviceView(), 
        gpTable(), 
        elecWorkingSpaces.vectorView(), 
        m_elecValuesAndDerss.matrixView(m_elePotentialP1, numGPs() * 2 * domainDim()));
}

void GPUElectroelasticityAssembler::allocateGPElecData()
{ m_GPData_As.resize(numGPs() * (dimTensor() * domainDim() + domainDim() + 1 + domainDim() * domainDim())); }

void GPUElectroelasticityAssembler::constructElecSolution(const DeviceVectorView<double> &solVector, const DeviceNestedArrayView<double> &fixedDoFs) const
{
    int minGrid, blockSize;
    int CPSize = totalNumControlPoints();
    cudaOccupancyMaxPotentialBlockSize(&minGrid, 
        &blockSize, constructElecSolutionKernel, 0, CPSize);
    int gridSize = (CPSize + blockSize - 1) / blockSize;
    constructElecSolutionKernel<<<gridSize, blockSize>>>(
        solVector, fixedDoFs, 
        multiBasisDeviceView(), 
        sparseSystemDeviceView(), 
        m_electricPotentialPatch.deviceView(), CPSize);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
}

void GPUElectroelasticityAssembler::assemble(
    const DeviceVectorView<double> &solVector, int numIter, const DeviceNestedArrayView<double> &fixedDoFs)
{
    setMatrixAndRHSZeros();
    //printf("Solution vector:\n");
    //solVector.print();
    constructDispSolution(solVector, fixedDoFs);
    constructElecSolution(solVector, fixedDoFs);
    DeviceNestedArrayView<double> fixedDofs_assemble;
    getFixedDofsForAssemble(numIter, fixedDofs_assemble);
    DeviceArray<double> parameterValues(options().realValues());
    setGPDataZeros();
    m_GPData_As.setZero();
    int dim = domainDim();
    int totalGPs = numGPs();
    int dimTnsr = dimTensor();
    int offset = 0;
    double * GPData = GPDataPtr();
    DeviceMatrixView<double> geoJacobianInvs(GPData + offset, dim, totalGPs * dim);
    offset += geoJacobianInvs.size();
    DeviceVectorView<double> measures(GPData + offset, totalGPs);
    offset += measures.size();
    DeviceVectorView<double> weightForces(GPData + offset, totalGPs);
    offset += weightForces.size();
    DeviceVectorView<double> weightBodys(GPData + offset, totalGPs);
    offset += weightBodys.size();
    DeviceMatrixView<double> Fs(GPData + offset, dim, totalGPs * dim);
    offset += Fs.size();
    DeviceMatrixView<double> Ss(GPData + offset, dim, totalGPs * dim);
    offset += Ss.size();
    DeviceMatrixView<double> Cs(GPData + offset, dimTnsr, totalGPs * dimTnsr);
    offset = 0;
    DeviceMatrixView<double> As(m_GPData_As.data(), dimTnsr, totalGPs * dim);
    offset += As.size();
    DeviceMatrixView<double> elecDisp(m_GPData_As.data() + offset, dim, totalGPs);
    offset += elecDisp.size();
    DeviceVectorView<double> Js(m_GPData_As.data() + offset, totalGPs);
    offset += Js.size();
    DeviceMatrixView<double> RCGinv(m_GPData_As.data() + offset, dim, totalGPs * dim);

    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, evaluateGPKernel, 0, totalGPs);
    int gridSize = (totalGPs + blockSize - 1) / blockSize;
    evaluateGPKernel<<<gridSize, blockSize>>>(numDerivatives(), totalGPs, parameterValues.vectorView(), displacementView(), m_electricPotentialPatch.deviceView(), geometryView(), gpTable(), wts().vectorView(), geoValuesAndDerssView(), dispValuesAndDerssView(), m_elecValuesAndDerss.matrixView(m_elePotentialP1, numGPs() * 2 * domainDim()), geoJacobianInvs, measures, weightForces, weightBodys, Js, Fs, Ss, Cs, RCGinv, elecDisp, As);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronize failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA device synchronize failed");
    }

    //Ss.print();

    //assembleMatrix(fixedDofs_assemble, geoJacobianInvs, measures, weightForces, weightBodys, Fs, Ss, Cs);
    //printCSRMatrix();

    blockSize = N_D();
    gridSize = N_D() * N_D() * numElements();
    assembleMatrixKernel<<<gridSize, blockSize>>>(numDerivatives(), 0, numElements(), N_D(), parameterValues.vectorView(), displacementView(), m_electricPotentialPatch.deviceView(), sparseSystemDeviceView(), fixedDofs_assemble, gpTable(), geoJacobianInvs, measures, weightForces, weightBodys, Js, dispValuesAndDerssView(), m_elecValuesAndDerss.matrixView(m_elePotentialP1, numGPs() * 2 * domainDim()), Fs, Ss, Cs, RCGinv, As);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronize failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA device synchronize failed");
    }

    //printCSRMatrix();

    gridSize = N_D() * numElements();
    assembleRHSKernel<<<gridSize, blockSize>>>(numDerivatives(), 0, numElements(), N_D(), displacementView(), m_electricPotentialPatch.deviceView(), sparseSystemDeviceView(), gpTable(), geoJacobianInvs, weightForces, weightBodys, dispValuesAndDerssView(), m_elecValuesAndDerss.matrixView(m_elePotentialP1, numGPs() * 2 * domainDim()), elecDisp, bodyForce().vectorView(), Fs, Ss);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronize failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA device synchronize failed");
    }

    //std::cout << "RHS:\n" << hostRHS() << std::endl; 
}

void GPUElectroelasticityAssembler::setElecBasisPatches()
{
    basisHost().giveBasis(m_electricPotentialPatchHost, m_electricPotentialTargetDim);
    m_electricPotentialPatch = MultiPatchDeviceData(m_electricPotentialPatchHost);
}

void GPUElectroelasticityAssembler::refreshFixedDofs()
{
    std::vector<DofMapper> dofMappers_stdVec(targetDim() + m_electricPotentialTargetDim);
    basisHost().getMappers(true, boundaryConditions(), dofMappers_stdVec, true);
    m_electricPotentialBasisHost.getMapper(true, boundaryConditions(), targetDim(), dofMappers_stdVec.back(), true);
    std::vector<Eigen::VectorXd> ddof(targetDim() + m_electricPotentialTargetDim);
    for (int unk = 0; unk < targetDim(); ++unk)
        computeDirichletDofs(unk, dofMappers_stdVec, ddof, basisHost());
    for (int unk = targetDim(); unk < targetDim() + m_electricPotentialTargetDim; ++unk)
        computeDirichletDofs(unk, dofMappers_stdVec, ddof, m_electricPotentialBasisHost);
    setDdof(ddof);
}
