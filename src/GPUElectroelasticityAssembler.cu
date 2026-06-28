#include "GPUElectroelasticityAssembler.h"

#include <GPUAssemblySupport.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <memory>
#include <thread>

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
void evaluateGPKernel(int numDerivatives, int GPStartId,
                      int inputGPStartId, int numGPBatched,
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
    //electricPotential.patch(0).print();
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numGPBatched; idx += blockDim.x * gridDim.x)
    {
        int GPIdx = GPStartId + idx;
        int inputGPIdx = inputGPStartId + idx;
        int dim = multiPatch.domainDim();
        int dimTensor = (dim * (dim + 1)) / 2;

        //int GPIdx = idx;
        DeviceVectorView<double> pt(pts.data() + inputGPIdx * dim, dim);
        double wt = wts[inputGPIdx];
        
        int patch_idx(0);
        int point_idx = displacement.threadPatch(GPIdx, patch_idx);
        PatchDeviceView geoPatch = multiPatch.patch(patch_idx);
        int geoP1 = geoPatch.basis().knotsOrder(0) + 1;
        DeviceMatrixView<double> geoValuesAndDers(geoValuesAndDerss.data() + inputGPIdx * geoP1 * (numDerivatives + 1) * dim, geoP1, (numDerivatives + 1) * dim);

        PatchDeviceView dispPatch = displacement.patch(patch_idx);
        PatchDeviceView elecPatch = electricPotential.patch(patch_idx);
        //elecPatch.print();
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
        int P1 = dispBasis.knotsOrder(0) + 1;
        DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDerss.data() + inputGPIdx * P1 * (numDerivatives + 1) * dim, P1, (numDerivatives + 1) * dim);
        TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patch_idx);
        int eleP1 = elecBasis.knotsOrder(0) + 1;
        DeviceMatrixView<double> elecValuesAndDers(elecValuesAndDerss.data() + inputGPIdx * eleP1 * (numDerivatives + 1) * dim, eleP1, (numDerivatives + 1) * dim);

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
            //printf("geo JacobianInv at GP %d:\n", idx);
            //geoJacobianInv.print();
            //printf("electric jacobian at GP %d:\n", idx);
            //elecJacobian.print();
        }
        //printf("Electric field at GP %d:\n", idx);
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
                          int inputGPStartId, int numElementsBatched, int N_D,
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
            int inputGPIdx = inputGPStartId + GPIdx;
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
            DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDerss.data() + inputGPIdx * P1 * (numDerivatives + 1) * dim, P1, (numDerivatives + 1) * dim);
            DeviceMatrixView<double> elecValuesAndDers(elecValuesAndDerss.data() + inputGPIdx * elecP1 * (numDerivatives + 1) * dim, elecP1, (numDerivatives + 1) * dim);
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
                    stiffnessEntryData *= - weightBody;
                    atomicAdd(&localMatrix(di, dim), stiffnessEntryData);
                    //atomicAdd(&localMatrix(dim, di), stiffnessEntryData);
                }
                //materialTangentTemp.print();
                for (int dj = 0; dj < dim; dj++){
                    double materialTangent = 0;
                    //if(blockIdx.x == 334 && threadId == 32)
                    //    printf("materialTangent: %f\n", materialTangent);
                    DeviceMatrixView<double> materialTangentMat(&materialTangent, 1, 1);
                    double B_j_djData[6] = {0.0}; //max 3D
                    DeviceVectorView<double> B_j_dj(B_j_djData, dimTensor);
                    setBSingleDim<double>(dj, B_j_dj, F, physGrad_j);
                    materialTangentTemp.times(B_j_dj, materialTangentMat);
                    if (di == dj)
                        materialTangent += geometricTangent;
                    double stiffnessEntry = weightBody *materialTangent;
                    atomicAdd(&localMatrix(di, dj), stiffnessEntry);
                    if (di == 0) {
                        double tempData[6] = {0.0}; //max 3D
                        DeviceMatrixView<double> temp(tempData, dimTensor, 1);
                        //printf("dj = %d\n\n", dj);
                        //physPGrad_i.print();
                        //printf("\n");
                        //A.print();
                        //printf("\n");
                        A.times(physPGrad_i, temp);
                        //temp.print();
                        //printf("\n");
                        double mixStiffnessEntryData = 0.0;
                        DeviceMatrixView<double> mixStiffnessEntry(&mixStiffnessEntryData, 1, 1);
                        temp.transposeTime(B_j_dj, mixStiffnessEntry);
                        //printf("%f\n\n", mixStiffnessEntryData);
                        mixStiffnessEntryData *= - weightBody;
                        atomicAdd(&localMatrix(dim, dj), mixStiffnessEntryData);
                    }
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
        int inputGPIdx = inputGPStartId + GPIdx;
        DeviceVectorView<double> pt(pts.data() + inputGPIdx * dim, dim);
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
                       int inputGPStartId, int numElementsBatched, int N_D,
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
            int inputGPIdx = inputGPStartId + GPIdx;
            DeviceMatrixView<double> geoJacobianInv(geoJacobianInvs.data() + GPIdx * dim * dim, dim, dim);
            double weightForce = weightForces[GPIdx];
            double weightBody = weightBodys[GPIdx];
            //printf("weightBody = %f\n", weightBody);
            DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDerss.data() + inputGPIdx * P1 * (numDerivatives + 1) * dim, P1, (numDerivatives + 1) * dim);
            DeviceMatrixView<double> elecValuesAndDers(elecValuesAndDerss.data() + inputGPIdx * elecP1 * (numDerivatives + 1) * dim, elecP1, (numDerivatives + 1) * dim);
            DeviceVectorView<double> elecDisp(elecDisps.data() + GPIdx * dim, dim);
            //printf("elecDisp at GP %d:\n", GPIdx);
            //elecDisp.print();
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
            //printf("physGrad_elec_i at GP %d:\n", GPIdx);
            //physGrad_elec_i.print();
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
        int inputGPIdx = inputGPStartId + GPIdx;
        DeviceVectorView<double> pt(pts.data() + inputGPIdx * dim, dim);
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
    opt.addReal("neumann_load_scaling", "Multiplier for Neumann boundary and corner loads", 1.0);
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
    const std::vector<double> parameterValuesHost = options().realValues();
    DeviceArray<double> parameterValues(parameterValuesHost);
    const bool enableMultiGPU =
        siga::gpuasm::envFlag("SIGA_MULTIGPU_ASSEMBLY", false);
    const bool enableStreamingAssembly = siga::gpuasm::envFlag(
        "SIGA_ELECTRO_STREAMING_ASSEMBLY",
        siga::gpuasm::envFlag("SIGA_GPU_STREAMING_ASSEMBLY", enableMultiGPU));
    if (enableStreamingAssembly)
    {
        if (m_GPData_As.size() != 0)
            m_GPData_As.resize(0);

        const bool reportMemory = siga::gpuasm::envFlag(
            "SIGA_ELECTRO_MEMORY_REPORT",
            siga::gpuasm::envFlag("SIGA_GPU_MEMORY_REPORT", true));
        const bool replicateSecondaryInputs = siga::gpuasm::envFlag(
            "SIGA_ELECTRO_REPLICATE_INPUTS",
            siga::gpuasm::envFlag("SIGA_GPU_REPLICATE_INPUTS", true));
        const bool useLocalCSRAssembly = siga::gpuasm::envFlag(
            "SIGA_ELECTRO_LOCAL_CSR_ASSEMBLY",
            siga::gpuasm::envFlag("SIGA_GPU_LOCAL_CSR_ASSEMBLY", true));

        int primaryDevice = 0;
        cudaError_t err = cudaGetDevice(&primaryDevice);
        if (err != cudaSuccess)
            throw std::runtime_error(
                "cudaGetDevice failed before electroelastic assembly");

        std::vector<int> assemblyDevices =
            siga::gpuasm::usableAssemblyDevices(enableMultiGPU);
        siga::gpuasm::printDeviceSelection("Electroelastic assembly",
                                           assemblyDevices,
                                           enableMultiGPU);
        const int numAssemblyDevices =
            static_cast<int>(assemblyDevices.size());
        if (numElements() <= 0 || numGPs() % numElements() != 0)
            throw std::runtime_error(
                "Electroelastic streaming assembly requires a uniform positive Gauss-point count per element");
        const int gpsPerElement = numGPs() / numElements();
        const int dim = domainDim();
        const int dimTnsr = dimTensor();
        const int baseStride = numDoublesPerGP();
        const int electroStride = dimTnsr * dim + dim + 1 + dim * dim;
        const int matrixValuesSize = csrMatrix().numNonZeros();
        const int rhsSize = numDofs();
        const int numFields = targetDim() + m_electricPotentialTargetDim;

        const SparseSystemDeviceView primarySystemView =
            sparseSystemDeviceView();
        const MultiPatchDeviceView primaryGeometryView = geometryView();
        const MultiPatchDeviceView primaryDisplacementView =
            displacementView();
        const MultiPatchDeviceView primaryElectricView =
            m_electricPotentialPatch.deviceView();
        const DeviceMatrixView<double> primaryGPTableView = gpTable();
        const DeviceVectorView<double> primaryWeightsView =
            wts().vectorView();
        const DeviceMatrixView<double> primaryGeoValuesView =
            geoValuesAndDerssView();
        const DeviceMatrixView<double> primaryDispValuesView =
            dispValuesAndDerssView();
        const DeviceMatrixView<double> primaryElecValuesView =
            m_elecValuesAndDerss.matrixView(
                m_elePotentialP1,
                numGPs() * (numDerivatives() + 1) * domainDim());
        const DeviceVectorView<double> primaryBodyForceView =
            bodyForce().vectorView();

        const int oneShotChunkElements =
            (numElements() + numAssemblyDevices - 1) / numAssemblyDevices;
        int requestedChunkElementLimit = siga::gpuasm::envInt(
            "SIGA_ELECTRO_CHUNK_ELEMENTS",
            siga::gpuasm::envInt("SIGA_GPU_CHUNK_ELEMENTS",
                                 oneShotChunkElements));
        if (requestedChunkElementLimit <= 0)
            requestedChunkElementLimit = oneShotChunkElements;
        requestedChunkElementLimit =
            std::min(numElements(),
                     std::max(1, requestedChunkElementLimit));
        double memorySafetyFraction = siga::gpuasm::envDouble(
            "SIGA_ELECTRO_MEMORY_FRACTION",
            siga::gpuasm::envDouble("SIGA_GPU_MEMORY_FRACTION", 0.90));
        memorySafetyFraction =
            std::min(1.0, std::max(0.10, memorySafetyFraction));

        auto staticInputBytes = [&](int gpCount)
        {
            unsigned long long bytes =
                siga::gpuasm::multiPatchReplicaBytes(primaryGeometryView) +
                siga::gpuasm::multiPatchReplicaBytes(primaryDisplacementView) +
                siga::gpuasm::multiPatchReplicaBytes(primaryElectricView) +
                siga::gpuasm::bytesForCount(
                    static_cast<long long>(gpCount) *
                        primaryGPTableView.rows(),
                    sizeof(double)) +
                siga::gpuasm::bytesForCount(gpCount, sizeof(double)) +
                siga::gpuasm::vectorBytes(primaryBodyForceView);
            if (numGPs() > 0)
            {
                bytes += siga::gpuasm::bytesForCount(
                    static_cast<long long>(gpCount) *
                        primaryGeoValuesView.size() / numGPs(),
                    sizeof(double));
                bytes += siga::gpuasm::bytesForCount(
                    static_cast<long long>(gpCount) *
                        primaryDispValuesView.size() / numGPs(),
                    sizeof(double));
                bytes += siga::gpuasm::bytesForCount(
                    static_cast<long long>(gpCount) *
                        primaryElecValuesView.size() / numGPs(),
                    sizeof(double));
            }
            return bytes;
        };

        auto requiredBytesForDevice =
            [&](int idx, int maxGPCount, int maxRowCount,
                int maxMatrixValueCount,
                std::vector<std::pair<std::string, unsigned long long>>* parts)
        {
            const unsigned long long matrixOutputBytes =
                idx == 0 ? 0
                         : siga::gpuasm::bytesForCount(maxMatrixValueCount,
                                                       sizeof(double));
            const unsigned long long rhsOutputBytes =
                idx == 0 ? 0
                         : siga::gpuasm::bytesForCount(maxRowCount,
                                                       sizeof(double));
            const unsigned long long materialBytes =
                siga::gpuasm::bytesForCount(parameterValuesHost.size(),
                                            sizeof(double));
            const unsigned long long baseGPBytes =
                siga::gpuasm::bytesForCount(
                    static_cast<long long>(baseStride) * maxGPCount,
                    sizeof(double));
            const unsigned long long electroGPBytes =
                siga::gpuasm::bytesForCount(
                    static_cast<long long>(electroStride) * maxGPCount,
                    sizeof(double));
            unsigned long long sparseBytes = 0;
            unsigned long long staticBytes = 0;
            unsigned long long fixedDofsBytes = 0;
            if (idx != 0)
            {
                sparseBytes = siga::gpuasm::sparseMetadataBytes(
                    primarySystemView, maxRowCount + 1,
                    maxMatrixValueCount);
                if (replicateSecondaryInputs)
                {
                    staticBytes = staticInputBytes(maxGPCount);
                    fixedDofsBytes =
                        siga::gpuasm::nestedArrayBytes(fixedDofs_assemble);
                }
            }
            if (parts)
            {
                *parts = {
                    {"matrix output buffer", matrixOutputBytes},
                    {"RHS output buffer", rhsOutputBytes},
                    {"material parameters", materialBytes},
                    {"electroelastic base GP data", baseGPBytes},
                    {"electroelastic coupled GP data", electroGPBytes},
                    {"sparse metadata and local CSR", sparseBytes},
                    {"replicated static input data", staticBytes},
                    {"fixed dofs replica", fixedDofsBytes}
                };
            }
            return matrixOutputBytes + rhsOutputBytes + materialBytes +
                   baseGPBytes + electroGPBytes + sparseBytes + staticBytes +
                   fixedDofsBytes;
        };

        siga::gpuasm::AssemblySchedule schedule;
        int chunkElementLimit = requestedChunkElementLimit;
        bool scheduleFitsMemory = false;
        for (int attempt = 0; attempt < 32; ++attempt)
        {
            cudaSetDevice(primaryDevice);
            schedule = siga::gpuasm::buildAssemblySchedule(
                chunkElementLimit, numElements(), gpsPerElement,
                numAssemblyDevices, N_D(), numFields, rhsSize,
                matrixValuesSize,
                useLocalCSRAssembly && numAssemblyDevices > 1,
                primarySystemView, primaryDisplacementView,
                primaryElectricView, primaryGPTableView);

            bool fitsMemory = true;
            double worstRatio = 1.0;
            int worstDeviceIdx = -1;
            unsigned long long worstRequired = 0;
            unsigned long long worstAvailable = 0;
            for (int idx = 0; idx < numAssemblyDevices; ++idx)
            {
                schedule.requiredBytes[idx] = requiredBytesForDevice(
                    idx, schedule.maxGPCounts[idx],
                    schedule.maxRowCounts[idx],
                    schedule.maxMatrixValueCounts[idx], nullptr);
                cudaSetDevice(assemblyDevices[idx]);
                size_t freeMem = 0;
                size_t totalMem = 0;
                err = cudaMemGetInfo(&freeMem, &totalMem);
                if (err != cudaSuccess)
                    throw std::runtime_error(
                        std::string("cudaMemGetInfo failed while sizing electroelastic assembly chunks: ") +
                        cudaGetErrorString(err));
                const unsigned long long available =
                    static_cast<unsigned long long>(
                        static_cast<double>(freeMem) * memorySafetyFraction);
                const unsigned long long required =
                    schedule.requiredBytes[idx];
                if (required > available && chunkElementLimit > 1)
                {
                    fitsMemory = false;
                    const double ratio =
                        static_cast<double>(available) /
                        static_cast<double>(std::max(1ULL, required));
                    if (ratio < worstRatio)
                    {
                        worstRatio = ratio;
                        worstDeviceIdx = idx;
                        worstRequired = required;
                        worstAvailable = available;
                    }
                }
                else if (required > available)
                {
                    throw std::runtime_error(
                        "Electroelastic assembly cannot fit even a one-element chunk on CUDA device " +
                        std::to_string(assemblyDevices[idx]) +
                        ". Required " + siga::gpuasm::gibString(required) +
                        ", available with safety margin " +
                        siga::gpuasm::gibString(available) + ".");
                }
            }
            cudaSetDevice(primaryDevice);
            if (fitsMemory)
            {
                scheduleFitsMemory = true;
                break;
            }
            int nextChunkElementLimit = static_cast<int>(
                std::floor(chunkElementLimit * worstRatio * 0.95));
            if (nextChunkElementLimit >= chunkElementLimit)
                nextChunkElementLimit = chunkElementLimit - 1;
            if (nextChunkElementLimit < 1)
                nextChunkElementLimit = 1;
            if (reportMemory)
            {
                std::cout << "Electroelastic adaptive assembly chunking: chunk limit "
                          << chunkElementLimit << " elements needs "
                          << siga::gpuasm::gibString(worstRequired)
                          << " on device " << assemblyDevices[worstDeviceIdx]
                          << ", memory target "
                          << siga::gpuasm::gibString(worstAvailable)
                          << "; reducing to " << nextChunkElementLimit
                          << " elements/chunk\n";
            }
            chunkElementLimit = nextChunkElementLimit;
        }
        if (!scheduleFitsMemory)
            throw std::runtime_error(
                "Electroelastic adaptive assembly chunking could not find a memory-fitting chunk size.");
        if (reportMemory)
        {
            int totalChunks = 0;
            for (const auto& chunks : schedule.chunksByDevice)
                totalChunks += static_cast<int>(chunks.size());
            std::cout << "Electroelastic adaptive assembly schedule: "
                      << totalChunks << " chunks, " << schedule.rounds
                      << " rounds, max " << schedule.chunkElementLimit
                      << " elements/chunk";
            if (schedule.chunkElementLimit < oneShotChunkElements)
                std::cout << " (streaming enabled)";
            std::cout << "\n";
        }

        struct ElectroGPViews
        {
            DeviceMatrixView<double> geoJacobianInvs;
            DeviceVectorView<double> measures;
            DeviceVectorView<double> weightForces;
            DeviceVectorView<double> weightBodys;
            DeviceMatrixView<double> Fs;
            DeviceMatrixView<double> Ss;
            DeviceMatrixView<double> Cs;
            DeviceMatrixView<double> As;
            DeviceMatrixView<double> elecDisp;
            DeviceVectorView<double> Js;
            DeviceMatrixView<double> RCGinv;
        };

        auto makeGPViews = [&](double* baseData, double* electroData,
                               int gpCount)
        {
            ElectroGPViews views;
            int offset = 0;
            views.geoJacobianInvs =
                DeviceMatrixView<double>(baseData + offset, dim,
                                         gpCount * dim);
            offset += views.geoJacobianInvs.size();
            views.measures =
                DeviceVectorView<double>(baseData + offset, gpCount);
            offset += views.measures.size();
            views.weightForces =
                DeviceVectorView<double>(baseData + offset, gpCount);
            offset += views.weightForces.size();
            views.weightBodys =
                DeviceVectorView<double>(baseData + offset, gpCount);
            offset += views.weightBodys.size();
            views.Fs =
                DeviceMatrixView<double>(baseData + offset, dim,
                                         gpCount * dim);
            offset += views.Fs.size();
            views.Ss =
                DeviceMatrixView<double>(baseData + offset, dim,
                                         gpCount * dim);
            offset += views.Ss.size();
            views.Cs =
                DeviceMatrixView<double>(baseData + offset, dimTnsr,
                                         gpCount * dimTnsr);

            offset = 0;
            views.As =
                DeviceMatrixView<double>(electroData + offset, dimTnsr,
                                         gpCount * dim);
            offset += views.As.size();
            views.elecDisp =
                DeviceMatrixView<double>(electroData + offset, dim, gpCount);
            offset += views.elecDisp.size();
            views.Js = DeviceVectorView<double>(electroData + offset, gpCount);
            offset += views.Js.size();
            views.RCGinv =
                DeviceMatrixView<double>(electroData + offset, dim,
                                         gpCount * dim);
            return views;
        };

        struct ElectroAssemblyDeviceBuffer
        {
            siga::gpuasm::SparseOutputBuffer output;
            DeviceArray<double> materialParameters;
            DeviceArray<double> baseGPData;
            DeviceArray<double> electroGPData;
            siga::gpuasm::MultiPatchReplica geometry;
            siga::gpuasm::MultiPatchReplica displacement;
            siga::gpuasm::MultiPatchReplica electricPotential;
            DeviceArray<double> gpTable;
            DeviceArray<double> weights;
            DeviceArray<double> geoValuesAndDerss;
            DeviceArray<double> dispValuesAndDerss;
            DeviceArray<double> elecValuesAndDerss;
            DeviceArray<double> bodyForce;
            siga::gpuasm::NestedArrayReplica<double> fixedDofs;

            ElectroAssemblyDeviceBuffer(
                int matrixSize, int rhsSize, int baseGPDataSize,
                int electroGPDataSize,
                const std::vector<double>& materialParametersHost)
                : output(matrixSize, rhsSize),
                  materialParameters(materialParametersHost),
                  baseGPData(baseGPDataSize),
                  electroGPData(electroGPDataSize)
            {
            }

            void copyStaticModelData(MultiPatchDeviceView geometryView,
                                     MultiPatchDeviceView displacementView,
                                     MultiPatchDeviceView electricView,
                                     DeviceVectorView<double> bodyForceView,
                                     int sourceDevice, int targetDevice)
            {
                geometry.updateStaticData(geometryView, sourceDevice,
                                          targetDevice, "geometry");
                displacement.updateStaticData(displacementView, sourceDevice,
                                              targetDevice, "displacement");
                electricPotential.updateStaticData(electricView, sourceDevice,
                                                   targetDevice,
                                                   "electric potential");
                siga::gpuasm::peerCopyInto(bodyForce, bodyForceView,
                                           sourceDevice, targetDevice,
                                           "body force");
            }

            void copyStaticChunkInputData(
                DeviceMatrixView<double> gpTableView,
                DeviceVectorView<double> weightsView,
                DeviceMatrixView<double> geoValuesView,
                DeviceMatrixView<double> dispValuesView,
                DeviceMatrixView<double> elecValuesView, int totalGPCount,
                int gpStart, int gpCount, int sourceDevice, int targetDevice)
            {
                const int gpTableStride = gpTableView.rows();
                siga::gpuasm::peerCopySliceInto(
                    gpTable, gpTableView.data() + gpStart * gpTableStride,
                    gpCount * gpTableStride, sourceDevice, targetDevice,
                    "local Gauss-point table");
                siga::gpuasm::peerCopySliceInto(
                    weights, weightsView.data() + gpStart, gpCount,
                    sourceDevice, targetDevice, "local Gauss weights");
                const int geoStride = geoValuesView.size() / totalGPCount;
                const int dispStride = dispValuesView.size() / totalGPCount;
                const int elecStride = elecValuesView.size() / totalGPCount;
                siga::gpuasm::peerCopySliceInto(
                    geoValuesAndDerss,
                    geoValuesView.data() + gpStart * geoStride,
                    gpCount * geoStride, sourceDevice, targetDevice,
                    "local geometry values and derivatives");
                siga::gpuasm::peerCopySliceInto(
                    dispValuesAndDerss,
                    dispValuesView.data() + gpStart * dispStride,
                    gpCount * dispStride, sourceDevice, targetDevice,
                    "local displacement values and derivatives");
                siga::gpuasm::peerCopySliceInto(
                    elecValuesAndDerss,
                    elecValuesView.data() + gpStart * elecStride,
                    gpCount * elecStride, sourceDevice, targetDevice,
                    "local electric values and derivatives");
            }

            void updateDynamicInputData(
                MultiPatchDeviceView displacementView,
                MultiPatchDeviceView electricView,
                DeviceNestedArrayView<double> fixedDofsView,
                int sourceDevice, int targetDevice)
            {
                displacement.updateControlPoints(displacementView,
                                                 sourceDevice, targetDevice,
                                                 "displacement");
                electricPotential.updateControlPoints(electricView,
                                                      sourceDevice,
                                                      targetDevice,
                                                      "electric potential");
                fixedDofs.update(fixedDofsView, sourceDevice, targetDevice,
                                 "fixed dofs");
            }
        };

        std::vector<std::unique_ptr<ElectroAssemblyDeviceBuffer>> buffers(
            numAssemblyDevices);
        for (int idx = 0; idx < numAssemblyDevices; ++idx)
        {
            cudaSetDevice(assemblyDevices[idx]);
            const int localMatrixValuesSize =
                idx == 0 ? 0 : schedule.maxMatrixValueCounts[idx];
            const int localRhsSize =
                idx == 0 ? 0 : schedule.maxRowCounts[idx];
            const int localBaseGPDataSize =
                baseStride * schedule.maxGPCounts[idx];
            const int localElectroGPDataSize =
                electroStride * schedule.maxGPCounts[idx];
            std::vector<std::pair<std::string, unsigned long long>> parts;
            const unsigned long long requiredBytes = requiredBytesForDevice(
                idx, schedule.maxGPCounts[idx],
                schedule.maxRowCounts[idx],
                schedule.maxMatrixValueCounts[idx], &parts);
            std::ostringstream label;
            label << "electroelastic assembly streaming buffer for GPU "
                  << idx << " (max elements "
                  << schedule.maxElementCounts[idx] << ", max GP "
                  << schedule.maxGPCounts[idx] << ")";
            siga::gpuasm::printCudaMemoryReport(label.str(), requiredBytes,
                                                reportMemory, parts);
            buffers[idx] = std::make_unique<ElectroAssemblyDeviceBuffer>(
                localMatrixValuesSize, localRhsSize, localBaseGPDataSize,
                localElectroGPDataSize, parameterValuesHost);
            if (idx != 0)
            {
                buffers[idx]->output.copySparseBaseMetadata(
                    primarySystemView, primaryDevice, assemblyDevices[idx]);
                if (replicateSecondaryInputs)
                    buffers[idx]->copyStaticModelData(
                        primaryGeometryView, primaryDisplacementView,
                        primaryElectricView, primaryBodyForceView,
                        primaryDevice, assemblyDevices[idx]);
            }
        }

        if (replicateSecondaryInputs)
        {
            for (int idx = 1; idx < numAssemblyDevices; ++idx)
            {
                cudaSetDevice(assemblyDevices[idx]);
                buffers[idx]->updateDynamicInputData(
                    primaryDisplacementView, primaryElectricView,
                    fixedDofs_assemble, primaryDevice, assemblyDevices[idx]);
            }
        }
        cudaSetDevice(primaryDevice);

        const int gpTableColsPerGP = primaryGPTableView.cols() / numGPs();
        const int geoValuesColsPerGP =
            primaryGeoValuesView.cols() / numGPs();
        const int dispValuesColsPerGP =
            primaryDispValuesView.cols() / numGPs();
        const int elecValuesColsPerGP =
            primaryElecValuesView.cols() / numGPs();

        auto inputGPStartForDevice =
            [&](int idx, const siga::gpuasm::AssemblyChunk& chunk)
        {
            return (idx != 0 && replicateSecondaryInputs) ? 0
                                                          : chunk.gpStart;
        };
        auto systemViewForDevice = [&](int idx)
        {
            if (idx == 0)
                return primarySystemView;
            return buffers[idx]->output.sparseSystemView(primarySystemView);
        };
        auto geometryViewForDevice = [&](int idx)
        {
            if (idx == 0 || !replicateSecondaryInputs)
                return primaryGeometryView;
            return buffers[idx]->geometry.view();
        };
        auto displacementViewForDevice = [&](int idx)
        {
            if (idx == 0 || !replicateSecondaryInputs)
                return primaryDisplacementView;
            return buffers[idx]->displacement.view();
        };
        auto electricViewForDevice = [&](int idx)
        {
            if (idx == 0 || !replicateSecondaryInputs)
                return primaryElectricView;
            return buffers[idx]->electricPotential.view();
        };
        auto gpTableViewForDevice =
            [&](int idx, const siga::gpuasm::AssemblyChunk& chunk)
        {
            if (idx == 0 || !replicateSecondaryInputs)
                return primaryGPTableView;
            return DeviceMatrixView<double>(
                buffers[idx]->gpTable.data(), primaryGPTableView.rows(),
                chunk.gpCount * gpTableColsPerGP);
        };
        auto weightsViewForDevice = [&](int idx)
        {
            if (idx == 0 || !replicateSecondaryInputs)
                return primaryWeightsView;
            return buffers[idx]->weights.vectorView();
        };
        auto geoValuesViewForDevice =
            [&](int idx, const siga::gpuasm::AssemblyChunk& chunk)
        {
            if (idx == 0 || !replicateSecondaryInputs)
                return primaryGeoValuesView;
            return DeviceMatrixView<double>(
                buffers[idx]->geoValuesAndDerss.data(),
                primaryGeoValuesView.rows(),
                chunk.gpCount * geoValuesColsPerGP);
        };
        auto dispValuesViewForDevice =
            [&](int idx, const siga::gpuasm::AssemblyChunk& chunk)
        {
            if (idx == 0 || !replicateSecondaryInputs)
                return primaryDispValuesView;
            return DeviceMatrixView<double>(
                buffers[idx]->dispValuesAndDerss.data(),
                primaryDispValuesView.rows(),
                chunk.gpCount * dispValuesColsPerGP);
        };
        auto elecValuesViewForDevice =
            [&](int idx, const siga::gpuasm::AssemblyChunk& chunk)
        {
            if (idx == 0 || !replicateSecondaryInputs)
                return primaryElecValuesView;
            return DeviceMatrixView<double>(
                buffers[idx]->elecValuesAndDerss.data(),
                primaryElecValuesView.rows(),
                chunk.gpCount * elecValuesColsPerGP);
        };
        auto bodyForceViewForDevice = [&](int idx)
        {
            if (idx == 0 || !replicateSecondaryInputs)
                return primaryBodyForceView;
            return buffers[idx]->bodyForce.vectorView();
        };
        auto fixedDofsViewForDevice = [&](int idx)
        {
            if (idx == 0 || !replicateSecondaryInputs)
                return fixedDofs_assemble;
            return buffers[idx]->fixedDofs.view();
        };

        auto prepareChunkForDevice =
            [&](int idx, const siga::gpuasm::AssemblyChunk& chunk)
        {
            cudaSetDevice(assemblyDevices[idx]);
            if (idx == 0)
                return;
            ElectroAssemblyDeviceBuffer& buffer = *buffers[idx];
            buffer.output.updateLocalSparseWindow(
                primarySystemView, chunk.rowPtrHost, chunk.rowStart,
                chunk.matrixValueStart, primaryDevice, assemblyDevices[idx]);
            buffer.output.clearActiveOutput();
            if (replicateSecondaryInputs)
                buffer.copyStaticChunkInputData(
                    primaryGPTableView, primaryWeightsView,
                    primaryGeoValuesView, primaryDispValuesView,
                    primaryElecValuesView, numGPs(), chunk.gpStart,
                    chunk.gpCount, primaryDevice, assemblyDevices[idx]);
        };

        auto runChunkGroup = [&](int round, const auto& launcher)
        {
            if (numAssemblyDevices == 1)
            {
                cudaSetDevice(primaryDevice);
                if (round < static_cast<int>(schedule.chunksByDevice[0].size()))
                    launcher(0, schedule.chunksByDevice[0][round],
                             primarySystemView);
                return;
            }
            std::vector<std::thread> workers;
            std::vector<std::exception_ptr> errors(numAssemblyDevices);
            workers.reserve(numAssemblyDevices);
            for (int idx = 0; idx < numAssemblyDevices; ++idx)
            {
                if (round >=
                    static_cast<int>(schedule.chunksByDevice[idx].size()))
                    continue;
                const int device = assemblyDevices[idx];
                const siga::gpuasm::AssemblyChunk chunk =
                    schedule.chunksByDevice[idx][round];
                const SparseSystemDeviceView systemView =
                    systemViewForDevice(idx);
                workers.emplace_back([&, idx, device, chunk, systemView]()
                {
                    try
                    {
                        cudaSetDevice(device);
                        launcher(idx, chunk, systemView);
                    }
                    catch (...)
                    {
                        errors[idx] = std::current_exception();
                    }
                });
            }
            for (auto& worker : workers)
                worker.join();
            cudaSetDevice(primaryDevice);
            for (const auto& error : errors)
                if (error)
                    std::rethrow_exception(error);
        };

        const auto launchPrecomputeChunk =
            [&](int idx, const siga::gpuasm::AssemblyChunk& chunk,
                SparseSystemDeviceView)
        {
            if (chunk.elementCount <= 0)
                return;
            ElectroAssemblyDeviceBuffer& buffer = *buffers[idx];
            ElectroGPViews views = makeGPViews(buffer.baseGPData.data(),
                                               buffer.electroGPData.data(),
                                               chunk.gpCount);
            int minGridLocal = 0;
            int blockSizeLocal = 0;
            cudaOccupancyMaxPotentialBlockSize(&minGridLocal,
                &blockSizeLocal, evaluateGPKernel, 0, chunk.gpCount);
            const int gridSizeLocal =
                (chunk.gpCount + blockSizeLocal - 1) / blockSizeLocal;
            evaluateGPKernel<<<gridSizeLocal, blockSizeLocal>>>(
                numDerivatives(), chunk.gpStart,
                inputGPStartForDevice(idx, chunk), chunk.gpCount,
                buffer.materialParameters.vectorView(),
                displacementViewForDevice(idx), electricViewForDevice(idx),
                geometryViewForDevice(idx), gpTableViewForDevice(idx, chunk),
                weightsViewForDevice(idx),
                geoValuesViewForDevice(idx, chunk),
                dispValuesViewForDevice(idx, chunk),
                elecValuesViewForDevice(idx, chunk),
                views.geoJacobianInvs, views.measures, views.weightForces,
                views.weightBodys, views.Js, views.Fs, views.Ss, views.Cs,
                views.RCGinv, views.elecDisp, views.As);
            cudaError_t syncErr = cudaDeviceSynchronize();
            if (syncErr != cudaSuccess)
                throw std::runtime_error(
                    "CUDA synchronize failed in chunk-local electroelastic GP evaluation");
        };

        const auto launchMatrixChunk =
            [&](int idx, const siga::gpuasm::AssemblyChunk& chunk,
                SparseSystemDeviceView systemView)
        {
            if (chunk.elementCount <= 0)
                return;
            ElectroAssemblyDeviceBuffer& buffer = *buffers[idx];
            ElectroGPViews views = makeGPViews(buffer.baseGPData.data(),
                                               buffer.electroGPData.data(),
                                               chunk.gpCount);
            const int gridSizeLocal = N_D() * N_D() * chunk.elementCount;
            assembleMatrixKernel<<<gridSizeLocal, N_D()>>>(
                numDerivatives(), chunk.elementStart,
                inputGPStartForDevice(idx, chunk), chunk.elementCount,
                N_D(), buffer.materialParameters.vectorView(),
                displacementViewForDevice(idx), electricViewForDevice(idx),
                systemView, fixedDofsViewForDevice(idx),
                gpTableViewForDevice(idx, chunk), views.geoJacobianInvs,
                views.measures, views.weightForces, views.weightBodys,
                views.Js, dispValuesViewForDevice(idx, chunk),
                elecValuesViewForDevice(idx, chunk), views.Fs, views.Ss,
                views.Cs, views.RCGinv, views.As);
            cudaError_t syncErr = cudaDeviceSynchronize();
            if (syncErr != cudaSuccess)
                throw std::runtime_error(
                    "CUDA synchronize failed in chunk-local electroelastic matrix assembly");
        };

        const auto launchRHSChunk =
            [&](int idx, const siga::gpuasm::AssemblyChunk& chunk,
                SparseSystemDeviceView systemView)
        {
            if (chunk.elementCount <= 0)
                return;
            ElectroAssemblyDeviceBuffer& buffer = *buffers[idx];
            ElectroGPViews views = makeGPViews(buffer.baseGPData.data(),
                                               buffer.electroGPData.data(),
                                               chunk.gpCount);
            const int gridSizeLocal = N_D() * chunk.elementCount;
            assembleRHSKernel<<<gridSizeLocal, N_D()>>>(
                numDerivatives(), chunk.elementStart,
                inputGPStartForDevice(idx, chunk), chunk.elementCount,
                N_D(), displacementViewForDevice(idx),
                electricViewForDevice(idx), systemView,
                gpTableViewForDevice(idx, chunk), views.geoJacobianInvs,
                views.weightForces, views.weightBodys,
                dispValuesViewForDevice(idx, chunk),
                elecValuesViewForDevice(idx, chunk), views.elecDisp,
                bodyForceViewForDevice(idx), views.Fs, views.Ss);
            cudaError_t syncErr = cudaDeviceSynchronize();
            if (syncErr != cudaSuccess)
                throw std::runtime_error(
                    "CUDA synchronize failed in chunk-local electroelastic RHS assembly");
        };

        for (int round = 0; round < schedule.rounds; ++round)
        {
            for (int idx = 0; idx < numAssemblyDevices; ++idx)
            {
                if (round >=
                    static_cast<int>(schedule.chunksByDevice[idx].size()))
                    continue;
                prepareChunkForDevice(idx,
                                      schedule.chunksByDevice[idx][round]);
            }
            cudaSetDevice(primaryDevice);
            runChunkGroup(round, launchPrecomputeChunk);
            runChunkGroup(round, launchMatrixChunk);
            runChunkGroup(round, launchRHSChunk);
            cudaSetDevice(primaryDevice);
            for (int idx = 1; idx < numAssemblyDevices; ++idx)
            {
                if (round >=
                    static_cast<int>(schedule.chunksByDevice[idx].size()))
                    continue;
                siga::gpuasm::reduceSparseOutputBuffer(
                    csrMatrix().values(), rhs(), buffers[idx]->output);
            }
        }

        cudaSetDevice(primaryDevice);
        return;
    }
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
    evaluateGPKernel<<<gridSize, blockSize>>>(numDerivatives(), 0, 0, totalGPs, parameterValues.vectorView(), displacementView(), m_electricPotentialPatch.deviceView(), geometryView(), gpTable(), wts().vectorView(), geoValuesAndDerssView(), dispValuesAndDerssView(), m_elecValuesAndDerss.matrixView(m_elePotentialP1, numGPs() * 2 * domainDim()), geoJacobianInvs, measures, weightForces, weightBodys, Js, Fs, Ss, Cs, RCGinv, elecDisp, As);
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
    assembleMatrixKernel<<<gridSize, blockSize>>>(numDerivatives(), 0, 0, numElements(), N_D(), parameterValues.vectorView(), displacementView(), m_electricPotentialPatch.deviceView(), sparseSystemDeviceView(), fixedDofs_assemble, gpTable(), geoJacobianInvs, measures, weightForces, weightBodys, Js, dispValuesAndDerssView(), m_elecValuesAndDerss.matrixView(m_elePotentialP1, numGPs() * 2 * domainDim()), Fs, Ss, Cs, RCGinv, As);
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
    assembleRHSKernel<<<gridSize, blockSize>>>(numDerivatives(), 0, 0, numElements(), N_D(), displacementView(), m_electricPotentialPatch.deviceView(), sparseSystemDeviceView(), gpTable(), geoJacobianInvs, weightForces, weightBodys, dispValuesAndDerssView(), m_elecValuesAndDerss.matrixView(m_elePotentialP1, numGPs() * 2 * domainDim()), elecDisp, bodyForce().vectorView(), Fs, Ss);
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
