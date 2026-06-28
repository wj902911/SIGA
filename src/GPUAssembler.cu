#include <GPUAssembler.h>
#include <GPUAssemblySupport.h>

#include <chrono>
#include <exception>
#include <memory>
#include <thread>

//#define TIME_INITIALIZATION

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
    //result.printControlPoints();
}

__global__
void zeroFunctionControlPointsKernel(MultiPatchDeviceView result, int totalEntries)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalEntries; idx += blockDim.x * gridDim.x)
    {
        int patch(0);
        int component(0);
        int point_idx = result.threadPatchAndDof(idx, patch, component);
        result.setCoefficients(patch, point_idx, component, 0.0);
    }
}

__global__
void recoverCauchyStressAtNodesKernel(int numDerivatives,
                                      int totalNumGPs,
                                      MultiPatchDeviceView displacement,
                                      MultiPatchDeviceView cauchyStress,
                                      DeviceMatrixView<double> pts,
                                      DeviceVectorView<double> weightForces,
                                      DeviceMatrixView<double> dispValuesAndDerss,
                                      DeviceMatrixView<double> Fs,
                                      DeviceMatrixView<double> Ss,
                                      DeviceVectorView<double> nodalWeights)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalNumGPs; idx += blockDim.x * gridDim.x)
    {
        int dim = displacement.domainDim();
        int dimTensor = dim * (dim + 1) / 2;
        int patch_idx(0);
        displacement.threadPatch(idx, patch_idx);

        TensorBsplineBasisDeviceView basis = displacement.basis(patch_idx);
        int P1 = basis.knotsOrder(0) + 1;
        int numActive = basis.numActiveControlPoints();
        DeviceVectorView<double> pt(pts.data() + idx * dim, dim);
        DeviceMatrixView<double> dispValuesAndDers(
            dispValuesAndDerss.data() + idx * P1 * (numDerivatives + 1) * dim,
            P1, (numDerivatives + 1) * dim);

        DeviceMatrixView<double> F(Fs.data() + idx * dim * dim, dim, dim);
        DeviceMatrixView<double> S(Ss.data() + idx * dim * dim, dim, dim);

        double FSData[3 * 3] = {0.0};
        double sigmaData[3 * 3] = {0.0};
        DeviceMatrixView<double> FS(FSData, dim, dim);
        DeviceMatrixView<double> sigma(sigmaData, dim, dim);
        F.times(S, FS);
        FS.timeTranspose(F, sigma);
        sigma.times(1.0 / F.determinant());

        double sigmaVecData[6] = {0.0};
        DeviceVectorView<double> sigmaVec(sigmaVecData, dimTensor);
        voigtStressView(sigmaVec, sigma);

        int patchControlPointOffset = 0;
        for (int p = 0; p < patch_idx; ++p)
            patchControlPointOffset += cauchyStress.numControlPoints(p);

        for (int r = 0; r < numActive; ++r)
        {
            int localControlPoint = basis.activeIndex(pt, r);
            int globalControlPoint = patchControlPointOffset + localControlPoint;
            double N = tensorBasisValue(r, P1, dim, numDerivatives, dispValuesAndDers);
            double weight = N * weightForces[idx];

            atomicAdd(&nodalWeights[globalControlPoint], weight);
            DeviceMatrixView<double> stressControlPoints = cauchyStress.controlPoints(patch_idx);
            for (int c = 0; c < dimTensor; ++c)
                atomicAdd(&stressControlPoints(localControlPoint, c), weight * sigmaVec[c]);
        }
    }
}

__global__
void recoverDeformationGradientAtNodesKernel(int numDerivatives,
                                             int totalNumGPs,
                                             MultiPatchDeviceView displacement,
                                             MultiPatchDeviceView deformationGradient,
                                             DeviceMatrixView<double> pts,
                                             DeviceVectorView<double> weightForces,
                                             DeviceMatrixView<double> dispValuesAndDerss,
                                             DeviceMatrixView<double> Fs,
                                             DeviceVectorView<double> nodalWeights)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalNumGPs; idx += blockDim.x * gridDim.x)
    {
        int dim = displacement.domainDim();
        int dim2 = dim * dim;
        int patch_idx(0);
        displacement.threadPatch(idx, patch_idx);

        TensorBsplineBasisDeviceView basis = displacement.basis(patch_idx);
        int P1 = basis.knotsOrder(0) + 1;
        int numActive = basis.numActiveControlPoints();
        DeviceVectorView<double> pt(pts.data() + idx * dim, dim);
        DeviceMatrixView<double> dispValuesAndDers(
            dispValuesAndDerss.data() + idx * P1 * (numDerivatives + 1) * dim,
            P1, (numDerivatives + 1) * dim);

        DeviceMatrixView<double> F(Fs.data() + idx * dim * dim, dim, dim);

        int patchControlPointOffset = 0;
        for (int p = 0; p < patch_idx; ++p)
            patchControlPointOffset += deformationGradient.numControlPoints(p);

        DeviceMatrixView<double> deformationGradientControlPoints =
            deformationGradient.controlPoints(patch_idx);
        for (int r = 0; r < numActive; ++r)
        {
            int localControlPoint = basis.activeIndex(pt, r);
            int globalControlPoint = patchControlPointOffset + localControlPoint;
            double N = tensorBasisValue(r, P1, dim, numDerivatives, dispValuesAndDers);
            double weight = N * weightForces[idx];

            atomicAdd(&nodalWeights[globalControlPoint], weight);
            for (int a = 0; a < dim; ++a)
                for (int A = 0; A < dim; ++A)
                {
                    int c = a * dim + A;
                    if (c < dim2)
                        atomicAdd(&deformationGradientControlPoints(localControlPoint, c),
                                  weight * F(a, A));
                }
        }
    }
}

__device__
double kinematicTensorBasisPartial(int r, int P1, int dim, int numDerivatives,
                                   DeviceMatrixView<double> valuesAndDers,
                                   int derivativeDir1, int derivativeDir2)
{
    int tensorCoordData[3] = {0}; // max 3D
    DeviceVectorView<int> tensorCoord(tensorCoordData, dim);
    getTensorCoordinate(dim, P1, r, tensorCoordData);

    double value = 1.0;
    for (int d = 0; d < dim; ++d)
    {
        int order = 0;
        if (d == derivativeDir1)
            ++order;
        if (d == derivativeDir2)
            ++order;
        value *= valuesAndDers(tensorCoord[d],
                               (numDerivatives + 1) * d + order);
    }
    return value;
}

__device__
void kinematicTensorBasisHessianParam(int r, int P1, int dim,
                                      int numDerivatives,
                                      DeviceMatrixView<double> valuesAndDers,
                                      double* hessian)
{
    for (int a = 0; a < dim; ++a)
        for (int b = 0; b < dim; ++b)
            hessian[a * dim + b] = kinematicTensorBasisPartial(
                r, P1, dim, numDerivatives, valuesAndDers, a, b);
}

__device__
void kinematicPatchParamHessian(PatchDeviceView patch,
                                DeviceVectorView<double> pt,
                                DeviceMatrixView<double> valuesAndDers,
                                int numDerivatives, int component,
                                double* hessian)
{
    const int dim = patch.domainDim();
    const int P1 = patch.basis().knotsOrder(0) + 1;
    const int numActive = patch.basis().numActiveControlPoints();

    for (int a = 0; a < dim * dim; ++a)
        hessian[a] = 0.0;

    for (int r = 0; r < numActive; ++r)
    {
        double basisHessian[9] = {0.0};
        kinematicTensorBasisHessianParam(r, P1, dim, numDerivatives,
                                         valuesAndDers, basisHessian);
        const double cp = patch.activeControlPointComponent(pt, r, component);
        for (int a = 0; a < dim * dim; ++a)
            hessian[a] += cp * basisHessian[a];
    }
}

__device__
void kinematicPhysicalBasisHessian(int r, int P1, int dim, int numDerivatives,
                                   DeviceMatrixView<double> basisValuesAndDers,
                                   double* geoHessians,
                                   DeviceMatrixView<double> geoJacobianInv,
                                   double* result)
{
    double gradParamData[3] = {0.0};
    DeviceVectorView<double> gradParam(gradParamData, dim);
    double hessParam[9] = {0.0};
    tensorBasisDerivative(r, P1, dim, numDerivatives, basisValuesAndDers,
                          gradParam);
    kinematicTensorBasisHessianParam(r, P1, dim, numDerivatives,
                                     basisValuesAndDers, hessParam);

    double corrected[9] = {0.0};
    for (int i = 0; i < dim; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            double correction = 0.0;
            for (int k = 0; k < dim; ++k)
                for (int c = 0; c < dim; ++c)
                    correction += gradParam[k] * geoJacobianInv(k, c) *
                                  geoHessians[c * dim * dim + i * dim + j];
            corrected[i * dim + j] = hessParam[i * dim + j] - correction;
        }
    }

    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
        {
            double value = 0.0;
            for (int i = 0; i < dim; ++i)
                for (int j = 0; j < dim; ++j)
                    value += geoJacobianInv(i, A) * geoJacobianInv(j, B) *
                             corrected[i * dim + j];
            result[A * dim + B] = value;
        }
}

__device__
void kinematicPhysicalFieldHessians(PatchDeviceView fieldPatch,
                                    DeviceVectorView<double> pt,
                                    DeviceMatrixView<double> fieldValuesAndDers,
                                    int numDerivatives, double* geoHessians,
                                    DeviceMatrixView<double> geoJacobianInv,
                                    double* result)
{
    const int dim = fieldPatch.domainDim();
    const int P1 = fieldPatch.basis().knotsOrder(0) + 1;
    const int numActive = fieldPatch.basis().numActiveControlPoints();

    for (int a = 0; a < dim * dim * dim; ++a)
        result[a] = 0.0;

    for (int r = 0; r < numActive; ++r)
    {
        double basisHessian[9] = {0.0};
        kinematicPhysicalBasisHessian(r, P1, dim, numDerivatives,
                                      fieldValuesAndDers, geoHessians,
                                      geoJacobianInv, basisHessian);
        for (int comp = 0; comp < dim; ++comp)
        {
            const double cp =
                fieldPatch.activeControlPointComponent(pt, r, comp);
            for (int a = 0; a < dim * dim; ++a)
                result[comp * dim * dim + a] += cp * basisHessian[a];
        }
    }
}

__device__
void computeGreenLagrangeStrainGradient(int dim, DeviceMatrixView<double> F,
                                         const double* gradF,
                                         double* greenStrainGradient)
{
    const int dim2 = dim * dim;
    for (int I = 0; I < dim; ++I)
        for (int J = 0; J < dim; ++J)
            for (int K = 0; K < dim; ++K)
            {
                double value = 0.0;
                for (int a = 0; a < dim; ++a)
                    value += gradF[a * dim2 + I * dim + K] * F(a, J) +
                             F(a, I) * gradF[a * dim2 + J * dim + K];
                greenStrainGradient[(I * dim + J) * dim + K] = 0.5 * value;
            }
}

__global__
void recoverKinematicGradientsAtNodesKernel(
    int numDerivatives,
    int totalNumGPs,
    MultiPatchDeviceView displacement,
    MultiPatchDeviceView geometry,
    MultiPatchDeviceView deformationGradientGradient,
    MultiPatchDeviceView greenLagrangeStrainGradient,
    DeviceMatrixView<double> pts,
    DeviceVectorView<double> weightForces,
    DeviceMatrixView<double> geoValuesAndDerss,
    DeviceMatrixView<double> dispValuesAndDerss,
    DeviceMatrixView<double> geoJacobianInvs,
    DeviceMatrixView<double> Fs,
    DeviceVectorView<double> nodalWeights)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalNumGPs; idx += blockDim.x * gridDim.x)
    {
        const int dim = displacement.domainDim();
        const int dim2 = dim * dim;
        const int dim3 = dim2 * dim;

        int patchIdx = 0;
        displacement.threadPatch(idx, patchIdx);

        PatchDeviceView geoPatch = geometry.patch(patchIdx);
        PatchDeviceView dispPatch = displacement.patch(patchIdx);
        TensorBsplineBasisDeviceView basis = displacement.basis(patchIdx);
        const int geoP1 = geoPatch.basis().knotsOrder(0) + 1;
        const int dispP1 = basis.knotsOrder(0) + 1;
        const int numActive = basis.numActiveControlPoints();

        DeviceVectorView<double> pt(pts.data() + idx * dim, dim);
        DeviceMatrixView<double> geoValuesAndDers(
            geoValuesAndDerss.data() +
                idx * geoP1 * (numDerivatives + 1) * dim,
            geoP1, (numDerivatives + 1) * dim);
        DeviceMatrixView<double> dispValuesAndDers(
            dispValuesAndDerss.data() +
                idx * dispP1 * (numDerivatives + 1) * dim,
            dispP1, (numDerivatives + 1) * dim);
        DeviceMatrixView<double> geoJacobianInv(
            geoJacobianInvs.data() + idx * dim2, dim, dim);
        DeviceMatrixView<double> F(Fs.data() + idx * dim2, dim, dim);

        double geoHessians[27] = {0.0};
        for (int a = 0; a < dim; ++a)
            kinematicPatchParamHessian(geoPatch, pt, geoValuesAndDers,
                                       numDerivatives, a,
                                       geoHessians + a * dim2);

        double gradF[27] = {0.0};
        kinematicPhysicalFieldHessians(dispPatch, pt, dispValuesAndDers,
                                       numDerivatives, geoHessians,
                                       geoJacobianInv, gradF);

        double greenStrainGradient[27] = {0.0};
        computeGreenLagrangeStrainGradient(dim, F, gradF,
                                           greenStrainGradient);

        int patchControlPointOffset = 0;
        for (int p = 0; p < patchIdx; ++p)
            patchControlPointOffset +=
                deformationGradientGradient.numControlPoints(p);

        DeviceMatrixView<double> gradFControlPoints =
            deformationGradientGradient.controlPoints(patchIdx);
        DeviceMatrixView<double> greenStrainGradientControlPoints =
            greenLagrangeStrainGradient.controlPoints(patchIdx);
        for (int r = 0; r < numActive; ++r)
        {
            const int localControlPoint = basis.activeIndex(pt, r);
            const int globalControlPoint =
                patchControlPointOffset + localControlPoint;
            const double N = tensorBasisValue(r, dispP1, dim, numDerivatives,
                                              dispValuesAndDers);
            const double weight = N * weightForces[idx];

            atomicAdd(&nodalWeights[globalControlPoint], weight);
            for (int c = 0; c < dim3; ++c)
            {
                atomicAdd(&gradFControlPoints(localControlPoint, c),
                          weight * gradF[c]);
                atomicAdd(&greenStrainGradientControlPoints(localControlPoint, c),
                          weight * greenStrainGradient[c]);
            }
        }
    }
}

__global__
void normalizeRecoveredStressKernel(MultiPatchDeviceView cauchyStress,
                                    DeviceVectorView<double> nodalWeights,
                                    int totalControlPoints)
{
    int dimTensor = cauchyStress.targetDim();
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalControlPoints * dimTensor; idx += blockDim.x * gridDim.x)
    {
        int patch(0);
        int component(0);
        int localControlPoint = cauchyStress.threadPatchAndDof(idx, patch, component);

        int globalControlPoint = localControlPoint;
        for (int p = 0; p < patch; ++p)
            globalControlPoint += cauchyStress.numControlPoints(p);

        double weight = nodalWeights[globalControlPoint];
        if (weight != 0.0)
        {
            DeviceMatrixView<double> stressControlPoints = cauchyStress.controlPoints(patch);
            stressControlPoints(localControlPoint, component) /= weight;
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
#endif

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
    }
    __syncthreads();
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

int followerMomentHostActiveIndex(const DofMapper& mapper, int dof,
                                  int patchIndex)
{
    const std::vector<int>& dofs = mapper.getDofs(0);
    const std::vector<int> offsets = mapper.getOffset();
    return dofs[offsets[patchIndex] + dof] + mapper.getShift();
}

bool followerMomentHostIsFree(const DofMapper& mapper, int activeIndex)
{
    return activeIndex < mapper.getCurElimId() + mapper.getShift();
}

void appendFollowerMomentCOOPattern(
    const BoundaryConditions& boundaryConditions,
    const MultiBasis& multiBasis,
    const SparseSystem& sparseSystem,
    const std::vector<DofMapper>& dofMappers,
    int domainDim,
    int targetDim,
    std::vector<int>& rows,
    std::vector<int>& cols)
{
    const BoundaryConditions::bcContainer& followerMomentSides =
        boundaryConditions.followerMomentSides();
    if (followerMomentSides.empty())
        return;

    if (domainDim != 2 || targetDim != 2)
        throw std::runtime_error(
            "Follower moment boundary conditions currently support 2D displacement problems only.");

    for (BoundaryConditions::bcContainer::const_iterator it =
             followerMomentSides.begin();
         it != followerMomentSides.end(); ++it)
    {
        const int patchIdx = it->patchIndex();
        if (it->side().direction() != 0)
            throw std::runtime_error(
                "Follower moment boundary conditions currently support west/east beam-end sides only.");

        const Eigen::VectorXi boundaryDofs =
            multiBasis.basis(patchIdx).boundary(it->side());

        for (int rowDofIdx = 0; rowDofIdx < boundaryDofs.size(); ++rowDofIdx)
        {
            const int rowDof = boundaryDofs[rowDofIdx];
            for (int rowComp = 0; rowComp < targetDim; ++rowComp)
            {
                const DofMapper& rowMapper = dofMappers[rowComp];
                const int rowActive =
                    followerMomentHostActiveIndex(rowMapper, rowDof, patchIdx);
                if (!followerMomentHostIsFree(rowMapper, rowActive))
                    continue;

                int row = sparseSystem.rowBlockOffset(rowComp) + rowActive;
#if defined(USE_PERMUTATION)
                row = sparseSystem.permOld2New()[row];
#endif

                for (int colDofIdx = 0; colDofIdx < boundaryDofs.size();
                     ++colDofIdx)
                {
                    const int colDof = boundaryDofs[colDofIdx];
                    for (int colComp = 0; colComp < targetDim; ++colComp)
                    {
                        const DofMapper& colMapper = dofMappers[colComp];
                        const int colActive = followerMomentHostActiveIndex(
                            colMapper, colDof, patchIdx);
                        if (!followerMomentHostIsFree(colMapper, colActive))
                            continue;

                        int col =
                            sparseSystem.colBlockOffset(colComp) + colActive;
#if defined(USE_PERMUTATION)
                        col = sparseSystem.permOld2New()[col];
#endif
                        rows.push_back(row);
                        cols.push_back(col);
                    }
                }
            }
        }
    }
}

__global__
void evaluateBasisValuesAndDerivativesAtGPsKernel(int numDerivatives, 
                            int totalNumGPs, int dim,
                            MultiPatchDeviceView displacement,
                            MultiPatchDeviceView multiPatch,
                            DeviceMatrixView<double> pts,
                            DeviceVectorView<double> geoWorkingSpaces,
                            DeviceVectorView<double> dispWorkingSpaces,
                            DeviceMatrixView<double> geoValuesAndDerss,
                            DeviceMatrixView<double> dispValuesAndDerss)
{
    for (int tidx = blockIdx.x * blockDim.x + threadIdx.x; tidx < totalNumGPs * dim; tidx += blockDim.x * gridDim.x) {
        int GPIdx = tidx / dim;
        int d = tidx % dim;
        int patch_idx(0);
        displacement.threadPatch(GPIdx, patch_idx);
        DeviceVectorView<double> pt(pts.data() + GPIdx * dim, dim);
        PatchDeviceView geoPatch = multiPatch.patch(patch_idx);
        int geoP1 = geoPatch.basis().knotsOrder(0) + 1;
        double* geoWorkingSpace = geoWorkingSpaces.data() + GPIdx * geoP1 * (geoP1 + 4) * dim;
        DeviceMatrixView<double> geoValuesAndDers(geoValuesAndDerss.data() + GPIdx * geoP1 * (numDerivatives + 1) * dim, geoP1, (numDerivatives + 1) * dim);
        geoPatch.basis().evalAllDers_into(d, dim, pt, numDerivatives, geoWorkingSpace, geoValuesAndDers);
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
        int P1 = dispBasis.knotsOrder(0) + 1;
        double* dispWorkingSpace = dispWorkingSpaces.data() + GPIdx * P1 * (P1 + 4) * dim;
        DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDerss.data() + GPIdx * P1 * (numDerivatives+1) * dim, P1, (numDerivatives+1)*dim);
        dispBasis.evalAllDers_into(d, dim, pt, numDerivatives, dispWorkingSpace, dispValuesAndDers);
    }
}

__global__
void computeGPTableKernel(int totalNumGPs,
                    MultiPatchDeviceView displacement,
                    MultiGaussPointsDeviceView multiGaussPoints,
                    DeviceMatrixView<double> pts,
                    DeviceVectorView<double> wts)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < totalNumGPs; idx += blockDim.x * gridDim.x) {
        int dim = displacement.domainDim();
        int patch_idx(0);
        int point_idx = displacement.threadPatch(idx, patch_idx);
        DeviceVectorView<double> pt(pts.data() + idx * dim, dim);
        wts[idx] = displacement.gsPoint(point_idx, patch_idx, multiGaussPoints[patch_idx], pt);
    }
}

__global__
void assembleRHSWithGPDataKernel(int numDerivatives, int EleStartId,
                      int inputGPStartId, int numElementsBatched, int N_D,
                      MultiPatchDeviceView displacement,
                      SparseSystemDeviceView system,
                      DeviceMatrixView<double> pts,
                      DeviceMatrixView<double> geoJacobianInvs,
                      DeviceVectorView<double> weightForces,
                      DeviceVectorView<double> weightBodys,
                      DeviceMatrixView<double> dispValuesAndDerss,
                      DeviceVectorView<double> bodyForce,
                      DeviceMatrixView<double> Fs, 
                      DeviceMatrixView<double> Ss)
{
    int totalNumShapeFunctions = numElementsBatched * N_D;
    for (int bidx = blockIdx.x; bidx < totalNumShapeFunctions; bidx += gridDim.x) {
        __shared__ int shapeFuncIdx, ele_idx, patch_idx, idx;
        __shared__ double localRHSData[3]; //max 3D

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
        DeviceVectorView<double> localRHS(localRHSData, dim);
        for (int i = threadId; i < dim; i += blockDim.x * blockDim.y)
            localRHSData[i] = 0.0;
        __syncthreads();

        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
        int P1 = dispBasis.knotsOrder(0) + 1;
        for (int i_GP = threadIdx.x; i_GP < N_D; i_GP += blockDim.x) {
            //int GPIdx = ele_idx * N_D + i_GP;
            int GPIdx = idx * N_D + i_GP;
            int inputGPIdx = inputGPStartId + GPIdx;
            DeviceMatrixView<double> geoJacobianInv(geoJacobianInvs.data() + GPIdx * dim * dim, dim, dim);
            double weightForce = weightForces[GPIdx];
            double weightBody = weightBodys[GPIdx];
            DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDerss.data() + inputGPIdx * P1 * (numDerivatives + 1) * dim, P1, (numDerivatives + 1) * dim);
            DeviceMatrixView<double> F(Fs.data() + GPIdx * dim * dim, dim, dim);
            DeviceMatrixView<double> S(Ss.data() + GPIdx * dim * dim, dim, dim);
            double dN_iData[3] = {0.0}; //max 3D
            DeviceVectorView<double> dN_i(dN_iData, dim);
            tensorBasisDerivative(shapeFuncIdx, P1, dim, numDerivatives, dispValuesAndDers, dN_i);
#if 1
            double physGrad_iData[3] = {0.0}; //max 3D
            DeviceVectorView<double> physGrad_i(physGrad_iData, dim);
            geoJacobianInv.transposeTime(dN_i, physGrad_i);
            for (int di = 0; di < dim; di++) {
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
                B_i_diTrans.times(Svec, residualEntryMat);
                residualEntry = -residualEntry * weightBody + weightForce * bodyForce[di] * tensorBasisValue(shapeFuncIdx, P1, dim, numDerivatives, dispValuesAndDers);
                atomicAdd(&localRHS[di], residualEntry);
            }
#endif
        }
        __syncthreads();
        //printf("bidx:%d, localRHS: \n", bidx);
        //localRHS.print();
        //int GPIdx = ele_idx * N_D + threadIdx.x;
#if 1
        //int GPIdx = idx * N_D + threadIdx.x;
        int GPIdx = idx * N_D;
        int inputGPIdx = inputGPStartId + GPIdx;
        DeviceVectorView<double> pt(pts.data() + inputGPIdx * dim, dim);
        for (int di = threadId; di < dim; di += blockDim.x * blockDim.y) {
            int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx).activeIndex(pt, shapeFuncIdx), patch_idx, di);
            system.pushToRhs(localRHS(di), globalIndex_i, di);
            //printf("bidx:%d, di:%d, globalIndex_i:%d, localRHS(di):%f\n", bidx, di, globalIndex_i, localRHS(di));
        }
#endif
    }

}

__global__
void assembleNeumannBoundaryConditionKernel(int totalNumBoundaryGPs,
                                            int numActive,
                                            MultiPatchDeviceView displacement,
                                            MultiPatchDeviceView geometry,
                                            SparseSystemDeviceView system,
                                            MultiGaussPointsDeviceView gaussPoints,
                                            DeviceVectorView<int> bcOffsets,
                                            DeviceVectorView<int> bcPatches,
                                            DeviceVectorView<int> bcSideIndexes,
                                            DeviceMatrixView<double> bcValues)
{
    const int dim = displacement.domainDim();
    const int targetDim = displacement.targetDim();

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalNumBoundaryGPs * numActive;
         idx += blockDim.x * gridDim.x)
    {
        const int shapeFuncIdx = idx % numActive;
        const int boundaryGPIdx = idx / numActive;

        int bcIdx = 0;
        while (bcIdx + 1 < bcOffsets.size() && boundaryGPIdx >= bcOffsets[bcIdx + 1])
            ++bcIdx;

        const int patchIdx = bcPatches[bcIdx];
        BoxSide_d side(bcSideIndexes[bcIdx]);
        const int fixedDir = side.direction();
        const int localBoundaryGPIdx = boundaryGPIdx - bcOffsets[bcIdx];

        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
        const int P1 = dispBasis.knotsOrder(0) + 1;

        double fullPtData[3] = {0.0, 0.0, 0.0};
        double boundaryPtData[2] = {0.0, 0.0};
        DeviceVectorView<double> fullPt(fullPtData, dim);
        DeviceVectorView<double> boundaryPt(boundaryPtData, dim - 1);

        double weight = 1.0;
        int gpRemainder = localBoundaryGPIdx;
        for (int d = 0, bd = 0; d < dim; ++d)
        {
            if (d == fixedDir)
            {
                fullPt[d] = side.parameter()
                    ? dispBasis.knotVector(d).domainEnd()
                    : dispBasis.knotVector(d).domainBegin();
                continue;
            }

            const int totalInDir = dispBasis.totalNumGPsInDir(d);
            const int oneDimGPIdx = gpRemainder % totalInDir;
            gpRemainder /= totalInDir;

            double coordinate = 0.0;
            weight *= dispBasis.gsPoint(oneDimGPIdx, d, gaussPoints[patchIdx], coordinate);
            fullPt[d] = coordinate;
            boundaryPt[bd++] = coordinate;
        }

        PatchDeviceView geoPatch = geometry.patch(patchIdx);
        double boundaryJacobianData[3 * 2] = {0.0};
        DeviceMatrixView<double> boundaryJacobian(boundaryJacobianData, targetDim, dim - 1);
        geoPatch.boundaryJacobian(side, boundaryPt, boundaryJacobian);

        double measure = 1.0;
        if (dim == 2)
        {
            double tangentData[3] = {0.0, 0.0, 0.0};
            DeviceVectorView<double> tangent(tangentData, targetDim);
            for (int i = 0; i < targetDim; ++i)
                tangent[i] = boundaryJacobian(i, 0);
            measure = tangent.norm_device();
        }
        else if (dim == 3)
        {
            const double ax = boundaryJacobian(0, 0);
            const double ay = boundaryJacobian(1, 0);
            const double az = boundaryJacobian(2, 0);
            const double bx = boundaryJacobian(0, 1);
            const double by = boundaryJacobian(1, 1);
            const double bz = boundaryJacobian(2, 1);
            const double cx = ay * bz - az * by;
            const double cy = az * bx - ax * bz;
            const double cz = ax * by - ay * bx;
            measure = sqrt(cx * cx + cy * cy + cz * cz);
        }

        double dispValuesAndDersData[5 * 3] = {0.0};
        DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDersData, P1, dim);
        dispBasis.evalAllDers_into(fullPt, 0, dispValuesAndDers);

        const double N = tensorBasisValue(shapeFuncIdx, P1, dim, 0, dispValuesAndDers);
        const int activeIndex = dispBasis.activeIndex(fullPt, shapeFuncIdx);
        const double weightedBasis = N * weight * measure;

        for (int di = 0; di < targetDim; ++di)
        {
            const int globalIndex = system.mapColIndex(activeIndex, patchIdx, di);
            system.pushToRhs(weightedBasis * bcValues(di, bcIdx), globalIndex, di);
        }
    }
}

__global__
void assembleDoubleStressBoundaryConditionKernel(int totalNumBoundaryGPs,
                                                int numActive,
                                                MultiPatchDeviceView displacement,
                                                MultiPatchDeviceView geometry,
                                                SparseSystemDeviceView system,
                                                MultiGaussPointsDeviceView gaussPoints,
                                                DeviceVectorView<int> bcOffsets,
                                                DeviceVectorView<int> bcPatches,
                                                DeviceVectorView<int> bcSideIndexes,
                                                DeviceMatrixView<double> bcValues)
{
    const int dim = displacement.domainDim();
    const int targetDim = displacement.targetDim();

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalNumBoundaryGPs * numActive;
         idx += blockDim.x * gridDim.x)
    {
        const int shapeFuncIdx = idx % numActive;
        const int boundaryGPIdx = idx / numActive;

        int bcIdx = 0;
        while (bcIdx + 1 < bcOffsets.size() && boundaryGPIdx >= bcOffsets[bcIdx + 1])
            ++bcIdx;

        const int patchIdx = bcPatches[bcIdx];
        BoxSide_d side(bcSideIndexes[bcIdx]);
        const int fixedDir = side.direction();
        const int localBoundaryGPIdx = boundaryGPIdx - bcOffsets[bcIdx];

        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
        const int P1 = dispBasis.knotsOrder(0) + 1;

        double fullPtData[3] = {0.0, 0.0, 0.0};
        double boundaryPtData[2] = {0.0, 0.0};
        DeviceVectorView<double> fullPt(fullPtData, dim);
        DeviceVectorView<double> boundaryPt(boundaryPtData, dim - 1);

        double weight = 1.0;
        int gpRemainder = localBoundaryGPIdx;
        for (int d = 0, bd = 0; d < dim; ++d)
        {
            if (d == fixedDir)
            {
                fullPt[d] = side.parameter()
                    ? dispBasis.knotVector(d).domainEnd()
                    : dispBasis.knotVector(d).domainBegin();
                continue;
            }

            const int totalInDir = dispBasis.totalNumGPsInDir(d);
            const int oneDimGPIdx = gpRemainder % totalInDir;
            gpRemainder /= totalInDir;

            double coordinate = 0.0;
            weight *= dispBasis.gsPoint(oneDimGPIdx, d, gaussPoints[patchIdx], coordinate);
            fullPt[d] = coordinate;
            boundaryPt[bd++] = coordinate;
        }

        PatchDeviceView geoPatch = geometry.patch(patchIdx);
        double geoValuesAndDersData[5 * 2 * 3] = {0.0};
        DeviceMatrixView<double> geoValuesAndDers(geoValuesAndDersData, P1, 2 * dim);
        geoPatch.basis().evalAllDers_into(fullPt, 1, geoValuesAndDers);

        double geoJacobianData[3 * 3] = {0.0};
        double geoJacobianInvTransData[3 * 3] = {0.0};
        DeviceMatrixView<double> geoJacobian(geoJacobianData, dim, dim);
        DeviceMatrixView<double> geoJacobianInvTrans(geoJacobianInvTransData, dim, dim);
        geoPatch.jacobian(fullPt, geoValuesAndDers, 1, geoJacobian);
        const double orientation = geoJacobian.determinant() >= 0.0 ? 1.0 : -1.0;

        double boundaryJacobianData[3 * 2] = {0.0};
        DeviceMatrixView<double> boundaryJacobian(boundaryJacobianData, targetDim, dim - 1);
        geoPatch.boundaryJacobian(side, boundaryPt, boundaryJacobian);

        double normalData[3] = {0.0, 0.0, 0.0};
        DeviceVectorView<double> normal(normalData, targetDim);
        double measure = 1.0;
        if (dim == 2 && targetDim == 2)
        {
            const int sideIndex = side.index();
            const double sideOrientation =
                ((sideIndex + (sideIndex + 1) / 2) % 2) ? 1.0 : -1.0;
            const double sgn = sideOrientation * orientation;
            const int dir = side.direction();

            double unormalData[2] = {0.0, 0.0};
            if (dir == 0)
            {
                unormalData[0] = sgn * geoJacobian(1, 1);
                unormalData[1] = -sgn * geoJacobian(0, 1);
            }
            else
            {
                unormalData[0] = sgn * geoJacobian(1, 0);
                unormalData[1] = -sgn * geoJacobian(0, 0);
            }

            measure = sqrt(unormalData[0] * unormalData[0] +
                           unormalData[1] * unormalData[1]);
            if (measure <= 0.0)
                continue;
            normal[0] = unormalData[0] / measure;
            normal[1] = unormalData[1] / measure;
        }
        else if (dim == 3 && targetDim == 3)
        {
            const double ax = boundaryJacobian(0, 0);
            const double ay = boundaryJacobian(1, 0);
            const double az = boundaryJacobian(2, 0);
            const double bx = boundaryJacobian(0, 1);
            const double by = boundaryJacobian(1, 1);
            const double bz = boundaryJacobian(2, 1);
            normal[0] = ay * bz - az * by;
            normal[1] = az * bx - ax * bz;
            normal[2] = ax * by - ay * bx;

            const double outwardSign = orientation * (side.parameter() ? 1.0 : -1.0);
            normal[0] *= outwardSign;
            normal[1] *= outwardSign;
            normal[2] *= outwardSign;

            measure = normal.norm_device();
            if (measure <= 0.0)
                continue;

            normal[0] /= measure;
            normal[1] /= measure;
            normal[2] /= measure;
        }
        else
        {
            continue;
        }

        double dispValuesAndDersData[5 * 2 * 3] = {0.0};
        DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDersData, P1, 2 * dim);
        dispBasis.evalAllDers_into(fullPt, 1, dispValuesAndDers);

        double dNData[3] = {0.0, 0.0, 0.0};
        DeviceVectorView<double> dN(dNData, dim);
        tensorBasisDerivative(shapeFuncIdx, P1, dim, 1, dispValuesAndDers, dN);

        geoJacobian.inverse(geoJacobianInvTrans);
        geoJacobianInvTrans.transpose();

        double physGradData[3] = {0.0, 0.0, 0.0};
        DeviceVectorView<double> physGrad(physGradData, dim);
        geoJacobianInvTrans.times(dN, physGrad);

        double normalDerivative = 0.0;
        for (int d = 0; d < dim; ++d)
            normalDerivative += physGrad[d] * normal[d];

        const int activeIndex = dispBasis.activeIndex(fullPt, shapeFuncIdx);
        const double weightedNormalDerivative = weight * measure * normalDerivative;
        for (int di = 0; di < targetDim; ++di)
        {
            const int globalIndex = system.mapColIndex(activeIndex, patchIdx, di);
            system.pushToRhs(weightedNormalDerivative * bcValues(di, bcIdx), globalIndex, di);
        }
    }
}

__device__
void followerMomentBoundaryPoint(int localBoundaryGPIdx,
                                 TensorBsplineBasisDeviceView dispBasis,
                                 BoxSide_d side,
                                 MultiGaussPointsDeviceView gaussPoints,
                                 int patchIdx,
                                 DeviceVectorView<double> fullPt,
                                 DeviceVectorView<double> boundaryPt,
                                 double& weight)
{
    const int dim = dispBasis.dim();
    const int fixedDir = side.direction();
    int gpRemainder = localBoundaryGPIdx;
    weight = 1.0;

    for (int d = 0, bd = 0; d < dim; ++d)
    {
        if (d == fixedDir)
        {
            fullPt[d] = side.parameter()
                ? dispBasis.knotVector(d).domainEnd()
                : dispBasis.knotVector(d).domainBegin();
            continue;
        }

        const int totalInDir = dispBasis.totalNumGPsInDir(d);
        const int oneDimGPIdx = gpRemainder % totalInDir;
        gpRemainder /= totalInDir;

        double coordinate = 0.0;
        weight *= dispBasis.gsPoint(oneDimGPIdx, d, gaussPoints[patchIdx], coordinate);
        fullPt[d] = coordinate;
        boundaryPt[bd++] = coordinate;
    }
}

static constexpr int FOLLOWER_MOMENT_STATS_STRIDE = 9;
static constexpr int FOLLOWER_MOMENT_DERIVATIVE_STRIDE = 6;

__device__
int followerMomentBoundaryDerivativeDirection(BoxSide_d side)
{
    const int fixedDir = side.direction();
    return fixedDir == 0 ? 1 : 0;
}

__device__
void followerMomentCurrentBoundaryKinematics(MultiPatchDeviceView displacement,
                                             MultiPatchDeviceView geometry,
                                             int patchIdx,
                                             BoxSide_d side,
                                             DeviceVectorView<double> fullPt,
                                             DeviceVectorView<double> boundaryPt,
                                             DeviceVectorView<double> currentPoint,
                                             DeviceVectorView<double> tangent,
                                             DeviceVectorView<double> normal,
                                             double& measure,
                                             double& orientation)
{
    PatchDeviceView geoPatch = geometry.patch(patchIdx);
    PatchDeviceView dispPatch = displacement.patch(patchIdx);

    double geoPointData[2] = {0.0, 0.0};
    double dispPointData[2] = {0.0, 0.0};
    DeviceVectorView<double> geoPoint(geoPointData, 2);
    DeviceVectorView<double> dispPoint(dispPointData, 2);
    geoPatch.evaluate(fullPt, geoPoint);
    dispPatch.evaluate(fullPt, dispPoint);
    for (int i = 0; i < 2; ++i)
        currentPoint[i] = geoPoint[i] + dispPoint[i];

    double geoJacobianData[4] = {0.0, 0.0, 0.0, 0.0};
    double dispJacobianData[4] = {0.0, 0.0, 0.0, 0.0};
    DeviceMatrixView<double> geoJacobian(geoJacobianData, 2, 2);
    DeviceMatrixView<double> dispJacobian(dispJacobianData, 2, 2);
    geoPatch.jacobian(fullPt, 1, geoJacobian);
    dispPatch.jacobian(fullPt, 1, dispJacobian);

    const int tangentDirection = followerMomentBoundaryDerivativeDirection(side);
    const double tx = geoJacobian(0, tangentDirection) +
                      dispJacobian(0, tangentDirection);
    const double ty = geoJacobian(1, tangentDirection) +
                      dispJacobian(1, tangentDirection);
    measure = sqrt(tx * tx + ty * ty);
    if (measure <= 0.0)
    {
        tangent[0] = 1.0;
        tangent[1] = 0.0;
        normal[0] = 0.0;
        normal[1] = 1.0;
        orientation = 1.0;
        return;
    }

    tangent[0] = tx / measure;
    tangent[1] = ty / measure;

    const bool clockwiseNormal = side.parameter() == (side.direction() == 0);
    if (clockwiseNormal)
    {
        normal[0] = tangent[1];
        normal[1] = -tangent[0];
    }
    else
    {
        normal[0] = -tangent[1];
        normal[1] = tangent[0];
    }
    orientation = tangent[0] * normal[1] - tangent[1] * normal[0];
}

__device__
void followerMomentSectionNormal(BoxSide_d side,
                                 DeviceVectorView<double> tangent,
                                 DeviceVectorView<double> normal,
                                 double& orientation)
{
    const bool clockwiseNormal = side.parameter() == (side.direction() == 0);
    if (clockwiseNormal)
    {
        normal[0] = tangent[1];
        normal[1] = -tangent[0];
    }
    else
    {
        normal[0] = -tangent[1];
        normal[1] = tangent[0];
    }
    orientation = tangent[0] * normal[1] - tangent[1] * normal[0];
}

__device__
void followerMomentSectionNormalDerivative(BoxSide_d side,
                                           DeviceVectorView<double> tangentDerivative,
                                           DeviceVectorView<double> normalDerivative)
{
    const bool clockwiseNormal = side.parameter() == (side.direction() == 0);
    if (clockwiseNormal)
    {
        normalDerivative[0] = tangentDerivative[1];
        normalDerivative[1] = -tangentDerivative[0];
    }
    else
    {
        normalDerivative[0] = -tangentDerivative[1];
        normalDerivative[1] = tangentDerivative[0];
    }
}

__global__
void computeFollowerMomentBoundaryCentroidKernel(int totalNumBoundaryGPs,
                                                 MultiPatchDeviceView displacement,
                                                 MultiPatchDeviceView geometry,
                                                 MultiGaussPointsDeviceView gaussPoints,
                                                 DeviceVectorView<int> bcOffsets,
                                                 DeviceVectorView<int> bcPatches,
                                                 DeviceVectorView<int> bcSideIndexes,
                                                 DeviceVectorView<double> stats)
{
    for (int boundaryGPIdx = blockIdx.x * blockDim.x + threadIdx.x;
         boundaryGPIdx < totalNumBoundaryGPs;
         boundaryGPIdx += blockDim.x * gridDim.x)
    {
        int bcIdx = 0;
        while (bcIdx + 1 < bcOffsets.size() && boundaryGPIdx >= bcOffsets[bcIdx + 1])
            ++bcIdx;

        const int patchIdx = bcPatches[bcIdx];
        BoxSide_d side(bcSideIndexes[bcIdx]);
        const int localBoundaryGPIdx = boundaryGPIdx - bcOffsets[bcIdx];
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);

        double fullPtData[2] = {0.0, 0.0};
        double boundaryPtData[1] = {0.0};
        DeviceVectorView<double> fullPt(fullPtData, 2);
        DeviceVectorView<double> boundaryPt(boundaryPtData, 1);

        double weight = 1.0;
        followerMomentBoundaryPoint(localBoundaryGPIdx, dispBasis, side,
                                    gaussPoints, patchIdx, fullPt, boundaryPt,
                                    weight);

        double currentPointData[2] = {0.0, 0.0};
        double tangentData[2] = {0.0, 0.0};
        double normalData[2] = {0.0, 0.0};
        DeviceVectorView<double> currentPoint(currentPointData, 2);
        DeviceVectorView<double> tangent(tangentData, 2);
        DeviceVectorView<double> normal(normalData, 2);
        double measure = 0.0;
        double orientation = 1.0;
        followerMomentCurrentBoundaryKinematics(displacement, geometry, patchIdx,
                                                side, fullPt, boundaryPt,
                                                currentPoint, tangent, normal,
                                                measure, orientation);
        const double weightedMeasure = weight * measure;
        const int statsBase = FOLLOWER_MOMENT_STATS_STRIDE * bcIdx;
        atomicAdd(&stats[statsBase + 0], weightedMeasure);
        atomicAdd(&stats[statsBase + 1], weightedMeasure * currentPoint[0]);
        atomicAdd(&stats[statsBase + 2], weightedMeasure * currentPoint[1]);
        atomicAdd(&stats[statsBase + 4], weightedMeasure * tangent[0]);
        atomicAdd(&stats[statsBase + 5], weightedMeasure * tangent[1]);
    }
}

__global__
void normalizeFollowerMomentBoundaryCentroidKernel(int numFollowerMomentSides,
                                                   DeviceVectorView<double> stats)
{
    for (int bcIdx = blockIdx.x * blockDim.x + threadIdx.x;
         bcIdx < numFollowerMomentSides;
         bcIdx += blockDim.x * gridDim.x)
    {
        const int statsBase = FOLLOWER_MOMENT_STATS_STRIDE * bcIdx;
        const double measure = stats[statsBase + 0];
        if (measure > 0.0)
        {
            stats[statsBase + 1] /= measure;
            stats[statsBase + 2] /= measure;
            const double tangentNorm = sqrt(
                stats[statsBase + 4] * stats[statsBase + 4] +
                stats[statsBase + 5] * stats[statsBase + 5]);
            stats[statsBase + 6] = tangentNorm;
            if (tangentNorm > 0.0)
            {
                stats[statsBase + 4] /= tangentNorm;
                stats[statsBase + 5] /= tangentNorm;
            }
        }
    }
}

__global__
void computeFollowerMomentBoundaryInertiaKernel(int totalNumBoundaryGPs,
                                                MultiPatchDeviceView displacement,
                                                MultiPatchDeviceView geometry,
                                                MultiGaussPointsDeviceView gaussPoints,
                                                DeviceVectorView<int> bcOffsets,
                                                DeviceVectorView<int> bcPatches,
                                                DeviceVectorView<int> bcSideIndexes,
                                                DeviceVectorView<double> stats)
{
    for (int boundaryGPIdx = blockIdx.x * blockDim.x + threadIdx.x;
         boundaryGPIdx < totalNumBoundaryGPs;
         boundaryGPIdx += blockDim.x * gridDim.x)
    {
        int bcIdx = 0;
        while (bcIdx + 1 < bcOffsets.size() && boundaryGPIdx >= bcOffsets[bcIdx + 1])
            ++bcIdx;

        const int patchIdx = bcPatches[bcIdx];
        BoxSide_d side(bcSideIndexes[bcIdx]);
        const int localBoundaryGPIdx = boundaryGPIdx - bcOffsets[bcIdx];
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);

        double fullPtData[2] = {0.0, 0.0};
        double boundaryPtData[1] = {0.0};
        DeviceVectorView<double> fullPt(fullPtData, 2);
        DeviceVectorView<double> boundaryPt(boundaryPtData, 1);

        double weight = 1.0;
        followerMomentBoundaryPoint(localBoundaryGPIdx, dispBasis, side,
                                    gaussPoints, patchIdx, fullPt, boundaryPt,
                                    weight);

        double currentPointData[2] = {0.0, 0.0};
        double tangentData[2] = {0.0, 0.0};
        double normalData[2] = {0.0, 0.0};
        DeviceVectorView<double> currentPoint(currentPointData, 2);
        DeviceVectorView<double> tangent(tangentData, 2);
        DeviceVectorView<double> normal(normalData, 2);
        double measure = 0.0;
        double orientation = 1.0;
        followerMomentCurrentBoundaryKinematics(displacement, geometry, patchIdx,
                                                side, fullPt, boundaryPt,
                                                currentPoint, tangent, normal,
                                                measure, orientation);

        const int statsBase = FOLLOWER_MOMENT_STATS_STRIDE * bcIdx;
        const double dx = currentPoint[0] - stats[statsBase + 1];
        const double dy = currentPoint[1] - stats[statsBase + 2];
        const double q = dx * stats[statsBase + 4] + dy * stats[statsBase + 5];
        atomicAdd(&stats[statsBase + 3], weight * measure * q * q);
        atomicAdd(&stats[statsBase + 7], weight * measure * q * dx);
        atomicAdd(&stats[statsBase + 8], weight * measure * q * dy);
    }
}

__global__
void computeFollowerMomentBoundaryDerivativeStatsKernel(
    int totalNumBoundaryGPs,
    int numActive,
    MultiPatchDeviceView displacement,
    MultiPatchDeviceView geometry,
    MultiGaussPointsDeviceView gaussPoints,
    DeviceVectorView<int> bcOffsets,
    DeviceVectorView<int> bcPatches,
    DeviceVectorView<int> bcSideIndexes,
    DeviceVectorView<int> bcDofOffsets,
    DeviceVectorView<double> derivativeStats)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalNumBoundaryGPs * numActive * 2;
         idx += blockDim.x * gridDim.x)
    {
        const int component = idx % 2;
        const int shapeFuncIdx = (idx / 2) % numActive;
        const int boundaryGPIdx = idx / (2 * numActive);

        int bcIdx = 0;
        while (bcIdx + 1 < bcOffsets.size() && boundaryGPIdx >= bcOffsets[bcIdx + 1])
            ++bcIdx;

        const int patchIdx = bcPatches[bcIdx];
        BoxSide_d side(bcSideIndexes[bcIdx]);
        const int localBoundaryGPIdx = boundaryGPIdx - bcOffsets[bcIdx];
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
        const int P1 = dispBasis.knotsOrder(0) + 1;

        double fullPtData[2] = {0.0, 0.0};
        double boundaryPtData[1] = {0.0};
        DeviceVectorView<double> fullPt(fullPtData, 2);
        DeviceVectorView<double> boundaryPt(boundaryPtData, 1);

        double weight = 1.0;
        followerMomentBoundaryPoint(localBoundaryGPIdx, dispBasis, side,
                                    gaussPoints, patchIdx, fullPt, boundaryPt,
                                    weight);

        double currentPointData[2] = {0.0, 0.0};
        double tangentData[2] = {0.0, 0.0};
        double normalData[2] = {0.0, 0.0};
        DeviceVectorView<double> currentPoint(currentPointData, 2);
        DeviceVectorView<double> tangent(tangentData, 2);
        DeviceVectorView<double> normal(normalData, 2);
        double measure = 0.0;
        double orientation = 1.0;
        followerMomentCurrentBoundaryKinematics(displacement, geometry, patchIdx,
                                                side, fullPt, boundaryPt,
                                                currentPoint, tangent, normal,
                                                measure, orientation);

        double dispValuesAndDersData[5 * 4] = {0.0};
        DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDersData, P1, 4);
        dispBasis.evalAllDers_into(fullPt, 1, dispValuesAndDers);

        const double N = tensorBasisValue(shapeFuncIdx, P1, 2, 1, dispValuesAndDers);
        double dNData[2] = {0.0, 0.0};
        DeviceVectorView<double> dN(dNData, 2);
        tensorBasisDerivative(shapeFuncIdx, P1, 2, 1, dispValuesAndDers, dN);
        const double boundaryDerivative =
            dN[followerMomentBoundaryDerivativeDirection(side)];

        const double measureDerivative = tangent[component] * boundaryDerivative;
        const double deltaX = component == 0 ? N : 0.0;
        const double deltaY = component == 1 ? N : 0.0;

        const int activeIndex = dispBasis.activeIndex(fullPt, shapeFuncIdx);
        const int dofIndex = bcDofOffsets[bcIdx] + activeIndex * 2 + component;
        const int derivativeBase = FOLLOWER_MOMENT_DERIVATIVE_STRIDE * dofIndex;

        atomicAdd(&derivativeStats[derivativeBase + 0],
                  weight * measureDerivative);
        atomicAdd(&derivativeStats[derivativeBase + 1],
                  weight * (measureDerivative * currentPoint[0] +
                            measure * deltaX));
        atomicAdd(&derivativeStats[derivativeBase + 2],
                  weight * (measureDerivative * currentPoint[1] +
                            measure * deltaY));
        atomicAdd(&derivativeStats[derivativeBase + 4],
                  weight * boundaryDerivative * (component == 0 ? 1.0 : 0.0));
        atomicAdd(&derivativeStats[derivativeBase + 5],
                  weight * boundaryDerivative * (component == 1 ? 1.0 : 0.0));
    }
}

__global__
void computeFollowerMomentBoundaryInertiaDerivativeKernel(
    int totalNumBoundaryGPs,
    int numActive,
    MultiPatchDeviceView displacement,
    MultiPatchDeviceView geometry,
    MultiGaussPointsDeviceView gaussPoints,
    DeviceVectorView<int> bcOffsets,
    DeviceVectorView<int> bcPatches,
    DeviceVectorView<int> bcSideIndexes,
    DeviceVectorView<int> bcDofOffsets,
    DeviceVectorView<double> stats,
    DeviceVectorView<double> derivativeStats)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalNumBoundaryGPs * numActive * 2;
         idx += blockDim.x * gridDim.x)
    {
        const int component = idx % 2;
        const int shapeFuncIdx = (idx / 2) % numActive;
        const int boundaryGPIdx = idx / (2 * numActive);

        int bcIdx = 0;
        while (bcIdx + 1 < bcOffsets.size() && boundaryGPIdx >= bcOffsets[bcIdx + 1])
            ++bcIdx;

        const int patchIdx = bcPatches[bcIdx];
        BoxSide_d side(bcSideIndexes[bcIdx]);
        const int localBoundaryGPIdx = boundaryGPIdx - bcOffsets[bcIdx];
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
        const int P1 = dispBasis.knotsOrder(0) + 1;

        double fullPtData[2] = {0.0, 0.0};
        double boundaryPtData[1] = {0.0};
        DeviceVectorView<double> fullPt(fullPtData, 2);
        DeviceVectorView<double> boundaryPt(boundaryPtData, 1);

        double weight = 1.0;
        followerMomentBoundaryPoint(localBoundaryGPIdx, dispBasis, side,
                                    gaussPoints, patchIdx, fullPt, boundaryPt,
                                    weight);

        double currentPointData[2] = {0.0, 0.0};
        double tangentData[2] = {0.0, 0.0};
        double normalData[2] = {0.0, 0.0};
        DeviceVectorView<double> currentPoint(currentPointData, 2);
        DeviceVectorView<double> tangent(tangentData, 2);
        DeviceVectorView<double> normal(normalData, 2);
        double measure = 0.0;
        double orientation = 1.0;
        followerMomentCurrentBoundaryKinematics(displacement, geometry, patchIdx,
                                                side, fullPt, boundaryPt,
                                                currentPoint, tangent, normal,
                                                measure, orientation);

        double dispValuesAndDersData[5 * 4] = {0.0};
        DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDersData, P1, 4);
        dispBasis.evalAllDers_into(fullPt, 1, dispValuesAndDers);

        const double N = tensorBasisValue(shapeFuncIdx, P1, 2, 1, dispValuesAndDers);
        double dNData[2] = {0.0, 0.0};
        DeviceVectorView<double> dN(dNData, 2);
        tensorBasisDerivative(shapeFuncIdx, P1, 2, 1, dispValuesAndDers, dN);
        const double boundaryDerivative =
            dN[followerMomentBoundaryDerivativeDirection(side)];
        const double measureDerivative = tangent[component] * boundaryDerivative;

        const int activeIndex = dispBasis.activeIndex(fullPt, shapeFuncIdx);
        const int dofIndex = bcDofOffsets[bcIdx] + activeIndex * 2 + component;
        const int derivativeBase = FOLLOWER_MOMENT_DERIVATIVE_STRIDE * dofIndex;
        const int statsBase = FOLLOWER_MOMENT_STATS_STRIDE * bcIdx;

        const double totalMeasure = stats[statsBase + 0];
        const double centroidX = stats[statsBase + 1];
        const double centroidY = stats[statsBase + 2];
        const double tangentX = stats[statsBase + 4];
        const double tangentY = stats[statsBase + 5];
        const double tangentNorm = stats[statsBase + 6];

        const double dx = currentPoint[0] - centroidX;
        const double dy = currentPoint[1] - centroidY;
        const double q = dx * tangentX + dy * tangentY;
        const double deltaDotT = N * (component == 0 ? tangentX : tangentY);

        atomicAdd(&derivativeStats[derivativeBase + 3],
                  weight * (measureDerivative * q * q +
                            2.0 * measure * q * deltaDotT));
    }
}

__global__
void addFollowerMomentBoundaryGlobalInertiaDerivativeKernel(
    int numFollowerMomentSides,
    DeviceVectorView<int> bcDofOffsets,
    DeviceVectorView<double> stats,
    DeviceVectorView<double> derivativeStats)
{
    const int totalDofs = bcDofOffsets[numFollowerMomentSides];
    for (int dofIndex = blockIdx.x * blockDim.x + threadIdx.x;
         dofIndex < totalDofs;
         dofIndex += blockDim.x * gridDim.x)
    {
        int bcIdx = 0;
        while (bcIdx + 1 < bcDofOffsets.size() &&
               dofIndex >= bcDofOffsets[bcIdx + 1])
            ++bcIdx;

        const int statsBase = FOLLOWER_MOMENT_STATS_STRIDE * bcIdx;
        const int derivativeBase =
            FOLLOWER_MOMENT_DERIVATIVE_STRIDE * dofIndex;
        const double tangentNorm = stats[statsBase + 6];
        if (tangentNorm <= 0.0)
            continue;

        const double tangentX = stats[statsBase + 4];
        const double tangentY = stats[statsBase + 5];
        const double rawTangentDerivativeX =
            derivativeStats[derivativeBase + 4];
        const double rawTangentDerivativeY =
            derivativeStats[derivativeBase + 5];
        const double projection =
            tangentX * rawTangentDerivativeX +
            tangentY * rawTangentDerivativeY;
        const double tangentDerivativeX =
            (rawTangentDerivativeX - tangentX * projection) / tangentNorm;
        const double tangentDerivativeY =
            (rawTangentDerivativeY - tangentY * projection) / tangentNorm;

        const double qMomentX = stats[statsBase + 7];
        const double qMomentY = stats[statsBase + 8];
        derivativeStats[derivativeBase + 3] +=
            2.0 * (qMomentX * tangentDerivativeX +
                   qMomentY * tangentDerivativeY);
    }
}

__global__
void assembleFollowerMomentBoundaryConditionKernel(int totalNumBoundaryGPs,
                                                   int numActive,
                                                   MultiPatchDeviceView displacement,
                                                   MultiPatchDeviceView geometry,
                                                   SparseSystemDeviceView system,
                                                   DeviceNestedArrayView<double> eliminatedDofs,
                                                   MultiGaussPointsDeviceView gaussPoints,
                                                   DeviceVectorView<int> bcOffsets,
                                                   DeviceVectorView<int> bcPatches,
                                                   DeviceVectorView<int> bcSideIndexes,
                                                   DeviceVectorView<int> bcDofOffsets,
                                                   DeviceVectorView<double> momentValues,
                                                   DeviceVectorView<double> stats,
                                                   DeviceVectorView<double> derivativeStats)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalNumBoundaryGPs * numActive;
         idx += blockDim.x * gridDim.x)
    {
        const int shapeFuncIdx = idx % numActive;
        const int boundaryGPIdx = idx / numActive;

        int bcIdx = 0;
        while (bcIdx + 1 < bcOffsets.size() && boundaryGPIdx >= bcOffsets[bcIdx + 1])
            ++bcIdx;

        const int patchIdx = bcPatches[bcIdx];
        BoxSide_d side(bcSideIndexes[bcIdx]);
        const int localBoundaryGPIdx = boundaryGPIdx - bcOffsets[bcIdx];
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
        const int P1 = dispBasis.knotsOrder(0) + 1;

        double fullPtData[2] = {0.0, 0.0};
        double boundaryPtData[1] = {0.0};
        DeviceVectorView<double> fullPt(fullPtData, 2);
        DeviceVectorView<double> boundaryPt(boundaryPtData, 1);

        double weight = 1.0;
        followerMomentBoundaryPoint(localBoundaryGPIdx, dispBasis, side,
                                    gaussPoints, patchIdx, fullPt, boundaryPt,
                                    weight);

        double currentPointData[2] = {0.0, 0.0};
        double tangentData[2] = {0.0, 0.0};
        double normalData[2] = {0.0, 0.0};
        DeviceVectorView<double> currentPoint(currentPointData, 2);
        DeviceVectorView<double> tangent(tangentData, 2);
        DeviceVectorView<double> normal(normalData, 2);
        double measure = 0.0;
        double orientation = 1.0;
        followerMomentCurrentBoundaryKinematics(displacement, geometry, patchIdx,
                                                side, fullPt, boundaryPt,
                                                currentPoint, tangent, normal,
                                                measure, orientation);
        const double localTangentX = tangent[0];
        const double localTangentY = tangent[1];
        const int statsBase = FOLLOWER_MOMENT_STATS_STRIDE * bcIdx;
        tangent[0] = stats[statsBase + 4];
        tangent[1] = stats[statsBase + 5];
        followerMomentSectionNormal(side, tangent, normal, orientation);

        const double inertia = stats[statsBase + 3];
        if (inertia <= 0.0 || fabs(orientation) <= 0.0)
            continue;

        double dispValuesAndDersData[5 * 4] = {0.0};
        DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDersData, P1, 4);
        dispBasis.evalAllDers_into(fullPt, 1, dispValuesAndDers);

        const double N = tensorBasisValue(shapeFuncIdx, P1, 2, 1, dispValuesAndDers);
        const int activeIndex = dispBasis.activeIndex(fullPt, shapeFuncIdx);
        const double dx = currentPoint[0] - stats[statsBase + 1];
        const double dy = currentPoint[1] - stats[statsBase + 2];
        const double q = dx * tangent[0] + dy * tangent[1];
        const double alpha = momentValues[bcIdx] / (orientation * inertia);
        const double weightedBasis = N * weight * measure;

        const int globalIndex0 = system.mapColIndex(activeIndex, patchIdx, 0);
        const int globalIndex1 = system.mapColIndex(activeIndex, patchIdx, 1);
        system.pushToRhs(weightedBasis * alpha * q * normal[0], globalIndex0, 0);
        system.pushToRhs(weightedBasis * alpha * q * normal[1], globalIndex1, 1);

        for (int colShapeFuncIdx = 0; colShapeFuncIdx < numActive; ++colShapeFuncIdx)
        {
            const double N_col = tensorBasisValue(colShapeFuncIdx, P1, 2, 1, dispValuesAndDers);
            double dNColData[2] = {0.0, 0.0};
            DeviceVectorView<double> dNCol(dNColData, 2);
            tensorBasisDerivative(colShapeFuncIdx, P1, 2, 1,
                                  dispValuesAndDers, dNCol);
            const double boundaryDerivative =
                dNCol[followerMomentBoundaryDerivativeDirection(side)];
            const int activeIndexCol = dispBasis.activeIndex(fullPt, colShapeFuncIdx);
            for (int rowComp = 0; rowComp < 2; ++rowComp)
            {
                const int activeIndexRow = activeIndex;
                for (int colComp = 0; colComp < 2; ++colComp)
                {
                    const double localMeasureDerivative =
                        (colComp == 0 ? localTangentX : localTangentY) *
                        boundaryDerivative;
                    const double deltaDotT = N_col * tangent[colComp];

                    const double dFext =
                        N * weight *
                        (localMeasureDerivative * alpha * q * normal[rowComp] +
                         measure * alpha * deltaDotT * normal[rowComp]);
                    const int globalIndexRow =
                        system.mapColIndex(activeIndexRow, patchIdx, rowComp);
                    const int globalIndexCol =
                        system.mapColIndex(activeIndexCol, patchIdx, colComp);
                    system.pushToMatrix(-dFext, globalIndexRow, globalIndexCol,
                                        eliminatedDofs, rowComp, colComp);
                }
            }
        }
    }
}

__global__
void assembleFollowerMomentBoundaryGlobalTangentKernel(
    int totalNumBoundaryGPs,
    int numActive,
    MultiPatchDeviceView displacement,
    MultiPatchDeviceView geometry,
    SparseSystemDeviceView system,
    DeviceNestedArrayView<double> eliminatedDofs,
    MultiGaussPointsDeviceView gaussPoints,
    DeviceVectorView<int> bcOffsets,
    DeviceVectorView<int> bcPatches,
    DeviceVectorView<int> bcSideIndexes,
    DeviceVectorView<int> bcDofOffsets,
    DeviceVectorView<int> bcBoundaryDofOffsets,
    DeviceVectorView<int> bcBoundaryDofs,
    DeviceVectorView<double> momentValues,
    DeviceVectorView<double> stats,
    DeviceVectorView<double> derivativeStats)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalNumBoundaryGPs * numActive;
         idx += blockDim.x * gridDim.x)
    {
        const int shapeFuncIdx = idx % numActive;
        const int boundaryGPIdx = idx / numActive;

        int bcIdx = 0;
        while (bcIdx + 1 < bcOffsets.size() &&
               boundaryGPIdx >= bcOffsets[bcIdx + 1])
            ++bcIdx;

        const int patchIdx = bcPatches[bcIdx];
        BoxSide_d side(bcSideIndexes[bcIdx]);
        const int localBoundaryGPIdx = boundaryGPIdx - bcOffsets[bcIdx];
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
        const int P1 = dispBasis.knotsOrder(0) + 1;

        double fullPtData[2] = {0.0, 0.0};
        double boundaryPtData[1] = {0.0};
        DeviceVectorView<double> fullPt(fullPtData, 2);
        DeviceVectorView<double> boundaryPt(boundaryPtData, 1);

        double weight = 1.0;
        followerMomentBoundaryPoint(localBoundaryGPIdx, dispBasis, side,
                                    gaussPoints, patchIdx, fullPt, boundaryPt,
                                    weight);

        double currentPointData[2] = {0.0, 0.0};
        double tangentData[2] = {0.0, 0.0};
        double normalData[2] = {0.0, 0.0};
        DeviceVectorView<double> currentPoint(currentPointData, 2);
        DeviceVectorView<double> tangent(tangentData, 2);
        DeviceVectorView<double> normal(normalData, 2);
        double measure = 0.0;
        double orientation = 1.0;
        followerMomentCurrentBoundaryKinematics(displacement, geometry,
                                                patchIdx, side, fullPt,
                                                boundaryPt, currentPoint,
                                                tangent, normal, measure,
                                                orientation);

        const int statsBase = FOLLOWER_MOMENT_STATS_STRIDE * bcIdx;
        tangent[0] = stats[statsBase + 4];
        tangent[1] = stats[statsBase + 5];
        followerMomentSectionNormal(side, tangent, normal, orientation);

        const double inertia = stats[statsBase + 3];
        if (inertia <= 0.0 || fabs(orientation) <= 0.0)
            continue;

        double dispValuesAndDersData[5 * 4] = {0.0};
        DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDersData, P1,
                                                   4);
        dispBasis.evalAllDers_into(fullPt, 1, dispValuesAndDers);

        const double N =
            tensorBasisValue(shapeFuncIdx, P1, 2, 1, dispValuesAndDers);
        const int activeIndexRow =
            dispBasis.activeIndex(fullPt, shapeFuncIdx);
        const double dx = currentPoint[0] - stats[statsBase + 1];
        const double dy = currentPoint[1] - stats[statsBase + 2];
        const double q = dx * tangent[0] + dy * tangent[1];
        const double alpha = momentValues[bcIdx] / (orientation * inertia);
        const double totalMeasure = stats[statsBase + 0];
        const double tangentNorm = stats[statsBase + 6];

        const int boundaryStart = bcBoundaryDofOffsets[bcIdx];
        const int boundaryEnd = bcBoundaryDofOffsets[bcIdx + 1];
        for (int boundaryDofIdx = boundaryStart;
             boundaryDofIdx < boundaryEnd; ++boundaryDofIdx)
        {
            const int activeIndexCol = bcBoundaryDofs[boundaryDofIdx];
            for (int rowComp = 0; rowComp < 2; ++rowComp)
            {
                for (int colComp = 0; colComp < 2; ++colComp)
                {
                    const int dofIndex =
                        bcDofOffsets[bcIdx] + activeIndexCol * 2 + colComp;
                    const int derivativeBase =
                        FOLLOWER_MOMENT_DERIVATIVE_STRIDE * dofIndex;

                    double centroidDerivativeX = 0.0;
                    double centroidDerivativeY = 0.0;
                    if (totalMeasure > 0.0)
                    {
                        centroidDerivativeX =
                            (derivativeStats[derivativeBase + 1] -
                             stats[statsBase + 1] *
                                 derivativeStats[derivativeBase + 0]) /
                            totalMeasure;
                        centroidDerivativeY =
                            (derivativeStats[derivativeBase + 2] -
                             stats[statsBase + 2] *
                                 derivativeStats[derivativeBase + 0]) /
                            totalMeasure;
                    }

                    double tangentDerivativeData[2] = {0.0, 0.0};
                    double normalDerivativeData[2] = {0.0, 0.0};
                    DeviceVectorView<double> tangentDerivative(
                        tangentDerivativeData, 2);
                    DeviceVectorView<double> normalDerivative(
                        normalDerivativeData, 2);

                    if (tangentNorm > 0.0)
                    {
                        const double rawTangentDerivativeX =
                            derivativeStats[derivativeBase + 4];
                        const double rawTangentDerivativeY =
                            derivativeStats[derivativeBase + 5];
                        const double projection =
                            tangent[0] * rawTangentDerivativeX +
                            tangent[1] * rawTangentDerivativeY;
                        tangentDerivative[0] =
                            (rawTangentDerivativeX - tangent[0] * projection) /
                            tangentNorm;
                        tangentDerivative[1] =
                            (rawTangentDerivativeY - tangent[1] * projection) /
                            tangentNorm;
                    }
                    followerMomentSectionNormalDerivative(
                        side, tangentDerivative, normalDerivative);

                    const double centroidDerivativeDotT =
                        centroidDerivativeX * tangent[0] +
                        centroidDerivativeY * tangent[1];
                    const double qDerivative =
                        -centroidDerivativeDotT +
                        dx * tangentDerivative[0] +
                        dy * tangentDerivative[1];
                    const double alphaDerivative =
                        -alpha * derivativeStats[derivativeBase + 3] / inertia;

                    const double dFext =
                        N * weight * measure *
                        ((alphaDerivative * q + alpha * qDerivative) *
                             normal[rowComp] +
                         alpha * q * normalDerivative[rowComp]);
                    const int globalIndexRow =
                        system.mapColIndex(activeIndexRow, patchIdx, rowComp);
                    const int globalIndexCol =
                        system.mapColIndex(activeIndexCol, patchIdx, colComp);
                    system.pushToMatrix(-dFext, globalIndexRow, globalIndexCol,
                                        eliminatedDofs, rowComp, colComp);
                }
            }
        }
    }
}

__global__
void assembleNeumannCornerPointLoadKernel(int numCornerLoads,
                                          SparseSystemDeviceView system,
                                          DeviceVectorView<int> cornerPatches,
                                          DeviceVectorView<int> cornerDofs,
                                          DeviceMatrixView<double> loadValues)
{
    const int targetDim = loadValues.rows();
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < numCornerLoads * targetDim;
         idx += blockDim.x * gridDim.x)
    {
        const int component = idx % targetDim;
        const int loadIdx = idx / targetDim;
        const int activeRow = system.mapColIndex(cornerDofs[loadIdx],
                                                 cornerPatches[loadIdx],
                                                 component);
        system.pushToRhs(loadValues(component, loadIdx), activeRow, component);
    }
}

__global__
void assembleMatrixWithGPDataKernel(int numDerivatives, int EleStartId,
                      int inputGPStartId, int numElementsBatched, int N_D,
                      MultiPatchDeviceView displacement,
                      SparseSystemDeviceView system,
                      DeviceNestedArrayView<double> eliminatedDofs,
                      DeviceMatrixView<double> pts,
                      //DeviceVectorView<double> wts,
                      DeviceMatrixView<double> geoJacobianInvs,
                      DeviceVectorView<double> measures,
                      DeviceVectorView<double> weightForces,
                      DeviceVectorView<double> weightBodys,
                      DeviceMatrixView<double> dispValuesAndDerss,
                      DeviceMatrixView<double> Fs, 
                      DeviceMatrixView<double> Ss, 
                      DeviceMatrixView<double> Cs)
{
    int totalNumShapeFunctions = numElementsBatched * N_D * N_D;
    //printf("totalNumShapeFunctions: %d\n", totalNumShapeFunctions);
    for (int bidx = blockIdx.x; bidx < totalNumShapeFunctions; bidx += gridDim.x)
    {
        __shared__ int patch_idx, ele_idx, shapeFuncCoord[2], idx;
        __shared__ double localMatrxData[3*3]; //max 3D

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

        for (int i = threadId; i < dim * dim; i += blockDim.x * blockDim.y)
            localMatrxData[i] = 0.0;
        __syncthreads();
        //localMatrix.print();

        DeviceMatrixView<double> localMatrix(localMatrxData, dim, dim);
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
        int P1 = dispBasis.knotsOrder(0) + 1;
        int i = shapeFuncCoord[0];
        int j = shapeFuncCoord[1];
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
            DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDerss.data() + inputGPIdx * P1 * (numDerivatives + 1) * dim, P1, (numDerivatives + 1) * dim);
            //dispValuesAndDers.print();
            DeviceMatrixView<double> F(Fs.data() + GPIdx * dim * dim, dim, dim);
            //F.print();
            DeviceMatrixView<double> S(Ss.data() + GPIdx * dim * dim, dim, dim);
            //S.print();
            DeviceMatrixView<double> C(Cs.data() + GPIdx * dimTensor * dimTensor, dimTensor, dimTensor);
            //if(blockIdx.x == 334 && threadId == 31)
            //    C.print();

            double dN_iData[3] = {0.0}; //max 3D
            double dN_jData[3] = {0.0}; //max 3D
            DeviceVectorView<double> dN_i(dN_iData, dim);
            DeviceVectorView<double> dN_j(dN_jData, dim);
            tensorBasisDerivative(i, P1, dim, numDerivatives, dispValuesAndDers, dN_i);
            //dN_i.print();
            tensorBasisDerivative(j, P1, dim, numDerivatives, dispValuesAndDers, dN_j);
            //dN_j.print();
            double physGrad_jData[3] = {0.0}; //max 3D
            double physGrad_iData[3] = {0.0}; //max 3D
            DeviceVectorView<double> physGrad_i(physGrad_iData, dim);
            DeviceVectorView<double> physGrad_j(physGrad_jData, dim);
            geoJacobianInv.transposeTime(dN_i, physGrad_i);
            //if(blockIdx.x == 334 && threadId == 32)
            //    physGrad_i.print();
            geoJacobianInv.transposeTime(dN_j, physGrad_j);
            //physGrad_j.print();
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
                    double stiffnessEntry = weightForce *materialTangent;
#if 1
                    atomicAdd(&localMatrix(di, dj), stiffnessEntry);
#endif
                    //if(blockIdx.x == 334 && threadIdx.x == 32) {
                    //    localMatrix.print();
                    //    printf("di:%d, dj:%d, stiffnessEntry: %f\n", di, dj, stiffnessEntry);
                    //}
                }
            }
        }
        __syncthreads();
        //printf("bidx:%d, localMatrix:\n", bidx);
        //localMatrix.print();
        //int GPIdx = ele_idx * N_D + threadIdx.x;
#if 1
        //int GPIdx = idx * N_D + threadIdx.x;
        int GPIdx = idx * N_D;
        int inputGPIdx = inputGPStartId + GPIdx;
        DeviceVectorView<double> pt(pts.data() + inputGPIdx * dim, dim);
        for (int tid = threadId; tid < dim * dim; tid += blockDim.x * blockDim.y) {
            int di = tid % dim;
            int dj = tid / dim;
            int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx).activeIndex(pt, i), patch_idx, di);
            int globalIndex_j = system.mapColIndex(displacement.basis(patch_idx).activeIndex(pt, j), patch_idx, dj);
            system.pushToMatrix(localMatrix(di, dj), globalIndex_i, globalIndex_j, eliminatedDofs, di, dj);
            //printf("tid:%d, i:%d, j:%d, di:%d, dj:%d, globalIndex_i:%d, globalIndex_j:%d, value:%f\n", tid, i, j, di, dj, globalIndex_i, globalIndex_j, localMatrix(di, dj));
        }
#endif
    }
}

__device__
void computeMaterialResponse(int materialLaw,
                                   double youngsModulus,
                                   double poissonsRatio,
                                   DeviceMatrixView<double> F,
                                   DeviceMatrixView<double> S,
                                   DeviceMatrixView<double> C);

__global__
void evaluateGPKernel_withoutComputingGPTableAndDers(
                    int numDerivatives, int GPStartId,
                    int inputGPStartId, int numGPBatched,
                    DeviceVectorView<double> parameters,
                    MultiPatchDeviceView displacement,
                    MultiPatchDeviceView multiPatch,
                    DeviceMatrixView<double> pts,
                    DeviceVectorView<double> wts,
                    DeviceMatrixView<double> geoValuesAndDerss,
                    DeviceMatrixView<double> dispValuesAndDerss,
                    DeviceMatrixView<double> geoJacobianInvs,
                    DeviceVectorView<double> measures,
                    DeviceVectorView<double> weightForces,
                    DeviceVectorView<double> weightBodys,
                    DeviceMatrixView<double> Fs, 
                    DeviceMatrixView<double> Ss, 
                    DeviceMatrixView<double> Cs)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numGPBatched; idx += blockDim.x * gridDim.x) {
        

        int dim = multiPatch.domainDim();
        int dimTensor = (dim * (dim + 1)) / 2;

        int GPIdx = GPStartId + idx;
        int inputGPIdx = inputGPStartId + idx;
        DeviceVectorView<double> pt(pts.data() + inputGPIdx * dim, dim);
        double wt = wts[inputGPIdx];
        
        int patch_idx(0);
        int point_idx = displacement.threadPatch(GPIdx, patch_idx);
        PatchDeviceView geoPatch = multiPatch.patch(patch_idx);
        int geoP1 = geoPatch.basis().knotsOrder(0) + 1;
        DeviceMatrixView<double> geoValuesAndDers(geoValuesAndDerss.data() + inputGPIdx * geoP1 * (numDerivatives + 1) * dim, geoP1, (numDerivatives + 1) * dim);

        PatchDeviceView dispPatch = displacement.patch(patch_idx);
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
        int P1 = dispBasis.knotsOrder(0) + 1;
        DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDerss.data() + inputGPIdx * P1 * (numDerivatives + 1) * dim, P1, (numDerivatives + 1) * dim);

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
        DeviceMatrixView<double> C(Cs.data() + idx * dimTensor * dimTensor, dimTensor, dimTensor);
        {
            const int parameterOffset = patch_idx * 3;
            int materialLaw = static_cast<int>(parameters[parameterOffset + 2]);
            double YM = parameters[parameterOffset + 1];
            double PR = parameters[parameterOffset + 0];
            computeMaterialResponse(materialLaw, YM, PR, F, S, C);
        }
    }
}

__global__
void evaluateGPKernel(int numDerivatives, int GPStartId, int numGPBatched,
                      DeviceVectorView<double> parameters,
                      MultiPatchDeviceView displacement,
                      MultiPatchDeviceView multiPatch,
                      MultiGaussPointsDeviceView multiGaussPoints,
                      DeviceMatrixView<double> pts,
                      //DeviceVectorView<double> wts,
                      DeviceMatrixView<double> geoJacobianInvs,
                      DeviceVectorView<double> measures,
                      DeviceVectorView<double> weightForces,
                      DeviceVectorView<double> weightBodys,
                      DeviceMatrixView<double> geoValuesAndDerss,
                      DeviceMatrixView<double> dispValuesAndDerss,
                      DeviceVectorView<double> geoWorkingSpaces,
                      DeviceVectorView<double> dispWorkingSpaces,
                      DeviceMatrixView<double> Fs, 
                      DeviceMatrixView<double> Ss, 
                      DeviceMatrixView<double> Cs)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < numGPBatched; idx += blockDim.x * gridDim.x)
    {
        //printf("idx:%d\n", idx);
        int GPIdx = GPStartId + idx;
        //int CPdim = multiPatch.targetDim();
        int dim = multiPatch.domainDim();

        int patch_idx(0);
        int point_idx = displacement.threadPatch(GPIdx, patch_idx);
        const int parameterOffset = patch_idx * 3;
        double YM = parameters[parameterOffset + 1];
        double PR = parameters[parameterOffset + 0];
        int materialLaw = static_cast<int>(parameters[parameterOffset + 2]);
        DeviceVectorView<double> pt(pts.data() + idx * dim, dim);
        //wts[idx] = displacement.gsPoint(point_idx, patch_idx, multiGaussPoints[patch_idx], pt);
        double wt = displacement.gsPoint(point_idx, patch_idx, multiGaussPoints[patch_idx], pt);
        double geoJacobianData[3*3] = {0.0}; //max 3D
        DeviceMatrixView<double> geoJacobian(geoJacobianData, dim, dim);
        PatchDeviceView geoPatch = multiPatch.patch(patch_idx);
        int geoP1 = geoPatch.basis().knotsOrder(0) + 1;
        double* geoWorkingSpace = geoWorkingSpaces.data() + idx * geoP1 * (geoP1 + 4) * dim;
        DeviceMatrixView<double> geoValuesAndDers(geoValuesAndDerss.data() + idx * geoP1 * (numDerivatives + 1) * dim, geoP1, (numDerivatives + 1) * dim);
        geoPatch.basis().evalAllDers_into(pt, numDerivatives, geoWorkingSpace, geoValuesAndDers);
        geoPatch.jacobian(pt, geoValuesAndDers, numDerivatives, geoJacobian);
        DeviceMatrixView<double> geoJacobianInv(geoJacobianInvs.data() + idx * dim * dim, dim, dim);
        geoJacobian.inverse(geoJacobianInv);
        //printf("geoJacobianInv:\n");
        //geoJacobianInv.print();
        measures[idx] = geoJacobian.determinant();
        weightForces[idx] = wt * measures[idx];
        weightBodys[idx] = wt * measures[idx];
        //printf("weightBodys[idx]: %f\n", weightBodys[idx]);
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
        PatchDeviceView dispPatch = displacement.patch(patch_idx);
        int P1 = dispBasis.knotsOrder(0) + 1;
        //printf("P1:%d\n", P1);
        //printf("dispValuesAndDerss.size(): %d\n", dispValuesAndDerss.size());
        double* dispWorkingSpace = dispWorkingSpaces.data() + idx * P1 * (P1 + 4) * dim;
        DeviceMatrixView<double> dispValuesAndDers(dispValuesAndDerss.data() + idx * P1 * (numDerivatives+1) * dim, P1, (numDerivatives+1)*dim);
        dispBasis.evalAllDers_into(pt, numDerivatives, dispWorkingSpace, dispValuesAndDers);
        //dispValuesAndDers.print();
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
        //F.print();

        DeviceMatrixView<double> S(Ss.data() + idx * dim * dim, dim, dim);
        int dimTensor = (dim * (dim + 1)) / 2;
        DeviceMatrixView<double> C(Cs.data() + idx * dimTensor * dimTensor, dimTensor, dimTensor);
        {
            computeMaterialResponse(materialLaw, YM, PR, F, S, C);
        }
    }
}

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
        __shared__ int dimTensor, ele_idx;
        __shared__ double wt, ptData[3], measure, weightForce, weightBody;
        __shared__ double lambda, mu, J;

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
            
            const int parameterOffset = patch_idx * 3;
            double YM = parameters[parameterOffset + 1];
            double PR = parameters[parameterOffset + 0];
            lambda = YM * PR / ( ( 1. + PR ) * ( 1. - 2. * PR ) );
            mu = YM / ( 2. * ( 1. + PR ) );
            //printf("blockIdx.x=%d, patch_idx=%d, blockCoord=(%d, %d), wt=%f, pt=(%f, %f)\n", 
            //       blockIdx.x, patch_idx, blockCoord[0], blockCoord[1], wt, pt[0], pt[1]);
        }
        __syncthreads();
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
        PatchDeviceView geoPatch = multiPatch.patch(patch_idx);
        PatchDeviceView dispPatch = displacement.patch(patch_idx);
        int N_D = dispBasis.numActiveControlPoints();
        int N_D_geo = geoPatch.basis().numActiveControlPoints();

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

            if (threadIdx.x < dimTensor && threadIdx.y < dimTensor) {
                matrixViewTraceTensor_parallel(threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, C, RCGinv, RCGinv);
                symmetricIdentityViewTensor_parallel(threadIdx.x, blockDim.x, threadIdx.y, blockDim.y, Ctemp, RCGinv);
            }
            __syncthreads();

            if(threadId == 0) {
                C.times(lambda*J*J);
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
                int globalIndex_i = system.mapColIndex(displacement.basis(patch_idx).activeIndex(pt, i), patch_idx, di);
                for (int j = blockCoord[1] * numActivePerBlock + threadIdx.y; 
                     j < (blockCoord[1] + 1) * numActivePerBlock && j < N_D; 
                     j += blockDim.y)
                {
                    for (int dj = 0; dj < dim; dj++)
                    {
                        int localCol = (j - blockCoord[1] * numActivePerBlock) * dim + dj;
                        int globalIndex_j = system.mapColIndex(displacement.basis(patch_idx).activeIndex(pt, j), patch_idx, dj);
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
        __shared__ int dimTensor;
        __shared__ double wt, ptData[3], measure, weightForce, weightBody;
        __shared__ double lambda, mu, J;
        
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
            
            const int parameterOffset = patch_idx * 3;
            double YM = parameters[parameterOffset + 1];
            double PR = parameters[parameterOffset + 0];
            lambda = YM * PR / ( ( 1. + PR ) * ( 1. - 2. * PR ) );
            mu = YM / ( 2. * ( 1. + PR ) );
            //printf("blockIdx.x=%d, patch_idx=%d, blockCoord=(%d, %d), wt=%f, pt=(%f, %f)\n", 
            //       blockIdx.x, patch_idx, blockCoord[0], blockCoord[1], wt, pt[0], pt[1]);
        }
        __syncthreads();
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patch_idx);
        PatchDeviceView geoPatch = multiPatch.patch(patch_idx);
        PatchDeviceView dispPatch = displacement.patch(patch_idx);
        int N_D = dispBasis.numActiveControlPoints();
        int N_D_geo = geoPatch.basis().numActiveControlPoints();

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
        
        int numDerivatives = 1;
        int CPdim = multiPatch.targetDim();
        int dim = multiPatch.domainDim();
        
        int patch_idx(0);
        int point_idx = displacement.threadPatch(idx, patch_idx);
        const int parameterOffset = patch_idx * 3;
        double YM = parameters[parameterOffset + 1];
        double PR = parameters[parameterOffset + 0];
        double lambda = YM * PR / ( ( 1. + PR ) * ( 1. - 2. * PR ) );
        double mu = YM / ( 2. * ( 1. + PR ) );
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
                           const Eigen::VectorXd &bodyForce,
                           bool baseInitial,
                           int numDerivatives)
:   m_multiPatch(multiPatch), m_multiBasis(multiBasis), 
    m_boundaryConditions(bc), m_bodyForce(bodyForce),
    m_multiGaussPoints(multiBasis), 
    m_multiPatchHost(multiPatch), m_multiBasisHost(multiBasis)
{
    int targetDim = multiPatch.getCPDim();
    m_targetDim = targetDim;
    m_domainDim = multiPatch.getBasisDim();
    m_dimTensor = (m_domainDim * (m_domainDim + 1)) / 2;
    m_numElements = multiBasis.totalNumElements();
    m_N_D = multiBasis.numActive();
    m_totalGPs = multiBasis.totalNumGPs();
    m_numDerivatives = numDerivatives;
    m_geoP1 = multiPatch.knotOrder() + 1;
    m_dispP1 = multiBasis.knotOrder() + 1;
    if (!baseInitial) {
        std::vector<DofMapper> dofMappers_stdVec(targetDim);
        multiBasis.getMappers(true, m_boundaryConditions, 
                          dofMappers_stdVec, true);
#if 0
        m_sparseSystemHost = SparseSystem(dofMappers_stdVec, 
                              Eigen::VectorXi::Ones(targetDim));
#else
        SparseSystem sparseSystem(dofMappers_stdVec, Eigen::VectorXi::Ones(targetDim));
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
#if 1
        int numActivePerBlock = std::min(16, m_N_D);
        int numBlocksPerElement = (m_N_D + numActivePerBlock - 1) / numActivePerBlock;
        dim3 blockSize(numActivePerBlock, numActivePerBlock);
        int gridSize = m_numElements * numBlocksPerElement * numBlocksPerElement;
#else
        int minGrid, blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                             countEntrysKernel, 0, numElements);
        int gridSize = (numElements + blockSize - 1) / blockSize;
#endif
#ifdef TIME_INITIALIZATION
	    auto start = std::chrono::high_resolution_clock::now();
#endif
        countEntrysKernel<<<gridSize, blockSize>>>(m_numElements, numBlocksPerElement, 
                                               numActivePerBlock,
                                               m_displacement.deviceView(),
                                               m_multiPatch.deviceView(),
                                               m_multiGaussPoints.view(),
                                               m_sparseSystem.deviceView(),
                                               m_ddof.view(),
                                               entryCountDevicePtr);
#endif
        err = cudaDeviceSynchronize();
#ifdef TIME_INITIALIZATION
	    auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Counted matrix entries in " << elapsed.count() << " s." << std::endl;
#endif
        assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler constructor during counting matrix entries");
        int entryCountHost;
        err = cudaMemcpy(&entryCountHost, entryCountDevicePtr, sizeof(int), cudaMemcpyDeviceToHost);
        assert(err == cudaSuccess && "cudaMemcpy failed in GPUAssembler constructor during counting matrix entries");
            
        std::vector<int> followerMomentCOORows;
        std::vector<int> followerMomentCOOCols;
        appendFollowerMomentCOOPattern(m_boundaryConditions, multiBasis,
                                       sparseSystem, dofMappers_stdVec,
                                       m_domainDim, m_targetDim,
                                       followerMomentCOORows,
                                       followerMomentCOOCols);
        const int followerMomentEntryCount =
            static_cast<int>(followerMomentCOORows.size());

        //m_sparseSystem.setNumMatrixEntries(entryCountHost);
        //m_sparseSystem.resizeMatrixData(entryCountHost);

        DeviceArray<int> cooRows(entryCountHost + followerMomentEntryCount);
        DeviceArray<int> cooCols(entryCountHost + followerMomentEntryCount);
        //DeviceArray<double> cooValues(entryCountHost);

        err = cudaMemset(entryCountDevicePtr, 0, sizeof(int));
        assert(err == cudaSuccess && "cudaMemset failed in GPUAssembler constructor during counting matrix entries");
#ifdef TIME_INITIALIZATION
        start = std::chrono::high_resolution_clock::now();
#endif
        computeCOOKernel<<<gridSize, blockSize>>>(m_numElements, entryCountDevicePtr,
                                numBlocksPerElement, numActivePerBlock,
                                m_displacement.deviceView(),
                                m_multiPatch.deviceView(),
                                m_multiGaussPoints.view(),
                                m_sparseSystem.deviceView(),
                                //m_ddof.view(),
                                cooRows.vectorView(),
                                cooCols.vectorView());

        err = cudaDeviceSynchronize();
#ifdef TIME_INITIALIZATION
	    end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Constructed COO matrix in " << elapsed.count() << " s." << std::endl;
#endif
        assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler constructor during COO construction");

        if (followerMomentEntryCount > 0)
        {
            err = cudaMemcpy(cooRows.data() + entryCountHost,
                             followerMomentCOORows.data(),
                             followerMomentEntryCount * sizeof(int),
                             cudaMemcpyHostToDevice);
            assert(err == cudaSuccess && "cudaMemcpy failed while appending follower moment COO rows");
            err = cudaMemcpy(cooCols.data() + entryCountHost,
                             followerMomentCOOCols.data(),
                             followerMomentEntryCount * sizeof(int),
                             cudaMemcpyHostToDevice);
            assert(err == cudaSuccess && "cudaMemcpy failed while appending follower moment COO cols");
        }

#ifdef TIME_INITIALIZATION
        start = std::chrono::high_resolution_clock::now();
#endif
        m_sparseSystem.setCSRMatrixFromCOO(sparseSystem.matrixRows(), 
                                       sparseSystem.matrixCols(),
                                       cooRows.vectorView(), 
                                       cooCols.vectorView());
#ifdef TIME_INITIALIZATION
	    end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Converted COO to CSR in " << elapsed.count() << " s." << std::endl;
#endif

        //m_sparseSystem.csrMatrix().sparsePrint_host();

        err = cudaFree(entryCountDevicePtr);
        assert(err == cudaSuccess && "cudaFree failed in GPUAssembler constructor during counting matrix entries");

        m_options = defaultOptions();

        int numDoublePerGP = /*m_domainDim // pts
                       + */m_domainDim * m_domainDim * 3 // geoJacobianInvs, Fs, Ss
                       + 3 // measures, weightForces, weightBodys
                       //+ m_geoP1 * (m_numDerivatives + 1) * m_domainDim // geoValuesAndDers
                       //+ m_dispP1 * (m_numDerivatives + 1) * m_domainDim // dispValuesAndDers
                       //+ (m_geoP1 * (m_geoP1 + 4) + m_dispP1 * (m_dispP1 + 4)) * m_domainDim // working space
                       + m_dimTensor * m_dimTensor; // Cs

        size_t bytesPerGP = numDoublePerGP * sizeof(double);
        size_t totalBytes = bytesPerGP * m_totalGPs;
        //printf("Total bytes needed for GPData: %zu\n", totalBytes);
        size_t freeMem = 0, totalMem = 0;
        err = cudaMemGetInfo(&freeMem, &totalMem);
        if (err != cudaSuccess)
            std::cerr << "Error during cudaMemGetInfo: " << cudaGetErrorString(err) << std::endl;
        double safetyFactor = 0.8;
        size_t usableMem = static_cast<size_t>(freeMem * safetyFactor);
        //printf("Usable memory for GPData: %zu bytes\n", usableMem);
#if 0
        if (usableMem < totalBytes)
            m_numBatches = (totalBytes + usableMem - 1) / usableMem;
        else
            m_numBatches = 1;
        if (m_numBatches > std::numeric_limits<int>::max())
            m_numBatches = std::numeric_limits<int>::max();
        printf("Number of batches: %d\n", m_numBatches);
        m_batchElements = (numElements + m_numBatches - 1) / m_numBatches;
        m_batchSize = m_batchElements * m_N_D;
        m_GPData.resize(numDoublePerGP * m_batchSize);
#endif
        m_GPData.resize(numDoublePerGP * m_totalGPs);
        //printf("Size of GPData: %zu bytes\n", m_GPData.size() * sizeof(double));

        m_GPTable.resize(m_totalGPs * m_domainDim);
        m_wts.resize(m_totalGPs);
        int minGrid, blockSize_GPTable;
        cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize_GPTable, 
                             countEntrysKernel, 0, m_totalGPs);
        gridSize = (m_totalGPs + blockSize_GPTable - 1) / blockSize_GPTable;
        computeGPTableKernel<<<gridSize, blockSize_GPTable>>>(m_totalGPs, 
            m_displacement.deviceView(), m_multiGaussPoints.view(),
            m_GPTable.matrixView(m_domainDim, m_totalGPs), m_wts.vectorView());
        err = cudaDeviceSynchronize();
        assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler constructor during GP table computation");

        m_geoValuesAndDerss.resize(m_geoP1 * m_totalGPs * (m_numDerivatives + 1) * m_domainDim);
        m_dispValuesAndDerss.resize(m_dispP1 * m_totalGPs * (m_numDerivatives + 1) * m_domainDim);
        DeviceArray<double> geoWorkingSpaces(m_totalGPs * m_geoP1 * (m_geoP1 + 4) * m_domainDim);
        DeviceArray<double> dispWorkingSpaces(m_totalGPs * m_dispP1 * (m_dispP1 + 4) * m_domainDim);

        int blockSize_evaluateValAbdDers;
        cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize_evaluateValAbdDers, evaluateBasisValuesAndDerivativesAtGPsKernel, 0, m_totalGPs * m_domainDim);
        gridSize = (m_totalGPs * m_domainDim + blockSize_evaluateValAbdDers - 1) / blockSize_evaluateValAbdDers;
        evaluateBasisValuesAndDerivativesAtGPsKernel<<<gridSize, blockSize_evaluateValAbdDers>>>(
            m_numDerivatives, m_totalGPs, 
            m_domainDim, m_displacement.deviceView(),
            m_multiPatch.deviceView(), 
            m_GPTable.matrixView(m_domainDim, m_totalGPs), 
            geoWorkingSpaces.vectorView(), dispWorkingSpaces.vectorView(),
            m_geoValuesAndDerss.matrixView(m_geoP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim), 
            m_dispValuesAndDerss.matrixView(m_dispP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim));
        err = cudaDeviceSynchronize();
        assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler constructor during basis values and derivatives evaluation at GPs");
    }
    
}

OptionList GPUAssembler::defaultOptions()
{
    OptionList opt;
    opt.addReal("youngs_modulus", "Young's modulus", 1.0);
    opt.addReal("poissons_ratio", "Poisson's ratio", 0.3);
    opt.addReal("neumann_load_scaling", "Multiplier for Neumann boundary and corner loads", 1.0);
    opt.addInt("material_law", "0: StVK, 1: neo-Hookean", 1);
    opt.addSwitch("use_nonsymmetric_newton_solver",
                  "Use a nonsymmetric direct solver for Newton systems", false);
    opt.addSwitch("print_timing",
                  "Print assembly and solve timing diagnostics", false);
    return opt;
}

__device__
void computeMaterialResponse(int materialLaw,
                                   double youngsModulus,
                                   double poissonsRatio,
                                   DeviceMatrixView<double> F,
                                   DeviceMatrixView<double> S,
                                   DeviceMatrixView<double> C)
{
    const int dim = F.rows();
    const int dimTensor = (dim * (dim + 1)) / 2;
    const double lambda = youngsModulus * poissonsRatio /
        ((1.0 + poissonsRatio) * (1.0 - 2.0 * poissonsRatio));
    const double mu = youngsModulus / (2.0 * (1.0 + poissonsRatio));

    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            S(i, j) = 0.0;
    for (int i = 0; i < dimTensor; ++i)
        for (int j = 0; j < dimTensor; ++j)
            C(i, j) = 0.0;

    double FTransData[3 * 3] = {0.0};
    double rightCauchyGreenData[3 * 3] = {0.0};
    DeviceMatrixView<double> FTrans(FTransData, dim, dim);
    DeviceMatrixView<double> rightCauchyGreen(rightCauchyGreenData, dim, dim);
    F.transpose(FTrans);
    FTrans.times(F, rightCauchyGreen);

    if (materialLaw == 0)
    {
        double traceE = 0.0;
        for (int A = 0; A < dim; ++A)
            traceE += 0.5 * (rightCauchyGreen(A, A) - 1.0);

        for (int A = 0; A < dim; ++A)
            for (int B = 0; B < dim; ++B)
            {
                const double EAB =
                    0.5 * (rightCauchyGreen(A, B) - (A == B ? 1.0 : 0.0));
                S(A, B) = lambda * traceE * (A == B ? 1.0 : 0.0) +
                          2.0 * mu * EAB;
            }

        for (int i = 0; i < dimTensor; ++i)
        {
            const int A = voigt(dim, i, 0);
            const int B = voigt(dim, i, 1);
            for (int j = 0; j < dimTensor; ++j)
            {
                const int Cidx = voigt(dim, j, 0);
                const int D = voigt(dim, j, 1);
                C(i, j) = lambda * (A == B ? 1.0 : 0.0) *
                                  (Cidx == D ? 1.0 : 0.0) +
                          mu * ((A == Cidx && B == D ? 1.0 : 0.0) +
                                (A == D && B == Cidx ? 1.0 : 0.0));
            }
        }
        return;
    }

    const double J = F.determinant();
    double rightCauchyGreenInvData[3 * 3] = {0.0};
    DeviceMatrixView<double> rightCauchyGreenInv(rightCauchyGreenInvData,
                                                 dim, dim);
    rightCauchyGreen.inverse(rightCauchyGreenInv);

    rightCauchyGreenInv.times(lambda * (J * J - 1.0) / 2.0 - mu, S);
    S.tracePlus(mu);

    matrixViewTraceTensor(C, rightCauchyGreenInv, rightCauchyGreenInv);
    C.times(lambda * J * J);
    double CtempData[6 * 6] = {0.0};
    DeviceMatrixView<double> Ctemp(CtempData, dimTensor, dimTensor);
    symmetricIdentityViewTensor(Ctemp, rightCauchyGreenInv);
    Ctemp.times(mu - lambda * (J * J - 1.0) / 2.0);
    C.plus(Ctemp);
}

void GPUAssembler::setDefaultOptions(const OptionList &opt)
{ m_options = opt; }

void GPUAssembler::setPatchRealOption(const std::string& label,
                                      const std::vector<double>& values)
{
    if (static_cast<int>(values.size()) != numPatches())
        throw std::invalid_argument("Patch real option '" + label +
            "' must have one value per patch");
    m_patchRealOptions[label] = values;
}

void GPUAssembler::setPatchIntOption(const std::string& label,
                                     const std::vector<int>& values)
{
    if (static_cast<int>(values.size()) != numPatches())
        throw std::invalid_argument("Patch int option '" + label +
            "' must have one value per patch");
    m_patchIntOptions[label] = values;
}

std::vector<double> GPUAssembler::patchRealOptionValues(
    const std::string& label) const
{
    const auto it = m_patchRealOptions.find(label);
    if (it != m_patchRealOptions.end())
        return it->second;

    return std::vector<double>(static_cast<std::size_t>(numPatches()),
                               m_options.getReal(label));
}

std::vector<int> GPUAssembler::patchIntOptionValues(
    const std::string& label) const
{
    const auto it = m_patchIntOptions.find(label);
    if (it != m_patchIntOptions.end())
        return it->second;

    return std::vector<int>(static_cast<std::size_t>(numPatches()),
                            m_options.getInt(label));
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
            ddof[unk_][ii] = it->value(unk_ > (m_targetDim - 1) ? unk_ - m_targetDim : unk_);
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
        ddof[unk_][ii] = it->value(unk_ > (m_targetDim - 1) ? unk_ - m_targetDim : unk_);
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
                                     GPUFunction& displacementFunction) const
{
    int minGrid, blockSize;
    int CPSize = m_displacementHost.CPSize();
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
        constructSolutionKernel, 0, CPSize);

    int gridSize = (CPSize + blockSize - 1) / blockSize;
    constructSolutionKernel<<<gridSize, blockSize>>>(solVector, fixedDoFs,
                                                     m_multiBasis.deviceView(),
                                                     m_sparseSystem.deviceView(),
                                                     displacementFunction.multiPatchDeviceView(),
                                                     CPSize);
    cudaError_t err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructSolution");
}

void GPUAssembler::constructCauchyStressFunction(const DeviceVectorView<double>& solVector,
                                                 const DeviceNestedArrayView<double>& fixedDoFs,
                                                 GPUFunction& cauchyStressFunction)
{
    int minGrid, blockSize;
    int CPSize = m_displacementHost.CPSize();
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        constructSolutionKernel, 0, CPSize);
    int gridSize = (CPSize + blockSize - 1) / blockSize;
    constructSolutionKernel<<<gridSize, blockSize>>>(solVector, fixedDoFs,
                                                     m_multiBasis.deviceView(),
                                                     m_sparseSystem.deviceView(),
                                                     m_displacement.deviceView(),
                                                     CPSize);
    cudaError_t err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructCauchyStressFunction constructSolutionKernel");

    constructCauchyStressFunctionFromDisplacement(m_displacement.deviceView(), cauchyStressFunction);
}

void GPUAssembler::constructCauchyStressFunction(GPUFunction& displacementFunction,
                                                 GPUFunction& cauchyStressFunction)
{
    assert(displacementFunction.domainDim() == m_domainDim &&
           "Displacement function domain dimension must match assembler domain dimension");
    assert(displacementFunction.targetDim() == m_targetDim &&
           "Displacement function target dimension must match assembler target dimension");

    constructCauchyStressFunctionFromDisplacement(displacementFunction.multiPatchDeviceView(),
                                                  cauchyStressFunction);
}

void GPUAssembler::constructDeformationGradientFunction(const DeviceVectorView<double>& solVector,
                                                        const DeviceNestedArrayView<double>& fixedDoFs,
                                                        GPUFunction& deformationGradientFunction)
{
    constructDispSolution(solVector, fixedDoFs);
    constructDeformationGradientFunctionFromDisplacement(m_displacement.deviceView(),
                                                         deformationGradientFunction);
}

void GPUAssembler::constructDeformationGradientFunction(GPUFunction& displacementFunction,
                                                        GPUFunction& deformationGradientFunction)
{
    assert(displacementFunction.domainDim() == m_domainDim &&
           "Displacement function domain dimension must match assembler domain dimension");
    assert(displacementFunction.targetDim() == m_targetDim &&
           "Displacement function target dimension must match assembler target dimension");

    constructDeformationGradientFunctionFromDisplacement(
        displacementFunction.multiPatchDeviceView(), deformationGradientFunction);
}

void GPUAssembler::constructKinematicGradientFunctions(
    const DeviceVectorView<double>& solVector,
    const DeviceNestedArrayView<double>& fixedDoFs,
    GPUFunction& deformationGradientGradientFunction,
    GPUFunction& greenLagrangeStrainGradientFunction)
{
    constructDispSolution(solVector, fixedDoFs);
    constructKinematicGradientFunctionsFromDisplacement(
        m_displacement.deviceView(), deformationGradientGradientFunction,
        greenLagrangeStrainGradientFunction);
}

void GPUAssembler::constructKinematicGradientFunctions(
    GPUFunction& displacementFunction,
    GPUFunction& deformationGradientGradientFunction,
    GPUFunction& greenLagrangeStrainGradientFunction)
{
    assert(displacementFunction.domainDim() == m_domainDim &&
           "Displacement function domain dimension must match assembler domain dimension");
    assert(displacementFunction.targetDim() == m_targetDim &&
           "Displacement function target dimension must match assembler target dimension");

    constructKinematicGradientFunctionsFromDisplacement(
        displacementFunction.multiPatchDeviceView(),
        deformationGradientGradientFunction,
        greenLagrangeStrainGradientFunction);
}

void GPUAssembler::constructKinematicGradientFunctionsFromDisplacement(
    MultiPatchDeviceView displacementView,
    GPUFunction& deformationGradientGradientFunction,
    GPUFunction& greenLagrangeStrainGradientFunction)
{
    if (m_numDerivatives < 2)
        throw std::invalid_argument(
            "Kinematic gradient recovery requires numDerivatives >= 2");

    const int dim3 = m_domainDim * m_domainDim * m_domainDim;
    assert(deformationGradientGradientFunction.domainDim() == m_domainDim &&
           "Grad F function domain dimension must match assembler domain dimension");
    assert(deformationGradientGradientFunction.targetDim() == dim3 &&
           "Grad F function target dimension must be dim * dim * dim");
    assert(greenLagrangeStrainGradientFunction.domainDim() == m_domainDim &&
           "Grad Green-Lagrange strain function domain dimension must match assembler domain dimension");
    assert(greenLagrangeStrainGradientFunction.targetDim() == dim3 &&
           "Grad Green-Lagrange strain function target dimension must be dim * dim * dim");

    int minGrid, blockSize;
    int gridSize;
    cudaError_t err;
    const int totalControlPoints = m_displacementHost.getTotalNumControlPoints();
    const int totalKinematicGradientEntries = totalControlPoints * dim3;

    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        zeroFunctionControlPointsKernel, 0, totalKinematicGradientEntries);
    gridSize = (totalKinematicGradientEntries + blockSize - 1) / blockSize;
    zeroFunctionControlPointsKernel<<<gridSize, blockSize>>>(
        deformationGradientGradientFunction.multiPatchDeviceView(),
        totalKinematicGradientEntries);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructKinematicGradientFunctions zero Grad F kernel");

    zeroFunctionControlPointsKernel<<<gridSize, blockSize>>>(
        greenLagrangeStrainGradientFunction.multiPatchDeviceView(),
        totalKinematicGradientEntries);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructKinematicGradientFunctions zero Grad E kernel");

    m_GPData.setZero();
    int offset = 0;
    DeviceMatrixView<double> geoJacobianInvs(m_GPData.data() + offset, m_domainDim, m_totalGPs * m_domainDim);
    offset += geoJacobianInvs.size();
    DeviceVectorView<double> measures(m_GPData.data() + offset, m_totalGPs);
    offset += measures.size();
    DeviceVectorView<double> weightForces(m_GPData.data() + offset, m_totalGPs);
    offset += weightForces.size();
    DeviceVectorView<double> weightBodys(m_GPData.data() + offset, m_totalGPs);
    offset += weightBodys.size();
    DeviceMatrixView<double> Fs(m_GPData.data() + offset, m_domainDim, m_totalGPs * m_domainDim);
    offset += Fs.size();
    DeviceMatrixView<double> Ss(m_GPData.data() + offset, m_domainDim, m_totalGPs * m_domainDim);
    offset += Ss.size();
    DeviceMatrixView<double> Cs(m_GPData.data() + offset, m_dimTensor, m_totalGPs * m_dimTensor);
    offset += Cs.size();

    const std::vector<double> patchPoissonsRatios =
        patchRealOptionValues("poissons_ratio");
    const std::vector<double> patchYoungsModuli =
        patchRealOptionValues("youngs_modulus");
    const std::vector<int> patchMaterialLaws =
        patchIntOptionValues("material_law");
    std::vector<double> materialParameters;
    materialParameters.reserve(static_cast<std::size_t>(3 * numPatches()));
    for (int p = 0; p < numPatches(); ++p)
    {
        materialParameters.push_back(patchPoissonsRatios[static_cast<std::size_t>(p)]);
        materialParameters.push_back(patchYoungsModuli[static_cast<std::size_t>(p)]);
        materialParameters.push_back(static_cast<double>(
            patchMaterialLaws[static_cast<std::size_t>(p)]));
    }
    DeviceArray<double> parameterValues(materialParameters);
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        evaluateGPKernel_withoutComputingGPTableAndDers, 0, m_totalGPs);
    gridSize = (m_totalGPs + blockSize - 1) / blockSize;
    evaluateGPKernel_withoutComputingGPTableAndDers<<<gridSize, blockSize>>>(
        m_numDerivatives, 0, 0, m_totalGPs,
        parameterValues.vectorView(),
        displacementView,
        m_multiPatch.deviceView(),
        m_GPTable.matrixView(m_domainDim, m_totalGPs),
        m_wts.vectorView(),
        m_geoValuesAndDerss.matrixView(m_geoP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim),
        m_dispValuesAndDerss.matrixView(m_dispP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim),
        geoJacobianInvs, measures, weightForces, weightBodys,
        Fs, Ss, Cs);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructKinematicGradientFunctions GP evaluation");

    DeviceArray<double> nodalWeights(totalControlPoints);
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        recoverKinematicGradientsAtNodesKernel, 0, m_totalGPs);
    gridSize = (m_totalGPs + blockSize - 1) / blockSize;
    recoverKinematicGradientsAtNodesKernel<<<gridSize, blockSize>>>(
        m_numDerivatives, m_totalGPs,
        displacementView,
        m_multiPatch.deviceView(),
        deformationGradientGradientFunction.multiPatchDeviceView(),
        greenLagrangeStrainGradientFunction.multiPatchDeviceView(),
        m_GPTable.matrixView(m_domainDim, m_totalGPs),
        weightForces,
        m_geoValuesAndDerss.matrixView(m_geoP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim),
        m_dispValuesAndDerss.matrixView(m_dispP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim),
        geoJacobianInvs,
        Fs,
        nodalWeights.vectorView());
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructKinematicGradientFunctions recovery");

    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        normalizeRecoveredStressKernel, 0, totalKinematicGradientEntries);
    gridSize = (totalKinematicGradientEntries + blockSize - 1) / blockSize;
    normalizeRecoveredStressKernel<<<gridSize, blockSize>>>(
        deformationGradientGradientFunction.multiPatchDeviceView(),
        nodalWeights.vectorView(),
        totalControlPoints);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructKinematicGradientFunctions normalize Grad F");

    normalizeRecoveredStressKernel<<<gridSize, blockSize>>>(
        greenLagrangeStrainGradientFunction.multiPatchDeviceView(),
        nodalWeights.vectorView(),
        totalControlPoints);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructKinematicGradientFunctions normalize Grad E");
}

void GPUAssembler::constructDeformationGradientFunctionFromDisplacement(
    MultiPatchDeviceView displacementView,
    GPUFunction& deformationGradientFunction)
{
    const int dim2 = m_domainDim * m_domainDim;
    assert(deformationGradientFunction.domainDim() == m_domainDim &&
           "Deformation gradient function domain dimension must match assembler domain dimension");
    assert(deformationGradientFunction.targetDim() == dim2 &&
           "Deformation gradient function target dimension must be dim * dim");

    int minGrid, blockSize;
    int gridSize;
    cudaError_t err;
    const int totalControlPoints = m_displacementHost.getTotalNumControlPoints();
    const int totalDeformationGradientEntries = totalControlPoints * dim2;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        zeroFunctionControlPointsKernel, 0, totalDeformationGradientEntries);
    gridSize = (totalDeformationGradientEntries + blockSize - 1) / blockSize;
    zeroFunctionControlPointsKernel<<<gridSize, blockSize>>>(
        deformationGradientFunction.multiPatchDeviceView(),
        totalDeformationGradientEntries);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructDeformationGradientFunction zeroFunctionControlPointsKernel");

    m_GPData.setZero();
    int offset = 0;
    DeviceMatrixView<double> geoJacobianInvs(m_GPData.data() + offset, m_domainDim, m_totalGPs * m_domainDim);
    offset += geoJacobianInvs.size();
    DeviceVectorView<double> measures(m_GPData.data() + offset, m_totalGPs);
    offset += measures.size();
    DeviceVectorView<double> weightForces(m_GPData.data() + offset, m_totalGPs);
    offset += weightForces.size();
    DeviceVectorView<double> weightBodys(m_GPData.data() + offset, m_totalGPs);
    offset += weightBodys.size();
    DeviceMatrixView<double> Fs(m_GPData.data() + offset, m_domainDim, m_totalGPs * m_domainDim);
    offset += Fs.size();
    DeviceMatrixView<double> Ss(m_GPData.data() + offset, m_domainDim, m_totalGPs * m_domainDim);
    offset += Ss.size();
    DeviceMatrixView<double> Cs(m_GPData.data() + offset, m_dimTensor, m_totalGPs * m_dimTensor);
    offset += Cs.size();

    const std::vector<double> patchPoissonsRatios =
        patchRealOptionValues("poissons_ratio");
    const std::vector<double> patchYoungsModuli =
        patchRealOptionValues("youngs_modulus");
    const std::vector<int> patchMaterialLaws =
        patchIntOptionValues("material_law");
    std::vector<double> materialParameters;
    materialParameters.reserve(static_cast<std::size_t>(3 * numPatches()));
    for (int p = 0; p < numPatches(); ++p)
    {
        materialParameters.push_back(patchPoissonsRatios[static_cast<std::size_t>(p)]);
        materialParameters.push_back(patchYoungsModuli[static_cast<std::size_t>(p)]);
        materialParameters.push_back(static_cast<double>(
            patchMaterialLaws[static_cast<std::size_t>(p)]));
    }
    DeviceArray<double> parameterValues(materialParameters);
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        evaluateGPKernel_withoutComputingGPTableAndDers, 0, m_totalGPs);
    gridSize = (m_totalGPs + blockSize - 1) / blockSize;
    evaluateGPKernel_withoutComputingGPTableAndDers<<<gridSize, blockSize>>>(
        m_numDerivatives, 0, 0, m_totalGPs,
        parameterValues.vectorView(),
        displacementView,
        m_multiPatch.deviceView(),
        m_GPTable.matrixView(m_domainDim, m_totalGPs),
        m_wts.vectorView(),
        m_geoValuesAndDerss.matrixView(m_geoP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim),
        m_dispValuesAndDerss.matrixView(m_dispP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim),
        geoJacobianInvs, measures, weightForces, weightBodys,
        Fs, Ss, Cs);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructDeformationGradientFunction evaluateGPKernel_withoutComputingGPTableAndDers");

    DeviceArray<double> nodalWeights(totalControlPoints);
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        recoverDeformationGradientAtNodesKernel, 0, m_totalGPs);
    gridSize = (m_totalGPs + blockSize - 1) / blockSize;
    recoverDeformationGradientAtNodesKernel<<<gridSize, blockSize>>>(
        m_numDerivatives, m_totalGPs,
        displacementView,
        deformationGradientFunction.multiPatchDeviceView(),
        m_GPTable.matrixView(m_domainDim, m_totalGPs),
        weightForces,
        m_dispValuesAndDerss.matrixView(m_dispP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim),
        Fs,
        nodalWeights.vectorView());
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructDeformationGradientFunction recoverDeformationGradientAtNodesKernel");

    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        normalizeRecoveredStressKernel, 0, totalDeformationGradientEntries);
    gridSize = (totalDeformationGradientEntries + blockSize - 1) / blockSize;
    normalizeRecoveredStressKernel<<<gridSize, blockSize>>>(
        deformationGradientFunction.multiPatchDeviceView(),
        nodalWeights.vectorView(),
        totalControlPoints);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructDeformationGradientFunction normalizeRecoveredStressKernel");
}

void GPUAssembler::constructCauchyStressFunctionFromDisplacement(MultiPatchDeviceView displacementView,
                                                                 GPUFunction& cauchyStressFunction)
{
    assert(cauchyStressFunction.domainDim() == m_domainDim &&
           "Cauchy stress function domain dimension must match assembler domain dimension");
    assert(cauchyStressFunction.targetDim() == m_dimTensor &&
           "Cauchy stress function target dimension must be dim * (dim + 1) / 2");

    int minGrid, blockSize;
    int gridSize;
    cudaError_t err;
    const int totalControlPoints = m_displacementHost.getTotalNumControlPoints();
    const int totalStressEntries = totalControlPoints * m_dimTensor;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        zeroFunctionControlPointsKernel, 0, totalStressEntries);
    gridSize = (totalStressEntries + blockSize - 1) / blockSize;
    zeroFunctionControlPointsKernel<<<gridSize, blockSize>>>(
        cauchyStressFunction.multiPatchDeviceView(), totalStressEntries);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructCauchyStressFunction zeroFunctionControlPointsKernel");

    m_GPData.setZero();
    int offset = 0;
    DeviceMatrixView<double> geoJacobianInvs(m_GPData.data() + offset, m_domainDim, m_totalGPs * m_domainDim);
    offset += geoJacobianInvs.size();
    DeviceVectorView<double> measures(m_GPData.data() + offset, m_totalGPs);
    offset += measures.size();
    DeviceVectorView<double> weightForces(m_GPData.data() + offset, m_totalGPs);
    offset += weightForces.size();
    DeviceVectorView<double> weightBodys(m_GPData.data() + offset, m_totalGPs);
    offset += weightBodys.size();
    DeviceMatrixView<double> Fs(m_GPData.data() + offset, m_domainDim, m_totalGPs * m_domainDim);
    offset += Fs.size();
    DeviceMatrixView<double> Ss(m_GPData.data() + offset, m_domainDim, m_totalGPs * m_domainDim);
    offset += Ss.size();
    DeviceMatrixView<double> Cs(m_GPData.data() + offset, m_dimTensor, m_totalGPs * m_dimTensor);
    offset += Cs.size();

    const std::vector<double> patchPoissonsRatios =
        patchRealOptionValues("poissons_ratio");
    const std::vector<double> patchYoungsModuli =
        patchRealOptionValues("youngs_modulus");
    const std::vector<int> patchMaterialLaws =
        patchIntOptionValues("material_law");
    std::vector<double> materialParameters;
    materialParameters.reserve(static_cast<std::size_t>(3 * numPatches()));
    for (int p = 0; p < numPatches(); ++p)
    {
        materialParameters.push_back(patchPoissonsRatios[static_cast<std::size_t>(p)]);
        materialParameters.push_back(patchYoungsModuli[static_cast<std::size_t>(p)]);
        materialParameters.push_back(static_cast<double>(
            patchMaterialLaws[static_cast<std::size_t>(p)]));
    }
    DeviceArray<double> parameterValues(materialParameters);
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        evaluateGPKernel_withoutComputingGPTableAndDers, 0, m_totalGPs);
    gridSize = (m_totalGPs + blockSize - 1) / blockSize;
    evaluateGPKernel_withoutComputingGPTableAndDers<<<gridSize, blockSize>>>(
        m_numDerivatives, 0, 0, m_totalGPs,
        parameterValues.vectorView(),
        displacementView,
        m_multiPatch.deviceView(),
        m_GPTable.matrixView(m_domainDim, m_totalGPs),
        m_wts.vectorView(),
        m_geoValuesAndDerss.matrixView(m_geoP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim),
        m_dispValuesAndDerss.matrixView(m_dispP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim),
        geoJacobianInvs, measures, weightForces, weightBodys,
        Fs, Ss, Cs);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructCauchyStressFunction evaluateGPKernel_withoutComputingGPTableAndDers");

    DeviceArray<double> nodalWeights(totalControlPoints);
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        recoverCauchyStressAtNodesKernel, 0, m_totalGPs);
    gridSize = (m_totalGPs + blockSize - 1) / blockSize;
    recoverCauchyStressAtNodesKernel<<<gridSize, blockSize>>>(
        m_numDerivatives, m_totalGPs,
        displacementView,
        cauchyStressFunction.multiPatchDeviceView(),
        m_GPTable.matrixView(m_domainDim, m_totalGPs),
        weightForces,
        m_dispValuesAndDerss.matrixView(m_dispP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim),
        Fs, Ss,
        nodalWeights.vectorView());
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructCauchyStressFunction recoverCauchyStressAtNodesKernel");

    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        normalizeRecoveredStressKernel, 0, totalStressEntries);
    gridSize = (totalStressEntries + blockSize - 1) / blockSize;
    normalizeRecoveredStressKernel<<<gridSize, blockSize>>>(
        cauchyStressFunction.multiPatchDeviceView(),
        nodalWeights.vectorView(),
        totalControlPoints);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler::constructCauchyStressFunction normalizeRecoveredStressKernel");
}

void GPUAssembler::constructDispSolution(const DeviceVectorView<double> &solVector, 
                                     const DeviceNestedArrayView<double> &fixedDoFs) const
{
    int minGrid, blockSize;
    int CPSize = m_displacementHost.CPSize();
    cudaOccupancyMaxPotentialBlockSize(&minGrid, 
        &blockSize, constructSolutionKernel, 0, CPSize);
    int gridSize = (CPSize + blockSize - 1) / blockSize;
    constructSolutionKernel<<<gridSize, blockSize>>>(solVector, fixedDoFs,
                                                     m_multiBasis.deviceView(),
                                                     m_sparseSystem.deviceView(),
                                                     m_displacement.deviceView(),
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

    //int domainDim = m_multiPatchHost.getBasisDim();
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
    const std::vector<double> patchPoissonsRatios =
        patchRealOptionValues("poissons_ratio");
    const std::vector<double> patchYoungsModuli =
        patchRealOptionValues("youngs_modulus");
    const std::vector<int> patchMaterialLaws =
        patchIntOptionValues("material_law");
    std::vector<double> materialParameters;
    materialParameters.reserve(static_cast<std::size_t>(3 * numPatches()));
    for (int p = 0; p < numPatches(); ++p)
    {
        materialParameters.push_back(patchPoissonsRatios[static_cast<std::size_t>(p)]);
        materialParameters.push_back(patchYoungsModuli[static_cast<std::size_t>(p)]);
        materialParameters.push_back(static_cast<double>(
            patchMaterialLaws[static_cast<std::size_t>(p)]));
    }
    DeviceArray<double> parameterValues(materialParameters);

    const bool enableMultiGPU =
        siga::gpuasm::envFlag("SIGA_MULTIGPU_ASSEMBLY", false);
    const bool enableStreamingAssembly =
        siga::gpuasm::envFlag("SIGA_GPU_STREAMING_ASSEMBLY", enableMultiGPU);
    if (enableStreamingAssembly)
    {
        const bool reportMemory =
            siga::gpuasm::envFlag("SIGA_GPU_MEMORY_REPORT", true);
        const bool replicateSecondaryInputs =
            siga::gpuasm::envFlag("SIGA_GPU_REPLICATE_INPUTS", true);
        const bool useLocalCSRAssembly =
            siga::gpuasm::envFlag("SIGA_GPU_LOCAL_CSR_ASSEMBLY", true);
        const bool printTiming =
            siga::gpuasm::envFlag("SIGA_GPU_MATRIX_TIMING", false);

        int primaryDevice = 0;
        err = cudaGetDevice(&primaryDevice);
        if (err != cudaSuccess)
            throw std::runtime_error("cudaGetDevice failed before elasticity assembly");

        std::vector<int> assemblyDevices =
            siga::gpuasm::usableAssemblyDevices(enableMultiGPU);
        siga::gpuasm::printDeviceSelection("Elasticity assembly",
                                           assemblyDevices,
                                           enableMultiGPU);
        const int numAssemblyDevices =
            static_cast<int>(assemblyDevices.size());
        if (m_numElements <= 0 || m_totalGPs % m_numElements != 0)
            throw std::runtime_error(
                "Elasticity streaming assembly requires a uniform positive Gauss-point count per element");
        const int gpsPerElement = m_totalGPs / m_numElements;
        const int stride = numDoublesPerGP();
        const int matrixValuesSize = csrMatrix().numNonZeros();
        const int rhsSize = numDofs();
        const int numFields = m_targetDim;

        const SparseSystemDeviceView primarySystemView =
            m_sparseSystem.deviceView();
        const MultiPatchDeviceView primaryGeometryView =
            m_multiPatch.deviceView();
        const MultiPatchDeviceView primaryDisplacementView =
            m_displacement.deviceView();
        const DeviceMatrixView<double> primaryGPTableView =
            m_GPTable.matrixView(m_domainDim, m_totalGPs);
        const DeviceVectorView<double> primaryWeightsView =
            m_wts.vectorView();
        const DeviceMatrixView<double> primaryGeoValuesView =
            m_geoValuesAndDerss.matrixView(
                m_geoP1,
                m_totalGPs * (m_numDerivatives + 1) * m_domainDim);
        const DeviceMatrixView<double> primaryDispValuesView =
            m_dispValuesAndDerss.matrixView(
                m_dispP1,
                m_totalGPs * (m_numDerivatives + 1) * m_domainDim);
        const DeviceVectorView<double> primaryBodyForceView =
            m_bodyForce.vectorView();

        const int oneShotChunkElements =
            (m_numElements + numAssemblyDevices - 1) / numAssemblyDevices;
        int requestedChunkElementLimit =
            siga::gpuasm::envInt("SIGA_GPU_CHUNK_ELEMENTS",
                                 oneShotChunkElements);
        if (requestedChunkElementLimit <= 0)
            requestedChunkElementLimit = oneShotChunkElements;
        requestedChunkElementLimit =
            std::min(m_numElements,
                     std::max(1, requestedChunkElementLimit));
        double memorySafetyFraction =
            siga::gpuasm::envDouble("SIGA_GPU_MEMORY_FRACTION", 0.90);
        memorySafetyFraction =
            std::min(1.0, std::max(0.10, memorySafetyFraction));

        auto staticInputBytes = [&](int gpCount)
        {
            unsigned long long bytes =
                siga::gpuasm::multiPatchReplicaBytes(primaryGeometryView) +
                siga::gpuasm::multiPatchReplicaBytes(primaryDisplacementView) +
                siga::gpuasm::bytesForCount(
                    static_cast<long long>(gpCount) *
                        primaryGPTableView.rows(),
                    sizeof(double)) +
                siga::gpuasm::bytesForCount(gpCount, sizeof(double)) +
                siga::gpuasm::vectorBytes(primaryBodyForceView);

            if (m_totalGPs > 0)
            {
                bytes += siga::gpuasm::bytesForCount(
                    static_cast<long long>(gpCount) *
                        primaryGeoValuesView.size() / m_totalGPs,
                    sizeof(double));
                bytes += siga::gpuasm::bytesForCount(
                    static_cast<long long>(gpCount) *
                        primaryDispValuesView.size() / m_totalGPs,
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
                siga::gpuasm::bytesForCount(materialParameters.size(),
                                            sizeof(double));
            const unsigned long long gpDataBytes =
                siga::gpuasm::bytesForCount(
                    static_cast<long long>(stride) * maxGPCount,
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
                    {"elasticity GP data", gpDataBytes},
                    {"sparse metadata and local CSR", sparseBytes},
                    {"replicated static input data", staticBytes},
                    {"fixed dofs replica", fixedDofsBytes}
                };
            }

            return matrixOutputBytes + rhsOutputBytes + materialBytes +
                   gpDataBytes + sparseBytes + staticBytes + fixedDofsBytes;
        };

        siga::gpuasm::AssemblySchedule schedule;
        int chunkElementLimit = requestedChunkElementLimit;
        bool scheduleFitsMemory = false;
        for (int attempt = 0; attempt < 32; ++attempt)
        {
            cudaSetDevice(primaryDevice);
            schedule = siga::gpuasm::buildAssemblySchedule(
                chunkElementLimit, m_numElements, gpsPerElement,
                numAssemblyDevices, m_N_D, numFields, rhsSize,
                matrixValuesSize,
                useLocalCSRAssembly && numAssemblyDevices > 1,
                primarySystemView, primaryDisplacementView,
                primaryDisplacementView, primaryGPTableView);

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
                        std::string("cudaMemGetInfo failed while sizing elasticity assembly chunks: ") +
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
                        "Elasticity assembly cannot fit even a one-element chunk on CUDA device " +
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
                std::cout << "Elasticity adaptive assembly chunking: chunk limit "
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
                "Elasticity adaptive assembly chunking could not find a memory-fitting chunk size.");

        if (reportMemory)
        {
            int totalChunks = 0;
            for (const auto& chunks : schedule.chunksByDevice)
                totalChunks += static_cast<int>(chunks.size());
            std::cout << "Elasticity adaptive assembly schedule: "
                      << totalChunks << " chunks, " << schedule.rounds
                      << " rounds, max " << schedule.chunkElementLimit
                      << " elements/chunk";
            if (schedule.chunkElementLimit < oneShotChunkElements)
                std::cout << " (streaming enabled)";
            std::cout << "\n";
        }

        struct ElasticGPViews
        {
            DeviceMatrixView<double> geoJacobianInvs;
            DeviceVectorView<double> measures;
            DeviceVectorView<double> weightForces;
            DeviceVectorView<double> weightBodys;
            DeviceMatrixView<double> Fs;
            DeviceMatrixView<double> Ss;
            DeviceMatrixView<double> Cs;
        };

        auto makeGPViews = [&](double* data, int gpCount)
        {
            ElasticGPViews views;
            int offsetLocal = 0;
            views.geoJacobianInvs =
                DeviceMatrixView<double>(data + offsetLocal, m_domainDim,
                                         gpCount * m_domainDim);
            offsetLocal += views.geoJacobianInvs.size();
            views.measures =
                DeviceVectorView<double>(data + offsetLocal, gpCount);
            offsetLocal += views.measures.size();
            views.weightForces =
                DeviceVectorView<double>(data + offsetLocal, gpCount);
            offsetLocal += views.weightForces.size();
            views.weightBodys =
                DeviceVectorView<double>(data + offsetLocal, gpCount);
            offsetLocal += views.weightBodys.size();
            views.Fs = DeviceMatrixView<double>(data + offsetLocal,
                                                m_domainDim,
                                                gpCount * m_domainDim);
            offsetLocal += views.Fs.size();
            views.Ss = DeviceMatrixView<double>(data + offsetLocal,
                                                m_domainDim,
                                                gpCount * m_domainDim);
            offsetLocal += views.Ss.size();
            views.Cs = DeviceMatrixView<double>(data + offsetLocal,
                                                m_dimTensor,
                                                gpCount * m_dimTensor);
            return views;
        };

        struct ElasticAssemblyDeviceBuffer
        {
            int device = -1;
            siga::gpuasm::SparseOutputBuffer output;
            DeviceArray<double> materialParameters;
            DeviceArray<double> gpData;
            siga::gpuasm::MultiPatchReplica geometry;
            siga::gpuasm::MultiPatchReplica displacement;
            DeviceArray<double> gpTable;
            DeviceArray<double> weights;
            DeviceArray<double> geoValuesAndDerss;
            DeviceArray<double> dispValuesAndDerss;
            DeviceArray<double> bodyForce;
            siga::gpuasm::NestedArrayReplica<double> fixedDofs;

            ElasticAssemblyDeviceBuffer(
                int device_, int matrixSize, int rhsSize, int gpDataSize,
                const std::vector<double>& materialParametersHost)
                : device(device_),
                  output(matrixSize, rhsSize),
                  materialParameters(materialParametersHost),
                  gpData(gpDataSize)
            {
            }

            void copyStaticModelData(MultiPatchDeviceView geometryView,
                                     MultiPatchDeviceView displacementView,
                                     DeviceVectorView<double> bodyForceView,
                                     int sourceDevice, int targetDevice)
            {
                geometry.updateStaticData(geometryView, sourceDevice,
                                          targetDevice, "geometry");
                displacement.updateStaticData(displacementView, sourceDevice,
                                              targetDevice, "displacement");
                siga::gpuasm::peerCopyInto(bodyForce, bodyForceView,
                                           sourceDevice, targetDevice,
                                           "body force");
            }

            void copyStaticChunkInputData(
                DeviceMatrixView<double> gpTableView,
                DeviceVectorView<double> weightsView,
                DeviceMatrixView<double> geoValuesView,
                DeviceMatrixView<double> dispValuesView, int totalGPCount,
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
            }

            void updateDynamicInputData(
                MultiPatchDeviceView displacementView,
                DeviceNestedArrayView<double> fixedDofsView,
                int sourceDevice, int targetDevice)
            {
                displacement.updateControlPoints(displacementView,
                                                 sourceDevice, targetDevice,
                                                 "displacement");
                fixedDofs.update(fixedDofsView, sourceDevice, targetDevice,
                                 "fixed dofs");
            }
        };

        std::vector<std::unique_ptr<ElasticAssemblyDeviceBuffer>> buffers(
            numAssemblyDevices);
        for (int idx = 0; idx < numAssemblyDevices; ++idx)
        {
            cudaSetDevice(assemblyDevices[idx]);
            const int localMatrixValuesSize =
                idx == 0 ? 0 : schedule.maxMatrixValueCounts[idx];
            const int localRhsSize =
                idx == 0 ? 0 : schedule.maxRowCounts[idx];
            const int localGPDataSize = stride * schedule.maxGPCounts[idx];

            std::vector<std::pair<std::string, unsigned long long>> parts;
            const unsigned long long requiredBytes = requiredBytesForDevice(
                idx, schedule.maxGPCounts[idx],
                schedule.maxRowCounts[idx],
                schedule.maxMatrixValueCounts[idx], &parts);
            std::ostringstream label;
            label << "elasticity assembly streaming buffer for GPU " << idx
                  << " (max elements " << schedule.maxElementCounts[idx]
                  << ", max GP " << schedule.maxGPCounts[idx] << ")";
            siga::gpuasm::printCudaMemoryReport(label.str(), requiredBytes,
                                                reportMemory, parts);

            buffers[idx] = std::make_unique<ElasticAssemblyDeviceBuffer>(
                assemblyDevices[idx], localMatrixValuesSize, localRhsSize,
                localGPDataSize, materialParameters);
            if (idx != 0)
            {
                buffers[idx]->output.copySparseBaseMetadata(
                    primarySystemView, primaryDevice, assemblyDevices[idx]);
                if (replicateSecondaryInputs)
                    buffers[idx]->copyStaticModelData(
                        primaryGeometryView, primaryDisplacementView,
                        primaryBodyForceView, primaryDevice,
                        assemblyDevices[idx]);
            }
        }

        const auto inputRefreshStartTime =
            std::chrono::high_resolution_clock::now();
        if (replicateSecondaryInputs)
        {
            for (int idx = 1; idx < numAssemblyDevices; ++idx)
            {
                cudaSetDevice(assemblyDevices[idx]);
                buffers[idx]->updateDynamicInputData(
                    primaryDisplacementView, fixedDofs_assemble,
                    primaryDevice, assemblyDevices[idx]);
            }
        }
        const auto inputRefreshEndTime =
            std::chrono::high_resolution_clock::now();
        cudaSetDevice(primaryDevice);

        const int gpTableColsPerGP = primaryGPTableView.cols() / m_totalGPs;
        const int geoValuesColsPerGP =
            primaryGeoValuesView.cols() / m_totalGPs;
        const int dispValuesColsPerGP =
            primaryDispValuesView.cols() / m_totalGPs;

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

            ElasticAssemblyDeviceBuffer& buffer = *buffers[idx];
            buffer.output.updateLocalSparseWindow(
                primarySystemView, chunk.rowPtrHost, chunk.rowStart,
                chunk.matrixValueStart, primaryDevice, assemblyDevices[idx]);
            buffer.output.clearActiveOutput();
            if (replicateSecondaryInputs)
                buffer.copyStaticChunkInputData(
                    primaryGPTableView, primaryWeightsView,
                    primaryGeoValuesView, primaryDispValuesView, m_totalGPs,
                    chunk.gpStart, chunk.gpCount, primaryDevice,
                    assemblyDevices[idx]);
        };

        auto runChunkGroup = [&](int round, const auto& launcher)
        {
            if (numAssemblyDevices == 1)
            {
                cudaSetDevice(primaryDevice);
                if (round < static_cast<int>(schedule.chunksByDevice[0].size()))
                {
                    const siga::gpuasm::AssemblyChunk& chunk =
                        schedule.chunksByDevice[0][round];
                    launcher(0, chunk, primarySystemView);
                }
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

            ElasticAssemblyDeviceBuffer& buffer = *buffers[idx];
            ElasticGPViews views =
                makeGPViews(buffer.gpData.data(), chunk.gpCount);
            int localMinGrid = 0;
            int localBlockSize = 0;
            cudaOccupancyMaxPotentialBlockSize(
                &localMinGrid, &localBlockSize,
                evaluateGPKernel_withoutComputingGPTableAndDers, 0,
                chunk.gpCount);
            int localGridSize =
                (chunk.gpCount + localBlockSize - 1) / localBlockSize;
            evaluateGPKernel_withoutComputingGPTableAndDers<<<
                localGridSize, localBlockSize>>>(
                m_numDerivatives, chunk.gpStart,
                inputGPStartForDevice(idx, chunk), chunk.gpCount,
                buffer.materialParameters.vectorView(),
                displacementViewForDevice(idx), geometryViewForDevice(idx),
                gpTableViewForDevice(idx, chunk), weightsViewForDevice(idx),
                geoValuesViewForDevice(idx, chunk),
                dispValuesViewForDevice(idx, chunk), views.geoJacobianInvs,
                views.measures, views.weightForces, views.weightBodys,
                views.Fs, views.Ss, views.Cs);
            cudaError_t syncErr = cudaDeviceSynchronize();
            if (syncErr != cudaSuccess)
                throw std::runtime_error(
                    "CUDA synchronize failed in chunk-local elasticity GP evaluation");
        };

        const auto launchMatrixChunk =
            [&](int idx, const siga::gpuasm::AssemblyChunk& chunk,
                SparseSystemDeviceView systemView)
        {
            if (chunk.elementCount <= 0)
                return;

            ElasticAssemblyDeviceBuffer& buffer = *buffers[idx];
            ElasticGPViews views =
                makeGPViews(buffer.gpData.data(), chunk.gpCount);
            const int chunkGridSize = m_N_D * m_N_D * chunk.elementCount;
            assembleMatrixWithGPDataKernel<<<chunkGridSize, m_N_D>>>(
                m_numDerivatives, chunk.elementStart,
                inputGPStartForDevice(idx, chunk), chunk.elementCount,
                m_N_D, displacementViewForDevice(idx), systemView,
                fixedDofsViewForDevice(idx),
                gpTableViewForDevice(idx, chunk), views.geoJacobianInvs,
                views.measures, views.weightForces, views.weightBodys,
                dispValuesViewForDevice(idx, chunk), views.Fs, views.Ss,
                views.Cs);
            cudaError_t syncErr = cudaDeviceSynchronize();
            if (syncErr != cudaSuccess)
                throw std::runtime_error(
                    "CUDA synchronize failed in chunk-local elasticity matrix assembly");
        };

        const auto launchRHSChunk =
            [&](int idx, const siga::gpuasm::AssemblyChunk& chunk,
                SparseSystemDeviceView systemView)
        {
            if (chunk.elementCount <= 0)
                return;

            ElasticAssemblyDeviceBuffer& buffer = *buffers[idx];
            ElasticGPViews views =
                makeGPViews(buffer.gpData.data(), chunk.gpCount);
            const int chunkGridSize = m_N_D * chunk.elementCount;
            assembleRHSWithGPDataKernel<<<chunkGridSize, m_N_D>>>(
                m_numDerivatives, chunk.elementStart,
                inputGPStartForDevice(idx, chunk), chunk.elementCount,
                m_N_D, displacementViewForDevice(idx), systemView,
                gpTableViewForDevice(idx, chunk), views.geoJacobianInvs,
                views.weightForces, views.weightBodys,
                dispValuesViewForDevice(idx, chunk),
                bodyForceViewForDevice(idx), views.Fs, views.Ss);
            cudaError_t syncErr = cudaDeviceSynchronize();
            if (syncErr != cudaSuccess)
                throw std::runtime_error(
                    "CUDA synchronize failed in chunk-local elasticity RHS assembly");
        };

        auto reduceChunkGroup = [&](int round)
        {
            cudaSetDevice(primaryDevice);
            for (int idx = 1; idx < numAssemblyDevices; ++idx)
            {
                if (round >=
                    static_cast<int>(schedule.chunksByDevice[idx].size()))
                    continue;
                siga::gpuasm::reduceSparseOutputBuffer(
                    csrMatrix().values(), rhs(), buffers[idx]->output);
            }
            cudaSetDevice(primaryDevice);
        };

        std::chrono::duration<double, std::milli> chunkInputMilliseconds(0.0);
        std::chrono::duration<double, std::milli> precomputeMilliseconds(0.0);
        std::chrono::duration<double, std::milli> matrixMilliseconds(0.0);
        std::chrono::duration<double, std::milli> rhsMilliseconds(0.0);
        std::chrono::duration<double, std::milli> reductionMilliseconds(0.0);

        for (int round = 0; round < schedule.rounds; ++round)
        {
            const auto chunkInputStartTime =
                std::chrono::high_resolution_clock::now();
            for (int idx = 0; idx < numAssemblyDevices; ++idx)
            {
                if (round >=
                    static_cast<int>(schedule.chunksByDevice[idx].size()))
                    continue;
                prepareChunkForDevice(idx,
                                      schedule.chunksByDevice[idx][round]);
            }
            cudaSetDevice(primaryDevice);
            const auto chunkInputEndTime =
                std::chrono::high_resolution_clock::now();
            chunkInputMilliseconds += chunkInputEndTime - chunkInputStartTime;

            const auto precomputeStartTime =
                std::chrono::high_resolution_clock::now();
            runChunkGroup(round, launchPrecomputeChunk);
            const auto precomputeEndTime =
                std::chrono::high_resolution_clock::now();
            precomputeMilliseconds += precomputeEndTime - precomputeStartTime;

            const auto matrixStartTime =
                std::chrono::high_resolution_clock::now();
            runChunkGroup(round, launchMatrixChunk);
            const auto matrixEndTime =
                std::chrono::high_resolution_clock::now();
            matrixMilliseconds += matrixEndTime - matrixStartTime;

            const auto rhsStartTime = std::chrono::high_resolution_clock::now();
            runChunkGroup(round, launchRHSChunk);
            const auto rhsEndTime = std::chrono::high_resolution_clock::now();
            rhsMilliseconds += rhsEndTime - rhsStartTime;

            const auto reductionStartTime =
                std::chrono::high_resolution_clock::now();
            reduceChunkGroup(round);
            const auto reductionEndTime =
                std::chrono::high_resolution_clock::now();
            reductionMilliseconds += reductionEndTime - reductionStartTime;
        }

        if (printTiming)
        {
            const std::chrono::duration<double, std::milli>
                inputMilliseconds =
                    (inputRefreshEndTime - inputRefreshStartTime) +
                    chunkInputMilliseconds;
            std::cout << "Elasticity assembly path: GPUs "
                      << numAssemblyDevices << ", replicated inputs: "
                      << (replicateSecondaryInputs ? "on" : "off")
                      << ", local CSR: "
                      << (useLocalCSRAssembly && numAssemblyDevices > 1
                              ? "on"
                              : "off")
                      << ", chunk limit: " << schedule.chunkElementLimit
                      << ", rounds: " << schedule.rounds
                      << ", matrix wall time: "
                      << matrixMilliseconds.count() << " ms\n";
            std::cout << "Elasticity assembly phases: input refresh "
                      << inputMilliseconds.count() << " ms, precompute "
                      << precomputeMilliseconds.count() << " ms, matrix "
                      << matrixMilliseconds.count() << " ms, RHS "
                      << rhsMilliseconds.count() << " ms, reduction "
                      << reductionMilliseconds.count() << " ms\n";
        }

        cudaSetDevice(primaryDevice);
        assembleNeumannBoundaryCondition();
        assembleDoubleStressBoundaryCondition();
        assembleFollowerMomentBoundaryCondition(fixedDofs_assemble);
        assembleNeumannCornerPointLoads();
        return;
    }

    //int* entryCountDevicePtr;
    //err = cudaMalloc((void**)&entryCountDevicePtr, sizeof(int));
    //assert(err == cudaSuccess && "cudaMalloc failed in GPUAssembler constructor during counting matrix entries");
    //err = cudaMemset(entryCountDevicePtr, 0, sizeof(int));
    //assert(err == cudaSuccess && "cudaMemset failed in GPUAssembler constructor during counting matrix entries");
    //int numElements = m_multiBasisHost.totalNumElements();
    //int N_D = m_multiBasisHost.numActive();
    //int totalGPs = m_multiBasisHost.totalNumGPs();
    //int numDerivatives = 1;
    //int geoP1 = m_multiPatchHost.knotOrder() + 1;
    //int dispP1 = m_multiBasisHost.knotOrder() + 1;
    //int dimTensor = (domainDim * (domainDim + 1)) / 2;
    //int numDoublePerGP = domainDim // pts
    //                   + domainDim * domainDim * 3 // geoJacobianInvs, Fs, Ss
    //                   + 3 // measures, weightForces, weightBodys
    //                   + geoP1 * (numDerivatives + 1) * domainDim // geoValuesAndDers
    //                   + dispP1 * (numDerivatives + 1) * domainDim // dispValuesAndDers
    //                   + (geoP1 * (geoP1 + 4) + dispP1 * (dispP1 + 4)) * domainDim // working space
    //                   + dimTensor * dimTensor; // Cs
    //size_t bytesPerGP = numDoublePerGP * sizeof(double);
    //size_t totalBytes = bytesPerGP * totalGPs;
    //size_t freeMem = 0, totalMem = 0;
    //err = cudaMemGetInfo(&freeMem, &totalMem);
    //if (err != cudaSuccess)
    //    std::cerr << "Error during cudaMemGetInfo: " << cudaGetErrorString(err) << std::endl;
    //double safetyFactor = 0.8;
    //size_t usableMem = static_cast<size_t>(freeMem * safetyFactor);
    //int numBatches = 0;
    //if (usableMem < totalBytes)
    //    numBatches = (totalBytes + usableMem - 1) / usableMem;
    //else
     //   numBatches = 1;
    //if (numBatches > std::numeric_limits<int>::max())
    //    numBatches = std::numeric_limits<int>::max();
    //printf("Number of batches: %d\n", numBatches);
    //size_t batchElements = (numElements + numBatches - 1) / numBatches;
    //size_t batchSize = batchElements * N_D;
    //DeviceArray<double> GPData(numDoublePerGP * batchSize);
    m_GPData.setZero();
    int offset = 0;
    //DeviceMatrixView<double> pts(m_GPData.data() + offset, m_domainDim, m_batchSize);
    //offset += pts.size();
    //DeviceVectorView<double> wts(m_GPData.data() + offset, m_batchSize);
    //offset += wts.size();
    DeviceMatrixView<double> geoJacobianInvs(m_GPData.data() + offset, m_domainDim, m_totalGPs * m_domainDim);
    offset += geoJacobianInvs.size();
    //int measuresStart = offset;
    DeviceVectorView<double> measures(m_GPData.data() + offset, m_totalGPs);
    offset += measures.size();
    //int weightForcesStart = offset;
    DeviceVectorView<double> weightForces(m_GPData.data() + offset, m_totalGPs);
    offset += weightForces.size();
    //int weightBodysStart = offset;
    DeviceVectorView<double> weightBodys(m_GPData.data() + offset, m_totalGPs);
    offset += weightBodys.size();
    //int geoValuesAndDerssStart = offset;
    //DeviceMatrixView<double> geoValuesAndDerss(m_GPData.data() + offset, m_geoP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim);
    //offset += geoValuesAndDerss.size();
    //int dispValuesAndDerssStart = offset;
    //DeviceMatrixView<double> dispValuesAndDerss(m_GPData.data() + offset, m_dispP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim);
    //offset += dispValuesAndDerss.size();
    //DeviceVectorView<double> geoWorkingSpaces(m_GPData.data() + offset, m_totalGPs * m_geoP1 * (m_geoP1 + 4) * m_domainDim);
    //offset += geoWorkingSpaces.size();
    //DeviceVectorView<double> dispWorkingSpaces(m_GPData.data() + offset, m_totalGPs * m_dispP1 * (m_dispP1 + 4) * m_domainDim);
    //offset += dispWorkingSpaces.size();
    //int FsStart = offset;
    DeviceMatrixView<double> Fs(m_GPData.data() + offset, m_domainDim, m_totalGPs * m_domainDim);
    offset += Fs.size();
    //int SsStart = offset;
    DeviceMatrixView<double> Ss(m_GPData.data() + offset, m_domainDim, m_totalGPs * m_domainDim);
    offset += Ss.size();
    //int CsStart = offset;
    DeviceMatrixView<double> Cs(m_GPData.data() + offset, m_dimTensor, m_totalGPs * m_dimTensor);
    offset += Cs.size();

    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
        evaluateGPKernel_withoutComputingGPTableAndDers, 0, m_totalGPs);
    gridSize = (m_totalGPs + blockSize - 1) / blockSize;
    evaluateGPKernel_withoutComputingGPTableAndDers<<<gridSize, blockSize>>>(m_numDerivatives, 0, 0, m_totalGPs,
        parameterValues.vectorView(), 
        m_displacement.deviceView(), 
        m_multiPatch.deviceView(), 
        m_GPTable.matrixView(m_domainDim, m_totalGPs), 
        m_wts.vectorView(), 
        m_geoValuesAndDerss.matrixView(m_geoP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim), 
        m_dispValuesAndDerss.matrixView(m_dispP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim),
        geoJacobianInvs, measures, weightForces, weightBodys,
        Fs, Ss, Cs);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (evaluateGPKernel_withoutComputingGPTableAndDers): " << cudaGetErrorString(err) << std::endl;

    int blockSize_assembleMatrix = m_N_D;
    //int gridSize_assembleMatrix = (N_D * N_D * batchElements + blockSize_assembleMatrix - 1) / blockSize_assembleMatrix;
    int gridSize_assembleMatrix = m_N_D * m_N_D * m_numElements;
    int blockSize_assembleRHs = m_N_D;
    //int gridSize_assembleRHs = (N_D * batchElements + blockSize_assembleRHs - 1) / blockSize_assembleRHs;
    int gridSize_assembleRHs = m_N_D * m_numElements;

    assembleMatrixWithGPDataKernel<<<gridSize_assembleMatrix, blockSize_assembleMatrix>>>(m_numDerivatives, 0, 0,
        m_numElements, m_N_D, 
        m_displacement.deviceView(), m_sparseSystem.deviceView(), fixedDofs_assemble,
        m_GPTable.matrixView(m_domainDim, m_totalGPs), geoJacobianInvs, measures, weightForces, weightBodys, m_dispValuesAndDerss.matrixView(m_dispP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim), 
        Fs, Ss, Cs);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after kernel assembleMatrixWithGPDataKernel launch: " << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (assembleMatrixWithGPDataKernel): " << cudaGetErrorString(err) << std::endl;
    
    assembleRHSWithGPDataKernel<<<gridSize_assembleRHs, blockSize_assembleRHs>>>(m_numDerivatives, 0, 0,
        m_numElements, m_N_D, m_displacement.deviceView(), m_sparseSystem.deviceView(),
        m_GPTable.matrixView(m_domainDim, m_totalGPs), 
        geoJacobianInvs, weightForces, weightBodys, 
        m_dispValuesAndDerss.matrixView(m_dispP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim), m_bodyForce.vectorView(), Fs, Ss);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after kernel assembleRHSWithGPDataKernel launch: " << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (assembleRHSWithGPDataKernel): " << cudaGetErrorString(err) << std::endl;

    assembleNeumannBoundaryCondition();
    assembleDoubleStressBoundaryCondition();
    assembleFollowerMomentBoundaryCondition(fixedDofs_assemble);
    assembleNeumannCornerPointLoads();

#if 0
    DeviceArray<double> measuresCopy1(m_GPData, measuresStart, measures.size());
    DeviceArray<double> weightForcesCopy1(m_GPData, weightForcesStart, weightForces.size());
    DeviceArray<double> weightBodysCopy1(m_GPData, weightBodysStart, weightBodys.size());
    DeviceArray<double> FsCopy1(m_GPData, FsStart, Fs.size());
    DeviceArray<double> SsCopy1(m_GPData, SsStart, Ss.size());
    DeviceArray<double> CsCopy1(m_GPData, CsStart, Cs.size());


    int blockSize_evaluateGPKernel = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize_evaluateGPKernel, 
        evaluateGPKernel, 0, m_batchSize);
    int gridSize_evaluateGPKernel = (m_batchSize + blockSize_evaluateGPKernel - 1) / blockSize_evaluateGPKernel;
    int blockSize_assembleMatrix = m_N_D;
    //int gridSize_assembleMatrix = (N_D * N_D * batchElements + blockSize_assembleMatrix - 1) / blockSize_assembleMatrix;
    int gridSize_assembleMatrix = m_N_D * m_N_D * m_batchElements;
    int blockSize_assembleRHs = m_N_D;
    //int gridSize_assembleRHs = (N_D * batchElements + blockSize_assembleRHs - 1) / blockSize_assembleRHs;
    int gridSize_assembleRHs = m_N_D * m_batchElements;

    for (int batch = 0; batch < m_numBatches; ++batch) {
            evaluateGPKernel<<<gridSize_evaluateGPKernel, blockSize_evaluateGPKernel>>>(m_numDerivatives, batch * m_batchSize,
            m_batchSize, parameterValues.vectorView(), m_displacement.deviceView(), m_multiPatch.deviceView(), 
            m_multiGaussPoints.view(), pts, /*wts,*/ geoJacobianInvs, measures, weightForces, weightBodys, geoValuesAndDerss, dispValuesAndDerss, geoWorkingSpaces, dispWorkingSpaces,
            Fs, Ss, Cs);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "Error after kernel evaluateGPKernel launch: " << cudaGetErrorString(err) << std::endl;
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            std::cerr << "CUDA error during device synchronization (evaluateGPKernel): " << cudaGetErrorString(err) << std::endl;

        assembleMatrixWithGPDataKernel<<<gridSize_assembleMatrix, blockSize_assembleMatrix>>>(m_numDerivatives, batch * m_batchElements, 0,
            m_batchElements, m_N_D, m_displacement.deviceView(), m_sparseSystem.deviceView(), fixedDofs_assemble,
            pts, geoJacobianInvs, measures, weightForces, weightBodys, dispValuesAndDerss, Fs, Ss, Cs);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "Error after kernel assembleMatrixWithGPDataKernel launch: " << cudaGetErrorString(err) << std::endl;
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            std::cerr << "CUDA error during device synchronization (assembleMatrixWithGPDataKernel): " << cudaGetErrorString(err) << std::endl;
        
        assembleRHSWithGPDataKernel<<<gridSize_assembleRHs, blockSize_assembleRHs>>>(m_numDerivatives, batch * m_batchElements, 0,
            m_batchElements, m_N_D, m_displacement.deviceView(), m_sparseSystem.deviceView(),
            pts, geoJacobianInvs, weightForces, weightBodys, dispValuesAndDerss, m_bodyForce.vectorView(), Fs, Ss);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "Error after kernel assembleRHSWithGPDataKernel launch: " << cudaGetErrorString(err) << std::endl;
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            std::cerr << "CUDA error during device synchronization (assembleRHSWithGPDataKernel): " << cudaGetErrorString(err) << std::endl;
    }
#endif
#if 0
    //m_sparseSystem.csrMatrix().print_host();
    //DeviceArray<double> matrixValuesA(m_sparseSystem.csrMatrix().valuesData());
    //std::cout << std::endl;
    //std::cout << m_sparseSystem.hostRHS();
    //DeviceArray<double> rhsValuesA(m_sparseSystem.RHS());
    //std::cout << std::endl << std::endl;
    //m_sparseSystem.matrixSetZero();
    //m_sparseSystem.RHSSetZero();
    DeviceArray<double> GPTable(m_GPData, 0, m_totalGPs * m_domainDim);
    DeviceArray<double> geoValuesAndDerssCopy(m_GPData, geoValuesAndDerssStart, geoValuesAndDerss.size());
    DeviceArray<double> dispValuesAndDerssCopy(m_GPData, dispValuesAndDerssStart, dispValuesAndDerss.size());
    m_GPTable.compare(GPTable);
    m_geoValuesAndDerss.compare(geoValuesAndDerssCopy);
    m_dispValuesAndDerss.compare(dispValuesAndDerssCopy);
    DeviceArray<double> measuresCopy2(m_GPData, measuresStart, measures.size());
    DeviceArray<double> weightForcesCopy2(m_GPData, weightForcesStart, weightForces.size());
    DeviceArray<double> weightBodysCopy2(m_GPData, weightBodysStart, weightBodys.size());
    DeviceArray<double> FsCopy2(m_GPData, FsStart, Fs.size());
    DeviceArray<double> SsCopy2(m_GPData, SsStart, Ss.size());
    DeviceArray<double> CsCopy2(m_GPData, CsStart, Cs.size());

    measuresCopy1.compare(measuresCopy2);
    weightForcesCopy1.compare(weightForcesCopy2);
    weightBodysCopy1.compare(weightBodysCopy2);
    FsCopy1.compare(FsCopy2);
    SsCopy1.compare(SsCopy2);
    CsCopy1.compare(CsCopy2);

#endif
#if 0
    int numActivePerBlock = std::min(16, m_N_D);
    dim3 blockSize2D(numActivePerBlock, numActivePerBlock);
    int numBlocksPerEle = (m_N_D + numActivePerBlock - 1) / numActivePerBlock;
    gridSize = numBlocksPerEle * numBlocksPerEle * m_numElements;
    int numDouble = m_geoP1 * (m_numDerivatives + 1) * m_domainDim + (m_geoP1 * m_geoP1 + 4 * m_geoP1) * m_domainDim + //geoValuesAndDers + workingSpace
                    m_dispP1 * (m_numDerivatives + 1) * m_domainDim + (m_dispP1 * m_dispP1 + 4 * m_dispP1) * m_domainDim + //dispValuesAndDers + workingSpace
                    numActivePerBlock * m_domainDim * numActivePerBlock * m_domainDim + //localMat
                    numActivePerBlock * m_domainDim; // localRHS
    size_t shmemBytes = numDouble * sizeof(double);
    assembleDomainKernel_perTileBlock_loopOverGps<<<gridSize, blockSize2D, shmemBytes>>>(
                    m_numDerivatives, m_numElements, 
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
    //m_sparseSystem.csrMatrix().print_host();
    //std::cout << std::endl;
    //std::cout << m_sparseSystem.hostRHS();
    //std::cout << std::endl << std::endl;
    DeviceArray<double> matrixValuesB(m_sparseSystem.csrMatrix().valuesData());
    DeviceArray<double> rhsValuesB(m_sparseSystem.RHS());

    matrixValuesA.compare(matrixValuesB);
    rhsValuesA.compare(rhsValuesB);
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

void GPUAssembler::refreshFixedDofs()
{
    std::vector<DofMapper> dofMappers_stdVec(m_targetDim);
    m_multiBasisHost.getMappers(true, m_boundaryConditions, dofMappers_stdVec, true);
    std::vector<Eigen::VectorXd> ddof(m_targetDim);
    for (int unk = 0; unk < m_targetDim; ++unk)
        computeDirichletDofs(unk, dofMappers_stdVec, ddof, m_multiBasisHost);
    m_ddof.setData(ddof);
}

void GPUAssembler::assembleNeumannBoundaryCondition()
{
    const BoundaryConditions::bcContainer& neumannSides = m_boundaryConditions.neumannSides();
    const int numNeumannSides = static_cast<int>(neumannSides.size());
    if (numNeumannSides == 0)
        return;

    const double neumannLoadScaling = options().getReal("neumann_load_scaling");
    std::vector<int> bcOffsets;
    std::vector<int> bcPatches;
    std::vector<int> bcSideIndexes;
    Eigen::MatrixXd bcValues(m_targetDim, numNeumannSides);
    bcOffsets.reserve(numNeumannSides + 1);
    bcPatches.reserve(numNeumannSides);
    bcSideIndexes.reserve(numNeumannSides);
    bcOffsets.push_back(0);
    bcValues.setZero();

    int totalNumBoundaryGPs = 0;
    for (BoundaryConditions::bcContainer::const_iterator it = neumannSides.begin();
         it != neumannSides.end(); ++it)
    {
        const int bcIdx = static_cast<int>(std::distance(neumannSides.begin(), it));
        const int patchIdx = it->patchIndex();
        const int fixedDir = it->side().direction();

        int numBoundaryGPs = 1;
        for (int d = 0; d < m_domainDim; ++d)
            if (d != fixedDir)
                numBoundaryGPs *= m_multiBasisHost.basis(patchIdx).getTotalNumGaussPoints(d);

        totalNumBoundaryGPs += numBoundaryGPs;
        bcOffsets.push_back(totalNumBoundaryGPs);
        bcPatches.push_back(patchIdx);
        bcSideIndexes.push_back(it->side().index());

        const Eigen::VectorXd values = it->valuesVector();
        const int numValues = static_cast<int>(values.size());
        for (int i = 0; i < m_targetDim && i < numValues; ++i)
            bcValues(i, bcIdx) = neumannLoadScaling * values[i];
    }

    if (totalNumBoundaryGPs == 0)
        return;

    DeviceArray<int> bcOffsetsDevice(bcOffsets);
    DeviceArray<int> bcPatchesDevice(bcPatches);
    DeviceArray<int> bcSideIndexesDevice(bcSideIndexes);
    DeviceArray<double> bcValuesDevice(bcValues);

    const int totalEntries = totalNumBoundaryGPs * m_N_D;
    int blockSize = 0;
    int minGrid = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        assembleNeumannBoundaryConditionKernel, 0, totalEntries);
    if (blockSize <= 0)
        blockSize = 128;
    const int gridSize = (totalEntries + blockSize - 1) / blockSize;

    assembleNeumannBoundaryConditionKernel<<<gridSize, blockSize>>>(
        totalNumBoundaryGPs,
        m_N_D,
        m_displacement.deviceView(),
        m_multiPatch.deviceView(),
        m_sparseSystem.deviceView(),
        m_multiGaussPoints.view(),
        bcOffsetsDevice.vectorView(),
        bcPatchesDevice.vectorView(),
        bcSideIndexesDevice.vectorView(),
        bcValuesDevice.matrixView(m_targetDim, numNeumannSides));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after assembleNeumannBoundaryConditionKernel launch: "
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during assembleNeumannBoundaryConditionKernel: "
                  << cudaGetErrorString(err) << std::endl;
}

void GPUAssembler::assembleDoubleStressBoundaryCondition()
{
    const int numDoubleStressSides = static_cast<int>(
        std::distance(m_boundaryConditions.doubleStressBegin(),
                      m_boundaryConditions.doubleStressEnd()));
    if (numDoubleStressSides == 0)
        return;

    if (m_domainDim != m_targetDim || (m_domainDim != 2 && m_domainDim != 3))
        throw std::runtime_error("Double stress boundary conditions currently support 2D or 3D displacement problems only.");

    const double neumannLoadScaling = options().getReal("neumann_load_scaling");
    std::vector<int> bcOffsets;
    std::vector<int> bcPatches;
    std::vector<int> bcSideIndexes;
    Eigen::MatrixXd bcValues(m_targetDim, numDoubleStressSides);
    bcOffsets.reserve(numDoubleStressSides + 1);
    bcPatches.reserve(numDoubleStressSides);
    bcSideIndexes.reserve(numDoubleStressSides);
    bcOffsets.push_back(0);
    bcValues.setZero();

    int totalNumBoundaryGPs = 0;
    for (BoundaryConditions::bcContainer::const_iterator it = m_boundaryConditions.doubleStressBegin();
         it != m_boundaryConditions.doubleStressEnd(); ++it)
    {
        const int bcIdx = static_cast<int>(
            std::distance(m_boundaryConditions.doubleStressBegin(), it));
        const int patchIdx = it->patchIndex();
        const int fixedDir = it->side().direction();

        int numBoundaryGPs = 1;
        for (int d = 0; d < m_domainDim; ++d)
            if (d != fixedDir)
                numBoundaryGPs *= m_multiBasisHost.basis(patchIdx).getTotalNumGaussPoints(d);

        totalNumBoundaryGPs += numBoundaryGPs;
        bcOffsets.push_back(totalNumBoundaryGPs);
        bcPatches.push_back(patchIdx);
        bcSideIndexes.push_back(it->side().index());

        const Eigen::VectorXd values = it->valuesVector();
        const int numValues = static_cast<int>(values.size());
        for (int i = 0; i < m_targetDim && i < numValues; ++i)
            bcValues(i, bcIdx) = neumannLoadScaling * values[i];
    }

    if (totalNumBoundaryGPs == 0)
        return;

    DeviceArray<int> bcOffsetsDevice(bcOffsets);
    DeviceArray<int> bcPatchesDevice(bcPatches);
    DeviceArray<int> bcSideIndexesDevice(bcSideIndexes);
    DeviceArray<double> bcValuesDevice(bcValues);

    const int totalEntries = totalNumBoundaryGPs * m_N_D;
    int blockSize = 0;
    int minGrid = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        assembleDoubleStressBoundaryConditionKernel, 0, totalEntries);
    if (blockSize <= 0)
        blockSize = 128;
    const int gridSize = (totalEntries + blockSize - 1) / blockSize;

    assembleDoubleStressBoundaryConditionKernel<<<gridSize, blockSize>>>(
        totalNumBoundaryGPs,
        m_N_D,
        m_displacement.deviceView(),
        m_multiPatch.deviceView(),
        m_sparseSystem.deviceView(),
        m_multiGaussPoints.view(),
        bcOffsetsDevice.vectorView(),
        bcPatchesDevice.vectorView(),
        bcSideIndexesDevice.vectorView(),
        bcValuesDevice.matrixView(m_targetDim, numDoubleStressSides));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after assembleDoubleStressBoundaryConditionKernel launch: "
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during assembleDoubleStressBoundaryConditionKernel: "
                  << cudaGetErrorString(err) << std::endl;
}

void GPUAssembler::assembleFollowerMomentBoundaryCondition(
    const DeviceNestedArrayView<double>& fixedDofs_assemble)
{
    const BoundaryConditions::bcContainer& followerMomentSides =
        m_boundaryConditions.followerMomentSides();
    const int numFollowerMomentSides =
        static_cast<int>(followerMomentSides.size());
    if (numFollowerMomentSides == 0)
        return;

    if (m_domainDim != 2 || m_targetDim != 2)
        throw std::runtime_error("Follower moment boundary conditions currently support 2D displacement problems only.");

    const double neumannLoadScaling = options().getReal("neumann_load_scaling");
    std::vector<int> bcOffsets;
    std::vector<int> bcPatches;
    std::vector<int> bcSideIndexes;
    std::vector<int> bcDofOffsets;
    std::vector<int> bcBoundaryDofOffsets;
    std::vector<int> bcBoundaryDofs;
    std::vector<double> momentValues(numFollowerMomentSides, 0.0);
    bcOffsets.reserve(numFollowerMomentSides + 1);
    bcPatches.reserve(numFollowerMomentSides);
    bcSideIndexes.reserve(numFollowerMomentSides);
    bcDofOffsets.reserve(numFollowerMomentSides + 1);
    bcBoundaryDofOffsets.reserve(numFollowerMomentSides + 1);
    bcOffsets.push_back(0);
    bcDofOffsets.push_back(0);
    bcBoundaryDofOffsets.push_back(0);

    int totalNumBoundaryGPs = 0;
    for (BoundaryConditions::bcContainer::const_iterator it = followerMomentSides.begin();
         it != followerMomentSides.end(); ++it)
    {
        const int bcIdx = static_cast<int>(
            std::distance(followerMomentSides.begin(), it));
        const int patchIdx = it->patchIndex();
        const int fixedDir = it->side().direction();
        if (fixedDir != 0)
            throw std::runtime_error("Follower moment boundary conditions currently support west/east beam-end sides only.");

        int numBoundaryGPs = 1;
        for (int d = 0; d < m_domainDim; ++d)
            if (d != fixedDir)
                numBoundaryGPs *= m_multiBasisHost.basis(patchIdx).getTotalNumGaussPoints(d);

        totalNumBoundaryGPs += numBoundaryGPs;
        bcOffsets.push_back(totalNumBoundaryGPs);
        bcPatches.push_back(patchIdx);
        bcSideIndexes.push_back(it->side().index());
        bcDofOffsets.push_back(
            bcDofOffsets.back() +
            m_multiBasisHost.basis(patchIdx).getNumControlPoints() *
                m_targetDim);
        const Eigen::VectorXi boundaryDofs =
            m_multiBasisHost.basis(patchIdx).boundary(it->side());
        for (int i = 0; i < boundaryDofs.size(); ++i)
            bcBoundaryDofs.push_back(boundaryDofs[i]);
        bcBoundaryDofOffsets.push_back(
            static_cast<int>(bcBoundaryDofs.size()));

        const Eigen::VectorXd values = it->valuesVector();
        if (values.size() > 0)
            momentValues[bcIdx] = neumannLoadScaling * values[0];
    }

    if (totalNumBoundaryGPs == 0)
        return;

    DeviceArray<int> bcOffsetsDevice(bcOffsets);
    DeviceArray<int> bcPatchesDevice(bcPatches);
    DeviceArray<int> bcSideIndexesDevice(bcSideIndexes);
    DeviceArray<int> bcDofOffsetsDevice(bcDofOffsets);
    DeviceArray<int> bcBoundaryDofOffsetsDevice(bcBoundaryDofOffsets);
    DeviceArray<int> bcBoundaryDofsDevice(bcBoundaryDofs);
    DeviceArray<double> momentValuesDevice(momentValues);
    DeviceArray<double> statsDevice(
        FOLLOWER_MOMENT_STATS_STRIDE * numFollowerMomentSides);
    statsDevice.setZero();
    DeviceArray<double> derivativeStatsDevice(
        FOLLOWER_MOMENT_DERIVATIVE_STRIDE * bcDofOffsets.back());
    derivativeStatsDevice.setZero();

    int blockSize = 0;
    int minGrid = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        computeFollowerMomentBoundaryCentroidKernel, 0, totalNumBoundaryGPs);
    if (blockSize <= 0)
        blockSize = 128;
    int gridSize = (totalNumBoundaryGPs + blockSize - 1) / blockSize;

    computeFollowerMomentBoundaryCentroidKernel<<<gridSize, blockSize>>>(
        totalNumBoundaryGPs,
        m_displacement.deviceView(),
        m_multiPatch.deviceView(),
        m_multiGaussPoints.view(),
        bcOffsetsDevice.vectorView(),
        bcPatchesDevice.vectorView(),
        bcSideIndexesDevice.vectorView(),
        statsDevice.vectorView());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after computeFollowerMomentBoundaryCentroidKernel launch: "
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during computeFollowerMomentBoundaryCentroidKernel: "
                  << cudaGetErrorString(err) << std::endl;

    const int normalizeBlockSize = 128;
    const int normalizeGridSize =
        (numFollowerMomentSides + normalizeBlockSize - 1) / normalizeBlockSize;
    normalizeFollowerMomentBoundaryCentroidKernel<<<normalizeGridSize, normalizeBlockSize>>>(
        numFollowerMomentSides,
        statsDevice.vectorView());

    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after normalizeFollowerMomentBoundaryCentroidKernel launch: "
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during normalizeFollowerMomentBoundaryCentroidKernel: "
                  << cudaGetErrorString(err) << std::endl;

    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        computeFollowerMomentBoundaryInertiaKernel, 0, totalNumBoundaryGPs);
    if (blockSize <= 0)
        blockSize = 128;
    gridSize = (totalNumBoundaryGPs + blockSize - 1) / blockSize;

    computeFollowerMomentBoundaryInertiaKernel<<<gridSize, blockSize>>>(
        totalNumBoundaryGPs,
        m_displacement.deviceView(),
        m_multiPatch.deviceView(),
        m_multiGaussPoints.view(),
        bcOffsetsDevice.vectorView(),
        bcPatchesDevice.vectorView(),
        bcSideIndexesDevice.vectorView(),
        statsDevice.vectorView());

    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after computeFollowerMomentBoundaryInertiaKernel launch: "
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during computeFollowerMomentBoundaryInertiaKernel: "
                  << cudaGetErrorString(err) << std::endl;

    const int totalDerivativeEntries =
        totalNumBoundaryGPs * m_N_D * m_targetDim;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        computeFollowerMomentBoundaryDerivativeStatsKernel, 0,
        totalDerivativeEntries);
    if (blockSize <= 0)
        blockSize = 128;
    gridSize = (totalDerivativeEntries + blockSize - 1) / blockSize;

    computeFollowerMomentBoundaryDerivativeStatsKernel<<<gridSize, blockSize>>>(
        totalNumBoundaryGPs,
        m_N_D,
        m_displacement.deviceView(),
        m_multiPatch.deviceView(),
        m_multiGaussPoints.view(),
        bcOffsetsDevice.vectorView(),
        bcPatchesDevice.vectorView(),
        bcSideIndexesDevice.vectorView(),
        bcDofOffsetsDevice.vectorView(),
        derivativeStatsDevice.vectorView());

    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after computeFollowerMomentBoundaryDerivativeStatsKernel launch: "
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during computeFollowerMomentBoundaryDerivativeStatsKernel: "
                  << cudaGetErrorString(err) << std::endl;

    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        computeFollowerMomentBoundaryInertiaDerivativeKernel, 0,
        totalDerivativeEntries);
    if (blockSize <= 0)
        blockSize = 128;
    gridSize = (totalDerivativeEntries + blockSize - 1) / blockSize;

    computeFollowerMomentBoundaryInertiaDerivativeKernel<<<gridSize, blockSize>>>(
        totalNumBoundaryGPs,
        m_N_D,
        m_displacement.deviceView(),
        m_multiPatch.deviceView(),
        m_multiGaussPoints.view(),
        bcOffsetsDevice.vectorView(),
        bcPatchesDevice.vectorView(),
        bcSideIndexesDevice.vectorView(),
        bcDofOffsetsDevice.vectorView(),
        statsDevice.vectorView(),
        derivativeStatsDevice.vectorView());

    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after computeFollowerMomentBoundaryInertiaDerivativeKernel launch: "
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during computeFollowerMomentBoundaryInertiaDerivativeKernel: "
                  << cudaGetErrorString(err) << std::endl;

    const int totalDerivativeDofs = bcDofOffsets.back();
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        addFollowerMomentBoundaryGlobalInertiaDerivativeKernel, 0,
        totalDerivativeDofs);
    if (blockSize <= 0)
        blockSize = 128;
    gridSize = (totalDerivativeDofs + blockSize - 1) / blockSize;

    addFollowerMomentBoundaryGlobalInertiaDerivativeKernel<<<gridSize, blockSize>>>(
        numFollowerMomentSides,
        bcDofOffsetsDevice.vectorView(),
        statsDevice.vectorView(),
        derivativeStatsDevice.vectorView());

    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after addFollowerMomentBoundaryGlobalInertiaDerivativeKernel launch: "
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during addFollowerMomentBoundaryGlobalInertiaDerivativeKernel: "
                  << cudaGetErrorString(err) << std::endl;

    const int totalEntries = totalNumBoundaryGPs * m_N_D;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        assembleFollowerMomentBoundaryConditionKernel, 0, totalEntries);
    if (blockSize <= 0)
        blockSize = 128;
    gridSize = (totalEntries + blockSize - 1) / blockSize;

    assembleFollowerMomentBoundaryConditionKernel<<<gridSize, blockSize>>>(
        totalNumBoundaryGPs,
        m_N_D,
        m_displacement.deviceView(),
        m_multiPatch.deviceView(),
        m_sparseSystem.deviceView(),
        fixedDofs_assemble,
        m_multiGaussPoints.view(),
        bcOffsetsDevice.vectorView(),
        bcPatchesDevice.vectorView(),
        bcSideIndexesDevice.vectorView(),
        bcDofOffsetsDevice.vectorView(),
        momentValuesDevice.vectorView(),
        statsDevice.vectorView(),
        derivativeStatsDevice.vectorView());

    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after assembleFollowerMomentBoundaryConditionKernel launch: "
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during assembleFollowerMomentBoundaryConditionKernel: "
                  << cudaGetErrorString(err) << std::endl;

    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        assembleFollowerMomentBoundaryGlobalTangentKernel, 0, totalEntries);
    if (blockSize <= 0)
        blockSize = 128;
    gridSize = (totalEntries + blockSize - 1) / blockSize;

    assembleFollowerMomentBoundaryGlobalTangentKernel<<<gridSize, blockSize>>>(
        totalNumBoundaryGPs,
        m_N_D,
        m_displacement.deviceView(),
        m_multiPatch.deviceView(),
        m_sparseSystem.deviceView(),
        fixedDofs_assemble,
        m_multiGaussPoints.view(),
        bcOffsetsDevice.vectorView(),
        bcPatchesDevice.vectorView(),
        bcSideIndexesDevice.vectorView(),
        bcDofOffsetsDevice.vectorView(),
        bcBoundaryDofOffsetsDevice.vectorView(),
        bcBoundaryDofsDevice.vectorView(),
        momentValuesDevice.vectorView(),
        statsDevice.vectorView(),
        derivativeStatsDevice.vectorView());

    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after assembleFollowerMomentBoundaryGlobalTangentKernel launch: "
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during assembleFollowerMomentBoundaryGlobalTangentKernel: "
                  << cudaGetErrorString(err) << std::endl;
}

void GPUAssembler::assembleNeumannCornerPointLoads()
{
    const int numCornerLoads = static_cast<int>(
        std::distance(m_boundaryConditions.neumannCornerBegin(),
                      m_boundaryConditions.neumannCornerEnd()));
    if (numCornerLoads == 0)
        return;

    const double neumannLoadScaling = options().getReal("neumann_load_scaling");
    std::vector<int> cornerPatches;
    std::vector<int> cornerDofs;
    Eigen::MatrixXd loadValues(m_targetDim, numCornerLoads);
    cornerPatches.reserve(numCornerLoads);
    cornerDofs.reserve(numCornerLoads);
    loadValues.setZero();

    int loadIdx = 0;
    for (BoundaryConditions::const_corner_iterator it = m_boundaryConditions.neumannCornerBegin();
         it != m_boundaryConditions.neumannCornerEnd(); ++it, ++loadIdx)
    {
        const int patchIdx = it->patchIndex();
        cornerPatches.push_back(patchIdx);
        cornerDofs.push_back(m_multiBasisHost.basis(patchIdx).corner(it->corner()));

        const Eigen::VectorXd values = it->valuesVector();
        const int numValues = static_cast<int>(values.size());
        for (int i = 0; i < m_targetDim && i < numValues; ++i)
            loadValues(i, loadIdx) = neumannLoadScaling * values[i];
    }

    DeviceArray<int> cornerPatchesDevice(cornerPatches);
    DeviceArray<int> cornerDofsDevice(cornerDofs);
    DeviceArray<double> loadValuesDevice(loadValues);

    const int totalEntries = numCornerLoads * m_targetDim;
    int blockSize = 0;
    int minGrid = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        assembleNeumannCornerPointLoadKernel, 0, totalEntries);
    if (blockSize <= 0)
        blockSize = 128;
    const int gridSize = (totalEntries + blockSize - 1) / blockSize;

    assembleNeumannCornerPointLoadKernel<<<gridSize, blockSize>>>(
        numCornerLoads,
        m_sparseSystem.deviceView(),
        cornerPatchesDevice.vectorView(),
        cornerDofsDevice.vectorView(),
        loadValuesDevice.matrixView(m_targetDim, numCornerLoads));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after assembleNeumannCornerPointLoadKernel launch: "
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during assembleNeumannCornerPointLoadKernel: "
                  << cudaGetErrorString(err) << std::endl;
}

void GPUAssembler::assembleMatrix(
    const DeviceNestedArrayView<double>& fixedDofs_assemble,
    const DeviceMatrixView<double>& geoJacobianInvs,
    const DeviceVectorView<double>& measures,
    const DeviceVectorView<double>& weightForces,
    const DeviceVectorView<double>& weightBodys,
    const DeviceMatrixView<double>& Fs,
    const DeviceMatrixView<double>& Ss,
    const DeviceMatrixView<double>& Cs)
{
    int blockSize = m_N_D;
    int gridSize = m_N_D * m_N_D * m_numElements;

    assembleMatrixWithGPDataKernel<<<gridSize, blockSize>>>(m_numDerivatives, 0, 0,
            m_numElements, m_N_D, 
            m_displacement.deviceView(), m_sparseSystem.deviceView(), fixedDofs_assemble,
            m_GPTable.matrixView(m_domainDim, m_totalGPs), geoJacobianInvs, measures, weightForces, weightBodys, m_dispValuesAndDerss.matrixView(m_dispP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim), 
            Fs, Ss, Cs);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after kernel assembleMatrixWithGPDataKernel launch: " << cudaGetErrorString(err) << std::endl;  
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (assembleMatrixWithGPDataKernel): " << cudaGetErrorString(err) << std::endl;
}

void GPUAssembler::denseMatrix(DeviceMatrixView<double> denseMat) const
{
    m_sparseSystem.csrMatrix().toDense(denseMat);
}

int GPUAssembler::numDispMatrixEntries() const
{
    int numActivePerBlock = std::min(16, m_N_D);
    int numBlocksPerElement = (m_N_D + numActivePerBlock - 1) / numActivePerBlock;
    dim3 blockSize(numActivePerBlock, numActivePerBlock);
    int gridSize = m_numElements * numBlocksPerElement * numBlocksPerElement;

    int* entryCountDevicePtr;
    cudaError_t err = cudaMalloc((void**)&entryCountDevicePtr, sizeof(int));
    assert(err == cudaSuccess && "cudaMalloc failed in GPUAssembler::numDispMatrixEntries");
    err = cudaMemset(entryCountDevicePtr, 0, sizeof(int));
    assert(err == cudaSuccess && "cudaMemset failed in GPUAssembler::numDispMatrixEntries");

    countEntrysKernel<<<gridSize, blockSize>>>(m_numElements, numBlocksPerElement, 
                                               numActivePerBlock,
                                               m_displacement.deviceView(),
                                               m_multiPatch.deviceView(),
                                               m_multiGaussPoints.view(),
                                               m_sparseSystem.deviceView(),
                                               m_ddof.view(),
                                               entryCountDevicePtr);

    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after kernel countEntrysKernel launch: " 
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (countEntrysKernel): " 
                  << cudaGetErrorString(err) << std::endl;

    int entryCountHost;
    err = cudaMemcpy(&entryCountHost, entryCountDevicePtr, sizeof(int), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess && "cudaMemcpy failed in GPUAssembler::numDispMatrixEntries");

    cudaFree(entryCountDevicePtr);

    return entryCountHost;

}

void GPUAssembler::computeCOO(DeviceVectorView<int> cooRows, 
                              DeviceVectorView<int> cooCols) const
{
    int numActivePerBlock = std::min(16, m_N_D);
    int numBlocksPerElement = (m_N_D + numActivePerBlock - 1) / numActivePerBlock;
    dim3 blockSize(numActivePerBlock, numActivePerBlock);
    int gridSize = m_numElements * numBlocksPerElement * numBlocksPerElement;

    int* entryCountDevicePtr;
    cudaError_t err = cudaMalloc((void**)&entryCountDevicePtr, sizeof(int));
    assert(err == cudaSuccess && "cudaMalloc failed in GPUAssembler::computeCOO");
    err = cudaMemset(entryCountDevicePtr, 0, sizeof(int));
    assert(err == cudaSuccess && "cudaMemset failed in GPUAssembler::computeCOO");

    computeCOOKernel<<<gridSize, blockSize>>>(m_numElements, 
                                entryCountDevicePtr,
                                numBlocksPerElement, numActivePerBlock,
                                m_displacement.deviceView(),
                                m_multiPatch.deviceView(),
                                m_multiGaussPoints.view(),
                                m_sparseSystem.deviceView(),
                                //m_ddof.view(),
                                cooRows,
                                cooCols);

    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after kernel computeCOOKernel launch: " 
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (computeCOOKernel): " 
                  << cudaGetErrorString(err) << std::endl;
    
    cudaFree(entryCountDevicePtr);
}

void GPUAssembler::computeGPTable()
{
    m_GPTable.resize(m_totalGPs * m_domainDim);
    m_wts.resize(m_totalGPs);
    int minGrid, blockSize_GPTable;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize_GPTable, 
                         countEntrysKernel, 0, m_totalGPs);
    int gridSize = (m_totalGPs + blockSize_GPTable - 1) / blockSize_GPTable;
    computeGPTableKernel<<<gridSize, blockSize_GPTable>>>(m_totalGPs, 
        m_displacement.deviceView(), m_multiGaussPoints.view(),
        m_GPTable.matrixView(m_domainDim, m_totalGPs), m_wts.vectorView());
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (computeGPTableKernel): " 
                  << cudaGetErrorString(err) << std::endl;
}

void GPUAssembler::evaluateBasisValuesAndDerivativesAtGPs()
{
    m_geoValuesAndDerss.resize(m_geoP1 * m_totalGPs * (m_numDerivatives + 1) * m_domainDim);
    m_dispValuesAndDerss.resize(m_dispP1 * m_totalGPs * (m_numDerivatives + 1) * m_domainDim);
    DeviceArray<double> geoWorkingSpaces(m_totalGPs * m_geoP1 * (m_geoP1 + 4) * m_domainDim);
    DeviceArray<double> dispWorkingSpaces(m_totalGPs * m_dispP1 * (m_dispP1 + 4) * m_domainDim);

    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, evaluateBasisValuesAndDerivativesAtGPsKernel, 0, m_totalGPs * m_domainDim);
    int gridSize = (m_totalGPs * m_domainDim + blockSize - 1) / blockSize;
    evaluateBasisValuesAndDerivativesAtGPsKernel<<<gridSize, blockSize>>>(
            m_numDerivatives, m_totalGPs, 
            m_domainDim, m_displacement.deviceView(),
            m_multiPatch.deviceView(), 
            m_GPTable.matrixView(m_domainDim, m_totalGPs), 
            geoWorkingSpaces.vectorView(), dispWorkingSpaces.vectorView(),
            m_geoValuesAndDerss.matrixView(m_geoP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim), 
            m_dispValuesAndDerss.matrixView(m_dispP1, m_totalGPs * (m_numDerivatives + 1) * m_domainDim));
    cudaError_t err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUAssembler constructor during basis values and derivatives evaluation at GPs");
}

void GPUAssembler::allocateGPData()
{ m_GPData.resize(numDoublesPerGP() * m_totalGPs); }

void GPUAssembler::setBasisPatches()
{
    m_multiBasisHost.giveBasis(m_displacementHost, m_targetDim);
    m_displacement = MultiPatchDeviceData(m_displacementHost);
}

void GPUAssembler::getFixedDofsForAssemble(int numIter, DeviceNestedArrayView<double> &fixedDofs_assemble) const
{
    if (numIter != 0)
        fixedDofs_assemble = m_ddof_zero.view();
    else
        fixedDofs_assemble = m_ddof.view();
}
