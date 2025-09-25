#include "Assembler.h"
#include "MultiPatch_d.h"
#include "MultiBasis_d.h"
#include "GaussPoints_d.h"
#include "Patch_d.h"
#include "device_launch_parameters.h"
#include "Boundary_d.h"
#include "DeviceMatrix.h"
#include "DeviceVector.h"
#include "DofMapper_d.h"
#include "DeviceObjectArray.h"
#include "Utility_h.h"
//#include "Utility_d.h"
#include <BoundaryCondition_d.h>

#if 0
__device__
double norm(double* vec, int dim)
{
    double norm = 0.0;
    for (int i = 0; i < dim; ++i)
    {
        norm += vec[i] * vec[i];
    }
    return sqrt(norm);
}

__device__ 
void normalize(double* vec, int dim)
{
    double n = norm(vec, dim);
    for (int i = 0; i < dim; ++i)
    {
        vec[i] /= n;
    }
}

__device__
void timeScalar(double* vec, double scalar, int dim)
{
    for (int i = 0; i < dim; ++i)
    {
        vec[i] *= scalar;
    }
}

__device__
void crossProduct(double* vec1, double* vec2, double* result)
{
    result[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
    result[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
    result[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];
}

__device__
void rowMinor(double* in, int in_rows, int in_cols, int row_delete, double* out)
{
    int idx = 0;
    for (int i = 0; i < in_rows; ++i)
    {
        if (i == row_delete)
            continue;
        for (int j = 0; j < in_cols; ++j)
        {
            out[idx++] = in[i * in_cols + j];
        }
    }
}

__device__
void firstMinor(double* in, int in_rows, int in_cols, int row_delete, int col_delete, double* out)
{
    int idx = 0;
    for (int i = 0; i < in_rows; ++i)
    {
        if (i == row_delete)
            continue;
        for (int j = 0; j < in_cols; ++j)
        {
            if (j == col_delete)
                continue;
            out[idx++] = in[i * in_cols + j];
        }
    }
}

__device__
double determinant_rowMajor(const double* matrix, int dim) 
{
    if (dim == 1) 
        return matrix[0];
    if (dim == 2) 
        return matrix[0] * matrix[3] - matrix[1] * matrix[2];

    double det = 0.0;
    for (int i = 0; i < dim; ++i) 
    {
        double submatrix[9] = { 0 };
        for (int j = 1; j < dim; ++j) 
        {
            for (int k = 0; k < dim; ++k) 
            {
                if (k < i) 
                    submatrix[(j - 1) * (dim - 1) + k] = matrix[j * dim + k];
                else if (k > i) 
                    submatrix[(j - 1) * (dim - 1) + k - 1] = matrix[j * dim + k];
            }
        }
        det += ((i % 2 == 0 ? 1 : -1) * matrix[i] * determinant_rowMajor(submatrix, dim - 1));
    }
    return det;
}
#endif

__device__
//void normal(double* firstGrads, int domainDim, int targetDim, double* result)
void normal(DeviceMatrix<double> firstGrads,  double* result)
{
    assert(firstGrads.rows() + 1 == firstGrads.cols());
    double alt_sgn = 1.0;

    for (int i = 0; i < firstGrads.cols(); ++i) 
    {
        DeviceMatrix<double> minor = firstGrads.rowMinor(i);
        //double* minor = new double[domainDim * domainDim];
        //rowMinor(firstGrads, targetDim, domainDim, i, minor);
        result[i] = alt_sgn * minor.determinant();
        alt_sgn = -alt_sgn;

        //delete[] minor;
    }
}

__device__
//void outerNormal(double* firstGrads, int domainDim, int targetDim, BoxSide_d s, double* result)
DeviceVector<double> outerNormal(DeviceMatrix<double> firstGrads, BoxSide_d s)
{
    int domainDim = firstGrads.cols();
    int targetDim = firstGrads.rows();
    assert(domainDim == targetDim && "The matrix must be square");
    //int orientation = determinant_rowMajor(firstGrads, domainDim) >= 0 ? 1 : -1;
    int orientation = firstGrads.determinant() >= 0 ? 1 : -1;
    const int sgn = sideOrientation(s) * orientation;
    const int dir = s.direction();

    DeviceVector<double> result(domainDim);

    if (targetDim == 1)
    {
        result(0) = sgn;
    }
    else
    {
        int alt_sgn = sgn;
        for (int i = 0; i < domainDim; ++i) 
        {
            //int minorDim = domainDim - 1;
            //double* minor = new double[minorDim  * minorDim];
            //firstMinor(firstGrads, targetDim, domainDim, i, dir, minor);
            //result[i] = alt_sgn * determinant_rowMajor(minor, minorDim);
            result(i) = alt_sgn * firstGrads.firstMinor(i, dir).determinant();
            alt_sgn = -alt_sgn;
            //delete[] minor;
        }
    }

    return result;
}

__device__
void tensorGrid_device(int idx, int* vecs_sizes, int dim, int num_patch, 
                       int& patch, int* pt_coords, int* starts = nullptr)
{
	int point_idx = idx;

	for (int i = 0; i < num_patch; ++i) 
    {
		int patch_offset = i * dim;
		int patch_points = 1;
		for (int j = 0; j < dim; ++j) 
			patch_points *= vecs_sizes[patch_offset + j];
		if (point_idx < patch_points) 
        {
			patch = i;
			break;
		}
		point_idx -= patch_points;
	}

	int patch_offset = patch * dim;
	int* patch_sizes = &vecs_sizes[patch_offset];
    int* patch_starts = starts ? &starts[patch_offset] : nullptr;

	for (int d = 0; d < dim; ++d) 
    {
		pt_coords[d] = point_idx % patch_sizes[d] + (patch_starts ? patch_starts[d] : 0);
		point_idx /= patch_sizes[d];
	}
}

__global__
void assembleDomain(int totalGPs, MultiPatch_d* displacement, MultiPatch_d* patches,
                    DeviceObjectArray<GaussPoints_d>* gaussPoints, DeviceVector<double>* bodyForce,
                    SparseSystem* system, DeviceObjectArray<DeviceVector<double>>* eliminatedDofs)
{
    //MultiBasis_d bases(*displacement);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < totalGPs; idx += blockDim.x * gridDim.x)
    {
        double YM = 1.0;
        double PR = 0.3;
        double lambda = YM * PR / ( ( 1. + PR ) * ( 1. - 2. * PR ) );
        double mu = YM / ( 2. * ( 1. + PR ) );

        int numDerivatives = 1;
        int CPdim = displacement->getCPDim();
        int dim = displacement->getBasisDim();
        int patch(0);
        int point_idx = displacement->threadPatch(idx, patch);
        DeviceVector<double> pt;
        double wt = displacement->gsPoint(point_idx, patch, (*gaussPoints)[patch], pt);

        DeviceMatrix<double> geoActiveCPs = patches->getActiveControlPoints(patch, pt);
        DeviceVector<int> localIndicesDisp = displacement->getActiveIndexes(patch, pt);
        DeviceObjectArray<DeviceVector<double>> geoValues;
        patches->evalAllDers_into(patch, pt, numDerivatives, geoValues);
        DeviceObjectArray<DeviceMatrix<double>> md;
        md.resize(numDerivatives+1);
        md[0] = geoValues[0].transpose() * geoActiveCPs;
        md[1] = geoValues[1].reshape(CPdim, geoActiveCPs.rows()) * geoActiveCPs;
        DeviceMatrix<double> geoJacobian = md[1].transpose();
        double measure = geoJacobian.determinant();
        double weightForce = wt * measure;
        double weightBody = wt * measure;
        
        DeviceMatrix<double> activeCPs = displacement->getActiveControlPoints(patch, pt);
        DeviceObjectArray<DeviceVector<double>> values;
        displacement->evalAllDers_into(patch, pt, numDerivatives, values);
        DeviceObjectArray<DeviceMatrix<double>> mdDisplacement;
        mdDisplacement.resize(numDerivatives+1);
        mdDisplacement[0] = values[0].transpose() * activeCPs;
        mdDisplacement[1] = values[1].reshape(CPdim, activeCPs.rows()) * activeCPs;
        //printf("patch %d, point %d:\n", patch, point_idx);
        //activeCPs.print();
        //printf("values[0]:\n");
        //values[0].print();
        //printf("values[1]:\n");
        //values[1].print();
        //mdDisplacement[1].transpose().print();
        DeviceMatrix<double> physGrad = geoJacobian.inverse().transpose() * values[1].reshape(CPdim, activeCPs.rows());
        DeviceMatrix<double> physDispJac = mdDisplacement[1].transpose() * geoJacobian.inverse();
        DeviceMatrix<double> I = DeviceMatrix<double>::Identity(CPdim);
        DeviceMatrix<double> F = physDispJac + I;
        double J = F.determinant();
        DeviceMatrix<double> RCG = F.transpose() * F;
        DeviceMatrix<double> E = 0.5 * (RCG - I);
        DeviceMatrix<double> RCGinv = RCG.inverse();
        DeviceMatrix<double> S = (lambda*(J*J-1)/2-mu)*RCGinv + mu*I;
        DeviceMatrix<double> C;
        matrixTraceTensor<double>(C,RCGinv,RCGinv);
        C=C*lambda*J*J;
        DeviceMatrix<double> Ctemp;
        symmetricIdentityTensor<double>(Ctemp, RCGinv);
        C=C+(mu-lambda*(J*J-1)/2)*Ctemp;
        int N_D = localIndicesDisp.rows();
        DeviceMatrix<double> B_i, B_j, materialTangentTemp, materialTangent, localMat;
        DeviceVector<double> geometricTangentTemp, localRhs, Svec, localResidual;
        localMat.setZero(dim*N_D,dim*N_D);
        localRhs.setZero(dim*N_D);
        for (int i = 0; i < N_D; i++)
        {
            setB<double>(B_i,F,physGrad.col(i));
            materialTangentTemp = B_i.transpose() * C;
            geometricTangentTemp = S * physGrad.col(i);
            for (int j = 0; j < N_D; j++)
            {
                setB<double>(B_j,F,physGrad.col(j));
                materialTangent = materialTangentTemp * B_j;
                double geometricTangent = geometricTangentTemp.dot(physGrad.col(j));
                for (int d = 0; d < dim; d++)
                    materialTangent(d,d) += geometricTangent;
                for (int di = 0; di < dim; di++)
                    for (int dj = 0; dj < dim; dj++)
                        localMat(di*N_D+i, dj*N_D+j) += weightBody * materialTangent(di,dj);
                //printf("patch %d, point %d, N_%d, N_%d:\n", patch, point_idx, i, j);
                //materialTangent.print();
                //printf("%f\n", geometricTangent);
            }
            voigtStress<double>(Svec, S);
            localResidual = B_i.transpose() * Svec;
            for (int d = 0; d < dim; d++)
                localRhs(d*N_D+i) -= weightBody * localResidual(d);
            //printf("patch %d, point %d, N_%d:\n", patch, point_idx, i);
            //localRhs.print();
        }

        for (int d = 0; d < dim; d++)
            localRhs.middleRows(d*N_D, N_D) += weightForce * bodyForce->operator()(d) * values[0];
            //localRhs.middleRows(d*N_D, N_D)+= weightForce * values[0];
            //weightForce * bodyForce->operator[](d) * values[0];
        //printf("patch %d, point %d:\n", patch, point_idx);
        //localRhs.print();
        //localMat.print();
        //printf("lambda: %f, mu: %f\n", lambda, mu);
        //localIndicesDisp.print();
        //phyGrads.print(); 
        //activeCPs.print();
        //printf("values:\n");
        //values[0].print();
        //printf("derivative:\n");
        //values[1].print();

        
        DeviceObjectArray<DeviceVector<int>> globalIndices(dim);
        DeviceVector<int> blockNumbers(dim);
        //printf("patch %d, point %d:\n", patch, point_idx);
        for (int d = 0; d < dim; d++)
        {
            //printf("Component %d global indices:\n", d);
            globalIndices[d] = system->mapColIndices(localIndicesDisp, patch, d);
            //globalIndices[d].print();
            blockNumbers(d) = d;
        }
        system->pushToRhs(localRhs, globalIndices, blockNumbers);
        system->pushToMatrix(localMat, globalIndices, *eliminatedDofs, blockNumbers, blockNumbers);
    }
    //system->rhs().print();
}

__global__
void constructSolutionKernel(const DeviceVector<double>* solVector,
                             const DeviceObjectArray<DeviceVector<double>>* fixedDoFs,
                             const MultiBasis_d* bases, const SparseSystem* system,
                             MultiPatch_d* result)
{
    int numDofs = result->getTotalNumControlPoints()*result->getCPDim();
    printf("Total num dofs: %d\n", numDofs);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < numDofs; idx += blockDim.x * gridDim.x)
    {
        printf("Thread %d\n", idx);
        //printf("Total num control points: %d\n", result->getTotalNumControlPoints());
        int patch(0);
        int unk(0);
        int point_idx = result->threadPatchAndDof(idx, patch, unk);
        //DeviceObjectArray<int> dofCoords = bases->dofCoords(point_idx, patch);
        //DeviceObjectArray<int> dofCoords = result->dofCoords(point_idx);
        printf("patch %d, point_idx %d, unknown:%d\n", patch, point_idx, unk);
        int index(0);
        //const DofMapper_d& mapper= system->colMapper(unk);
        if (system->m_mappers[unk].is_free(point_idx, patch))
        {
            printf("free dof\n");
            index = system->mapToGlobalColIndex(point_idx, patch, unk);
            printf("global index: %d\n", index);
            result->setCoefficients(patch, point_idx, unk, (*solVector)(index));
        }
        else
        {
            printf("fixed dof\n");
            index = system->m_mappers[unk].bindex(point_idx, patch);
            printf("global index: %d\n", index);
            result->setCoefficients(patch, point_idx, unk, 
                                    (*fixedDoFs)[unk](index));
        }
        printf("Thread %d finished\n", idx);
    }
    //printf("Patch 0 control points:\n");
    //result->patch(0).controlPoints().print();
    //printf("Patch 1 control points:\n");
    //result->patch(1).controlPoints().print();
}

#if 0
__global__
void assemble_kernel(int basisDim, int CPDim, int numPatches, int num_guspts, double *knots,
                     int *numKnots, int *orders, double *controlPoints, 
                     int* numCPs, int *numGpAndEle, double *knots_ref, int *numKnots_ref, 
                     int *orders_ref, double *controlPoints_ref, int* numCPs_ref)
{
    MultiPatch_d multiPatch_d(basisDim, CPDim, numPatches, knots, numKnots, orders, 
                              controlPoints, numCPs, numGpAndEle);

    MultiPatch_d multiPatch_ref(basisDim, CPDim, numPatches, knots_ref, numKnots_ref, orders_ref, 
                                controlPoints_ref, numCPs_ref, nullptr);

    int patch_idx = 0;
    int coords[6] = { 0 };
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < num_guspts; idx += blockDim.x * gridDim.x)
    {
		tensorGrid_device(idx, numGpAndEle, basisDim * 2, numPatches, patch_idx, coords);
        printf("patch_idx: %d\n", patch_idx);

        const int* numGpAndEle_patch = multiPatch_d.getPatchNumGpAndEle(patch_idx);
        const int* numKnots_patch = multiPatch_d.getPatchNumKnots(patch_idx);
        const int* orders_patch = multiPatch_d.getPatchKnotOrders(patch_idx);
        const double* knots_patch = multiPatch_d.getPatchKnots(patch_idx);
        const double* controlPoints_patch = multiPatch_d.getPatchControlPoints(patch_idx);
        int numControlPoints_patch = multiPatch_d.getPatchNumControlPoints(patch_idx);

        Patch_d patch_disp(basisDim, CPDim, numKnots_patch, knots_patch, orders_patch, 
                           controlPoints_patch, numControlPoints_patch);
#if 0
        const int* numGpAndEle_patch_ref = multiPatch_ref.getPatchNumGpAndEle(patch_idx);
        const int* numKnots_patch_ref = multiPatch_ref.getPatchNumKnots(patch_idx);
        const int* orders_patch_ref = multiPatch_ref.getPatchKnotOrders(patch_idx);
        const double* knots_patch_ref = multiPatch_ref.getPatchKnots(patch_idx);
#endif
        double lower[3] = { 0 };
		double upper[3] = { 0 };
        patch_disp.getThreadElementSupport(coords, lower, upper);
#if 0
        printf("lower: ");
        for (int d = 0; d < dim; ++d) 
        {
            printf("%f ", lower[d]);
        }
        printf("\n");
        printf("upper: ");
        for (int d = 0; d < dim; ++d) 
        {
            printf("%f ", upper[d]);
        }
        printf("\n");
        printf("patch_disp dim: %d \n", patch_disp.getDim());
#endif
        
        GaussPoints_d gaussPoints(basisDim, numGpAndEle_patch);
        double gaussPt[3] = { 0 };
        double weight = 1.0;
        gaussPoints.getThreadGaussPoint(lower, upper, coords, gaussPt, weight);
        printf("gaussPt_%d: ", idx);
        for (int d = 0; d < basisDim; ++d) 
        {
            printf("%f ", gaussPt[d]);
        }
        printf("\n");

        int maxDerOder = 1;
        int valuesSize = patch_disp.getNumBasisFunctions();
        int derivativesSize = patch_disp.getNumDerivatives(maxDerOder);
        double* values = new double[valuesSize];
        double* derivatives = new double[derivativesSize];
        patch_disp.getValuesAnddDerivatives(gaussPt, maxDerOder, values, derivatives);
        int numActiveControlPoints_patch = patch_disp.getNumActiveControlPoints();
        double* cp_thread = new double[valuesSize * CPDim];
        patch_disp.getActiveControlPoints(gaussPt, valuesSize, cp_thread);
        double* firstGrads = new double[basisDim * CPDim];
        patch_disp.getGPFirstOrderGradients(derivatives, numActiveControlPoints_patch, 
                                            cp_thread, firstGrads);
        printf("values: ");
        for (int i = 0; i < valuesSize; ++i) 
        {
            printf("%f ", values[i]);
        }
        printf("\n");
        printf("cp_thread: ");
        for(int i = 0; i < numActiveControlPoints_patch * CPDim; ++i) 
        {
            printf("%f ", cp_thread[i]);
        }
        printf("\n");
        printf("derivatives: ");
        for (int i = 0; i < derivativesSize; ++i) 
        {
            printf("%f ", derivatives[i]);
        }
        printf("\n");
        printf("firstGrads: ");
        for (int i = 0; i < basisDim * CPDim; ++i) 
        {
            printf("%f ", firstGrads[i]);
        }
        printf("\n");
        delete[] values, derivatives, cp_thread, firstGrads;
    }
    
}
#endif

__global__
void assembleNeumannBCsKernel(MultiBasis_d* bases, MultiPatch_d* patches,
                              DeviceObjectArray<boundary_condition_d>* neumannBCs)
{
    neumannBCs->operator[](0).values().print();
}

#if 0
__global__
void assembleNeumannBCs_kernel(int basisDim, int CPDim, int numPts, int numPatches, int numBds, 
                               int* PatchIdxs, double *knots, int *numKnots, int *orders, 
                               int *numGpAndEle, double *knots_ref, int *numKnots_ref, 
                               int *orders_ref, double* controlPoints_ref, int* numCPs_ref, 
                               int* sizes, int* starts, int* sides, double* neumannValues 
                               //, int* mapperData)
                               , SparseSystem* sparseSystem)
                               //)
{
    MultiBasis_d multiBasis_d(basisDim, numPatches, knots, numKnots, orders, numGpAndEle);

    MultiPatch_d multiPatch_ref(basisDim, CPDim, numPatches, knots_ref, numKnots_ref, orders_ref, 
                                controlPoints_ref, numCPs_ref, nullptr);

#if 0
    //printf("CPDim: %d\n", CPDim);
    DeviceObjectArray<DofMapper_d> dofMappers(CPDim);
    int patchMapperDataStart = 0;
    for (int i = 0; i < CPDim; ++i) 
    {
        dofMappers[i] = DofMapper_d(&mapperData[patchMapperDataStart]);
        patchMapperDataStart += mapperData[patchMapperDataStart];
    }
    //DeviceObjectArray<int> numCpldDofs = dofMappers[0].getNumCpldDofs();
    //printf("numCpldDofs: ");
    //numCpldDofs.print();
    //printf("curElimId_0: %d\n", dofMappers[0].getCurElimId());
    //printf("curElimId_1: %d\n", dofMappers[1].getCurElimId());
    //printf("curElimId_0: %d\n", (*dofMappers)[0].getCurElimId());
    //printf("curElimId_1: %d\n", (*dofMappers)[1].getCurElimId());
#endif


    int patch_idx = 0;
    int coords[6] = { 0 };
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < numPts; idx += blockDim.x * gridDim.x)
    {
        tensorGrid_device(idx, sizes, basisDim * 2, numBds, patch_idx, coords, starts);
        const int* bdSizes_patch = &sizes[patch_idx * basisDim * 2];
        BoxSide_d side(sides[patch_idx]);
        DeviceVector<double> neumannValues_thisBC(CPDim, &neumannValues[patch_idx * CPDim]);

        patch_idx = PatchIdxs[patch_idx];

#if 0
        printf("patch_idx: %d\n", patch_idx);
        printf("coords: ");
        for (int i = 0; i < basisDim * 2; ++i) 
        {
            printf("%d ", coords[i]);
        }
        printf("\n");
#endif

        const int* numKnots_patch = multiBasis_d.getPatchNumKnots(patch_idx);
        const int* orders_patch = multiBasis_d.getPatchKnotOrders(patch_idx);
        const double* knots_patch = multiBasis_d.getPatchKnots(patch_idx);
        const int* numGpAndEle_patch = multiBasis_d.getPatchNumGpAndEle(patch_idx);
        TensorBsplineBasis_d basis(basisDim, numKnots_patch, knots_patch, orders_patch);

        double lower[3] = { 0 };
		double upper[3] = { 0 };
        basis.getThreadElementSupport(coords, lower, upper);
#if 0        
        printf("lower: ");
        for (int d = 0; d < basisDim; ++d) 
        {
            printf("%f ", lower[d]);
        }
        printf("\n");
        printf("upper: ");
        for (int d = 0; d < basisDim; ++d) 
        {
            printf("%f ", upper[d]);
        }
        printf("\n");
#endif
        GaussPoints_d gaussPoints(basisDim, bdSizes_patch);
#if 0
        int numGaussPoints = 0;
        for (int i = 0; i < basisDim; ++i) 
        {
            numGaussPoints += gaussPoints.getNumGaussPoints()[i];
        }
        DeviceVector<double> gspts(numGaussPoints,gaussPoints.getGaussWeights());
        printf("Weights: ");
        gspts.transpose().print();
#endif
#if 0        
        const double* gaussPt = gaussPoints.getGaussPointVecs();
        int numGaussPoints = 1;
        for (int i = 0; i < dim; ++i) 
        {
            numGaussPoints *= bdSizes_patch[i];
        }
        for (int i = 0; i < numGaussPoints; ++i) 
        {
            printf("gaussPt_%d: ", i);
            for (int d = 0; d < dim; ++d) 
            {
                printf("%f ", gaussPt[i * dim + d]);
            }
            printf("\n");
        }
#endif
        double gaussPt[3] = { 0 };
        double GPWeight = 1.0;
        gaussPoints.getThreadGaussPoint(lower, upper, coords, gaussPt, GPWeight);
#if 0
        printf("gaussPt_%d: ", idx);
        for (int d = 0; d < dim; ++d) 
        {
            printf("%f ", gaussPt[d]);
        }
        printf("\n");
#endif
        int maxDerOder = 0;
        int valuesSize = basis.getNumValues();
        //int derivativesSize = TensorBsplineBasis_d::getNumDerivatives(dim, maxDerOder, valuesSize);
        double* values = new double[valuesSize];
        //double* derivatives = new double[derivativesSize];
        basis.getValuesAnddDerivatives(gaussPt, maxDerOder, values, nullptr);
#if 0
        printf("values: ");
        for (int i = 0; i < valuesSize; ++i) 
        {
            printf("%f ", values[i]);
        }
        printf("\n");
#endif
        int numAvtive = basis.getNumActiveControlPoints();
        int* activeIndexes = new int[numAvtive];
        basis.getActiveIndexes(gaussPt, activeIndexes, numAvtive);
#if 0
        printf("activeIndexes: ");
        for (int i = 0; i < numAvtive; ++i) 
        {
            printf("%d ", activeIndexes[i]);
        }
        printf("\n");
#endif
        const int* numKnots_patch_ref = multiPatch_ref.getPatchNumKnots(patch_idx);
        const int* orders_patch_ref = multiPatch_ref.getPatchKnotOrders(patch_idx);
        const double* knots_patch_ref = multiPatch_ref.getPatchKnots(patch_idx);
        const double* controlPoints_patch_ref = multiPatch_ref.getPatchControlPoints(patch_idx);
        int numControlPoints_patch_ref = multiPatch_ref.getPatchNumControlPoints(patch_idx);

        Patch_d patch_ref(basisDim, CPDim, numKnots_patch_ref, knots_patch_ref, orders_patch_ref, 
                          controlPoints_patch_ref, numControlPoints_patch_ref);

        int maxDerOder_ref = 1;
        double* values_ref = new double[valuesSize]; 
        int derivativesSize = TensorBsplineBasis_d::getNumDerivatives(basisDim, maxDerOder_ref, valuesSize);
        double* derivatives_ref = new double[derivativesSize];
        patch_ref.getValuesAnddDerivatives(gaussPt, maxDerOder_ref, values_ref, derivatives_ref);
#if 0
        printf("values_ref: ");
        for (int i = 0; i < valuesSize; ++i) 
        {
            printf("%f ", values_ref[i]);
        }
        printf("\n");
        printf("derivatives_ref: ");
        for (int i = 0; i < derivativesSize; ++i) 
        {
            printf("%f ", derivatives_ref[i]);
        }
        printf("\n");
#endif
        int numActiveControlPoints_patch = patch_ref.getNumActiveControlPoints();
        //printf("numActiveControlPoints_patch: %d\n", numActiveControlPoints_patch);
        double* cp_thread = new double[valuesSize * CPDim];
        patch_ref.getActiveControlPoints(gaussPt, valuesSize, cp_thread);
#if 0
        printf("cp_thread: ");
        for(int i = 0; i < numActiveControlPoints_patch * CPDim; ++i) 
        {
            printf("%f ", cp_thread[i]);
        }
        printf("\n");
#endif
        double* firstGrads = new double[basisDim * CPDim];
        patch_ref.getGPFirstOrderGradients(derivatives_ref, numActiveControlPoints_patch, 
                                           cp_thread, firstGrads);
#if 0
        printf("firstGrads: ");
        for (int i = 0; i < basisDim * CPDim; ++i) 
        {
            printf("%f ", firstGrads[i]);
        }
        printf("\n");
#endif
        DeviceVector<double> localRhs(numAvtive * CPDim);
        //double* unormal = new double[basisDim];

        DeviceMatrix<double> firstGradsMat(basisDim, CPDim, firstGrads);
        int domainDim = firstGradsMat.cols();
        int targetDim = firstGradsMat.rows();
        int orientation = firstGradsMat.determinant() >= 0 ? 1 : -1;
        const int sgn = sideOrientation(side) * orientation;
        const int dir = side.direction();
        DeviceMatrix<double> minor11(basisDim, CPDim);
        //minor = firstGradsMat;
        //printf("%d\n", minor.cols());
#if 0        
        DeviceVector unormal = outerNormal(firstGradsMat, side);

        printf("GPWeight: %f\n", GPWeight);
        double weight = GPWeight * unormal.norm();

        printf("weight: %f\n", weight);

        DeviceVector<double> values_DV(valuesSize, values);
        printf("values_DV:\n");
        values_DV.print();

        printf("neumannValues_thisBC:\n");
        neumannValues_thisBC.print();

        for (int d = 0; d < CPDim; ++d) 
        {
            localRhs.middleRows(d * numAvtive, numAvtive) = 
                weight * neumannValues_thisBC(d) * DeviceVector<double>(valuesSize, values);
        }

        printf("localRhs:\n");
        localRhs.print();

        DeviceVector<int> localIndicesDisp(valuesSize, activeIndexes);
        DeviceObjectArray<DeviceVector<int>> globalIndices(CPDim);
        DeviceVector<int> blockNumbers(CPDim);
        for (int d = 0; d < CPDim; ++d) 
        {
            //globalIndices[d] = (*dofMappers)[d].getGlobalIndices(localIndicesDisp, patch_idx);
            globalIndices[d] = sparseSystem->mapColIndices(localIndicesDisp, patch_idx, d);
            //globalIndices[d] = dofMappers[d].getGlobalIndices(localIndicesDisp, patch_idx);
            printf("globalIndices[%d]:\n", d);
            globalIndices[d].print();
            blockNumbers(d) = d;
        }

        delete[] values, activeIndexes, values_ref, derivatives_ref, 
                 localRhs, cp_thread, firstGrads, unormal;
#endif
    }
}
#endif

#if 0
__global__
void testKernel(DeviceObjectArray<DeviceObjectArray<int>>* mat)
{
    printf("mat[0][0]: ");
    printf("%d ", (*mat)[0][0]);
    printf("mat[1][0]: ");
    printf("%d ", (*mat)[1][0]);
}
#endif

#if 0
__global__
void dofMapperTestKernel( DeviceObjectArray<DofMapper_d>* dofMappers, int numMappers)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numMappers; i += blockDim.x * gridDim.x)
    {
        printf("DofMapper %d: curElimId = %d, freeSize = %d\n", 
               i, (*dofMappers)[i].getCurElimId(), (*dofMappers)[i].freeSize());
        (*dofMappers)[i].getDofs().print();
    }
}
#endif

__global__
void matrixTexsKernel(int rows, int column, double* matrix)
{
    DeviceMatrix<double> mat(rows, column, matrix);
    //mat(0,0) = 2.0;
    //printf("%f", mat(0,0));
    DeviceMatrix<double> minor = mat.firstMinor(0, 0);
    minor.print();
    //printf("%f", minor(0,0));
    //mat.firstMinor( 0, 0).print("First minor of the matrix:\n");
}

Assembler::Assembler(const MultiPatch& multiPatch, const MultiBasis& multiBasis, 
                     const BoundaryConditions& bc, const Eigen::VectorXd& bodyForce)
: m_multiPatch(multiPatch), m_boundaryConditions(bc), m_multiBasis(multiBasis), m_bodyForce(bodyForce)
{
    int targetDim = m_multiPatch.getCPDim();
    int domainDim = m_multiPatch.getBasisDim();
    std::vector<DofMapper> dofMappers_stdVec(targetDim);
    //m_dofMappers.clear();
    multiBasis.getMappers(true, m_boundaryConditions, dofMappers_stdVec, true);

#if 0
    //DofMapper_d* h_dofMappers = new DofMapper_d[targetDim];
    //for (int i = 0; i < targetDim; ++i)
        //h_dofMappers[i] = DofMapper_d(dofMappers_stdVec[i]);
    DeviceObjectArray<DofMapper_d> dofMappers(targetDim);
    for (int i = 0; i < targetDim; ++i)
    {
    #if 0
        // Copy each DofMapper_d to the DeviceObjectArray
        DofMapper_d dofMapper_d(dofMappers_stdVec[i]); // Create a temporary DofMapper_d
        //DofMapper_d* d_dofMapper_d = nullptr;
        //cudaMalloc((void**)&d_dofMapper_d, sizeof(DofMapper_d));
        //cudaMemcpy(d_dofMapper_d, &dofMapper_d, sizeof(DofMapper_d), cudaMemcpyHostToDevice);
        dofMappers.at(i) = dofMapper_d;
        //cudaFree(d_dofMapper_d); // Free the temporary device memory
    #else
        dofMappers.at(i) = DofMapper_d(dofMappers_stdVec[i]);
    #endif
    }

    #if 0
    DeviceObjectArray<DofMapper_d>* d_dofMappers = nullptr;
    cudaMalloc((void**)&d_dofMappers, sizeof(DeviceObjectArray<DofMapper_d>));
    cudaMemcpy(d_dofMappers, &dofMappers, sizeof(DeviceObjectArray<DofMapper_d>), 
               cudaMemcpyHostToDevice);
    dofMapperTestKernel<<<1, 1>>>(d_dofMappers, targetDim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after dofMapperTestKernel launch: " 
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (dofMapperTestKernel): " 
                  << cudaGetErrorString(err) << std::endl;
    cudaFree(d_dofMappers); // Free the temporary device memory after testing
    //dofMappers.parallelDataSetting(h_dofMappers, targetDim);
    //delete[] h_dofMappers;
    #endif
#endif

    //m_sparseSystem = SparseSystem(dofMappers, Eigen::VectorXi::Ones(targetDim));
    m_sparseSystem = SparseSystem(dofMappers_stdVec, Eigen::VectorXi::Ones(targetDim));

    m_ddof.resize(targetDim);
    for (int unk = 0; unk < targetDim; ++unk)
    {
        computeDirichletDofs(unk, dofMappers_stdVec);
    }

#if 0
    m_dims = thrust::device_vector<int>(targetDim, 1);
    m_row.resize(targetDim);
    m_col.resize(targetDim);
    m_rstr.resize(targetDim);
    m_cstr.resize(targetDim);

    int k = 0;
    for (int i = 0; i < targetDim; ++i)
        for (int j = 0; j < m_dims[i]; ++j)
        {
            m_row[k]=i;
            ++k;
        }

    m_col = m_row;
#endif
}

int Assembler::getNumPatches() const
{
    return m_multiPatch.getNumPatches();
}

int Assembler::numDofs() const
{
    return m_sparseSystem.numDofs();
}

__global__
void gaussPointsTest(GaussPoints_d* gspts)
{
    gspts->gaussPointsOnDir(0).print();
    gspts->gaussPointsOnDir(1).print();
}

void Assembler::assemble(const DeviceVector<double>& solVector)
{
    m_sparseSystem.matrix().setZero();
    m_sparseSystem.rhs().setZero();
#if 1
    MultiPatch displacement;
    int geoDim = m_multiPatch.getCPDim();
    for (int i = 0; i < m_multiPatch.getNumPatches(); ++i) 
    {
        Patch patch(m_multiBasis.basis(i), geoDim);
        displacement.addPatch(patch);
    }
    MultiPatch_d displacement_d(displacement);
    MultiPatch_d* d_displacement = nullptr;
    cudaMalloc((void**)&d_displacement, sizeof(MultiPatch_d));
    cudaMemcpy(d_displacement, &displacement_d, sizeof(MultiPatch_d), 
               cudaMemcpyHostToDevice);

    DeviceVector<double>* d_solVector = nullptr;
    cudaMalloc((void**)&d_solVector, sizeof(DeviceVector<double>));
    cudaMemcpy(d_solVector, &solVector, sizeof(DeviceVector<double>), 
               cudaMemcpyHostToDevice);
#endif

    DeviceObjectArray<DeviceVector<double>> fixedDoFs_d(m_ddof.size());
    for (int i = 0; i < m_ddof.size(); ++i)
    {
        DeviceVector<double> fixedDoF_d(m_ddof[i].size(), m_ddof[i].data());
        fixedDoFs_d.at(i) = fixedDoF_d;
    }
    DeviceObjectArray<DeviceVector<double>>* d_fixedDoFs = nullptr;
    cudaMalloc((void**)&d_fixedDoFs, sizeof(DeviceObjectArray<DeviceVector<double>>));
    cudaMemcpy(d_fixedDoFs, &fixedDoFs_d, sizeof(DeviceObjectArray<DeviceVector<double>>), 
               cudaMemcpyHostToDevice);
#if 1
    MultiPatch_d patches(m_multiPatch);
    MultiPatch_d* d_patches = nullptr;
    cudaMalloc((void**)&d_patches, sizeof(MultiPatch_d));
    cudaMemcpy(d_patches, &patches, sizeof(MultiPatch_d), cudaMemcpyHostToDevice);

    SparseSystem* d_sparseSystem = nullptr;
    cudaMalloc((void**)&d_sparseSystem, sizeof(SparseSystem));
    cudaMemcpy(d_sparseSystem, &m_sparseSystem, sizeof(SparseSystem), 
               cudaMemcpyHostToDevice);

    MultiBasis_d bases(m_multiBasis);
    MultiBasis_d* d_bases = nullptr;
    cudaMalloc((void**)&d_bases, sizeof(MultiBasis_d));
    cudaMemcpy(d_bases, &bases, sizeof(MultiBasis_d), cudaMemcpyHostToDevice);

    DeviceVector<double> bodyForce_d(m_bodyForce);
    DeviceVector<double>* d_bodyForce = nullptr;
    cudaMalloc((void**)&d_bodyForce, sizeof(DeviceVector<double>));
    cudaMemcpy(d_bodyForce, &bodyForce_d, sizeof(DeviceVector<double>), cudaMemcpyHostToDevice);

#endif

#if 1
    constructSolutionKernel<<<1, 1>>>(d_solVector, d_fixedDoFs, d_bases, 
                                      d_sparseSystem, d_displacement);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after kernel constructSolutionKernel launch: " 
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (constructSolutionKernel): " 
                  << cudaGetErrorString(err) << std::endl;
#endif

#if 1
    int dim = m_multiPatch.getBasisDim();
    int numPatches = m_multiPatch.getNumPatches();
    DeviceObjectArray<GaussPoints_d> gaussPoints(numPatches);
    for (int i = 0; i < numPatches; ++i) 
    {
        std::vector<int> numGPs(dim, 0);
        for (int d = 0; d < dim; ++d)
            numGPs[d] = m_multiBasis.basis(i).getNumGaussPoints(d);
    #if 0
        GaussPoints_d gaussPoints_d(dim, numGPs);
        GaussPoints_d* d_gaussPoints_d = nullptr;
        cudaMalloc((void**)&d_gaussPoints_d, sizeof(GaussPoints_d));
        cudaMemcpy(d_gaussPoints_d, &gaussPoints_d, sizeof(GaussPoints_d), 
                   cudaMemcpyHostToDevice);
        gaussPointsTest<<<1, 1>>>(d_gaussPoints_d);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "Error after gaussPointsTest launch: " 
                      << cudaGetErrorString(err) << std::endl;
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            std::cerr << "CUDA error during device synchronization (gaussPointsTest): " 
                      << cudaGetErrorString(err) << std::endl;
        cudaFree(d_gaussPoints_d); 
        gaussPoints.at(i) = gaussPoints_d;
    #else
        gaussPoints.at(i) = GaussPoints_d(dim, numGPs);
    #endif
    }
    DeviceObjectArray<GaussPoints_d>* d_gaussPoints = nullptr;
    cudaMalloc((void**)&d_gaussPoints, sizeof(DeviceObjectArray<GaussPoints_d>));
    cudaMemcpy(d_gaussPoints, &gaussPoints, sizeof(DeviceObjectArray<GaussPoints_d>), 
               cudaMemcpyHostToDevice);
#endif

    

    //size_t curr, limit;
    //cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    //printf("StackSize limit = %zu bytes\n", limit);
    cudaDeviceSetLimit(cudaLimitStackSize, 4*1024);

#if 1
    int totalGPs = m_multiBasis.totalNumGPs();
    assembleDomain<<<1, 1>>>(totalGPs, d_displacement, d_patches, d_gaussPoints, d_bodyForce, d_sparseSystem, d_fixedDoFs);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after kernel assembleDomain launch: " 
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (assembleDomain): " 
                  << cudaGetErrorString(err) << std::endl;
#endif
    
    cudaFree(d_patches);
    cudaFree(d_bases);
    cudaFree(d_displacement);
    cudaFree(d_gaussPoints);
    cudaFree(d_fixedDoFs);
    cudaFree(d_solVector);
    cudaFree(d_sparseSystem);
    cudaFree(d_bodyForce);

#if 0
    assemble_kernel<<<1, 1>>>(m_multiPatchData.getBasisDim(),
                              m_multiPatchData.getCPDim(),
                              m_multiPatchData.getNumPatches(),
                              m_multiPatch.getTotalNumGaussPoints(),
                              m_multiPatchData.getKnots_ptr(),
                              m_multiPatchData.getNumKnots_ptr(),
                              m_multiPatchData.getOrders_ptr(),
                              m_multiPatchData.getControlPoints_ptr(),
                              m_multiPatchData.getNumControlPoints_ptr(),
                              m_multiPatchData.getNumGpAndEle_ptr(),
                              m_multiPatchData.getKnots_ref_ptr(),
                              m_multiPatchData.getNumKnots_ref_ptr(),
                              m_multiPatchData.getOrders_ref_ptr(),
                              m_multiPatchData.getControlPoints_ptr(),
                              m_multiPatchData.getNumControlPoints_ptr());
    
	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
		std::cerr << "Error after kernel assembleKernel launch: " 
        << cudaGetErrorString(err) << std::endl;
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		std::cerr << "CUDA error during device synchronization (assembleKernel): " 
        << cudaGetErrorString(err) << std::endl;
#endif
}

void Assembler::assembleNeumannBCs()
{
    DeviceObjectArray<boundary_condition_d> 
    neumannBCs(m_boundaryConditions.neumannSides().size());
    int i = 0;
    for (BoundaryConditions::bcContainer::const_iterator 
        it = m_boundaryConditions.neumannBegin(); 
        it != m_boundaryConditions.neumannEnd(); ++it)
    {
        neumannBCs.at(i) = boundary_condition_d(*it);
        ++i;
    }
    DeviceObjectArray<boundary_condition_d>* d_neumannBCs = nullptr;
    cudaMalloc((void**)&d_neumannBCs, sizeof(DeviceObjectArray<boundary_condition_d>));
    cudaMemcpy(d_neumannBCs, &neumannBCs, sizeof(DeviceObjectArray<boundary_condition_d>), 
               cudaMemcpyHostToDevice);
    MultiPatch_d patches(m_multiPatch);
    MultiPatch_d* d_patches = nullptr;
    cudaMalloc((void**)&d_patches, sizeof(MultiPatch_d));
    cudaMemcpy(d_patches, &patches, sizeof(MultiPatch_d), cudaMemcpyHostToDevice);
    MultiBasis_d bases(m_multiBasis);
    MultiBasis_d* d_bases = nullptr;
    cudaMalloc((void**)&d_bases, sizeof(MultiBasis_d));
    cudaMemcpy(d_bases, &bases, sizeof(MultiBasis_d), cudaMemcpyHostToDevice);

    assembleNeumannBCsKernel<<<1, 1>>>(d_bases, d_patches, d_neumannBCs);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after kernel assembleNeumannBCsKernel launch: " 
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (assembleNeumannBCsKernel): " 
                  << cudaGetErrorString(err) << std::endl;
    cudaFree(d_neumannBCs);
    cudaFree(d_patches);
    cudaFree(d_bases);

#if 0
    int totalMapperDataSize = 0;
    for (int i = 0; i < m_dofMappers.size(); ++i)
    {
        totalMapperDataSize += m_dofMappers[i].getDataSize();
    }
    thrust::device_vector<int> mapperData;
    mapperData.reserve(totalMapperDataSize);
    for (int i = 0; i < m_dofMappers.size(); ++i)
    {
        thrust::device_vector<int> tmp = m_dofMappers[i].getDofMapperDataVec();
        mapperData.insert(mapperData.end(), tmp.begin(), tmp.end());
    }

    int cpDim = m_multiPatch.getCPDim();
    DofMapper_d* h_dofMappers = new DofMapper_d[cpDim];
    for (int i = 0; i < cpDim; ++i)
        h_dofMappers[i] = DofMapper_d(m_dofMappers[i]);
    DeviceObjectArray<DofMapper_d> dofMappers(cpDim);
    dofMappers.parallelDataSetting(h_dofMappers, cpDim);
    delete[] h_dofMappers;
    DeviceObjectArray<DofMapper_d>* d_dofMappers = nullptr;
    cudaMalloc((void**)&d_dofMappers, sizeof(DeviceObjectArray<DofMapper_d>));
    cudaMemcpy(d_dofMappers, &dofMappers, sizeof(DeviceObjectArray<DofMapper_d>), cudaMemcpyHostToDevice);
#endif
#if 0
    SparseSystem* d_sparseSystem = nullptr;
    cudaMalloc((void**)&d_sparseSystem, sizeof(SparseSystem));
    cudaMemcpy(d_sparseSystem, &m_sparseSystem, sizeof(SparseSystem), cudaMemcpyHostToDevice);

    thrust::device_vector<int> sizes;
    thrust::device_vector<int> starts;
    thrust::device_vector<int> PatchIdxs;
    thrust::device_vector<int> sides;
    thrust::device_vector<double> neumannValues;
    sizes.reserve(m_boundaryConditions.neumannSides().size() * m_multiPatch.getBasisDim() * 2);
    starts.reserve(m_boundaryConditions.neumannSides().size() * m_multiPatch.getBasisDim() * 2);
    PatchIdxs.reserve(m_boundaryConditions.neumannSides().size());
    sides.reserve(m_boundaryConditions.neumannSides().size());
    neumannValues.reserve(m_boundaryConditions.neumannSides().size() * m_multiPatch.getCPDim());

    int totalNumGaussPts = getBoundaryData_Neumann(sizes, starts, PatchIdxs, sides, neumannValues);

    //printThrustVector(neumannValues, "Neumann Values");

    thrust::device_vector<double> testMat(9, 1.0);
    matrixTexsKernel<<<1, 1>>>(3, 3, thrust::raw_pointer_cast(testMat.data()));
    if (cudaGetLastError() != cudaSuccess)
        std::cerr << "Error after matrixTexsKernel launch: " 
                  << cudaGetErrorString(cudaGetLastError()) << std::endl;
    if (cudaDeviceSynchronize() != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (matrixTexsKernel): " 
                  << cudaGetErrorString(cudaGetLastError()) << std::endl;
#if 0
    assembleNeumannBCs_kernel<<<1, 1>>>(m_multiPatchData.getBasisDim(),
                                        m_multiPatchData.getCPDim(),
                                        totalNumGaussPts,
                                        m_multiPatchData.getNumPatches(),
                                        m_boundaryConditions.neumannSides().size(),
                                        thrust::raw_pointer_cast(PatchIdxs.data()),
                                        m_multiPatchData.getKnots_ptr(),
                                        m_multiPatchData.getNumKnots_ptr(),
                                        m_multiPatchData.getOrders_ptr(),
                                        m_multiPatchData.getNumGpAndEle_ptr(),
                                        m_multiPatchData.getKnots_ref_ptr(),
                                        m_multiPatchData.getNumKnots_ref_ptr(),
                                        m_multiPatchData.getOrders_ref_ptr(),
                                        m_multiPatchData.getControlPoints_ptr(),
                                        m_multiPatchData.getNumControlPoints_ptr(),
                                        thrust::raw_pointer_cast(sizes.data()),
                                        thrust::raw_pointer_cast(starts.data()),
                                        thrust::raw_pointer_cast(sides.data()),
                                        thrust::raw_pointer_cast(neumannValues.data())
                                        //thrust::raw_pointer_cast(mapperData.data()));
                                        //);
                                        , d_sparseSystem);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after kernel assembleNeumannBCs launch: " 
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (assembleNeumannBCs): " 
                  << cudaGetErrorString(err) << std::endl;
#endif
    cudaFree(d_sparseSystem);
#if 0
    std::cout << "sizes: ";
    for (int i = 0; i < sizes.size(); ++i)
    {
        std::cout << sizes[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "starts: ";
    for (int i = 0; i < starts.size(); ++i)
    {
        std::cout << starts[i] << " ";
    }
    std::cout << std::endl;

    std::vector<int> test(3, 3);
    std::vector<int> test2(3, 1);
    //DeviceObjectArray<int> testArr1(2);
    //testArr1.at(0) = 4;
    //testArr1.at(1) = 8;
    //DeviceObjectArray<int> testArr2(2);
    //testArr2.at(0) = 1;
    //testArr2.at(1) = 3;
    DeviceObjectArray<int>* h_testArr = new DeviceObjectArray<int>[2];
    h_testArr[0] = DeviceObjectArray<int>(3, test.data());
    h_testArr[1] = DeviceObjectArray<int>(3, test2.data());
    DeviceObjectArray<DeviceObjectArray<int>> testArr(2);
    testArr.parallelDataSetting(h_testArr, 2);
    delete[] h_testArr;
    DeviceObjectArray<DeviceObjectArray<int>>* d_testArr = nullptr;
    cudaMalloc((void**)&d_testArr, sizeof(DeviceObjectArray<DeviceObjectArray<int>>));
    cudaMemcpy(d_testArr, &testArr, sizeof(DeviceObjectArray<DeviceObjectArray<int>>), cudaMemcpyHostToDevice);

    testKernel<<<1, 1>>>(d_testArr);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after kernel testKernel launch: " 
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during device synchronization (testKernel): " 
                  << cudaGetErrorString(err) << std::endl;


    cudaFree(d_testArr);
    d_testArr = nullptr;
#endif
#endif
}

void Assembler::computeDirichletDofs(int unk_, const std::vector<DofMapper> &mappers)
{
    //std::cout << m_sparseSystem.colMapper(unk_).getCurElimId();
    //DofMapper_d dofMapper = m_sparseSystem.colMapper_h(unk_); 
    DofMapper dofMapper = mappers[unk_]; 
    //dofMapper.getNumElimDofs().print();
    m_ddof[unk_].resize(dofMapper.boundarySize());
    const TensorBsplineBasis &basis = m_multiBasis.basis(unk_);

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
            m_ddof[unk_][ii] = it->value(unk_);
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
        m_ddof[unk_][ii] = it->value(unk_);
    }
}

#if 0
void Assembler::constructSolution(const DeviceVector<double> &solVector, 
                                  MultiPatch_d &displacement) const
{
    
}
#endif

int Assembler::getBoundaryData_Neumann(thrust::device_vector<int> &sizes, 
                                       thrust::device_vector<int> &starts,
                                       thrust::device_vector<int> &PatchIdxs,
                                       thrust::device_vector<int> &sides,
                                       thrust::device_vector<double> &values) const
{
    int totalNumGaussPoints = 0;
    for (BoundaryConditions::bcContainer::const_iterator 
         it = m_boundaryConditions.neumannBegin(); 
         it != m_boundaryConditions.neumannEnd(); ++it)
    {

        int dim = m_multiPatch.getBasisDim();
        BoxSide side = it -> side();
        sides.push_back(side);
        int fixDir = side.direction();
        bool param = side.parameter();
        int patch = it -> patchIndex();
        PatchIdxs.push_back(patch);
        std::vector<int> domain_sizes=m_multiPatch.getNumGpAndEle(patch);

        thrust::device_vector<int> patch_sizes(dim * 2);
        patch_sizes[fixDir] = 1;
        patch_sizes[fixDir + dim] = 1;
        thrust::device_vector<int> patch_starts(dim * 2);
        patch_starts[fixDir] = 0;
        patch_starts[fixDir + dim] = param ? domain_sizes[fixDir + dim] : 0;

        for (int d = 0; d < dim; ++d)
        {
            if (d != fixDir)
            {
                patch_sizes[d] = domain_sizes[d];
                patch_sizes[d + dim] = domain_sizes[d + dim];
                patch_starts[d] = 0;
                patch_starts[d + dim] = 0;
            }
        }

        sizes.insert(sizes.end(), patch_sizes.begin(), patch_sizes.end());
        starts.insert(starts.end(), patch_starts.begin(), patch_starts.end());

        thrust::device_vector<double> neumannValues = it->values();
        values.insert(values.end(), neumannValues.begin(), neumannValues.end());

        int totalNumGaussPoints_patch = 1;
        for (int i = 0; i < dim * 2; ++i)
        {
            totalNumGaussPoints_patch *= patch_sizes[i];
        }
        totalNumGaussPoints += totalNumGaussPoints_patch;
    }

    return totalNumGaussPoints;
}
