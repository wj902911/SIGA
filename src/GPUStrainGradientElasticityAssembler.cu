#include "GPUStrainGradientElasticityAssembler.h"

#include <Utility_d.h>

#include <stdexcept>

// Implementation notes
// --------------------
// This assembler mirrors gsVisitorStrainGradientElasticity from gismo, but keeps
// the contractions in small CUDA device helpers instead of building Eigen
// matrices at each quadrature point. All fixed-size scratch arrays are sized for
// the existing SIGA limit of dim <= 3.
//
// Indexing convention used below:
//   F/P1 rows:        a * dim + A
//   GradF/P2 rows:    a * dim * dim + A * dim + B
//   Hessian entries:  A * dim + B
//
// Lowercase indices are spatial/displacement components; uppercase indices are
// physical coordinate directions. This matches the notation in the gismo visitor.
namespace
{

__host__ __device__
int strainGradientDoublesPerGP(int dim)
{
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    return dim2         // geoJacobianInv
         + 2            // weightForce, weightBody
         + dim3         // geometry Hessians
         + dim2         // F
         + dim2         // P1
         + dim3         // P2
         + dim2 * dim2  // dP1/dF
         + dim2 * dim3  // dP1/dGradF
         + dim3 * dim3; // dP2/dGradF
}

__host__ __device__
void strainGradientGPOffsets(int dim, int& geoInvOffset,
                              int& weightForceOffset, int& weightBodyOffset,
                              int& geoHessOffset, int& FOffset,
                              int& P1Offset, int& P2Offset, int& dP1dFOffset,
                              int& dP1dGradFOffset, int& dP2dGradFOffset)
{
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    int offset = 0;
    geoInvOffset = offset; offset += dim2;
    weightForceOffset = offset; offset += 1;
    weightBodyOffset = offset; offset += 1;
    geoHessOffset = offset; offset += dim3;
    FOffset = offset; offset += dim2;
    P1Offset = offset; offset += dim2;
    P2Offset = offset; offset += dim3;
    dP1dFOffset = offset; offset += dim2 * dim2;
    dP1dGradFOffset = offset; offset += dim2 * dim3;
    dP2dGradFOffset = offset;
}

// Evaluates one tensor-product basis function derivative in parametric
// coordinates.
//
// derivativeDir1 and derivativeDir2 are not limited to 2D. They are the two
// slots needed to describe derivatives up to order two:
//   value:              derivativeDir1 = -1, derivativeDir2 = -1
//   first derivative:   derivativeDir1 = 0, 1, or 2
//   second derivative:  derivativeDir1 = 0, 1, or 2 and
//                       derivativeDir2 = 0, 1, or 2
//
// For example, in 3D:
//   (0, 0) gives d2N / dx0 dx0,
//   (0, 2) gives d2N / dx0 dx2,
//   (2, 2) gives d2N / dx2 dx2.
__device__
double tensorBasisPartial(int r, int P1, int dim, int numDerivatives,
                          DeviceMatrixView<double> valuesAndDers,
                          int derivativeDir1, int derivativeDir2 = -1)
{
    int tensorCoordData[3] = {0};
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
        value *= valuesAndDers(tensorCoord[d], (numDerivatives + 1) * d + order);
    }
    return value;
}

// Tensor-product basis value N_i.
__device__
double tensorBasisValueSG(int r, int P1, int dim, int numDerivatives,
                          DeviceMatrixView<double> valuesAndDers)
{
    return tensorBasisPartial(r, P1, dim, numDerivatives, valuesAndDers, -1);
}

// Parametric gradient dN_i / dxi.
__device__
void tensorBasisGradientSG(int r, int P1, int dim, int numDerivatives,
                           DeviceMatrixView<double> valuesAndDers,
                           double* grad)
{
    for (int a = 0; a < dim; ++a)
        grad[a] = tensorBasisPartial(r, P1, dim, numDerivatives, valuesAndDers, a);
}

// Parametric Hessian d2N_i / dxi_a dxi_b.
__device__
void tensorBasisHessianParamSG(int r, int P1, int dim, int numDerivatives,
                               DeviceMatrixView<double> valuesAndDers,
                               double* hessian)
{
    for (int a = 0; a < dim; ++a)
        for (int b = 0; b < dim; ++b)
            hessian[a * dim + b] =
                tensorBasisPartial(r, P1, dim, numDerivatives, valuesAndDers, a, b);
}

// Parametric Hessian of a mapped field component, e.g. geometry x_component or
// displacement u_component.
__device__
void patchParamHessian(PatchDeviceView patch, DeviceVectorView<double> pt,
                       DeviceMatrixView<double> valuesAndDers, int numDerivatives,
                       int component, double* hessian)
{
    const int dim = patch.domainDim();
    const int P1 = patch.basis().knotsOrder(0) + 1;
    const int numActive = patch.basis().numActiveControlPoints();

    for (int a = 0; a < dim * dim; ++a)
        hessian[a] = 0.0;

    for (int r = 0; r < numActive; ++r)
    {
        double basisHessian[9] = {0.0};
        tensorBasisHessianParamSG(r, P1, dim, numDerivatives, valuesAndDers, basisHessian);
        const double cp = patch.activeControlPointComponent(pt, r, component);
        for (int a = 0; a < dim * dim; ++a)
            hessian[a] += cp * basisHessian[a];
    }
}

// Converts a basis-function Hessian from parametric to physical coordinates.
// The correction term subtracts the geometry curvature contribution, matching
// gismo's transformDeriv2Hgrad behavior.
__device__
void physicalBasisHessian(int r, int P1, int dim, int numDerivatives,
                          DeviceMatrixView<double> basisValuesAndDers,
                          double* geoHessians,
                          DeviceMatrixView<double> geoJacobianInv,
                          double* result)
{
    double gradParam[3] = {0.0};
    double hessParam[9] = {0.0};
    tensorBasisGradientSG(r, P1, dim, numDerivatives, basisValuesAndDers, gradParam);
    tensorBasisHessianParamSG(r, P1, dim, numDerivatives, basisValuesAndDers, hessParam);

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

    for (int a = 0; a < dim; ++a)
        for (int b = 0; b < dim; ++b)
        {
            double value = 0.0;
            for (int i = 0; i < dim; ++i)
                for (int j = 0; j < dim; ++j)
                    value += geoJacobianInv(i, a) * geoJacobianInv(j, b) *
                             corrected[i * dim + j];
            result[a * dim + b] = value;
        }
}

// Physical Hessians of the current displacement field. These are GradF in the
// strain-gradient formulation, stored as component-major dim x dim x dim data.
__device__
void physicalFieldHessians(PatchDeviceView fieldPatch, DeviceVectorView<double> pt,
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
        physicalBasisHessian(r, P1, dim, numDerivatives, fieldValuesAndDers,
                             geoHessians, geoJacobianInv, basisHessian);
        for (int comp = 0; comp < dim; ++comp)
        {
            const double cp = fieldPatch.activeControlPointComponent(pt, r, comp);
            for (int a = 0; a < dim * dim; ++a)
                result[comp * dim * dim + a] += cp * basisHessian[a];
        }
    }
}

// Computes P1, P2 and their tangent blocks at one quadrature point.
//
// materialLaw:
//   0 = St. Venant-Kirchhoff
//   1 = neo-Hookean
//
// Outputs are flattened in the indexing convention described at the top of this
// file. The loop structure intentionally follows the original gismo visitor so
// future formula checks can be done term by term.
__device__
void computeKinematicsAndMaterial(
    int materialLaw, double youngsModulus, double poissonsRatio, double lengthScale,
    int dim, DeviceMatrixView<double> F, double* gradFHess,
    double* P1Vec, double* P2Vec, double* ParP1ParF,
    double* ParP1ParGradF, double* ParP2ParGradF)
{
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    const double lambda = youngsModulus * poissonsRatio /
        ((1.0 + poissonsRatio) * (1.0 - 2.0 * poissonsRatio));
    const double mu = youngsModulus / (2.0 * (1.0 + poissonsRatio));
    const double l2 = lengthScale * lengthScale;

    for (int i = 0; i < dim2; ++i)
    {
        P1Vec[i] = 0.0;
        for (int j = 0; j < dim2; ++j)
            ParP1ParF[i * dim2 + j] = 0.0;
        for (int j = 0; j < dim3; ++j)
            ParP1ParGradF[i * dim3 + j] = 0.0;
    }
    for (int i = 0; i < dim3; ++i)
    {
        P2Vec[i] = 0.0;
        for (int j = 0; j < dim3; ++j)
            ParP2ParGradF[i * dim3 + j] = 0.0;
    }

    double FInv[9] = {0.0};
    double FInvTrans[9] = {0.0};
    double FData[9] = {0.0};
    DeviceMatrixView<double> FCopy(FData, dim, dim);
    DeviceMatrixView<double> FInvView(FInv, dim, dim);
    for (int a = 0; a < dim; ++a)
        for (int A = 0; A < dim; ++A)
            FCopy(a, A) = F(a, A);
    FCopy.inverse(FInvView);
    for (int a = 0; a < dim; ++a)
        for (int A = 0; A < dim; ++A)
            FInvTrans[a * dim + A] = FInvView(A, a);
    const double J = FCopy.determinant();

    double E[9] = {0.0};
    double LCG[9] = {0.0};
    if (materialLaw == 0)
    {
        for (int A = 0; A < dim; ++A)
            for (int B = 0; B < dim; ++B)
            {
                double rcg = 0.0;
                for (int a = 0; a < dim; ++a)
                    rcg += F(a, A) * F(a, B);
                E[A * dim + B] = 0.5 * (rcg - (A == B ? 1.0 : 0.0));
            }
        for (int a = 0; a < dim; ++a)
            for (int b = 0; b < dim; ++b)
            {
                double value = 0.0;
                for (int A = 0; A < dim; ++A)
                    value += F(a, A) * F(b, A);
                LCG[a * dim + b] = value;
            }
    }

    double P1[9] = {0.0};
    if (materialLaw == 0)
    {
        double traceE = 0.0;
        for (int A = 0; A < dim; ++A)
            traceE += E[A * dim + A];
        for (int a = 0; a < dim; ++a)
            for (int A = 0; A < dim; ++A)
            {
                double value = 0.0;
                for (int B = 0; B < dim; ++B)
                {
                    const double S = lambda * traceE * (A == B ? 1.0 : 0.0) +
                                     2.0 * mu * E[A * dim + B];
                    value += F(a, B) * S;
                }
                P1[a * dim + A] = value;
            }
    }
    else
    {
        for (int a = 0; a < dim; ++a)
            for (int A = 0; A < dim; ++A)
                P1[a * dim + A] = mu * (F(a, A) - FInvTrans[a * dim + A]) +
                    0.5 * lambda * (J * J - 1.0) * FInvTrans[a * dim + A];
    }

    double P1Part2[9] = {0.0};
    double ParP1ParFPart2[81] = {0.0};
    double ParP1ParFPart3[81] = {0.0};
    double ParP1ParFPart4[81] = {0.0};
    double ParP1ParGradFPart2[243] = {0.0};
    double ParP1ParGradFPart3[243] = {0.0};
    double ParP2ParGradFPart2[729] = {0.0};
    double ParP2ParGradFPart3[729] = {0.0};

    for (int a = 0; a < dim; ++a)
        for (int A = 0; A < dim; ++A)
            for (int b = 0; b < dim; ++b)
                for (int B = 0; B < dim; ++B)
                {
                    const int rowF = a * dim + A;
                    const int colF = b * dim + B;
                    const double delta_ab = (a == b ? 1.0 : 0.0);
                    const double delta_AB = (A == B ? 1.0 : 0.0);

                    if (materialLaw == 0)
                    {
                        ParP1ParF[rowF * dim2 + colF] = F(a, A) * F(b, B);
                        ParP1ParFPart2[rowF * dim2 + colF] =
                            (E[0] + (dim > 1 ? E[dim + 1] : 0.0) +
                             (dim > 2 ? E[2 * dim + 2] : 0.0)) * delta_ab * delta_AB;
                        ParP1ParFPart3[rowF * dim2 + colF] =
                            LCG[a * dim + b] * delta_AB + F(a, B) * F(b, A) +
                            2.0 * E[A * dim + B] * delta_ab;
                    }
                    else
                    {
                        ParP1ParF[rowF * dim2 + colF] =
                            delta_ab * delta_AB +
                            FInvTrans[b * dim + A] * FInvTrans[a * dim + B];
                        ParP1ParFPart2[rowF * dim2 + colF] =
                            FInvTrans[b * dim + B] * FInvTrans[a * dim + A];
                        ParP1ParFPart3[rowF * dim2 + colF] =
                            FInvTrans[b * dim + A] * FInvTrans[a * dim + B];
                    }

                    for (int C = 0; C < dim; ++C)
                    {
                        P1Part2[rowF] += gradFHess[a * dim2 + B * dim + C] *
                            (gradFHess[b * dim2 + B * dim + C] * F(b, A) +
                             gradFHess[b * dim2 + A * dim + C] * F(b, B));

                        const int p2Row = a * dim2 + A * dim + B;
                        P2Vec[p2Row] += F(a, C) *
                            (gradFHess[b * dim2 + A * dim + B] * F(b, C) +
                             0.5 * gradFHess[b * dim2 + C * dim + B] * F(b, A) +
                             0.5 * gradFHess[b * dim2 + C * dim + A] * F(b, B));

                        const int gradCol = b * dim2 + B * dim + C;
                        for (int c = 0; c < dim; ++c)
                            ParP1ParGradF[rowF * dim3 + gradCol] += delta_ab *
                                (gradFHess[c * dim2 + B * dim + C] * F(c, A) +
                                 0.5 * (gradFHess[c * dim2 + A * dim + C] * F(c, B) +
                                        gradFHess[c * dim2 + A * dim + B] * F(c, C)));

                        ParP1ParGradFPart2[rowF * dim3 + gradCol] =
                            gradFHess[a * dim2 + B * dim + C] * F(b, A);

                        for (int D = 0; D < dim; ++D)
                        {
                            const double delta_BD = (B == D ? 1.0 : 0.0);
                            const double delta_AD = (A == D ? 1.0 : 0.0);
                            const double delta_BC = (B == C ? 1.0 : 0.0);
                            const double delta_AC = (A == C ? 1.0 : 0.0);
                            ParP1ParFPart4[rowF * dim2 + colF] +=
                                gradFHess[a * dim2 + C * dim + D] *
                                (gradFHess[b * dim2 + C * dim + D] * delta_AB +
                                 gradFHess[b * dim2 + A * dim + D] * delta_BC);

                            const int gradCol2 = b * dim2 + C * dim + D;
                            ParP2ParGradFPart2[p2Row * dim3 + gradCol2] =
                                (delta_BD * F(a, C) + delta_BC * F(a, D)) * F(b, A);
                            ParP2ParGradFPart3[p2Row * dim3 + gradCol2] =
                                (delta_AD * F(a, C) + delta_AC * F(a, D)) * F(b, B);
                            for (int e = 0; e < dim; ++e)
                                ParP2ParGradF[p2Row * dim3 + gradCol2] +=
                                    (delta_AC * delta_BD + delta_AD * delta_BC) *
                                    F(a, e) * F(b, e);

                            ParP1ParGradFPart3[rowF * dim3 + gradCol] +=
                                (gradFHess[a * dim2 + D * dim + C] * delta_AB +
                                 gradFHess[a * dim2 + D * dim + B] * delta_AC) *
                                F(b, D);
                        }
                    }
                }

    for (int i = 0; i < dim2; ++i)
    {
        P1Vec[i] = P1[i] + 0.5 * mu * l2 * P1Part2[i];
        for (int j = 0; j < dim2; ++j)
        {
            const int idx = i * dim2 + j;
            if (materialLaw == 0)
                ParP1ParF[idx] = lambda * (ParP1ParF[idx] + ParP1ParFPart2[idx]) +
                                 mu * ParP1ParFPart3[idx] +
                                 0.5 * mu * l2 * ParP1ParFPart4[idx];
            else
                ParP1ParF[idx] = mu * ParP1ParF[idx] +
                                 lambda * J * J * ParP1ParFPart2[idx] -
                                 0.5 * lambda * (J * J - 1.0) * ParP1ParFPart3[idx] +
                                 0.5 * mu * l2 * ParP1ParFPart4[idx];
        }
        for (int j = 0; j < dim3; ++j)
        {
            const int idx = i * dim3 + j;
            ParP1ParGradF[idx] = 0.5 * mu * l2 *
                (ParP1ParGradF[idx] + ParP1ParGradFPart2[idx] +
                 0.5 * ParP1ParGradFPart3[idx]);
        }
    }

    for (int i = 0; i < dim3; ++i)
    {
        P2Vec[i] *= 0.5 * mu * l2;
        for (int j = 0; j < dim3; ++j)
        {
            const int idx = i * dim3 + j;
            ParP2ParGradF[idx] = 0.25 * mu * l2 *
                (ParP2ParGradF[idx] + 0.5 * ParP2ParGradFPart2[idx] +
                 0.5 * ParP2ParGradFPart3[idx]);
        }
    }
}

// Builds the strain-displacement operator B and the second-gradient operator D
// for one scalar basis function and one displacement component.
__device__
void buildBAndD(int basisIndex, int component, int P1, int dim, int numDerivatives,
                DeviceMatrixView<double> valuesAndDers,
                double* geoHessians, DeviceMatrixView<double> geoJacobianInv,
                double* B, double* D)
{
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    for (int i = 0; i < dim2; ++i)
        B[i] = 0.0;
    for (int i = 0; i < dim3; ++i)
        D[i] = 0.0;

    double gradParam[3] = {0.0};
    tensorBasisGradientSG(basisIndex, P1, dim, numDerivatives, valuesAndDers, gradParam);
    double gradPhys[3] = {0.0};
    for (int a = 0; a < dim; ++a)
        for (int i = 0; i < dim; ++i)
            gradPhys[a] += geoJacobianInv(i, a) * gradParam[i];

    double hessianPhys[9] = {0.0};
    physicalBasisHessian(basisIndex, P1, dim, numDerivatives, valuesAndDers,
                         geoHessians, geoJacobianInv, hessianPhys);

    for (int A = 0; A < dim; ++A)
        B[(component * dim + A) * dim + component] = gradPhys[A];
    for (int A = 0; A < dim; ++A)
        for (int C = 0; C < dim; ++C)
            D[(component * dim2 + A * dim + C) * dim + component] =
                hessianPhys[A * dim + C];
}

__device__
void buildPhysicalGradientAndHessian(int basisIndex, int P1, int dim,
                                     int numDerivatives,
                                     DeviceMatrixView<double> valuesAndDers,
                                     double* geoHessians,
                                     DeviceMatrixView<double> geoJacobianInv,
                                     double* gradPhys,
                                     double* hessianPhys)
{
    double gradParam[3] = {0.0};
    tensorBasisGradientSG(basisIndex, P1, dim, numDerivatives, valuesAndDers, gradParam);
    for (int a = 0; a < dim; ++a)
    {
        gradPhys[a] = 0.0;
        for (int i = 0; i < dim; ++i)
            gradPhys[a] += geoJacobianInv(i, a) * gradParam[i];
    }

    physicalBasisHessian(basisIndex, P1, dim, numDerivatives, valuesAndDers,
                         geoHessians, geoJacobianInv, hessianPhys);
}

__device__
double strainGradientTangentEntry(int dim, int di, int dj,
                                  const double* grad_i,
                                  const double* hess_i,
                                  const double* grad_j,
                                  const double* hess_j,
                                  const double* dP1dF,
                                  const double* dP1dGradF,
                                  const double* dP2dGradF)
{
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    double tangent = 0.0;

    for (int A = 0; A < dim; ++A)
    {
        const int rowF_i = di * dim + A;
        for (int B = 0; B < dim; ++B)
        {
            const int rowF_j = dj * dim + B;
            tangent += grad_i[A] * dP1dF[rowF_i * dim2 + rowF_j] * grad_j[B];

            for (int C = 0; C < dim; ++C)
            {
                const int rowG_j = dj * dim2 + B * dim + C;
                tangent += grad_i[A] *
                           dP1dGradF[rowF_i * dim3 + rowG_j] *
                           hess_j[B * dim + C];
            }
        }
    }

    for (int A = 0; A < dim; ++A)
        for (int C = 0; C < dim; ++C)
        {
            const int rowG_i = di * dim2 + A * dim + C;
            for (int B = 0; B < dim; ++B)
            {
                const int rowF_j = dj * dim + B;
                tangent += hess_i[A * dim + C] *
                           dP1dGradF[rowF_j * dim3 + rowG_i] *
                           grad_j[B];

                for (int D = 0; D < dim; ++D)
                {
                    const int rowG_j = dj * dim2 + B * dim + D;
                    tangent += hess_i[A * dim + C] *
                               dP2dGradF[rowG_i * dim3 + rowG_j] *
                               hess_j[B * dim + D];
                }
            }
        }

    return tangent;
}

__device__
double strainGradientResidualEntry(int dim, int di,
                                   const double* grad_i,
                                   const double* hess_i,
                                   const double* P1Vec,
                                   const double* P2Vec)
{
    const int dim2 = dim * dim;
    double residual = 0.0;
    for (int A = 0; A < dim; ++A)
        residual += grad_i[A] * P1Vec[di * dim + A];
    for (int A = 0; A < dim; ++A)
        for (int C = 0; C < dim; ++C)
            residual += hess_i[A * dim + C] *
                        P2Vec[di * dim2 + A * dim + C];
    return residual;
}

/**
 * Precompute all Gauss-point quantities that are independent of the local
 * basis-pair entry. One thread owns one global Gauss point.
 *
 * This mirrors the evaluateGPKernel stage in GPUAssembler and
 * GPUElectroelasticityAssembler. The matrix/RHS kernels read this flat storage
 * instead of recomputing geometry, F, GradF, and constitutive tangent blocks for
 * every local matrix entry.
 */
__global__
void evaluateStrainGradientGPKernel(
    int numDerivatives, int totalNumGPs, int stride,
    int materialLaw, double youngsModulus, double poissonsRatio,
    double lengthScale, double localStiffening,
    MultiPatchDeviceView displacement, MultiPatchDeviceView geometry,
    DeviceMatrixView<double> pts, DeviceVectorView<double> wts,
    DeviceMatrixView<double> geoValuesAndDerss,
    DeviceMatrixView<double> dispValuesAndDerss,
    DeviceVectorView<double> sgGPData)
{
    for (int GPIdx = blockIdx.x * blockDim.x + threadIdx.x; GPIdx < totalNumGPs;
         GPIdx += blockDim.x * gridDim.x)
    {
        const int dim = geometry.domainDim();
        const int dim2 = dim * dim;
        const int dim3 = dim2 * dim;

        int geoInvOffset = 0;
        int weightForceOffset = 0;
        int weightBodyOffset = 0;
        int geoHessOffset = 0;
        int FOffset = 0;
        int P1Offset = 0;
        int P2Offset = 0;
        int dP1dFOffset = 0;
        int dP1dGradFOffset = 0;
        int dP2dGradFOffset = 0;
        strainGradientGPOffsets(dim, geoInvOffset, weightForceOffset,
                                weightBodyOffset, geoHessOffset, FOffset,
                                P1Offset, P2Offset, dP1dFOffset,
                                dP1dGradFOffset, dP2dGradFOffset);

        double* gp = sgGPData.data() + GPIdx * stride;
        DeviceVectorView<double> pt(pts.data() + GPIdx * dim, dim);

        int patchIdx = 0;
        displacement.threadPatch(GPIdx, patchIdx);
        PatchDeviceView geoPatch = geometry.patch(patchIdx);
        PatchDeviceView dispPatch = displacement.patch(patchIdx);
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);

        const int geoP1 = geoPatch.basis().knotsOrder(0) + 1;
        const int dispP1 = dispBasis.knotsOrder(0) + 1;
        DeviceMatrixView<double> geoValuesAndDers(
            geoValuesAndDerss.data() + GPIdx * geoP1 * (numDerivatives + 1) * dim,
            geoP1, (numDerivatives + 1) * dim);
        DeviceMatrixView<double> dispValuesAndDers(
            dispValuesAndDerss.data() + GPIdx * dispP1 * (numDerivatives + 1) * dim,
            dispP1, (numDerivatives + 1) * dim);

        DeviceMatrixView<double> geoJacobianInv(gp + geoInvOffset, dim, dim);
        double geoJacobianData[9] = {0.0};
        DeviceMatrixView<double> geoJacobian(geoJacobianData, dim, dim);
        geoPatch.jacobian(pt, geoValuesAndDers, numDerivatives, geoJacobian);
        geoJacobian.inverse(geoJacobianInv);
        const double measure = geoJacobian.determinant();
        gp[weightForceOffset] = wts[GPIdx] * measure;
        gp[weightBodyOffset] = wts[GPIdx] * pow(measure, -localStiffening) * measure;

        double* geoHessians = gp + geoHessOffset;
        for (int a = 0; a < dim; ++a)
            patchParamHessian(geoPatch, pt, geoValuesAndDers, numDerivatives, a,
                              geoHessians + a * dim2);

        double dispJacobianData[9] = {0.0};
        double physDispJacData[9] = {0.0};
        double FData[9] = {0.0};
        DeviceMatrixView<double> dispJacobian(dispJacobianData, dim, dim);
        DeviceMatrixView<double> physDispJac(physDispJacData, dim, dim);
        DeviceMatrixView<double> F(FData, dim, dim);
        dispPatch.jacobian(pt, dispValuesAndDers, numDerivatives, dispJacobian);
        dispJacobian.times(geoJacobianInv, physDispJac);
        physDispJac.plusIdentity(F);
        double* FStored = gp + FOffset;
        for (int a = 0; a < dim; ++a)
            for (int A = 0; A < dim; ++A)
                FStored[a * dim + A] = F(a, A);

        double gradFHess[27] = {0.0};
        physicalFieldHessians(dispPatch, pt, dispValuesAndDers, numDerivatives,
                              geoHessians, geoJacobianInv, gradFHess);

        computeKinematicsAndMaterial(materialLaw, youngsModulus, poissonsRatio,
            lengthScale, dim, F, gradFHess, gp + P1Offset, gp + P2Offset,
            gp + dP1dFOffset, gp + dP1dGradFOffset, gp + dP2dGradFOffset);
    }
}

/**
 * Assemble the tangent matrix for finite-strain strain-gradient elasticity.
 *
 * Thread/block ownership:
 *   Each CUDA block owns one local tangent block K_e(i, j), whose size is
 *   dim x dim. Each thread computes one Gauss-point contribution to the whole
 *   K_e(i, j) block. It evaluates all component entries K_e(i,di,j,dj), then
 *   atomically adds that dim x dim contribution into shared memory.
 *
 * Required precomputed data:
 *   sgGPData          - output of evaluateStrainGradientGPKernel. It stores
 *                       geometry inverse, weights, geometry Hessians, P1/P2,
 *                       and constitutive tangent blocks per Gauss point.
 *   dispValuesAndDerss- displacement basis values/derivatives. The local B and
 *                       D operators still depend on the basis index and are
 *                       therefore built in this kernel.
 *
 * Per-thread contribution:
 *   1. Decode block ownership (element, i, j) and local Gauss point q.
 *   2. Read GP data from global memory.
 *   3. Loop over component pairs (di, dj).
 *   4. Build B_i, D_i, B_j, D_j for this q and component pair.
 *   5. Contract the four tangent blocks:
 *
 *          B_i^T (dP1/dF)      B_j
 *        + B_i^T (dP1/dGradF)  D_j
 *        + D_i^T (dP1/dGradF)^T B_j
 *        + D_i^T (dP2/dGradF)  D_j.
 *
 * Global insertion:
 *   After all Gauss-point contributions are accumulated, the block pushes its
 *   dim x dim local tangent block to the global sparse system.
 *   SparseSystemDeviceView::pushToMatrix handles free/fixed dof logic. Free
 *   columns are added to the CSR matrix; eliminated fixed columns contribute to
 *   the RHS through `eliminatedDofs`, matching GPUAssembler's convention.
 *
 * Limits/assumptions:
 *   - Scratch arrays are statically sized for dim <= 3.
 *   - numDerivatives must be at least 2.
 *   - The displacement basis should be smooth enough for Hessian terms inside
 *     each patch.
 */
__global__
void assembleStrainGradientMatrixKernel(
    int numDerivatives, int numElements, int N_D, int stride,
    MultiPatchDeviceView displacement, MultiPatchDeviceView geometry,
    SparseSystemDeviceView system, DeviceNestedArrayView<double> eliminatedDofs,
    DeviceMatrixView<double> pts, DeviceMatrixView<double> dispValuesAndDerss,
    DeviceVectorView<double> sgGPData)
{
    extern __shared__ double localMatrix[];

    const int dim = geometry.domainDim();
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    const int threadId = threadIdx.x;

    int blockId = blockIdx.x;
    const int j = blockId % N_D; blockId /= N_D;
    const int i = blockId % N_D; blockId /= N_D;
    const int elementGlobal = blockId;
    if (elementGlobal >= numElements)
        return;

    for (int idx = threadId; idx < dim * dim; idx += blockDim.x)
        localMatrix[idx] = 0.0;
    __syncthreads();

    int patchIdx = 0;
    displacement.threadPatch_element(elementGlobal, patchIdx);
    TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
    const int numGPsInElement = dispBasis.numGPsInElement();
    const int P1 = dispBasis.knotsOrder(0) + 1;

    int geoInvOffset = 0;
    int weightForceOffset = 0;
    int weightBodyOffset = 0;
        int geoHessOffset = 0;
        int FOffset = 0;
        int P1Offset = 0;
    int P2Offset = 0;
    int dP1dFOffset = 0;
    int dP1dGradFOffset = 0;
    int dP2dGradFOffset = 0;
        strainGradientGPOffsets(dim, geoInvOffset, weightForceOffset,
                                weightBodyOffset, geoHessOffset, FOffset,
                                P1Offset, P2Offset, dP1dFOffset,
                                dP1dGradFOffset, dP2dGradFOffset);

    for (int q = threadId; q < numGPsInElement; q += blockDim.x)
    {
        const int GPIdx = elementGlobal * numGPsInElement + q;
        double* gp = sgGPData.data() + GPIdx * stride;
        DeviceMatrixView<double> geoJacobianInv(gp + geoInvOffset, dim, dim);
        double* geoHessians = gp + geoHessOffset;
        DeviceMatrixView<double> dispValuesAndDers(
            dispValuesAndDerss.data() + GPIdx * P1 * (numDerivatives + 1) * dim,
            P1, (numDerivatives + 1) * dim);

        double* ParP1ParF = gp + dP1dFOffset;
        double* ParP1ParGradF = gp + dP1dGradFOffset;
        double* ParP2ParGradF = gp + dP2dGradFOffset;

        double grad_i[3] = {0.0};
        double grad_j[3] = {0.0};
        double hess_i[9] = {0.0};
        double hess_j[9] = {0.0};
        buildPhysicalGradientAndHessian(i, P1, dim, numDerivatives,
                                        dispValuesAndDers, geoHessians,
                                        geoJacobianInv, grad_i, hess_i);
        buildPhysicalGradientAndHessian(j, P1, dim, numDerivatives,
                                        dispValuesAndDers, geoHessians,
                                        geoJacobianInv, grad_j, hess_j);

        for (int di = 0; di < dim; ++di)
        {
            for (int dj = 0; dj < dim; ++dj)
            {
                const double tangent = strainGradientTangentEntry(
                    dim, di, dj, grad_i, hess_i, grad_j, hess_j,
                    ParP1ParF, ParP1ParGradF, ParP2ParGradF);
                atomicAdd(&localMatrix[di * dim + dj], gp[weightBodyOffset] * tangent);
            }
        }
    }
    __syncthreads();

    double ptForIndexData[3] = {0.0};
    DeviceVectorView<double> ptForIndex(ptForIndexData, dim);
    for (int a = 0; a < dim; ++a)
        ptForIndex[a] = pts(a, elementGlobal * numGPsInElement);

    for (int localEntry = threadId; localEntry < dim * dim; localEntry += blockDim.x)
    {
        const int dj = localEntry % dim;
        const int di = localEntry / dim;
        const int globalIndexI = system.mapColIndex(dispBasis.activeIndex(ptForIndex, i), patchIdx, di);
        const int globalIndexJ = system.mapColIndex(dispBasis.activeIndex(ptForIndex, j), patchIdx, dj);
        system.pushToMatrix(localMatrix[localEntry], globalIndexI, globalIndexJ,
                            eliminatedDofs, di, dj);
    }
}

// RHS/residual assembly kernel.
// Each CUDA block owns one local residual block R_e(i), whose size is dim.
// Threads inside the block own Gauss-point contributions and accumulate all
// displacement components into shared memory before pushing the dim entries.
__global__
void assembleStrainGradientRHSKernel(
    int numDerivatives, int numElements, int N_D, int stride,
    double forceScaling,
    MultiPatchDeviceView displacement, MultiPatchDeviceView geometry,
    SparseSystemDeviceView system, DeviceMatrixView<double> pts,
    DeviceMatrixView<double> dispValuesAndDerss, DeviceVectorView<double> bodyForce,
    DeviceVectorView<double> sgGPData)
{
    extern __shared__ double localRHS[];

    const int dim = geometry.domainDim();
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    const int threadId = threadIdx.x;

    int blockId = blockIdx.x;
    const int i = blockId % N_D;
    const int elementGlobal = blockId / N_D;
    if (elementGlobal >= numElements)
        return;

    for (int idx = threadId; idx < dim; idx += blockDim.x)
        localRHS[idx] = 0.0;
    __syncthreads();

    int patchIdx = 0;
    displacement.threadPatch_element(elementGlobal, patchIdx);
    TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
    const int numGPsInElement = dispBasis.numGPsInElement();
    const int P1 = dispBasis.knotsOrder(0) + 1;

    int geoInvOffset = 0;
    int weightForceOffset = 0;
    int weightBodyOffset = 0;
        int geoHessOffset = 0;
        int FOffset = 0;
        int P1Offset = 0;
    int P2Offset = 0;
    int dP1dFOffset = 0;
    int dP1dGradFOffset = 0;
    int dP2dGradFOffset = 0;
        strainGradientGPOffsets(dim, geoInvOffset, weightForceOffset,
                                weightBodyOffset, geoHessOffset, FOffset,
                                P1Offset, P2Offset, dP1dFOffset,
                                dP1dGradFOffset, dP2dGradFOffset);

    for (int q = threadId; q < numGPsInElement; q += blockDim.x)
    {
        const int GPIdx = elementGlobal * numGPsInElement + q;
        double* gp = sgGPData.data() + GPIdx * stride;
        DeviceMatrixView<double> geoJacobianInv(gp + geoInvOffset, dim, dim);
        double* geoHessians = gp + geoHessOffset;
        DeviceMatrixView<double> dispValuesAndDers(
            dispValuesAndDerss.data() + GPIdx * P1 * (numDerivatives + 1) * dim,
            P1, (numDerivatives + 1) * dim);

        double* P1Vec = gp + P1Offset;
        double* P2Vec = gp + P2Offset;
        double grad_i[3] = {0.0};
        double hess_i[9] = {0.0};
        buildPhysicalGradientAndHessian(i, P1, dim, numDerivatives,
                                        dispValuesAndDers, geoHessians,
                                        geoJacobianInv, grad_i, hess_i);

        for (int di = 0; di < dim; ++di)
        {
            const double residual = strainGradientResidualEntry(dim, di, grad_i,
                                                                hess_i, P1Vec,
                                                                P2Vec);
            double rhs = -gp[weightBodyOffset] * residual;
            rhs += gp[weightForceOffset] * forceScaling * bodyForce[di] *
                   tensorBasisValueSG(i, P1, dim, numDerivatives, dispValuesAndDers);
            atomicAdd(&localRHS[di], rhs);
        }
    }
    __syncthreads();

    double ptForIndexData[3] = {0.0};
    DeviceVectorView<double> ptForIndex(ptForIndexData, dim);
    for (int a = 0; a < dim; ++a)
        ptForIndex[a] = pts(a, elementGlobal * numGPsInElement);

    for (int di = threadId; di < dim; di += blockDim.x)
    {
        const int globalIndexI = system.mapColIndex(dispBasis.activeIndex(ptForIndex, i), patchIdx, di);
        system.pushToRhs(localRHS[di], globalIndexI, di);
    }
}

__global__
void zeroStrainGradientFunctionControlPointsKernel(MultiPatchDeviceView result,
                                                   int totalEntries)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalEntries; idx += blockDim.x * gridDim.x)
    {
        int patch = 0;
        int component = 0;
        int pointIdx = result.threadPatchAndDof(idx, patch, component);
        result.setCoefficients(patch, pointIdx, component, 0.0);
    }
}

__global__
void recoverFirstPiolaStressAtNodesKernel(int numDerivatives,
                                          int totalNumGPs,
                                          int stride,
                                          MultiPatchDeviceView displacement,
                                          MultiPatchDeviceView firstPiolaStress,
                                          DeviceMatrixView<double> pts,
                                          DeviceMatrixView<double> dispValuesAndDerss,
                                          DeviceVectorView<double> sgGPData,
                                          DeviceVectorView<double> nodalWeights)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalNumGPs; idx += blockDim.x * gridDim.x)
    {
        const int dim = displacement.domainDim();
        const int dim2 = dim * dim;

        int geoInvOffset = 0;
        int weightForceOffset = 0;
        int weightBodyOffset = 0;
        int geoHessOffset = 0;
        int FOffset = 0;
        int P1Offset = 0;
        int P2Offset = 0;
        int dP1dFOffset = 0;
        int dP1dGradFOffset = 0;
        int dP2dGradFOffset = 0;
        strainGradientGPOffsets(dim, geoInvOffset, weightForceOffset,
                                weightBodyOffset, geoHessOffset, FOffset,
                                P1Offset, P2Offset, dP1dFOffset,
                                dP1dGradFOffset, dP2dGradFOffset);

        int patchIdx = 0;
        displacement.threadPatch(idx, patchIdx);
        TensorBsplineBasisDeviceView basis = displacement.basis(patchIdx);
        const int P1 = basis.knotsOrder(0) + 1;
        const int numActive = basis.numActiveControlPoints();
        DeviceVectorView<double> pt(pts.data() + idx * dim, dim);
        DeviceMatrixView<double> dispValuesAndDers(
            dispValuesAndDerss.data() + idx * P1 * (numDerivatives + 1) * dim,
            P1, (numDerivatives + 1) * dim);

        double* gp = sgGPData.data() + idx * stride;
        double* P1Vec = gp + P1Offset;
        const double weightForce = gp[weightForceOffset];

        int patchControlPointOffset = 0;
        for (int p = 0; p < patchIdx; ++p)
            patchControlPointOffset += firstPiolaStress.numControlPoints(p);

        DeviceMatrixView<double> stressControlPoints =
            firstPiolaStress.controlPoints(patchIdx);
        for (int r = 0; r < numActive; ++r)
        {
            const int localControlPoint = basis.activeIndex(pt, r);
            const int globalControlPoint =
                patchControlPointOffset + localControlPoint;
            const double N = tensorBasisValueSG(r, P1, dim, numDerivatives,
                                                dispValuesAndDers);
            const double weight = N * weightForce;

            atomicAdd(&nodalWeights[globalControlPoint], weight);
            for (int c = 0; c < dim2; ++c)
                atomicAdd(&stressControlPoints(localControlPoint, c),
                          weight * P1Vec[c]);
        }
    }
}

__global__
void recoverStrainGradientCauchyStressAtNodesKernel(int numDerivatives,
                                                    int totalNumGPs,
                                                    int stride,
                                                    MultiPatchDeviceView displacement,
                                                    MultiPatchDeviceView cauchyStress,
                                                    DeviceMatrixView<double> pts,
                                                    DeviceMatrixView<double> dispValuesAndDerss,
                                                    DeviceVectorView<double> sgGPData,
                                                    DeviceVectorView<double> nodalWeights)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalNumGPs; idx += blockDim.x * gridDim.x)
    {
        const int dim = displacement.domainDim();
        const int dimTensor = dim * (dim + 1) / 2;

        int geoInvOffset = 0;
        int weightForceOffset = 0;
        int weightBodyOffset = 0;
        int geoHessOffset = 0;
        int FOffset = 0;
        int P1Offset = 0;
        int P2Offset = 0;
        int dP1dFOffset = 0;
        int dP1dGradFOffset = 0;
        int dP2dGradFOffset = 0;
        strainGradientGPOffsets(dim, geoInvOffset, weightForceOffset,
                                weightBodyOffset, geoHessOffset, FOffset,
                                P1Offset, P2Offset, dP1dFOffset,
                                dP1dGradFOffset, dP2dGradFOffset);

        int patchIdx = 0;
        displacement.threadPatch(idx, patchIdx);
        TensorBsplineBasisDeviceView basis = displacement.basis(patchIdx);
        const int basisP1 = basis.knotsOrder(0) + 1;
        const int numActive = basis.numActiveControlPoints();
        DeviceVectorView<double> pt(pts.data() + idx * dim, dim);
        DeviceMatrixView<double> dispValuesAndDers(
            dispValuesAndDerss.data() + idx * basisP1 * (numDerivatives + 1) * dim,
            basisP1, (numDerivatives + 1) * dim);

        double* gp = sgGPData.data() + idx * stride;
        DeviceMatrixView<double> F(gp + FOffset, dim, dim);
        DeviceMatrixView<double> firstPiola(gp + P1Offset, dim, dim);
        const double weightForce = gp[weightForceOffset];

        double sigmaData[9] = {0.0};
        DeviceMatrixView<double> sigma(sigmaData, dim, dim);
        firstPiola.timeTranspose(F, sigma);
        sigma.times(1.0 / F.determinant());

        double sigmaVecData[6] = {0.0};
        DeviceVectorView<double> sigmaVec(sigmaVecData, dimTensor);
        voigtStressView(sigmaVec, sigma);

        int patchControlPointOffset = 0;
        for (int p = 0; p < patchIdx; ++p)
            patchControlPointOffset += cauchyStress.numControlPoints(p);

        DeviceMatrixView<double> stressControlPoints =
            cauchyStress.controlPoints(patchIdx);
        for (int r = 0; r < numActive; ++r)
        {
            const int localControlPoint = basis.activeIndex(pt, r);
            const int globalControlPoint =
                patchControlPointOffset + localControlPoint;
            const double N = tensorBasisValueSG(r, basisP1, dim, numDerivatives,
                                                dispValuesAndDers);
            const double weight = N * weightForce;

            atomicAdd(&nodalWeights[globalControlPoint], weight);
            for (int c = 0; c < dimTensor; ++c)
                atomicAdd(&stressControlPoints(localControlPoint, c),
                          weight * sigmaVec[c]);
        }
    }
}

__global__
void normalizeRecoveredFirstPiolaStressKernel(MultiPatchDeviceView firstPiolaStress,
                                              DeviceVectorView<double> nodalWeights,
                                              int totalControlPoints)
{
    const int targetDim = firstPiolaStress.targetDim();
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalControlPoints * targetDim; idx += blockDim.x * gridDim.x)
    {
        int patch = 0;
        int component = 0;
        int localControlPoint = firstPiolaStress.threadPatchAndDof(idx, patch,
                                                                   component);

        int globalControlPoint = localControlPoint;
        for (int p = 0; p < patch; ++p)
            globalControlPoint += firstPiolaStress.numControlPoints(p);

        const double weight = nodalWeights[globalControlPoint];
        if (weight != 0.0)
        {
            DeviceMatrixView<double> stressControlPoints =
                firstPiolaStress.controlPoints(patch);
            stressControlPoints(localControlPoint, component) /= weight;
        }
    }
}

} // namespace

// Requests second derivatives from the base assembler and then installs the
// strain-gradient option set.
GPUStrainGradientElasticityAssembler::GPUStrainGradientElasticityAssembler(
    const MultiPatch& multiPatch,
    const MultiBasis& multiBasis,
    const BoundaryConditions& bc,
    const Eigen::VectorXd& bodyForce)
    : GPUAssembler(multiPatch, multiBasis, bc, bodyForce, false, 2)
{
    setDefaultOptions();
}

// Keep this option order stable: OptionList::realValues() is map-ordered in the
// base assembler, but this assembler reads options by name on the host before
// launching kernels.
OptionList GPUStrainGradientElasticityAssembler::defaultStrainGradientOptions()
{
    OptionList opt;
    opt.addReal("youngs_modulus", "Young's modulus", 1.0);
    opt.addReal("poissons_ratio", "Poisson's ratio", 0.3);
    opt.addReal("length_scale", "Strain-gradient length scale", 0.0);
    opt.addReal("force_scaling", "Body-force scaling", 1.0);
    opt.addReal("neumann_load_scaling", "Multiplier for Neumann boundary and corner loads", 1.0);
    opt.addReal("local_stiffening", "Local stiffening exponent", 0.0);
    opt.addInt("material_law", "0: StVK, 1: neo-Hookean", 1);
    return opt;
}

void GPUStrainGradientElasticityAssembler::setDefaultOptions()
{
    GPUAssembler::setDefaultOptions(defaultStrainGradientOptions());
}

void GPUStrainGradientElasticityAssembler::constructFirstPiolaStressFunction(
    const DeviceVectorView<double>& solVector,
    const DeviceNestedArrayView<double>& fixedDoFs,
    GPUFunction& firstPiolaStressFunction)
{
    constructDispSolution(solVector, fixedDoFs);
    constructStrainGradientStressFunctions(displacementView(),
                                           &firstPiolaStressFunction, nullptr);
}

void GPUStrainGradientElasticityAssembler::constructFirstPiolaStressFunction(
    GPUFunction& displacementFunction,
    GPUFunction& firstPiolaStressFunction)
{
    assert(displacementFunction.domainDim() == domainDim() &&
           "Displacement function domain dimension must match assembler domain dimension");
    assert(displacementFunction.targetDim() == targetDim() &&
           "Displacement function target dimension must match assembler target dimension");

    constructStrainGradientStressFunctions(displacementFunction.multiPatchDeviceView(),
                                           &firstPiolaStressFunction, nullptr);
}

void GPUStrainGradientElasticityAssembler::constructStrainGradientStressFunctions(
    const DeviceVectorView<double>& solVector,
    const DeviceNestedArrayView<double>& fixedDoFs,
    GPUFunction& firstPiolaStressFunction,
    GPUFunction& cauchyStressFunction)
{
    constructDispSolution(solVector, fixedDoFs);
    constructStrainGradientStressFunctions(displacementView(),
                                           &firstPiolaStressFunction,
                                           &cauchyStressFunction);
}

void GPUStrainGradientElasticityAssembler::constructStrainGradientStressFunctions(
    GPUFunction& displacementFunction,
    GPUFunction& firstPiolaStressFunction,
    GPUFunction& cauchyStressFunction)
{
    assert(displacementFunction.domainDim() == domainDim() &&
           "Displacement function domain dimension must match assembler domain dimension");
    assert(displacementFunction.targetDim() == targetDim() &&
           "Displacement function target dimension must match assembler target dimension");

    constructStrainGradientStressFunctions(displacementFunction.multiPatchDeviceView(),
                                           &firstPiolaStressFunction,
                                           &cauchyStressFunction);
}

void GPUStrainGradientElasticityAssembler::constructStrainGradientStressFunctions(
    MultiPatchDeviceView displacementView,
    GPUFunction* firstPiolaStressFunction,
    GPUFunction* cauchyStressFunction)
{
    const int dim = domainDim();
    const int dim2 = dim * dim;
    const int dimTensor = dim * (dim + 1) / 2;
    if (firstPiolaStressFunction == nullptr && cauchyStressFunction == nullptr)
        throw std::invalid_argument("At least one strain-gradient stress function must be requested");
    if (firstPiolaStressFunction != nullptr)
    {
        if (firstPiolaStressFunction->domainDim() != dim)
            throw std::invalid_argument("First Piola stress function domain dimension must match assembler domain dimension");
        if (firstPiolaStressFunction->targetDim() != dim2)
            throw std::invalid_argument("First Piola stress function target dimension must be dim * dim");
    }
    if (cauchyStressFunction != nullptr)
    {
        if (cauchyStressFunction->domainDim() != dim)
            throw std::invalid_argument("Cauchy stress function domain dimension must match assembler domain dimension");
        if (cauchyStressFunction->targetDim() != dimTensor)
            throw std::invalid_argument("Cauchy stress function target dimension must be dim * (dim + 1) / 2");
    }

    int minGrid = 0;
    int blockSize = 0;
    int gridSize = 0;
    cudaError_t err;

    const int totalControlPoints = totalNumControlPoints();
    const int totalFirstPiolaEntries = totalControlPoints * dim2;
    const int totalCauchyEntries = totalControlPoints * dimTensor;
    if (firstPiolaStressFunction != nullptr)
    {
        cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
            zeroStrainGradientFunctionControlPointsKernel, 0, totalFirstPiolaEntries);
        gridSize = (totalFirstPiolaEntries + blockSize - 1) / blockSize;
        zeroStrainGradientFunctionControlPointsKernel<<<gridSize, blockSize>>>(
            firstPiolaStressFunction->multiPatchDeviceView(), totalFirstPiolaEntries);
        err = cudaDeviceSynchronize();
        assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUStrainGradientElasticityAssembler::constructStrainGradientStressFunctions zero first Piola kernel");
    }
    if (cauchyStressFunction != nullptr)
    {
        cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
            zeroStrainGradientFunctionControlPointsKernel, 0, totalCauchyEntries);
        gridSize = (totalCauchyEntries + blockSize - 1) / blockSize;
        zeroStrainGradientFunctionControlPointsKernel<<<gridSize, blockSize>>>(
            cauchyStressFunction->multiPatchDeviceView(), totalCauchyEntries);
        err = cudaDeviceSynchronize();
        assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUStrainGradientElasticityAssembler::constructStrainGradientStressFunctions zero Cauchy kernel");
    }

    const double youngsModulus = options().getReal("youngs_modulus");
    const double poissonsRatio = options().getReal("poissons_ratio");
    const double lengthScale = options().getReal("length_scale");
    const double localStiffening = options().getReal("local_stiffening");
    const int materialLaw = options().getInt("material_law");

    const int sgStride = strainGradientDoublesPerGP(dim);
    const int sgDataSize = sgStride * numGPs();
    if (m_sgGPData.size() != sgDataSize)
        m_sgGPData.resize(sgDataSize);
    else
        m_sgGPData.setZero();

    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        evaluateStrainGradientGPKernel, 0, numGPs());
    gridSize = (numGPs() + blockSize - 1) / blockSize;
    evaluateStrainGradientGPKernel<<<gridSize, blockSize>>>(
        numDerivatives(), numGPs(), sgStride, materialLaw, youngsModulus,
        poissonsRatio, lengthScale, localStiffening, displacementView,
        geometryView(), gpTable(), wts().vectorView(), geoValuesAndDerssView(),
        dispValuesAndDerssView(), m_sgGPData.vectorView());
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUStrainGradientElasticityAssembler::constructStrainGradientStressFunctions GP evaluation");

    if (firstPiolaStressFunction != nullptr)
    {
        DeviceArray<double> nodalWeights(totalControlPoints);
        cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
            recoverFirstPiolaStressAtNodesKernel, 0, numGPs());
        gridSize = (numGPs() + blockSize - 1) / blockSize;
        recoverFirstPiolaStressAtNodesKernel<<<gridSize, blockSize>>>(
            numDerivatives(), numGPs(), sgStride, displacementView,
            firstPiolaStressFunction->multiPatchDeviceView(), gpTable(),
            dispValuesAndDerssView(), m_sgGPData.vectorView(),
            nodalWeights.vectorView());
        err = cudaDeviceSynchronize();
        assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUStrainGradientElasticityAssembler::constructStrainGradientStressFunctions first Piola recovery");

        cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
            normalizeRecoveredFirstPiolaStressKernel, 0, totalFirstPiolaEntries);
        gridSize = (totalFirstPiolaEntries + blockSize - 1) / blockSize;
        normalizeRecoveredFirstPiolaStressKernel<<<gridSize, blockSize>>>(
            firstPiolaStressFunction->multiPatchDeviceView(),
            nodalWeights.vectorView(), totalControlPoints);
        err = cudaDeviceSynchronize();
        assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUStrainGradientElasticityAssembler::constructStrainGradientStressFunctions first Piola normalize");
    }

    if (cauchyStressFunction != nullptr)
    {
        DeviceArray<double> nodalWeights(totalControlPoints);
        cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
            recoverStrainGradientCauchyStressAtNodesKernel, 0, numGPs());
        gridSize = (numGPs() + blockSize - 1) / blockSize;
        recoverStrainGradientCauchyStressAtNodesKernel<<<gridSize, blockSize>>>(
            numDerivatives(), numGPs(), sgStride, displacementView,
            cauchyStressFunction->multiPatchDeviceView(), gpTable(),
            dispValuesAndDerssView(), m_sgGPData.vectorView(),
            nodalWeights.vectorView());
        err = cudaDeviceSynchronize();
        assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUStrainGradientElasticityAssembler::constructStrainGradientStressFunctions Cauchy recovery");

        cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
            normalizeRecoveredFirstPiolaStressKernel, 0, totalCauchyEntries);
        gridSize = (totalCauchyEntries + blockSize - 1) / blockSize;
        normalizeRecoveredFirstPiolaStressKernel<<<gridSize, blockSize>>>(
            cauchyStressFunction->multiPatchDeviceView(),
            nodalWeights.vectorView(), totalControlPoints);
        err = cudaDeviceSynchronize();
        assert(err == cudaSuccess && "cudaDeviceSynchronize failed in GPUStrainGradientElasticityAssembler::constructStrainGradientStressFunctions Cauchy normalize");
    }
}

// Assembly flow:
//   1. Rebuild the current displacement control points from free + fixed dofs.
//   2. Evaluate and cache expensive Gauss-point data once.
//   3. Choose the fixed-dof vector used for eliminated columns.
//   4. Launch element-owned matrix and RHS kernels that accumulate GP
//      contributions through shared memory.
//
// The precomputed GP table and basis derivative arrays are owned by GPUAssembler.
void GPUStrainGradientElasticityAssembler::assemble(
    const DeviceVectorView<double>& solVector,
    int numIter,
    const DeviceNestedArrayView<double>& fixedDoFs)
{
    setMatrixAndRHSZeros();
    constructDispSolution(solVector, fixedDoFs);

    DeviceNestedArrayView<double> fixedDofsAssemble;
    getFixedDofsForAssemble(numIter, fixedDofsAssemble);

    const double youngsModulus = options().getReal("youngs_modulus");
    const double poissonsRatio = options().getReal("poissons_ratio");
    const double lengthScale = options().getReal("length_scale");
    const double forceScaling = options().getReal("force_scaling");
    const double localStiffening = options().getReal("local_stiffening");
    const int materialLaw = options().getInt("material_law");

    const int sgStride = strainGradientDoublesPerGP(domainDim());
    const int sgDataSize = sgStride * numGPs();
    if (m_sgGPData.size() != sgDataSize)
        m_sgGPData.resize(sgDataSize);
    else
        m_sgGPData.setZero();

    int minGrid = 0;
    int blockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        evaluateStrainGradientGPKernel, 0, numGPs());
    int gridSize = (numGPs() + blockSize - 1) / blockSize;
    evaluateStrainGradientGPKernel<<<gridSize, blockSize>>>(
        numDerivatives(), numGPs(), sgStride, materialLaw, youngsModulus,
        poissonsRatio, lengthScale, localStiffening, displacementView(),
        geometryView(), gpTable(), wts().vectorView(), geoValuesAndDerssView(),
        dispValuesAndDerssView(), m_sgGPData.vectorView());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after evaluateStrainGradientGPKernel launch: "
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during evaluateStrainGradientGPKernel: "
                  << cudaGetErrorString(err) << std::endl;

    int blockSize_assembleMatrix = N_D();
    const int localMatrixEntries = domainDim() * domainDim();
    const size_t matrixSharedBytes = localMatrixEntries * sizeof(double);
    const int gridSize_assembleMatrix = N_D() * N_D() * numElements();
    assembleStrainGradientMatrixKernel<<<gridSize_assembleMatrix, blockSize_assembleMatrix, matrixSharedBytes>>>(
        numDerivatives(), numElements(), N_D(), sgStride, displacementView(),
        geometryView(), sparseSystemDeviceView(), fixedDofsAssemble, gpTable(),
        dispValuesAndDerssView(), m_sgGPData.vectorView());
    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after assembleStrainGradientMatrixKernel launch: "
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during assembleStrainGradientMatrixKernel: "
                  << cudaGetErrorString(err) << std::endl;

    int blockSize_assembleRHS = N_D();
    const int localRHSEntries = domainDim();
    const size_t rhsSharedBytes = localRHSEntries * sizeof(double);
    const int gridSize_assembleRHS = N_D() * numElements();
    assembleStrainGradientRHSKernel<<<gridSize_assembleRHS, blockSize_assembleRHS, rhsSharedBytes>>>(
        numDerivatives(), numElements(), N_D(), sgStride, forceScaling,
        displacementView(), geometryView(), sparseSystemDeviceView(), gpTable(),
        dispValuesAndDerssView(), bodyForce().vectorView(), m_sgGPData.vectorView());
    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error after assembleStrainGradientRHSKernel launch: "
                  << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA error during assembleStrainGradientRHSKernel: "
                  << cudaGetErrorString(err) << std::endl;

    assembleNeumannBoundaryCondition();
    assembleNeumannCornerPointLoads();
}
