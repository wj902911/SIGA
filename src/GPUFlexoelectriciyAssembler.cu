#include "GPUFlexoelectriciyAssembler.h"

#include <Utility_d.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{

bool flexoEnvFlag(const char* name, bool defaultValue)
{
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0')
        return defaultValue;
    return value[0] != '0' && value[0] != 'f' && value[0] != 'F' &&
           value[0] != 'n' && value[0] != 'N';
}

struct FlexoGPOffsets
{
    int geoInvOffset = 0;
    int weightForceOffset = 0;
    int weightBodyOffset = 0;
    int geoHessOffset = 0;
    int FOffset = 0;
    int gradFOffset = 0;
    int electricFieldOffset = 0;
    int SOffset = 0;
    int SgradOffset = 0;
    int DOffset = 0;
    int AOffset = 0;
    int SGradGradOffset = 0;
    int SEgradOffset = 0;
    int SgradEOffset = 0;
    int SElectricOffset = 0;
    int SgradElectricOffset = 0;
    int DEOffset = 0;
    int DGradOffset = 0;
    int KOffset = 0;
    int total = 0;
};

__host__ __device__
FlexoGPOffsets flexoMakeGPOffsets(int dim)
{
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    FlexoGPOffsets offsets;
    int offset = 0;
    offsets.geoInvOffset = offset; offset += dim2;
    offsets.weightForceOffset = offset; offset += 1;
    offsets.weightBodyOffset = offset; offset += 1;
    offsets.geoHessOffset = offset; offset += dim3;
    offsets.FOffset = offset; offset += dim2;
    offsets.gradFOffset = offset; offset += dim3;
    offsets.electricFieldOffset = offset; offset += dim;
    offsets.SOffset = offset; offset += dim2;
    offsets.SgradOffset = offset; offset += dim3;
    offsets.DOffset = offset; offset += dim;
    offsets.AOffset = offset; offset += dim2 * dim2;
    offsets.SGradGradOffset = offset; offset += dim3 * dim3;
    offsets.SEgradOffset = offset; offset += dim2 * dim3;
    offsets.SgradEOffset = offset; offset += dim3 * dim2;
    offsets.SElectricOffset = offset; offset += dim2 * dim;
    offsets.SgradElectricOffset = offset; offset += dim3 * dim;
    offsets.DEOffset = offset; offset += dim * dim2;
    offsets.DGradOffset = offset; offset += dim * dim3;
    offsets.KOffset = offset; offset += dim * dim;
    offsets.total = offset;
    return offsets;
}

__host__ __device__
int flexoDoublesPerGP(int dim)
{
    return flexoMakeGPOffsets(dim).total;
}

struct FlexoBasisOffsets
{
    int dispGradOffset = 0;
    int dispHessOffset = 0;
    int elecGradOffset = 0;
    int total = 0;
};

__host__ __device__
FlexoBasisOffsets flexoMakeBasisOffsets(int dim)
{
    const int dim2 = dim * dim;
    FlexoBasisOffsets offsets;
    int offset = 0;
    offsets.dispGradOffset = offset; offset += dim;
    offsets.dispHessOffset = offset; offset += dim2;
    offsets.elecGradOffset = offset; offset += dim;
    offsets.total = offset;
    return offsets;
}

__host__ __device__
int flexoDoublesPerGPBasis(int dim)
{
    return flexoMakeBasisOffsets(dim).total;
}

__device__
double tensorBasisPartialFlexo(int r, int P1, int dim, int numDerivatives,
                               DeviceMatrixView<double> valuesAndDers,
                               int derivativeDir1, int derivativeDir2 = -1)
{
    int tensorCoordData[3] = {0};
    getTensorCoordinate(dim, P1, r, tensorCoordData);

    double value = 1.0;
    for (int d = 0; d < dim; ++d)
    {
        int order = 0;
        if (d == derivativeDir1)
            ++order;
        if (d == derivativeDir2)
            ++order;
        value *= valuesAndDers(tensorCoordData[d], (numDerivatives + 1) * d + order);
    }
    return value;
}

__device__
double tensorBasisValueFlexo(int r, int P1, int dim, int numDerivatives,
                             DeviceMatrixView<double> valuesAndDers)
{
    return tensorBasisPartialFlexo(r, P1, dim, numDerivatives, valuesAndDers, -1);
}

__device__
void tensorBasisGradientFlexo(int r, int P1, int dim, int numDerivatives,
                              DeviceMatrixView<double> valuesAndDers,
                              double* grad)
{
    for (int a = 0; a < dim; ++a)
        grad[a] = tensorBasisPartialFlexo(r, P1, dim, numDerivatives, valuesAndDers, a);
}

__device__
void tensorBasisHessianParamFlexo(int r, int P1, int dim, int numDerivatives,
                                  DeviceMatrixView<double> valuesAndDers,
                                  double* hessian)
{
    for (int a = 0; a < dim; ++a)
        for (int b = 0; b < dim; ++b)
            hessian[a * dim + b] =
                tensorBasisPartialFlexo(r, P1, dim, numDerivatives, valuesAndDers, a, b);
}

__device__
void patchParamHessianFlexo(PatchDeviceView patch, DeviceVectorView<double> pt,
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
        tensorBasisHessianParamFlexo(r, P1, dim, numDerivatives, valuesAndDers,
                                     basisHessian);
        const double cp = patch.activeControlPointComponent(pt, r, component);
        for (int a = 0; a < dim * dim; ++a)
            hessian[a] += cp * basisHessian[a];
    }
}

__device__
void physicalBasisHessianFlexo(int r, int P1, int dim, int numDerivatives,
                               DeviceMatrixView<double> basisValuesAndDers,
                               double* geoHessians,
                               DeviceMatrixView<double> geoJacobianInv,
                               double* result)
{
    double gradParam[3] = {0.0};
    double hessParam[9] = {0.0};
    tensorBasisGradientFlexo(r, P1, dim, numDerivatives, basisValuesAndDers,
                             gradParam);
    tensorBasisHessianParamFlexo(r, P1, dim, numDerivatives, basisValuesAndDers,
                                 hessParam);

    double corrected[9] = {0.0};
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
        {
            double correction = 0.0;
            for (int k = 0; k < dim; ++k)
                for (int c = 0; c < dim; ++c)
                    correction += gradParam[k] * geoJacobianInv(k, c) *
                                  geoHessians[c * dim * dim + i * dim + j];
            corrected[i * dim + j] = hessParam[i * dim + j] - correction;
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

__device__
void buildPhysicalGradientAndHessianFlexo(int basisIndex, int P1, int dim,
                                          int numDerivatives,
                                          DeviceMatrixView<double> valuesAndDers,
                                          double* geoHessians,
                                          DeviceMatrixView<double> geoJacobianInv,
                                          double* gradPhys,
                                          double* hessianPhys)
{
    double gradParamData[3] = {0.0};
    DeviceVectorView<double> gradParam(gradParamData, dim);
    DeviceVectorView<double> gradPhysView(gradPhys, dim);
    tensorBasisDerivative(basisIndex, P1, dim, numDerivatives, valuesAndDers,
                          gradParam);
    geoJacobianInv.transposeTime(gradParam, gradPhysView);

    physicalBasisHessianFlexo(basisIndex, P1, dim, numDerivatives, valuesAndDers,
                              geoHessians, geoJacobianInv, hessianPhys);
}

__device__
void buildPhysicalGradientFlexo(int basisIndex, int P1, int dim,
                                 int numDerivatives,
                                 DeviceMatrixView<double> valuesAndDers,
                                 DeviceMatrixView<double> geoJacobianInv,
                                 double* gradPhys)
{
    double gradParamData[3] = {0.0};
    DeviceVectorView<double> gradParam(gradParamData, dim);
    DeviceVectorView<double> gradPhysView(gradPhys, dim);
    tensorBasisDerivative(basisIndex, P1, dim, numDerivatives, valuesAndDers,
                          gradParam);
    geoJacobianInv.transposeTime(gradParam, gradPhysView);
}

__device__
void physicalFieldHessiansFlexo(PatchDeviceView fieldPatch,
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
        physicalBasisHessianFlexo(r, P1, dim, numDerivatives, fieldValuesAndDers,
                                  geoHessians, geoJacobianInv, basisHessian);
        for (int comp = 0; comp < dim; ++comp)
        {
            const double cp = fieldPatch.activeControlPointComponent(pt, r, comp);
            for (int a = 0; a < dim * dim; ++a)
                result[comp * dim * dim + a] += cp * basisHessian[a];
        }
    }
}

template <typename Scalar>
__device__
void computeCAndGreenGradient(int dim, const Scalar* F, const Scalar* gradF,
                              Scalar* C, Scalar* Egrad)
{
    const int dim2 = dim * dim;
    for (int I = 0; I < dim; ++I)
        for (int J = 0; J < dim; ++J)
        {
            Scalar value = 0.0;
            for (int a = 0; a < dim; ++a)
                value += F[a * dim + I] * F[a * dim + J];
            C[I * dim + J] = value;
        }

    for (int I = 0; I < dim; ++I)
        for (int J = 0; J < dim; ++J)
            for (int K = 0; K < dim; ++K)
            {
                Scalar value = 0.0;
                for (int a = 0; a < dim; ++a)
                    value += gradF[a * dim2 + I * dim + K] * F[a * dim + J] +
                             F[a * dim + I] * gradF[a * dim2 + J * dim + K];
                Egrad[(I * dim + J) * dim + K] = 0.5 * value;
            }
}

__device__
double flexoMuContract(double muL, double muT, double muS,
                       int L, int I, int J, int K)
{
    if (L == I && I == J && J == K)
        return muL;
    if (I == J && K == L && I != K)
        return muT;
    if ((L == I && J == K && L != J) ||
        (L == J && I == K && L != I))
        return muS;
    return 0.0;
}

template <typename Scalar>
__device__
Scalar flexoDeterminant(int dim, const Scalar* A);

template <typename Scalar>
__device__
void flexoInverse(int dim, const Scalar* A, Scalar* AInv);

__device__
double flexoEnergy(int materialLaw, double youngsModulus, double poissonsRatio,
                   double lengthScale, double dielectricPermittivity,
                   double vacuumPermittivity, double muL, double muT,
                   double muS, int dim, const double* F,
                   const double* gradF, const double* electricField)
{
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    const double lambda = youngsModulus * poissonsRatio /
        ((1.0 + poissonsRatio) * (1.0 - 2.0 * poissonsRatio));
    const double shearModulus = youngsModulus / (2.0 * (1.0 + poissonsRatio));

    double CData[9] = {0.0};
    double CInvData[9] = {0.0};
    double EgradData[27] = {0.0};
    computeCAndGreenGradient(dim, F, gradF, CData, EgradData);
    flexoInverse(dim, CData, CInvData);
    const double J = flexoDeterminant(dim, F);

    double energy = 0.0;
    if (materialLaw == 0)
    {
        double traceE = 0.0;
        double E2 = 0.0;
        for (int I = 0; I < dim; ++I)
            for (int Jidx = 0; Jidx < dim; ++Jidx)
            {
                const double green =
                    0.5 * (CData[I * dim + Jidx] - (I == Jidx ? 1.0 : 0.0));
                if (I == Jidx)
                    traceE += green;
                E2 += green * green;
            }
        energy += 0.5 * lambda * traceE * traceE + shearModulus * E2;
    }
    else
    {
        double traceC = 0.0;
        for (int I = 0; I < dim; ++I)
            traceC += CData[I * dim + I];
        energy += 0.5 * lambda * log(J) * log(J) +
                  0.5 * shearModulus * (traceC - dim - 2.0 * log(J));
    }

    double cInvE[3] = {0.0};
    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
            cInvE[A] += CInvData[A * dim + B] * electricField[B];

    double eCe = 0.0;
    for (int A = 0; A < dim; ++A)
        eCe += electricField[A] * cInvE[A];
    energy -= 0.5 * dielectricPermittivity * J * eCe;

    const double lengthScale2 = lengthScale * lengthScale;
    const double dielectricContrast = dielectricPermittivity - vacuumPermittivity;
    const bool hasHbarCorrection = fabs(dielectricContrast) > 1.0e-14;
    for (int I = 0; I < dim; ++I)
        for (int Jidx = 0; Jidx < dim; ++Jidx)
            for (int K = 0; K < dim; ++K)
            {
                const int row = (I * dim + Jidx) * dim + K;
                for (int L = 0; L < dim; ++L)
                    for (int M = 0; M < dim; ++M)
                        for (int N = 0; N < dim; ++N)
                        {
                            const int col = (L * dim + M) * dim + N;
                            double hbar =
                                (lambda * (I == Jidx ? 1.0 : 0.0) *
                                          (L == M ? 1.0 : 0.0) +
                                 2.0 * shearModulus * (I == L ? 1.0 : 0.0) *
                                                       (Jidx == M ? 1.0 : 0.0)) *
                                lengthScale2 * (K == N ? 1.0 : 0.0);
                            if (hasHbarCorrection)
                            {
                                double flexoCorrection = 0.0;
                                for (int A = 0; A < dim; ++A)
                                    for (int B = 0; B < dim; ++B)
                                        flexoCorrection +=
                                            flexoMuContract(muL, muT, muS, A, I, Jidx, K) *
                                            J * CInvData[A * dim + B] *
                                            flexoMuContract(muL, muT, muS, B, L, M, N);
                                hbar -= flexoCorrection / dielectricContrast;
                            }
                            energy += 0.5 * EgradData[row] * hbar * EgradData[col];
                        }
            }

    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
            for (int I = 0; I < dim; ++I)
                for (int Jidx = 0; Jidx < dim; ++Jidx)
                    for (int K = 0; K < dim; ++K)
                    {
                        const double muBIJK =
                            flexoMuContract(muL, muT, muS, B, I, Jidx, K);
                        energy -= J * electricField[A] *
                                  CInvData[A * dim + B] * muBIJK *
                                  EgradData[(I * dim + Jidx) * dim + K];
                    }

    return energy;
}

struct FlexoDual
{
    double value;
    double derivative;

    __host__ __device__
    FlexoDual(double v = 0.0, double d = 0.0) : value(v), derivative(d) {}
};

__host__ __device__
FlexoDual operator+(FlexoDual a, FlexoDual b)
{ return FlexoDual(a.value + b.value, a.derivative + b.derivative); }

__host__ __device__
FlexoDual operator-(FlexoDual a, FlexoDual b)
{ return FlexoDual(a.value - b.value, a.derivative - b.derivative); }

__host__ __device__
FlexoDual operator-(FlexoDual a)
{ return FlexoDual(-a.value, -a.derivative); }

__host__ __device__
FlexoDual operator*(FlexoDual a, FlexoDual b)
{
    return FlexoDual(a.value * b.value,
                     a.derivative * b.value + a.value * b.derivative);
}

__host__ __device__
FlexoDual operator/(FlexoDual a, FlexoDual b)
{
    const double inv = 1.0 / b.value;
    return FlexoDual(a.value * inv,
                     (a.derivative * b.value - a.value * b.derivative) *
                     inv * inv);
}

__host__ __device__
FlexoDual& operator+=(FlexoDual& a, FlexoDual b)
{
    a.value += b.value;
    a.derivative += b.derivative;
    return a;
}

__host__ __device__
FlexoDual& operator-=(FlexoDual& a, FlexoDual b)
{
    a.value -= b.value;
    a.derivative -= b.derivative;
    return a;
}

__host__ __device__
FlexoDual flexoLog(FlexoDual a)
{ return FlexoDual(log(a.value), a.derivative / a.value); }

__host__ __device__
double flexoLog(double a)
{ return log(a); }

__host__ __device__
FlexoDual flexoSqrt(FlexoDual a)
{
    const double root = ::sqrt(a.value);
    return FlexoDual(root, 0.5 * a.derivative / root);
}

__host__ __device__
double flexoSqrt(double a)
{ return ::sqrt(a); }

template <typename Scalar>
__device__
Scalar flexoDeterminant(int dim, const Scalar* A)
{
    if (dim == 2)
        return A[0] * A[3] - A[1] * A[2];
    return A[0] * (A[4] * A[8] - A[5] * A[7]) -
           A[1] * (A[3] * A[8] - A[5] * A[6]) +
           A[2] * (A[3] * A[7] - A[4] * A[6]);
}

template <typename Scalar>
__device__
void flexoInverse(int dim, const Scalar* A, Scalar* AInv)
{
    const Scalar det = flexoDeterminant(dim, A);
    if (dim == 2)
    {
        AInv[0] = A[3] / det;
        AInv[1] = -A[1] / det;
        AInv[2] = -A[2] / det;
        AInv[3] = A[0] / det;
        return;
    }

    AInv[0] = (A[4] * A[8] - A[5] * A[7]) / det;
    AInv[1] = (A[2] * A[7] - A[1] * A[8]) / det;
    AInv[2] = (A[1] * A[5] - A[2] * A[4]) / det;
    AInv[3] = (A[5] * A[6] - A[3] * A[8]) / det;
    AInv[4] = (A[0] * A[8] - A[2] * A[6]) / det;
    AInv[5] = (A[2] * A[3] - A[0] * A[5]) / det;
    AInv[6] = (A[3] * A[7] - A[4] * A[6]) / det;
    AInv[7] = (A[1] * A[6] - A[0] * A[7]) / det;
    AInv[8] = (A[0] * A[4] - A[1] * A[3]) / det;
}

template <typename Scalar>
__device__
void flexoMaterialResponse(int materialLaw, double youngsModulus,
                           double poissonsRatio, double lengthScale,
                           double dielectricPermittivity,
                           double vacuumPermittivity, double muL,
                           double muT, double muS, int dim, const Scalar* F,
                           const Scalar* gradF, const Scalar* electricField,
                           bool includeMechanical, Scalar* dPsi_dF,
                           Scalar* dPsi_dGradF,
                           Scalar* electricDisplacement)
{
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    const double lambda = youngsModulus * poissonsRatio /
        ((1.0 + poissonsRatio) * (1.0 - 2.0 * poissonsRatio));
    const double shearModulus = youngsModulus / (2.0 * (1.0 + poissonsRatio));

    // Material kinematics used by the derivation:
    // C = F^T F, Egrad = Grad_X Green strain, and J = det(F).
    Scalar C[9] = {0.0};
    Scalar CInv[9] = {0.0};
    Scalar FInv[9] = {0.0};
    Scalar Egrad[27] = {0.0};
    computeCAndGreenGradient(dim, F, gradF, C, Egrad);
    flexoInverse(dim, C, CInv);
    flexoInverse(dim, F, FInv);
    const Scalar J = flexoDeterminant(dim, F);

    for (int i = 0; i < dim2; ++i)
        dPsi_dF[i] = 0.0;
    for (int i = 0; i < dim3; ++i)
        dPsi_dGradF[i] = 0.0;
    for (int i = 0; i < dim; ++i)
        electricDisplacement[i] = 0.0;

    // Mechanical stress contribution dPsi_mech/dF.
    // materialLaw == 0 uses StVK through S = dPsi/dE and P = F S;
    // otherwise use the same compressible Neo-Hookean stress as GPUAssembler.
    if (includeMechanical && materialLaw == 0)
    {
        Scalar green[9] = {0.0};
        Scalar traceE = 0.0;
        for (int A = 0; A < dim; ++A)
            for (int B = 0; B < dim; ++B)
            {
                green[A * dim + B] =
                    0.5 * (C[A * dim + B] - (A == B ? 1.0 : 0.0));
                if (A == B)
                    traceE = traceE + green[A * dim + B];
            }

        Scalar secondPK[9] = {0.0};
        for (int A = 0; A < dim; ++A)
            for (int B = 0; B < dim; ++B)
                secondPK[A * dim + B] =
                    lambda * traceE * (A == B ? 1.0 : 0.0) +
                    2.0 * shearModulus * green[A * dim + B];

        for (int a = 0; a < dim; ++a)
            for (int A = 0; A < dim; ++A)
                for (int B = 0; B < dim; ++B)
                    dPsi_dF[a * dim + A] =
                        dPsi_dF[a * dim + A] + F[a * dim + B] *
                        secondPK[B * dim + A];
    }
    else if (includeMechanical)
    {
        Scalar secondPK[9] = {0.0};
        const Scalar pressureTerm =
            lambda * (J * J - 1.0) * 0.5 - shearModulus;
        for (int A = 0; A < dim; ++A)
            for (int B = 0; B < dim; ++B)
                secondPK[A * dim + B] =
                    pressureTerm * CInv[A * dim + B] +
                    shearModulus * (A == B ? 1.0 : 0.0);

        for (int a = 0; a < dim; ++a)
            for (int A = 0; A < dim; ++A)
                for (int B = 0; B < dim; ++B)
                    dPsi_dF[a * dim + A] =
                        dPsi_dF[a * dim + A] +
                        F[a * dim + B] * secondPK[B * dim + A];
    }

    Scalar M[9] = {0.0};
    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
            M[A * dim + B] = J * CInv[A * dim + B];

    // Strain-gradient stress in Green-strain-gradient variables:
    // S_ABC = hbar_ABCDEF E_DEF - E_A (J C^{-1})_AB mu_BCDE.
    // hbar contains the length-scale elasticity and the eliminated
    // polarization correction mu J C^{-1} mu / (epsilon - epsilon0).
    const double dielectricContrast = dielectricPermittivity - vacuumPermittivity;
    const bool hasHbarCorrection = fabs(dielectricContrast) > 1.0e-14;
    Scalar strainGradientStress[27] = {0.0};

    for (int I = 0; I < dim; ++I)
        for (int Jidx = 0; Jidx < dim; ++Jidx)
            for (int K = 0; K < dim; ++K)
            {
                const int row = (I * dim + Jidx) * dim + K;
                for (int L = 0; L < dim; ++L)
                    for (int Midx = 0; Midx < dim; ++Midx)
                        for (int N = 0; N < dim; ++N)
                        {
                            const int col = (L * dim + Midx) * dim + N;
                            Scalar hbar =
                                (lambda * (I == Jidx ? 1.0 : 0.0) *
                                          (L == Midx ? 1.0 : 0.0) +
                                 2.0 * shearModulus * (I == L ? 1.0 : 0.0) *
                                                       (Jidx == Midx ? 1.0 : 0.0)) *
                                lengthScale * lengthScale *
                                (K == N ? 1.0 : 0.0);
                            if (hasHbarCorrection)
                            {
                                Scalar correction = 0.0;
                                for (int A = 0; A < dim; ++A)
                                    for (int B = 0; B < dim; ++B)
                                        correction = correction +
                                            flexoMuContract(muL, muT, muS, A, I, Jidx, K) *
                                            M[A * dim + B] *
                                            flexoMuContract(muL, muT, muS, B, L, Midx, N);
                                hbar = hbar - correction / dielectricContrast;
                            }
                            strainGradientStress[row] =
                                strainGradientStress[row] + hbar * Egrad[col];
                        }

                for (int A = 0; A < dim; ++A)
                    for (int B = 0; B < dim; ++B)
                        strainGradientStress[row] =
                            strainGradientStress[row] -
                            electricField[A] * M[A * dim + B] *
                            flexoMuContract(muL, muT, muS, B, I, Jidx, K);
            }

    // Derivative with respect to M = J C^{-1}. The stress split keeps the
    // hbar correction in the double stress only, so the local stress includes
    // only the dielectric and explicit flexoelectric M-dependence here.
    Scalar dPsi_dM[9] = {0.0};
    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
        {
            dPsi_dM[A * dim + B] =
                -0.5 * dielectricPermittivity * electricField[A] *
                electricField[B];
            for (int I = 0; I < dim; ++I)
                for (int Jidx = 0; Jidx < dim; ++Jidx)
                    for (int K = 0; K < dim; ++K)
                    {
                        const int row = (I * dim + Jidx) * dim + K;
                        dPsi_dM[A * dim + B] =
                            dPsi_dM[A * dim + B] -
                            electricField[A] *
                            flexoMuContract(muL, muT, muS, B, I, Jidx, K) *
                            Egrad[row];
                    }
        }

    Scalar trMC = 0.0;
    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
            trMC = trMC + dPsi_dM[A * dim + B] * CInv[A * dim + B];

    // Push dPsi/dM back to dPsi/dF through M(F) = J C^{-1}.
    for (int a = 0; a < dim; ++a)
        for (int R = 0; R < dim; ++R)
        {
            Scalar value = J * FInv[R * dim + a] * trMC;
            for (int A = 0; A < dim; ++A)
                for (int B = 0; B < dim; ++B)
                    for (int S = 0; S < dim; ++S)
                    {
                        value = value -
                            J * dPsi_dM[A * dim + B] *
                            CInv[A * dim + R] * F[a * dim + S] *
                            CInv[S * dim + B];
                        value = value -
                            J * dPsi_dM[A * dim + B] *
                            CInv[A * dim + S] * F[a * dim + S] *
                            CInv[R * dim + B];
                    }
            dPsi_dF[a * dim + R] = dPsi_dF[a * dim + R] + value;
        }

    // Push the strain-gradient stress back through E_ABC(F, GradF):
    // this is the part of dPsi/dF caused by Egrad depending on F.
    for (int a = 0; a < dim; ++a)
        for (int A = 0; A < dim; ++A)
            for (int I = 0; I < dim; ++I)
                for (int K = 0; K < dim; ++K)
                    dPsi_dF[a * dim + A] =
                        dPsi_dF[a * dim + A] +
                        0.5 * strainGradientStress[I * dim2 + A * dim + K] *
                        gradF[a * dim2 + I * dim + K] +
                        0.5 * strainGradientStress[A * dim2 + I * dim + K] *
                        gradF[a * dim2 + I * dim + K];

    // Hyperstress contribution Q = dPsi/dGradF from E_ABC(F, GradF).
    for (int a = 0; a < dim; ++a)
        for (int A = 0; A < dim; ++A)
            for (int B = 0; B < dim; ++B)
            {
                Scalar value = 0.0;
                for (int Jidx = 0; Jidx < dim; ++Jidx)
                    value = value +
                        0.5 * strainGradientStress[A * dim2 + Jidx * dim + B] *
                        F[a * dim + Jidx];
                for (int I = 0; I < dim; ++I)
                    value = value +
                        0.5 * strainGradientStress[I * dim2 + A * dim + B] *
                        F[a * dim + I];
                dPsi_dGradF[a * dim2 + A * dim + B] = value;
            }

    // Material electric displacement D = -dPsi/dE_field.
    // It contains the dielectric term epsilon J C^{-1} E and the
    // flexoelectric polarization term J C^{-1} mu Egrad.
    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
        {
            electricDisplacement[A] =
                electricDisplacement[A] +
                dielectricPermittivity * M[A * dim + B] * electricField[B];
            for (int I = 0; I < dim; ++I)
                for (int Jidx = 0; Jidx < dim; ++Jidx)
                    for (int K = 0; K < dim; ++K)
                        electricDisplacement[A] =
                            electricDisplacement[A] +
                            M[A * dim + B] *
                            flexoMuContract(muL, muT, muS, B, I, Jidx, K) *
                            Egrad[(I * dim + Jidx) * dim + K];
        }
}

__device__
void flexoDerivatives(int materialLaw, double youngsModulus,
                      double poissonsRatio, double lengthScale,
                      double dielectricPermittivity, double vacuumPermittivity,
                      double muL, double muT, double muS,
                      int dim, const double* F, const double* gradF,
                      const double* electricField, double* dPsi_dF,
                      double* dPsi_dGradF, double* electricDisplacement)
{
    flexoMaterialResponse(materialLaw, youngsModulus, poissonsRatio,
        lengthScale, dielectricPermittivity, vacuumPermittivity, muL, muT,
        muS, dim, F, gradF, electricField, true, dPsi_dF, dPsi_dGradF,
        electricDisplacement);
}

__device__
void flexoDirectionalDerivatives(int materialLaw, double youngsModulus,
                                 double poissonsRatio, double lengthScale,
                                 double dielectricPermittivity,
                                 double vacuumPermittivity, double muL,
                                 double muT, double muS, int dim,
                                 const double* F, const double* gradF,
                                 const double* electricField,
                                 const double* deltaF,
                                 const double* deltaGradF,
                                 const double* deltaElectricField,
                                 double* deltaP, double* deltaQ,
                                 double* deltaD)
{
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    FlexoDual Fd[9];
    FlexoDual Gd[27];
    FlexoDual Ed[3];
    FlexoDual Pd[9];
    FlexoDual Qd[27];
    FlexoDual Dd[3];

    for (int i = 0; i < dim2; ++i)
        Fd[i] = FlexoDual(F[i], deltaF[i]);
    for (int i = 0; i < dim3; ++i)
        Gd[i] = FlexoDual(gradF[i], deltaGradF[i]);
    for (int i = 0; i < dim; ++i)
        Ed[i] = FlexoDual(electricField[i], deltaElectricField[i]);

    flexoMaterialResponse(materialLaw, youngsModulus, poissonsRatio,
        lengthScale, dielectricPermittivity, vacuumPermittivity, muL, muT,
        muS, dim, Fd, Gd, Ed, false, Pd, Qd, Dd);

    for (int i = 0; i < dim2; ++i)
        deltaP[i] = Pd[i].derivative;
    for (int i = 0; i < dim3; ++i)
        deltaQ[i] = Qd[i].derivative;
    for (int i = 0; i < dim; ++i)
        deltaD[i] = Dd[i].derivative;
}

__device__
double baseMechanicalTangentFlexo(int materialLaw, double youngsModulus,
                                  double poissonsRatio, int dim, int row,
                                  int col, const double* FData,
                                  const double* grad_i_data,
                                  const double* grad_j_data)
{
    const int dimTensor = dim * (dim + 1) / 2;
    const double lambda = youngsModulus * poissonsRatio /
        ((1.0 + poissonsRatio) * (1.0 - 2.0 * poissonsRatio));
    const double shearModulus = youngsModulus / (2.0 * (1.0 + poissonsRatio));

    double FViewData[9] = {0.0};
    for (int a = 0; a < dim; ++a)
        for (int A = 0; A < dim; ++A)
            FViewData[A * dim + a] = FData[a * dim + A];
    DeviceMatrixView<double> F(FViewData, dim, dim);
    DeviceVectorView<double> grad_i(const_cast<double*>(grad_i_data), dim);
    DeviceVectorView<double> grad_j(const_cast<double*>(grad_j_data), dim);

    double SData[9] = {0.0};
    double CData[36] = {0.0};
    DeviceMatrixView<double> S(SData, dim, dim);
    DeviceMatrixView<double> C(CData, dimTensor, dimTensor);

    if (materialLaw == 0)
    {
        double CmatData[9] = {0.0};
        double FTransData[9] = {0.0};
        DeviceMatrixView<double> Cmat(CmatData, dim, dim);
        DeviceMatrixView<double> FTrans(FTransData, dim, dim);
        F.transpose(FTrans);
        FTrans.times(F, Cmat);

        double traceE = 0.0;
        for (int A = 0; A < dim; ++A)
            traceE += 0.5 * (Cmat(A, A) - 1.0);
        for (int A = 0; A < dim; ++A)
            for (int B = 0; B < dim; ++B)
            {
                const double EAB = 0.5 * (Cmat(A, B) - (A == B ? 1.0 : 0.0));
                S(A, B) = lambda * traceE * (A == B ? 1.0 : 0.0) +
                          2.0 * shearModulus * EAB;
            }

        for (int a = 0; a < dimTensor; ++a)
        {
            const int A = voigt(dim, a, 0);
            const int B = voigt(dim, a, 1);
            for (int b = 0; b < dimTensor; ++b)
            {
                const int Cidx = voigt(dim, b, 0);
                const int D = voigt(dim, b, 1);
                C(a, b) = lambda * (A == B ? 1.0 : 0.0) *
                                  (Cidx == D ? 1.0 : 0.0) +
                          shearModulus *
                              ((A == Cidx && B == D ? 1.0 : 0.0) +
                               (A == D && B == Cidx ? 1.0 : 0.0));
            }
        }
    }
    else
    {
        const double J = F.determinant();
        double rightCauchyGreenData[9] = {0.0};
        double FTransData[9] = {0.0};
        double rightCauchyGreenInvData[9] = {0.0};
        DeviceMatrixView<double> rightCauchyGreen(rightCauchyGreenData, dim, dim);
        DeviceMatrixView<double> FTrans(FTransData, dim, dim);
        DeviceMatrixView<double> rightCauchyGreenInv(rightCauchyGreenInvData,
                                                     dim, dim);

        F.transpose(FTrans);
        FTrans.times(F, rightCauchyGreen);
        rightCauchyGreen.inverse(rightCauchyGreenInv);
        rightCauchyGreenInv.times(lambda * (J * J - 1.0) / 2.0 - shearModulus,
                                  S);
        S.tracePlus(shearModulus);

        matrixViewTraceTensor(C, rightCauchyGreenInv, rightCauchyGreenInv);
        C.times(lambda * J * J);
        double CtempData[36] = {0.0};
        DeviceMatrixView<double> Ctemp(CtempData, dimTensor, dimTensor);
        symmetricIdentityViewTensor(Ctemp, rightCauchyGreenInv);
        Ctemp.times(shearModulus - lambda * (J * J - 1.0) / 2.0);
        C.plus(Ctemp);
    }

    double geometricTangentTempData[3] = {0.0};
    DeviceVectorView<double> geometricTangentTemp(geometricTangentTempData, dim);
    S.times(grad_i, geometricTangentTemp);
    double tangent = row == col ? geometricTangentTemp.dot(grad_j) : 0.0;

    double B_i_data[6] = {0.0};
    double B_j_data[6] = {0.0};
    double materialTempData[6] = {0.0};
    double materialTangent = 0.0;
    DeviceVectorView<double> B_i(B_i_data, dimTensor);
    DeviceVectorView<double> B_j(B_j_data, dimTensor);
    DeviceMatrixView<double> B_i_trans(B_i_data, 1, dimTensor);
    DeviceMatrixView<double> materialTemp(materialTempData, 1, dimTensor);
    DeviceMatrixView<double> materialTangentMat(&materialTangent, 1, 1);

    setBSingleDim<double>(row, B_i, F, grad_i);
    B_i_trans.times(C, materialTemp);
    setBSingleDim<double>(col, B_j, F, grad_j);
    materialTemp.times(B_j, materialTangentMat);
    return tangent + materialTangent;
}

__device__
double displacementResidualFlexo(int materialLaw, double youngsModulus,
                                 double poissonsRatio, double lengthScale,
                                 double dielectricPermittivity,
                                 double vacuumPermittivity, double muL,
                                 double muT, double muS, int dim, int component,
                                 const double* grad_i, const double* hess_i,
                                 const double* F, const double* gradF,
                                 const double* electricField)
{
    double dPsi_dF[9] = {0.0};
    double dPsi_dGradF[27] = {0.0};
    double D[3] = {0.0};
    flexoDerivatives(materialLaw, youngsModulus, poissonsRatio, lengthScale,
                     dielectricPermittivity, vacuumPermittivity, muL, muT, muS,
                     dim, F, gradF,
                     electricField, dPsi_dF, dPsi_dGradF, D);

    const int dim2 = dim * dim;
    double residual = 0.0;
    for (int A = 0; A < dim; ++A)
        residual += grad_i[A] * dPsi_dF[component * dim + A];
    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
            residual += hess_i[A * dim + B] *
                        dPsi_dGradF[component * dim2 + A * dim + B];
    return residual;
}

__device__
double potentialResidualFlexo(int materialLaw, double youngsModulus,
                              double poissonsRatio, double lengthScale,
                              double dielectricPermittivity,
                              double vacuumPermittivity, double muL,
                              double muT, double muS, int dim,
                              const double* gradPotential_i,
                              const double* F, const double* gradF,
                              const double* electricField)
{
    double dPsi_dF[9] = {0.0};
    double dPsi_dGradF[27] = {0.0};
    double D[3] = {0.0};
    flexoDerivatives(materialLaw, youngsModulus, poissonsRatio, lengthScale,
                     dielectricPermittivity, vacuumPermittivity, muL, muT, muS,
                     dim, F, gradF,
                     electricField, dPsi_dF, dPsi_dGradF, D);

    double residual = 0.0;
    for (int A = 0; A < dim; ++A)
        residual += gradPotential_i[A] * D[A];
    return residual;
}

template <typename Scalar>
__device__
void flexoGreenMaterialResponse(int materialLaw, double youngsModulus,
                                double poissonsRatio, double lengthScale,
                                double dielectricPermittivity,
                                double vacuumPermittivity, double muL,
                                double muT, double muS,
                                int includeHbarFlexoCorrection, int dim,
                                const Scalar* C, const Scalar* Egrad,
                                const Scalar* electricField,
                                Scalar* secondPK,
                                Scalar* strainGradientStress,
                                Scalar* electricDisplacement)
{
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    const double lambda = youngsModulus * poissonsRatio /
        ((1.0 + poissonsRatio) * (1.0 - 2.0 * poissonsRatio));
    const double shearModulus = youngsModulus / (2.0 * (1.0 + poissonsRatio));

    Scalar CInv[9] = {0.0};
    flexoInverse(dim, C, CInv);
    const Scalar J = flexoSqrt(flexoDeterminant(dim, C));

    for (int i = 0; i < dim2; ++i)
        secondPK[i] = 0.0;
    for (int i = 0; i < dim3; ++i)
        strainGradientStress[i] = 0.0;
    for (int i = 0; i < dim; ++i)
        electricDisplacement[i] = 0.0;

    if (materialLaw == 0)
    {
        Scalar traceE = 0.0;
        for (int A = 0; A < dim; ++A)
            traceE = traceE + 0.5 * (C[A * dim + A] - 1.0);
        for (int A = 0; A < dim; ++A)
            for (int B = 0; B < dim; ++B)
            {
                const Scalar green =
                    0.5 * (C[A * dim + B] - (A == B ? 1.0 : 0.0));
                secondPK[A * dim + B] =
                    lambda * traceE * (A == B ? 1.0 : 0.0) +
                    2.0 * shearModulus * green;
            }
    }
    else
    {
        const Scalar logJ = flexoLog(J);
        for (int A = 0; A < dim; ++A)
            for (int B = 0; B < dim; ++B)
                secondPK[A * dim + B] =
                    lambda * logJ * CInv[A * dim + B] +
                    shearModulus * ((A == B ? 1.0 : 0.0) -
                                    CInv[A * dim + B]);
    }

    Scalar M[9] = {0.0};
    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
            M[A * dim + B] = J * CInv[A * dim + B];

    const double dielectricContrast = dielectricPermittivity - vacuumPermittivity;
    const bool hasHbarCorrection =
        includeHbarFlexoCorrection && fabs(dielectricContrast) > 1.0e-14;
    for (int I = 0; I < dim; ++I)
        for (int Jidx = 0; Jidx < dim; ++Jidx)
            for (int K = 0; K < dim; ++K)
            {
                const int row = (I * dim + Jidx) * dim + K;
                for (int L = 0; L < dim; ++L)
                    for (int Midx = 0; Midx < dim; ++Midx)
                        for (int N = 0; N < dim; ++N)
                        {
                            const int col = (L * dim + Midx) * dim + N;
                            Scalar hbar =
                                (lambda * (I == Jidx ? 1.0 : 0.0) *
                                          (L == Midx ? 1.0 : 0.0) +
                                 2.0 * shearModulus * (I == L ? 1.0 : 0.0) *
                                                       (Jidx == Midx ? 1.0 : 0.0)) *
                                lengthScale * lengthScale *
                                (K == N ? 1.0 : 0.0);
                            if (hasHbarCorrection)
                            {
                                Scalar correction = 0.0;
                                for (int A = 0; A < dim; ++A)
                                    for (int B = 0; B < dim; ++B)
                                        correction = correction +
                                            flexoMuContract(muL, muT, muS, A, I, Jidx, K) *
                                            M[A * dim + B] *
                                            flexoMuContract(muL, muT, muS, B, L, Midx, N);
                                hbar = hbar - correction / dielectricContrast;
                            }
                            strainGradientStress[row] =
                                strainGradientStress[row] + hbar * Egrad[col];
                        }

                for (int A = 0; A < dim; ++A)
                    for (int B = 0; B < dim; ++B)
                        strainGradientStress[row] =
                            strainGradientStress[row] -
                            electricField[A] * M[A * dim + B] *
                            flexoMuContract(muL, muT, muS, B, I, Jidx, K);
            }

    Scalar dPsi_dM[9] = {0.0};
    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
        {
            dPsi_dM[A * dim + B] =
                -0.5 * dielectricPermittivity * electricField[A] *
                electricField[B];
            for (int I = 0; I < dim; ++I)
                for (int Jidx = 0; Jidx < dim; ++Jidx)
                    for (int K = 0; K < dim; ++K)
                    {
                        const int row = (I * dim + Jidx) * dim + K;
                        dPsi_dM[A * dim + B] =
                            dPsi_dM[A * dim + B] -
                            electricField[A] *
                            flexoMuContract(muL, muT, muS, B, I, Jidx, K) *
                            Egrad[row];
                    }
        }

    Scalar trMC = 0.0;
    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
            trMC = trMC + dPsi_dM[A * dim + B] * CInv[A * dim + B];
    for (int R = 0; R < dim; ++R)
        for (int S = 0; S < dim; ++S)
        {
            Scalar value = J * CInv[S * dim + R] * trMC;
            for (int A = 0; A < dim; ++A)
                for (int B = 0; B < dim; ++B)
                    value = value - 2.0 * J * dPsi_dM[A * dim + B] *
                        CInv[A * dim + R] * CInv[S * dim + B];
            secondPK[R * dim + S] = secondPK[R * dim + S] + value;
        }

    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
        {
            electricDisplacement[A] =
                electricDisplacement[A] +
                dielectricPermittivity * M[A * dim + B] * electricField[B];
            for (int I = 0; I < dim; ++I)
                for (int Jidx = 0; Jidx < dim; ++Jidx)
                    for (int K = 0; K < dim; ++K)
                        electricDisplacement[A] =
                            electricDisplacement[A] +
                            M[A * dim + B] *
                            flexoMuContract(muL, muT, muS, B, I, Jidx, K) *
                            Egrad[(I * dim + Jidx) * dim + K];
        }
}

__device__
double flexoDMinvDE(int dim, double J, const double* CInv,
                    int A, int B, int I, int Jidx)
{
    return J * CInv[Jidx * dim + I] * CInv[A * dim + B] -
           2.0 * J * CInv[A * dim + I] * CInv[Jidx * dim + B];
}

__device__
double flexoDCInvDE(int dim, const double* CInv, int P, int Q, int K, int L)
{
    return -2.0 * CInv[P * dim + K] * CInv[L * dim + Q];
}

__device__
double flexoD2MinvDE2(int dim, double J, const double* CInv,
                      int A, int B, int I, int Jidx, int K, int L)
{
    const double base =
        CInv[Jidx * dim + I] * CInv[A * dim + B] -
        2.0 * CInv[A * dim + I] * CInv[Jidx * dim + B];

    double value = J * CInv[L * dim + K] * base;
    value += J * (flexoDCInvDE(dim, CInv, Jidx, I, K, L) * CInv[A * dim + B] +
                  CInv[Jidx * dim + I] * flexoDCInvDE(dim, CInv, A, B, K, L));
    value -= 2.0 * J * (flexoDCInvDE(dim, CInv, A, I, K, L) * CInv[Jidx * dim + B] +
                        CInv[A * dim + I] * flexoDCInvDE(dim, CInv, Jidx, B, K, L));
    return value;
}

__device__
void flexoGreenMaterialResponseAndTangents(
    int materialLaw, double youngsModulus, double poissonsRatio,
    double lengthScale, double dielectricPermittivity,
    double vacuumPermittivity, double muL, double muT, double muS,
    int includeHbarFlexoCorrection, int dim, const double* C, const double* Egrad,
    const double* electricField, double* secondPK,
    double* strainGradientStress, double* electricDisplacement,
    double* tangentA, double* tangentSGradGrad, double* tangentSEgrad,
    double* tangentSgradE, double* tangentSElectric,
    double* tangentSgradElectric, double* tangentDE,
    double* tangentDGrad, double* tangentK)
{
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    const double lambda = youngsModulus * poissonsRatio /
        ((1.0 + poissonsRatio) * (1.0 - 2.0 * poissonsRatio));
    const double shearModulus = youngsModulus / (2.0 * (1.0 + poissonsRatio));
    const double lengthScale2 = lengthScale * lengthScale;
    const double dielectricContrast = dielectricPermittivity - vacuumPermittivity;
    const bool hasHbarCorrection =
        includeHbarFlexoCorrection && fabs(dielectricContrast) > 1.0e-14;

    flexoGreenMaterialResponse(materialLaw, youngsModulus, poissonsRatio,
        lengthScale, dielectricPermittivity, vacuumPermittivity, muL, muT, muS,
        includeHbarFlexoCorrection, dim, C, Egrad, electricField, secondPK, strainGradientStress,
        electricDisplacement);

    for (int i = 0; i < dim2 * dim2; ++i)
        tangentA[i] = 0.0;
    for (int i = 0; i < dim3 * dim3; ++i)
        tangentSGradGrad[i] = 0.0;
    for (int i = 0; i < dim2 * dim3; ++i)
        tangentSEgrad[i] = 0.0;
    for (int i = 0; i < dim3 * dim2; ++i)
        tangentSgradE[i] = 0.0;
    for (int i = 0; i < dim2 * dim; ++i)
        tangentSElectric[i] = 0.0;
    for (int i = 0; i < dim3 * dim; ++i)
        tangentSgradElectric[i] = 0.0;
    for (int i = 0; i < dim * dim2; ++i)
        tangentDE[i] = 0.0;
    for (int i = 0; i < dim * dim3; ++i)
        tangentDGrad[i] = 0.0;
    for (int i = 0; i < dim * dim; ++i)
        tangentK[i] = 0.0;

    double CInv[9] = {0.0};
    flexoInverse(dim, C, CInv);
    const double J = flexoSqrt(flexoDeterminant(dim, C));
    const double logJ = flexoLog(J);

    double Mmat[9] = {0.0};
    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
            Mmat[A * dim + B] = J * CInv[A * dim + B];

    double muG[3] = {0.0};
    for (int A = 0; A < dim; ++A)
        for (int I = 0; I < dim; ++I)
            for (int Jidx = 0; Jidx < dim; ++Jidx)
                for (int K = 0; K < dim; ++K)
                {
                    const int row = (I * dim + Jidx) * dim + K;
                    muG[A] += flexoMuContract(muL, muT, muS, A, I, Jidx, K) *
                              Egrad[row];
                }

    double Pmat[9] = {0.0};
    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
            Pmat[A * dim + B] =
                -0.5 * dielectricPermittivity * electricField[A] *
                electricField[B] -
                electricField[A] * muG[B];

    for (int I = 0; I < dim; ++I)
        for (int Jidx = 0; Jidx < dim; ++Jidx)
            for (int K = 0; K < dim; ++K)
                for (int L = 0; L < dim; ++L)
                {
                    const int rowE = I * dim + Jidx;
                    const int colE = K * dim + L;
                    double elastic = 0.0;
                    if (materialLaw == 0)
                    {
                        elastic =
                            lambda * (I == Jidx ? 1.0 : 0.0) *
                                     (K == L ? 1.0 : 0.0) +
                            2.0 * shearModulus * (I == K ? 1.0 : 0.0) *
                                                  (Jidx == L ? 1.0 : 0.0);
                    }
                    else
                    {
                        elastic =
                            lambda * CInv[L * dim + K] * CInv[I * dim + Jidx] +
                            2.0 * (shearModulus - lambda * logJ) *
                            CInv[I * dim + K] * CInv[L * dim + Jidx];
                    }

                    double materialM = 0.0;
                    for (int A = 0; A < dim; ++A)
                        for (int B = 0; B < dim; ++B)
                            materialM += Pmat[A * dim + B] *
                                flexoD2MinvDE2(dim, J, CInv, A, B, I, Jidx, K, L);
                    tangentA[rowE * dim2 + colE] = elastic + materialM;
                }

    for (int I = 0; I < dim; ++I)
        for (int Jidx = 0; Jidx < dim; ++Jidx)
            for (int K = 0; K < dim; ++K)
            {
                const int rowG = (I * dim + Jidx) * dim + K;
                for (int L = 0; L < dim; ++L)
                    for (int Midx = 0; Midx < dim; ++Midx)
                        for (int N = 0; N < dim; ++N)
                        {
                            const int colG = (L * dim + Midx) * dim + N;
                            double hbar =
                                (lambda * (I == Jidx ? 1.0 : 0.0) *
                                          (L == Midx ? 1.0 : 0.0) +
                                 2.0 * shearModulus * (I == L ? 1.0 : 0.0) *
                                                       (Jidx == Midx ? 1.0 : 0.0)) *
                                lengthScale2 * (K == N ? 1.0 : 0.0);
                            if (hasHbarCorrection)
                                for (int A = 0; A < dim; ++A)
                                    for (int B = 0; B < dim; ++B)
                                        hbar -=
                                            flexoMuContract(muL, muT, muS, A, I, Jidx, K) *
                                            Mmat[A * dim + B] *
                                            flexoMuContract(muL, muT, muS, B, L, Midx, N) /
                                            dielectricContrast;
                            tangentSGradGrad[rowG * dim3 + colG] = hbar;
                        }

                for (int L = 0; L < dim; ++L)
                    for (int Midx = 0; Midx < dim; ++Midx)
                    {
                        const int colE = L * dim + Midx;
                        double value = 0.0;
                        for (int A = 0; A < dim; ++A)
                            for (int B = 0; B < dim; ++B)
                            {
                                const double dM =
                                    flexoDMinvDE(dim, J, CInv, A, B, L, Midx);
                                value -= electricField[A] * dM *
                                         flexoMuContract(muL, muT, muS, B, I, Jidx, K);
                                if (hasHbarCorrection)
                                    for (int P = 0; P < dim; ++P)
                                        for (int Q = 0; Q < dim; ++Q)
                                            for (int R = 0; R < dim; ++R)
                                            {
                                                const int colG = (P * dim + Q) * dim + R;
                                                value -=
                                                    flexoMuContract(muL, muT, muS, A, I, Jidx, K) *
                                                    dM *
                                                    flexoMuContract(muL, muT, muS, B, P, Q, R) *
                                                    Egrad[colG] / dielectricContrast;
                                            }
                            }
                        tangentSgradE[rowG * dim2 + colE] = value;
                    }

                for (int Cidx = 0; Cidx < dim; ++Cidx)
                    for (int B = 0; B < dim; ++B)
                        tangentSgradElectric[rowG * dim + Cidx] -=
                            Mmat[Cidx * dim + B] *
                            flexoMuContract(muL, muT, muS, B, I, Jidx, K);
            }

    for (int I = 0; I < dim; ++I)
        for (int Jidx = 0; Jidx < dim; ++Jidx)
        {
            const int rowE = I * dim + Jidx;
            for (int L = 0; L < dim; ++L)
                for (int Midx = 0; Midx < dim; ++Midx)
                    for (int N = 0; N < dim; ++N)
                    {
                        const int colG = (L * dim + Midx) * dim + N;
                        double value = 0.0;
                        for (int A = 0; A < dim; ++A)
                            for (int B = 0; B < dim; ++B)
                            {
                                double dPdG =
                                    -electricField[A] *
                                    flexoMuContract(muL, muT, muS, B, L, Midx, N);
                                if (hasHbarCorrection)
                                    dPdG -= 0.5 *
                                        (flexoMuContract(muL, muT, muS, A, L, Midx, N) * muG[B] +
                                         muG[A] * flexoMuContract(muL, muT, muS, B, L, Midx, N)) /
                                        dielectricContrast;
                                value += flexoDMinvDE(dim, J, CInv, A, B, I, Jidx) * dPdG;
                            }
                        tangentSEgrad[rowE * dim3 + colG] = value;
                    }

            for (int Cidx = 0; Cidx < dim; ++Cidx)
            {
                double value = 0.0;
                for (int A = 0; A < dim; ++A)
                    for (int B = 0; B < dim; ++B)
                    {
                        const double dPde =
                            -0.5 * dielectricPermittivity *
                            ((A == Cidx ? 1.0 : 0.0) * electricField[B] +
                             electricField[A] * (B == Cidx ? 1.0 : 0.0)) -
                            (A == Cidx ? 1.0 : 0.0) * muG[B];
                        value += flexoDMinvDE(dim, J, CInv, A, B, I, Jidx) * dPde;
                    }
                tangentSElectric[rowE * dim + Cidx] = value;
            }
        }

    for (int A = 0; A < dim; ++A)
    {
        for (int I = 0; I < dim; ++I)
            for (int Jidx = 0; Jidx < dim; ++Jidx)
            {
                const int colE = I * dim + Jidx;
                double value = 0.0;
                for (int B = 0; B < dim; ++B)
                {
                    double electricVector = dielectricPermittivity *
                                            electricField[B] + muG[B];
                    value += flexoDMinvDE(dim, J, CInv, A, B, I, Jidx) *
                             electricVector;
                }
                tangentDE[A * dim2 + colE] = value;
            }

        for (int I = 0; I < dim; ++I)
            for (int Jidx = 0; Jidx < dim; ++Jidx)
                for (int K = 0; K < dim; ++K)
                {
                    const int colG = (I * dim + Jidx) * dim + K;
                    for (int B = 0; B < dim; ++B)
                        tangentDGrad[A * dim3 + colG] +=
                            Mmat[A * dim + B] *
                            flexoMuContract(muL, muT, muS, B, I, Jidx, K);
                }

        for (int B = 0; B < dim; ++B)
            tangentK[A * dim + B] = dielectricPermittivity * Mmat[A * dim + B];
    }
}

__device__
void flexoGreenVariations(int dim, int component,
                          const double* gradBasis, const double* hessBasis,
                          const double* F, const double* gradF,
                          double* deltaE, double* deltaEgrad)
{
    const int dim2 = dim * dim;
    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
            deltaE[A * dim + B] =
                0.5 * (gradBasis[A] * F[component * dim + B] +
                       F[component * dim + A] * gradBasis[B]);

    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
            for (int Cidx = 0; Cidx < dim; ++Cidx)
                deltaEgrad[(A * dim + B) * dim + Cidx] =
                    0.5 * (hessBasis[A * dim + Cidx] * F[component * dim + B] +
                           gradBasis[A] * gradF[component * dim2 + B * dim + Cidx] +
                           hessBasis[B * dim + Cidx] * F[component * dim + A] +
                           gradBasis[B] * gradF[component * dim2 + A * dim + Cidx]);
}

__device__
void flexoGreenSecondVariation(int dim, int rowComponent, int colComponent,
                               const double* gradTest, const double* hessTest,
                               const double* gradTrial, const double* hessTrial,
                               double* secondE, double* secondEgrad)
{
    for (int i = 0; i < dim * dim; ++i)
        secondE[i] = 0.0;
    for (int i = 0; i < dim * dim * dim; ++i)
        secondEgrad[i] = 0.0;
    if (rowComponent != colComponent)
        return;

    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
            secondE[A * dim + B] =
                0.5 * (gradTest[A] * gradTrial[B] +
                       gradTrial[A] * gradTest[B]);

    for (int A = 0; A < dim; ++A)
        for (int B = 0; B < dim; ++B)
            for (int Cidx = 0; Cidx < dim; ++Cidx)
                secondEgrad[(A * dim + B) * dim + Cidx] =
                    0.5 * (hessTest[A * dim + Cidx] * gradTrial[B] +
                           gradTest[A] * hessTrial[B * dim + Cidx] +
                           hessTest[B * dim + Cidx] * gradTrial[A] +
                           gradTest[B] * hessTrial[A * dim + Cidx]);
}

__device__
void flexoGreenDirectionalDerivatives(int materialLaw, double youngsModulus,
                                      double poissonsRatio, double lengthScale,
                                      double dielectricPermittivity,
                                      double vacuumPermittivity, double muL,
                                      double muT, double muS, int dim,
                                      const double* C, const double* Egrad,
                                      const double* electricField,
                                      const double* deltaE,
                                      const double* deltaEgrad,
                                      const double* deltaElectricField,
                                      double* deltaS,
                                      double* deltaSgrad,
                                      double* deltaD)
{
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    FlexoDual Cd[9];
    FlexoDual Gd[27];
    FlexoDual Ed[3];
    FlexoDual Sd[9];
    FlexoDual Sgd[27];
    FlexoDual Dd[3];

    for (int i = 0; i < dim2; ++i)
        Cd[i] = FlexoDual(C[i], 2.0 * deltaE[i]);
    for (int i = 0; i < dim3; ++i)
        Gd[i] = FlexoDual(Egrad[i], deltaEgrad[i]);
    for (int i = 0; i < dim; ++i)
        Ed[i] = FlexoDual(electricField[i], deltaElectricField[i]);

    flexoGreenMaterialResponse(materialLaw, youngsModulus, poissonsRatio,
        lengthScale, dielectricPermittivity, vacuumPermittivity, muL, muT,
        muS, 1, dim, Cd, Gd, Ed, Sd, Sgd, Dd);

    for (int i = 0; i < dim2; ++i)
        deltaS[i] = Sd[i].derivative;
    for (int i = 0; i < dim3; ++i)
        deltaSgrad[i] = Sgd[i].derivative;
    for (int i = 0; i < dim; ++i)
        deltaD[i] = Dd[i].derivative;
}

__global__
void evaluateFlexoEleBasisValuesAndDerivativesAtGPsKernel(
    int numDerivatives, int totalNumGPs, int dim, MultiPatchDeviceView displacement,
    MultiPatchDeviceView electricPotential, DeviceMatrixView<double> pts,
    DeviceVectorView<double> elecWorkingSpaces,
    DeviceMatrixView<double> elecValuesAndDerss)
{
    for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
         tidx < totalNumGPs * dim; tidx += blockDim.x * gridDim.x)
    {
        const int GPIdx = tidx / dim;
        const int d = tidx % dim;
        int patchIdx = 0;
        displacement.threadPatch(GPIdx, patchIdx);
        DeviceVectorView<double> pt(pts.data() + GPIdx * dim, dim);
        TensorBsplineBasisDeviceView eleBasis = electricPotential.basis(patchIdx);
        const int P1 = eleBasis.knotsOrder(0) + 1;
        double* elecWorkingSpace =
            elecWorkingSpaces.data() + GPIdx * P1 * (P1 + 4) * dim;
        DeviceMatrixView<double> elecValuesAndDers(
            elecValuesAndDerss.data() + GPIdx * P1 * (numDerivatives + 1) * dim,
            P1, (numDerivatives + 1) * dim);
        eleBasis.evalAllDers_into(d, dim, pt, numDerivatives, elecWorkingSpace,
                                  elecValuesAndDers);
    }
}

__global__
void constructFlexoElecSolutionKernel(DeviceVectorView<double> solVector,
                                      DeviceNestedArrayView<double> fixedDoFs,
                                      MultiBasisDeviceView multiBasis,
                                      SparseSystemDeviceView sparseSystem,
                                      MultiPatchDeviceView result, int CPSize)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < CPSize; idx += blockDim.x * gridDim.x)
    {
        int patch = 0;
        const int dim = multiBasis.domainDim();
        const int pointIdx = result.threadPatch_CPBase(idx, patch);
        if (sparseSystem.mapper(dim).is_free(pointIdx, patch))
        {
            const int index = sparseSystem.mapToGlobalColIndex(pointIdx, patch, dim);
            result.setCoefficients(patch, pointIdx, 0, solVector[index]);
        }
        else
        {
            const int index = sparseSystem.mapper(dim).bindex(pointIdx, patch);
            result.setCoefficients(patch, pointIdx, 0, fixedDoFs[dim][index]);
        }
    }
}

__global__
void zeroFlexoFunctionControlPointsKernel(MultiPatchDeviceView result,
                                          int totalEntries)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalEntries; idx += blockDim.x * gridDim.x)
    {
        int patch = 0;
        int component = 0;
        const int pointIdx = result.threadPatchAndDof(idx, patch, component);
        result.setCoefficients(patch, pointIdx, component, 0.0);
    }
}

__global__
void recoverElectricFieldAtNodesKernel(
    int numDerivatives, int totalNumGPs,
    MultiPatchDeviceView geometry,
    MultiPatchDeviceView electricPotential,
    MultiPatchDeviceView electricField,
    DeviceMatrixView<double> pts,
    DeviceVectorView<double> wts,
    DeviceMatrixView<double> geoValuesAndDerss,
    DeviceMatrixView<double> elecValuesAndDerss,
    DeviceVectorView<double> nodalWeights)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalNumGPs; idx += blockDim.x * gridDim.x)
    {
        const int dim = geometry.domainDim();
        int patchIdx = 0;
        geometry.threadPatch(idx, patchIdx);
        DeviceVectorView<double> pt(pts.data() + idx * dim, dim);

        PatchDeviceView geoPatch = geometry.patch(patchIdx);
        TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patchIdx);
        const int geoP1 = geoPatch.basis().knotsOrder(0) + 1;
        const int elecP1 = elecBasis.knotsOrder(0) + 1;
        DeviceMatrixView<double> geoValuesAndDers(
            geoValuesAndDerss.data() + idx * geoP1 * (numDerivatives + 1) * dim,
            geoP1, (numDerivatives + 1) * dim);
        DeviceMatrixView<double> elecValuesAndDers(
            elecValuesAndDerss.data() + idx * elecP1 * (numDerivatives + 1) * dim,
            elecP1, (numDerivatives + 1) * dim);

        double geoJacobianData[9] = {0.0};
        double geoJacobianInvData[9] = {0.0};
        DeviceMatrixView<double> geoJacobian(geoJacobianData, dim, dim);
        DeviceMatrixView<double> geoJacobianInv(geoJacobianInvData, dim, dim);
        geoPatch.jacobian(pt, geoValuesAndDers, numDerivatives, geoJacobian);
        geoJacobian.inverse(geoJacobianInv);

        double elecJacobianData[3] = {0.0};
        double elecGradData[3] = {0.0};
        DeviceMatrixView<double> elecJacobian(elecJacobianData, 1, dim);
        DeviceMatrixView<double> elecGrad(elecGradData, 1, dim);
        electricPotential.patch(patchIdx).jacobian(
            pt, elecValuesAndDers, numDerivatives, elecJacobian);
        elecJacobian.times(geoJacobianInv, elecGrad);

        double fieldData[3] = {0.0};
        for (int A = 0; A < dim; ++A)
            fieldData[A] = -elecGrad(0, A);

        int patchControlPointOffset = 0;
        for (int p = 0; p < patchIdx; ++p)
            patchControlPointOffset += electricField.numControlPoints(p);

        DeviceMatrixView<double> fieldControlPoints =
            electricField.controlPoints(patchIdx);
        const int numActive = elecBasis.numActiveControlPoints();
        const double measure = geoJacobian.determinant();
        for (int r = 0; r < numActive; ++r)
        {
            const int localControlPoint = elecBasis.activeIndex(pt, r);
            const int globalControlPoint =
                patchControlPointOffset + localControlPoint;
            const double N = tensorBasisValueFlexo(
                r, elecP1, dim, numDerivatives, elecValuesAndDers);
            const double weight = N * wts[idx] * measure;
            atomicAdd(&nodalWeights[globalControlPoint], weight);
            for (int A = 0; A < dim; ++A)
                atomicAdd(&fieldControlPoints(localControlPoint, A),
                          weight * fieldData[A]);
        }
    }
}

__global__
void normalizeRecoveredFlexoFunctionKernel(MultiPatchDeviceView result,
                                           DeviceVectorView<double> nodalWeights,
                                           int totalControlPoints)
{
    const int targetDim = result.targetDim();
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalControlPoints * targetDim; idx += blockDim.x * gridDim.x)
    {
        int patch = 0;
        int component = 0;
        const int localControlPoint =
            result.threadPatchAndDof(idx, patch, component);

        int globalControlPoint = localControlPoint;
        for (int p = 0; p < patch; ++p)
            globalControlPoint += result.numControlPoints(p);

        const double weight = nodalWeights[globalControlPoint];
        if (weight != 0.0)
        {
            DeviceMatrixView<double> controlPoints = result.controlPoints(patch);
            controlPoints(localControlPoint, component) /= weight;
        }
    }
}

__global__
void evaluateFlexoGPKernel(int numDerivatives, int totalNumGPs, int stride,
                           DeviceVectorView<double> materialParameters,
                           double localStiffening,
                           MultiPatchDeviceView displacement,
                           MultiPatchDeviceView electricPotential,
                           MultiPatchDeviceView geometry, DeviceMatrixView<double> pts,
                           DeviceVectorView<double> wts,
                           DeviceMatrixView<double> geoValuesAndDerss,
                           DeviceMatrixView<double> dispValuesAndDerss,
                           DeviceMatrixView<double> elecValuesAndDerss,
                           DeviceVectorView<double> flexoGPData)
{
    for (int GPIdx = blockIdx.x * blockDim.x + threadIdx.x; GPIdx < totalNumGPs;
         GPIdx += blockDim.x * gridDim.x)
    {
        const int dim = geometry.domainDim();
        const int dim2 = dim * dim;

        const FlexoGPOffsets offsets = flexoMakeGPOffsets(dim);

        double* gp = flexoGPData.data() + GPIdx * stride;
        DeviceVectorView<double> pt(pts.data() + GPIdx * dim, dim);

        int patchIdx = 0;
        displacement.threadPatch(GPIdx, patchIdx);
        PatchDeviceView geoPatch = geometry.patch(patchIdx);
        PatchDeviceView dispPatch = displacement.patch(patchIdx);
        PatchDeviceView elecPatch = electricPotential.patch(patchIdx);
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
        TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patchIdx);
        const int parameterOffset = patchIdx * 10;
        const int materialLaw =
            static_cast<int>(materialParameters[parameterOffset + 0]);
        const double youngsModulus = materialParameters[parameterOffset + 1];
        const double poissonsRatio = materialParameters[parameterOffset + 2];
        const double lengthScale = materialParameters[parameterOffset + 3];
        const double dielectricPermittivity = materialParameters[parameterOffset + 4];
        const double vacuumPermittivity = materialParameters[parameterOffset + 5];
        const double muL = materialParameters[parameterOffset + 6];
        const double muT = materialParameters[parameterOffset + 7];
        const double muS = materialParameters[parameterOffset + 8];
        const int includeHbarFlexoCorrection =
            static_cast<int>(materialParameters[parameterOffset + 9]);

        const int geoP1 = geoPatch.basis().knotsOrder(0) + 1;
        const int dispP1 = dispBasis.knotsOrder(0) + 1;
        const int elecP1 = elecBasis.knotsOrder(0) + 1;
        DeviceMatrixView<double> geoValuesAndDers(
            geoValuesAndDerss.data() + GPIdx * geoP1 * (numDerivatives + 1) * dim,
            geoP1, (numDerivatives + 1) * dim);
        DeviceMatrixView<double> dispValuesAndDers(
            dispValuesAndDerss.data() + GPIdx * dispP1 * (numDerivatives + 1) * dim,
            dispP1, (numDerivatives + 1) * dim);
        DeviceMatrixView<double> elecValuesAndDers(
            elecValuesAndDerss.data() + GPIdx * elecP1 * (numDerivatives + 1) * dim,
            elecP1, (numDerivatives + 1) * dim);

        DeviceMatrixView<double> geoJacobianInv(gp + offsets.geoInvOffset, dim, dim);
        double geoJacobianData[9] = {0.0};
        DeviceMatrixView<double> geoJacobian(geoJacobianData, dim, dim);
        geoPatch.jacobian(pt, geoValuesAndDers, numDerivatives, geoJacobian);
        geoJacobian.inverse(geoJacobianInv);
        const double measure = geoJacobian.determinant();
        gp[offsets.weightForceOffset] = wts[GPIdx] * measure;
        gp[offsets.weightBodyOffset] = wts[GPIdx] * pow(measure, -localStiffening) * measure;

        double* geoHessians = gp + offsets.geoHessOffset;
        for (int a = 0; a < dim; ++a)
            patchParamHessianFlexo(geoPatch, pt, geoValuesAndDers, numDerivatives,
                                   a, geoHessians + a * dim2);

        double dispJacobianData[9] = {0.0};
        double physDispJacData[9] = {0.0};
        DeviceMatrixView<double> dispJacobian(dispJacobianData, dim, dim);
        DeviceMatrixView<double> physDispJac(physDispJacData, dim, dim);
        dispPatch.jacobian(pt, dispValuesAndDers, numDerivatives, dispJacobian);
        dispJacobian.times(geoJacobianInv, physDispJac);
        for (int a = 0; a < dim; ++a)
            for (int A = 0; A < dim; ++A)
                gp[offsets.FOffset + a * dim + A] =
                    physDispJac(a, A) + (a == A ? 1.0 : 0.0);

        physicalFieldHessiansFlexo(dispPatch, pt, dispValuesAndDers,
                                    numDerivatives, geoHessians, geoJacobianInv,
                                    gp + offsets.gradFOffset);

        double elecJacobianData[3] = {0.0};
        double elecGradData[3] = {0.0};
        DeviceMatrixView<double> elecJacobian(elecJacobianData, 1, dim);
        DeviceMatrixView<double> elecGrad(elecGradData, 1, dim);
        elecPatch.jacobian(pt, elecValuesAndDers, numDerivatives, elecJacobian);
        elecJacobian.times(geoJacobianInv, elecGrad);
        for (int A = 0; A < dim; ++A)
            gp[offsets.electricFieldOffset + A] = -elecGrad(0, A);

        double C[9] = {0.0};
        double Egrad[27] = {0.0};
        computeCAndGreenGradient(dim, gp + offsets.FOffset,
                                 gp + offsets.gradFOffset, C, Egrad);
        flexoGreenMaterialResponseAndTangents(
            materialLaw, youngsModulus, poissonsRatio, lengthScale,
            dielectricPermittivity, vacuumPermittivity, muL, muT, muS,
            includeHbarFlexoCorrection, dim, C, Egrad, gp + offsets.electricFieldOffset,
            gp + offsets.SOffset, gp + offsets.SgradOffset,
            gp + offsets.DOffset, gp + offsets.AOffset,
            gp + offsets.SGradGradOffset, gp + offsets.SEgradOffset,
            gp + offsets.SgradEOffset, gp + offsets.SElectricOffset,
            gp + offsets.SgradElectricOffset, gp + offsets.DEOffset,
            gp + offsets.DGradOffset, gp + offsets.KOffset);
    }
}

__global__
void evaluateFlexoBasisDataKernel(int numDerivatives, int totalNumGPs, int N_D,
                                  int gpStride, int basisStride,
                                  MultiPatchDeviceView displacement,
                                  MultiPatchDeviceView electricPotential,
                                  DeviceMatrixView<double> dispValuesAndDerss,
                                  DeviceMatrixView<double> elecValuesAndDerss,
                                  DeviceVectorView<double> flexoGPData,
                                  DeviceVectorView<double> flexoBasisData)
{
    const int totalEntries = totalNumGPs * N_D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < totalEntries;
         idx += blockDim.x * gridDim.x)
    {
        const int basisIndex = idx % N_D;
        const int GPIdx = idx / N_D;
        const int dim = displacement.domainDim();

        int patchIdx = 0;
        displacement.threadPatch(GPIdx, patchIdx);
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
        TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patchIdx);
        const int dispP1 = dispBasis.knotsOrder(0) + 1;
        const int elecP1 = elecBasis.knotsOrder(0) + 1;

        DeviceMatrixView<double> dispValuesAndDers(
            dispValuesAndDerss.data() + GPIdx * dispP1 * (numDerivatives + 1) * dim,
            dispP1, (numDerivatives + 1) * dim);
        DeviceMatrixView<double> elecValuesAndDers(
            elecValuesAndDerss.data() + GPIdx * elecP1 * (numDerivatives + 1) * dim,
            elecP1, (numDerivatives + 1) * dim);

        const FlexoGPOffsets gpOffsets = flexoMakeGPOffsets(dim);
        double* gp = flexoGPData.data() + GPIdx * gpStride;
        DeviceMatrixView<double> geoJacobianInv(
            gp + gpOffsets.geoInvOffset, dim, dim);
        double* geoHessians = gp + gpOffsets.geoHessOffset;

        const FlexoBasisOffsets basisOffsets = flexoMakeBasisOffsets(dim);
        double* basisData =
            flexoBasisData.data() + (GPIdx * N_D + basisIndex) * basisStride;

        buildPhysicalGradientAndHessianFlexo(
            basisIndex, dispP1, dim, numDerivatives, dispValuesAndDers,
            geoHessians, geoJacobianInv,
            basisData + basisOffsets.dispGradOffset,
            basisData + basisOffsets.dispHessOffset);
        buildPhysicalGradientFlexo(
            basisIndex, elecP1, dim, numDerivatives, elecValuesAndDers,
            geoJacobianInv, basisData + basisOffsets.elecGradOffset);
    }
}

__global__
void assembleFlexoMatrixKernel(int numDerivatives, int numElements, int N_D,
                               int stride, int basisStride, int materialLaw,
                               double youngsModulus, double poissonsRatio,
                               double lengthScale,
                               double dielectricPermittivity,
                               double vacuumPermittivity,
                               double muL, double muT, double muS,
                               MultiPatchDeviceView displacement,
                               MultiPatchDeviceView electricPotential,
                               SparseSystemDeviceView system,
                               DeviceNestedArrayView<double> eliminatedDofs,
                               DeviceMatrixView<double> pts,
                               DeviceMatrixView<double> dispValuesAndDerss,
                               DeviceMatrixView<double> elecValuesAndDerss,
                               DeviceVectorView<double> flexoGPData,
                               DeviceVectorView<double> flexoBasisData)
{
    extern __shared__ double localMatrix[];

    const int dim = displacement.domainDim();
    const int numFields = dim + 1;
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    const int threadId = threadIdx.x;
    DeviceMatrixView<double> localMatrixView(localMatrix, numFields, numFields);

    int blockId = blockIdx.x;
    const int j = blockId % N_D; blockId /= N_D;
    const int i = blockId % N_D; blockId /= N_D;
    const int elementGlobal = blockId;
    if (elementGlobal >= numElements)
        return;

    for (int idx = threadId; idx < numFields * numFields; idx += blockDim.x)
        localMatrix[idx] = 0.0;
    __syncthreads();

    int patchIdx = 0;
    displacement.threadPatch_element(elementGlobal, patchIdx);
    TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
    TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patchIdx);
    const int numGPsInElement = dispBasis.numGPsInElement();

    const FlexoGPOffsets offsets = flexoMakeGPOffsets(dim);
    const FlexoBasisOffsets basisOffsets = flexoMakeBasisOffsets(dim);

    for (int q = threadId; q < numGPsInElement; q += blockDim.x)
    {
        const int GPIdx = elementGlobal * numGPsInElement + q;
        double* gp = flexoGPData.data() + GPIdx * stride;
        double* F = gp + offsets.FOffset;
        double* gradF = gp + offsets.gradFOffset;
        double* S = gp + offsets.SOffset;
        double* Sgrad = gp + offsets.SgradOffset;
        double* ABlock = gp + offsets.AOffset;
        double* SGradGradBlock = gp + offsets.SGradGradOffset;
        double* SEgradBlock = gp + offsets.SEgradOffset;
        double* SgradEBlock = gp + offsets.SgradEOffset;
        double* SElectricBlock = gp + offsets.SElectricOffset;
        double* SgradElectricBlock = gp + offsets.SgradElectricOffset;
        double* DEBlock = gp + offsets.DEOffset;
        double* DGradBlock = gp + offsets.DGradOffset;
        double* KBlock = gp + offsets.KOffset;

        const double* basisI =
            flexoBasisData.data() + (GPIdx * N_D + i) * basisStride;
        const double* basisJ =
            flexoBasisData.data() + (GPIdx * N_D + j) * basisStride;
        const double* grad_i = basisI + basisOffsets.dispGradOffset;
        const double* hess_i = basisI + basisOffsets.dispHessOffset;
        const double* gradP_i = basisI + basisOffsets.elecGradOffset;
        const double* grad_j = basisJ + basisOffsets.dispGradOffset;
        const double* hess_j = basisJ + basisOffsets.dispHessOffset;
        const double* gradP_j = basisJ + basisOffsets.elecGradOffset;

        double testEByRow[27] = {0.0};
        double testEgradByRow[81] = {0.0};
        double trialEByCol[27] = {0.0};
        double trialEgradByCol[81] = {0.0};
        double secondE[9] = {0.0};
        double secondEgrad[27] = {0.0};
        double deltaElectricField[3] = {0.0};

        for (int row = 0; row < dim; ++row)
            flexoGreenVariations(dim, row, grad_i, hess_i, F, gradF,
                                 testEByRow + row * dim2,
                                 testEgradByRow + row * dim3);
        for (int col = 0; col < dim; ++col)
            flexoGreenVariations(dim, col, grad_j, hess_j, F, gradF,
                                 trialEByCol + col * dim2,
                                 trialEgradByCol + col * dim3);
        for (int A = 0; A < dim; ++A)
            deltaElectricField[A] = -gradP_j[A];

        for (int col = 0; col < numFields; ++col)
        {
            const double* trialE =
                col < dim ? trialEByCol + col * dim2 : nullptr;
            const double* trialEgrad =
                col < dim ? trialEgradByCol + col * dim3 : nullptr;

            for (int row = 0; row < numFields; ++row)
            {
                double tangent = 0.0;
                if (row < dim)
                {
                    const double* testE = testEByRow + row * dim2;
                    const double* testEgrad = testEgradByRow + row * dim3;
                    const bool hasSecondVariation = col < dim && row == col;
                    if (hasSecondVariation)
                        flexoGreenSecondVariation(dim, row, col, grad_i, hess_i,
                                                  grad_j, hess_j, secondE,
                                                  secondEgrad);
                    for (int A = 0; A < dim; ++A)
                        for (int B = 0; B < dim; ++B)
                        {
                            const int testIdx = A * dim + B;
                            if (col < dim)
                            {
                                for (int Cidx = 0; Cidx < dim; ++Cidx)
                                    for (int Didx = 0; Didx < dim; ++Didx)
                                    {
                                        const int trialIdx = Cidx * dim + Didx;
                                        tangent += testE[testIdx] *
                                            ABlock[testIdx * dim2 + trialIdx] *
                                            trialE[trialIdx];
                                    }
                                for (int Cidx = 0; Cidx < dim; ++Cidx)
                                    for (int Didx = 0; Didx < dim; ++Didx)
                                        for (int Eidx = 0; Eidx < dim; ++Eidx)
                                        {
                                            const int trialGradIdx =
                                                (Cidx * dim + Didx) * dim + Eidx;
                                            tangent += testE[testIdx] *
                                                SEgradBlock[testIdx * dim3 + trialGradIdx] *
                                                trialEgrad[trialGradIdx];
                                        }
                            }
                            else
                            {
                                for (int Cidx = 0; Cidx < dim; ++Cidx)
                                    tangent += testE[testIdx] *
                                        SElectricBlock[testIdx * dim + Cidx] *
                                        deltaElectricField[Cidx];
                            }
                            if (hasSecondVariation)
                                tangent += S[testIdx] * secondE[testIdx];
                        }
                    for (int A = 0; A < dim; ++A)
                        for (int B = 0; B < dim; ++B)
                            for (int Cidx = 0; Cidx < dim; ++Cidx)
                            {
                                const int testGradIdx = (A * dim + B) * dim + Cidx;
                                if (col < dim)
                                {
                                    for (int Didx = 0; Didx < dim; ++Didx)
                                        for (int Eidx = 0; Eidx < dim; ++Eidx)
                                        {
                                            const int trialIdx = Didx * dim + Eidx;
                                            tangent += testEgrad[testGradIdx] *
                                                SgradEBlock[testGradIdx * dim2 + trialIdx] *
                                                trialE[trialIdx];
                                        }
                                    for (int Didx = 0; Didx < dim; ++Didx)
                                        for (int Eidx = 0; Eidx < dim; ++Eidx)
                                            for (int Fidx = 0; Fidx < dim; ++Fidx)
                                            {
                                                const int trialGradIdx =
                                                    (Didx * dim + Eidx) * dim + Fidx;
                                                tangent += testEgrad[testGradIdx] *
                                                    SGradGradBlock[testGradIdx * dim3 + trialGradIdx] *
                                                    trialEgrad[trialGradIdx];
                                            }
                                }
                                else
                                {
                                    for (int Didx = 0; Didx < dim; ++Didx)
                                        tangent += testEgrad[testGradIdx] *
                                            SgradElectricBlock[testGradIdx * dim + Didx] *
                                            deltaElectricField[Didx];
                                }
                                if (hasSecondVariation)
                                    tangent += Sgrad[testGradIdx] *
                                               secondEgrad[testGradIdx];
                            }
                }
                else
                {
                    for (int A = 0; A < dim; ++A)
                    {
                        if (col < dim)
                        {
                            for (int B = 0; B < dim; ++B)
                                for (int Cidx = 0; Cidx < dim; ++Cidx)
                                {
                                    const int trialIdx = B * dim + Cidx;
                                    tangent += gradP_i[A] *
                                        DEBlock[A * dim2 + trialIdx] *
                                        trialE[trialIdx];
                                }
                            for (int B = 0; B < dim; ++B)
                                for (int Cidx = 0; Cidx < dim; ++Cidx)
                                    for (int Didx = 0; Didx < dim; ++Didx)
                                    {
                                        const int trialGradIdx =
                                            (B * dim + Cidx) * dim + Didx;
                                        tangent += gradP_i[A] *
                                            DGradBlock[A * dim3 + trialGradIdx] *
                                            trialEgrad[trialGradIdx];
                                    }
                        }
                        else
                        {
                            for (int B = 0; B < dim; ++B)
                                tangent += gradP_i[A] * KBlock[A * dim + B] *
                                           deltaElectricField[B];
                        }
                    }
                }

                atomicAdd(&localMatrixView(row, col),
                          gp[offsets.weightBodyOffset] * tangent);
            }
        }
    }
    __syncthreads();

    double ptForIndexData[3] = {0.0};
    DeviceVectorView<double> ptForIndex(ptForIndexData, dim);
    for (int a = 0; a < dim; ++a)
        ptForIndex[a] = pts(a, elementGlobal * numGPsInElement);

    for (int localEntry = threadId; localEntry < numFields * numFields;
         localEntry += blockDim.x)
    {
        const int row = localEntry % numFields;
        const int col = localEntry / numFields;
        const int activeIndexI = row < dim
            ? dispBasis.activeIndex(ptForIndex, i)
            : elecBasis.activeIndex(ptForIndex, i);
        const int activeIndexJ = col < dim
            ? dispBasis.activeIndex(ptForIndex, j)
            : elecBasis.activeIndex(ptForIndex, j);
        const int globalIndexI = system.mapColIndex(activeIndexI, patchIdx, row);
        const int globalIndexJ = system.mapColIndex(activeIndexJ, patchIdx, col);
        system.pushToMatrix(localMatrixView(row, col), globalIndexI, globalIndexJ,
                            eliminatedDofs, row, col);
    }
}

__global__
void assembleFlexoMatrixUUKernel(int numElements, int N_D, int stride,
                                 int basisStride,
                                 MultiPatchDeviceView displacement,
                                 MultiPatchDeviceView electricPotential,
                                 SparseSystemDeviceView system,
                                 DeviceNestedArrayView<double> eliminatedDofs,
                                 DeviceMatrixView<double> pts,
                                 DeviceVectorView<double> flexoGPData,
                                 DeviceVectorView<double> flexoBasisData)
{
    extern __shared__ double localValue[];

    const int dim = displacement.domainDim();
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    const int threadId = threadIdx.x;

    int blockId = blockIdx.x;
    const int col = blockId % dim; blockId /= dim;
    const int row = blockId % dim; blockId /= dim;
    const int j = blockId % N_D; blockId /= N_D;
    const int i = blockId % N_D; blockId /= N_D;
    const int elementGlobal = blockId;
    if (elementGlobal >= numElements)
        return;

    if (threadId == 0)
        localValue[0] = 0.0;
    __syncthreads();

    int patchIdx = 0;
    displacement.threadPatch_element(elementGlobal, patchIdx);
    TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
    const int numGPsInElement = dispBasis.numGPsInElement();
    const FlexoGPOffsets offsets = flexoMakeGPOffsets(dim);
    const FlexoBasisOffsets basisOffsets = flexoMakeBasisOffsets(dim);

    for (int q = threadId; q < numGPsInElement; q += blockDim.x)
    {
        const int GPIdx = elementGlobal * numGPsInElement + q;
        double* gp = flexoGPData.data() + GPIdx * stride;
        const double* basisI =
            flexoBasisData.data() + (GPIdx * N_D + i) * basisStride;
        const double* basisJ =
            flexoBasisData.data() + (GPIdx * N_D + j) * basisStride;
        const double* grad_i = basisI + basisOffsets.dispGradOffset;
        const double* hess_i = basisI + basisOffsets.dispHessOffset;
        const double* grad_j = basisJ + basisOffsets.dispGradOffset;
        const double* hess_j = basisJ + basisOffsets.dispHessOffset;

        double testE[9] = {0.0};
        double testEgrad[27] = {0.0};
        double trialE[9] = {0.0};
        double trialEgrad[27] = {0.0};
        double secondE[9] = {0.0};
        double secondEgrad[27] = {0.0};
        flexoGreenVariations(dim, row, grad_i, hess_i,
                             gp + offsets.FOffset, gp + offsets.gradFOffset,
                             testE, testEgrad);
        flexoGreenVariations(dim, col, grad_j, hess_j,
                             gp + offsets.FOffset, gp + offsets.gradFOffset,
                             trialE, trialEgrad);
        const bool hasSecondVariation = row == col;
        if (hasSecondVariation)
            flexoGreenSecondVariation(dim, row, col, grad_i, hess_i,
                                      grad_j, hess_j, secondE, secondEgrad);

        double tangent = 0.0;
        double* S = gp + offsets.SOffset;
        double* Sgrad = gp + offsets.SgradOffset;
        double* ABlock = gp + offsets.AOffset;
        double* SGradGradBlock = gp + offsets.SGradGradOffset;
        double* SEgradBlock = gp + offsets.SEgradOffset;
        double* SgradEBlock = gp + offsets.SgradEOffset;
        for (int A = 0; A < dim; ++A)
            for (int B = 0; B < dim; ++B)
            {
                const int testIdx = A * dim + B;
                for (int Cidx = 0; Cidx < dim; ++Cidx)
                    for (int Didx = 0; Didx < dim; ++Didx)
                    {
                        const int trialIdx = Cidx * dim + Didx;
                        tangent += testE[testIdx] *
                            ABlock[testIdx * dim2 + trialIdx] *
                            trialE[trialIdx];
                    }
                for (int Cidx = 0; Cidx < dim; ++Cidx)
                    for (int Didx = 0; Didx < dim; ++Didx)
                        for (int Eidx = 0; Eidx < dim; ++Eidx)
                        {
                            const int trialGradIdx =
                                (Cidx * dim + Didx) * dim + Eidx;
                            tangent += testE[testIdx] *
                                SEgradBlock[testIdx * dim3 + trialGradIdx] *
                                trialEgrad[trialGradIdx];
                        }
                if (hasSecondVariation)
                    tangent += S[testIdx] * secondE[testIdx];
            }
        for (int A = 0; A < dim; ++A)
            for (int B = 0; B < dim; ++B)
                for (int Cidx = 0; Cidx < dim; ++Cidx)
                {
                    const int testGradIdx = (A * dim + B) * dim + Cidx;
                    for (int Didx = 0; Didx < dim; ++Didx)
                        for (int Eidx = 0; Eidx < dim; ++Eidx)
                        {
                            const int trialIdx = Didx * dim + Eidx;
                            tangent += testEgrad[testGradIdx] *
                                SgradEBlock[testGradIdx * dim2 + trialIdx] *
                                trialE[trialIdx];
                        }
                    for (int Didx = 0; Didx < dim; ++Didx)
                        for (int Eidx = 0; Eidx < dim; ++Eidx)
                            for (int Fidx = 0; Fidx < dim; ++Fidx)
                            {
                                const int trialGradIdx =
                                    (Didx * dim + Eidx) * dim + Fidx;
                                tangent += testEgrad[testGradIdx] *
                                    SGradGradBlock[testGradIdx * dim3 + trialGradIdx] *
                                    trialEgrad[trialGradIdx];
                            }
                    if (hasSecondVariation)
                        tangent += Sgrad[testGradIdx] * secondEgrad[testGradIdx];
                }
        atomicAdd(&localValue[0], gp[offsets.weightBodyOffset] * tangent);
    }
    __syncthreads();

    double ptForIndexData[3] = {0.0};
    DeviceVectorView<double> ptForIndex(ptForIndexData, dim);
    for (int a = 0; a < dim; ++a)
        ptForIndex[a] = pts(a, elementGlobal * numGPsInElement);
    const int activeIndexI = dispBasis.activeIndex(ptForIndex, i);
    const int activeIndexJ = dispBasis.activeIndex(ptForIndex, j);
    const int globalIndexI = system.mapColIndex(activeIndexI, patchIdx, row);
    const int globalIndexJ = system.mapColIndex(activeIndexJ, patchIdx, col);
    if (threadId == 0)
        system.pushToMatrix(localValue[0], globalIndexI, globalIndexJ,
                            eliminatedDofs, row, col);
}

__global__
void assembleFlexoMatrixUPhiKernel(int numElements, int N_D, int stride,
                                   int basisStride,
                                   MultiPatchDeviceView displacement,
                                   MultiPatchDeviceView electricPotential,
                                   SparseSystemDeviceView system,
                                   DeviceNestedArrayView<double> eliminatedDofs,
                                   DeviceMatrixView<double> pts,
                                   DeviceVectorView<double> flexoGPData,
                                   DeviceVectorView<double> flexoBasisData)
{
    extern __shared__ double localValue[];

    const int dim = displacement.domainDim();
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    const int threadId = threadIdx.x;

    int blockId = blockIdx.x;
    const int row = blockId % dim; blockId /= dim;
    const int j = blockId % N_D; blockId /= N_D;
    const int i = blockId % N_D; blockId /= N_D;
    const int elementGlobal = blockId;
    if (elementGlobal >= numElements)
        return;

    if (threadId == 0)
        localValue[0] = 0.0;
    __syncthreads();

    int patchIdx = 0;
    displacement.threadPatch_element(elementGlobal, patchIdx);
    TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
    TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patchIdx);
    const int numGPsInElement = dispBasis.numGPsInElement();
    const FlexoGPOffsets offsets = flexoMakeGPOffsets(dim);
    const FlexoBasisOffsets basisOffsets = flexoMakeBasisOffsets(dim);

    for (int q = threadId; q < numGPsInElement; q += blockDim.x)
    {
        const int GPIdx = elementGlobal * numGPsInElement + q;
        double* gp = flexoGPData.data() + GPIdx * stride;
        const double* basisI =
            flexoBasisData.data() + (GPIdx * N_D + i) * basisStride;
        const double* basisJ =
            flexoBasisData.data() + (GPIdx * N_D + j) * basisStride;
        const double* grad_i = basisI + basisOffsets.dispGradOffset;
        const double* hess_i = basisI + basisOffsets.dispHessOffset;
        const double* gradP_j = basisJ + basisOffsets.elecGradOffset;

        double testE[9] = {0.0};
        double testEgrad[27] = {0.0};
        flexoGreenVariations(dim, row, grad_i, hess_i,
                             gp + offsets.FOffset, gp + offsets.gradFOffset,
                             testE, testEgrad);

        double tangent = 0.0;
        double* SElectricBlock = gp + offsets.SElectricOffset;
        double* SgradElectricBlock = gp + offsets.SgradElectricOffset;
        for (int A = 0; A < dim; ++A)
            for (int B = 0; B < dim; ++B)
            {
                const int testIdx = A * dim + B;
                for (int Cidx = 0; Cidx < dim; ++Cidx)
                    tangent -= testE[testIdx] *
                        SElectricBlock[testIdx * dim + Cidx] * gradP_j[Cidx];
            }
        for (int A = 0; A < dim; ++A)
            for (int B = 0; B < dim; ++B)
                for (int Cidx = 0; Cidx < dim; ++Cidx)
                {
                    const int testGradIdx = (A * dim + B) * dim + Cidx;
                    for (int Didx = 0; Didx < dim; ++Didx)
                        tangent -= testEgrad[testGradIdx] *
                            SgradElectricBlock[testGradIdx * dim + Didx] *
                            gradP_j[Didx];
                }
        atomicAdd(&localValue[0], gp[offsets.weightBodyOffset] * tangent);
    }
    __syncthreads();

    double ptForIndexData[3] = {0.0};
    DeviceVectorView<double> ptForIndex(ptForIndexData, dim);
    for (int a = 0; a < dim; ++a)
        ptForIndex[a] = pts(a, elementGlobal * numGPsInElement);
    const int activeIndexI = dispBasis.activeIndex(ptForIndex, i);
    const int activeIndexJ = elecBasis.activeIndex(ptForIndex, j);
    const int globalIndexI = system.mapColIndex(activeIndexI, patchIdx, row);
    const int globalIndexJ = system.mapColIndex(activeIndexJ, patchIdx, dim);
    if (threadId == 0)
        system.pushToMatrix(localValue[0], globalIndexI, globalIndexJ,
                            eliminatedDofs, row, dim);
}

__global__
void assembleFlexoMatrixPhiUKernel(int numElements, int N_D, int stride,
                                   int basisStride,
                                   MultiPatchDeviceView displacement,
                                   MultiPatchDeviceView electricPotential,
                                   SparseSystemDeviceView system,
                                   DeviceNestedArrayView<double> eliminatedDofs,
                                   DeviceMatrixView<double> pts,
                                   DeviceVectorView<double> flexoGPData,
                                   DeviceVectorView<double> flexoBasisData)
{
    extern __shared__ double localValue[];

    const int dim = displacement.domainDim();
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    const int threadId = threadIdx.x;

    int blockId = blockIdx.x;
    const int col = blockId % dim; blockId /= dim;
    const int j = blockId % N_D; blockId /= N_D;
    const int i = blockId % N_D; blockId /= N_D;
    const int elementGlobal = blockId;
    if (elementGlobal >= numElements)
        return;

    if (threadId == 0)
        localValue[0] = 0.0;
    __syncthreads();

    int patchIdx = 0;
    displacement.threadPatch_element(elementGlobal, patchIdx);
    TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
    TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patchIdx);
    const int numGPsInElement = dispBasis.numGPsInElement();
    const FlexoGPOffsets offsets = flexoMakeGPOffsets(dim);
    const FlexoBasisOffsets basisOffsets = flexoMakeBasisOffsets(dim);

    for (int q = threadId; q < numGPsInElement; q += blockDim.x)
    {
        const int GPIdx = elementGlobal * numGPsInElement + q;
        double* gp = flexoGPData.data() + GPIdx * stride;
        const double* basisI =
            flexoBasisData.data() + (GPIdx * N_D + i) * basisStride;
        const double* basisJ =
            flexoBasisData.data() + (GPIdx * N_D + j) * basisStride;
        const double* gradP_i = basisI + basisOffsets.elecGradOffset;
        const double* grad_j = basisJ + basisOffsets.dispGradOffset;
        const double* hess_j = basisJ + basisOffsets.dispHessOffset;

        double trialE[9] = {0.0};
        double trialEgrad[27] = {0.0};
        flexoGreenVariations(dim, col, grad_j, hess_j,
                             gp + offsets.FOffset, gp + offsets.gradFOffset,
                             trialE, trialEgrad);

        double tangent = 0.0;
        double* DEBlock = gp + offsets.DEOffset;
        double* DGradBlock = gp + offsets.DGradOffset;
        for (int A = 0; A < dim; ++A)
        {
            for (int B = 0; B < dim; ++B)
                for (int Cidx = 0; Cidx < dim; ++Cidx)
                {
                    const int trialIdx = B * dim + Cidx;
                    tangent += gradP_i[A] *
                        DEBlock[A * dim2 + trialIdx] * trialE[trialIdx];
                }
            for (int B = 0; B < dim; ++B)
                for (int Cidx = 0; Cidx < dim; ++Cidx)
                    for (int Didx = 0; Didx < dim; ++Didx)
                    {
                        const int trialGradIdx =
                            (B * dim + Cidx) * dim + Didx;
                        tangent += gradP_i[A] *
                            DGradBlock[A * dim3 + trialGradIdx] *
                            trialEgrad[trialGradIdx];
                    }
        }
        atomicAdd(&localValue[0], gp[offsets.weightBodyOffset] * tangent);
    }
    __syncthreads();

    double ptForIndexData[3] = {0.0};
    DeviceVectorView<double> ptForIndex(ptForIndexData, dim);
    for (int a = 0; a < dim; ++a)
        ptForIndex[a] = pts(a, elementGlobal * numGPsInElement);
    const int activeIndexI = elecBasis.activeIndex(ptForIndex, i);
    const int activeIndexJ = dispBasis.activeIndex(ptForIndex, j);
    const int globalIndexI = system.mapColIndex(activeIndexI, patchIdx, dim);
    const int globalIndexJ = system.mapColIndex(activeIndexJ, patchIdx, col);
    if (threadId == 0)
        system.pushToMatrix(localValue[0], globalIndexI, globalIndexJ,
                            eliminatedDofs, dim, col);
}

__global__
void assembleFlexoMatrixPhiPhiKernel(int numElements, int N_D, int stride,
                                     int basisStride,
                                     MultiPatchDeviceView displacement,
                                     MultiPatchDeviceView electricPotential,
                                     SparseSystemDeviceView system,
                                     DeviceNestedArrayView<double> eliminatedDofs,
                                     DeviceMatrixView<double> pts,
                                     DeviceVectorView<double> flexoGPData,
                                     DeviceVectorView<double> flexoBasisData)
{
    extern __shared__ double localValue[];

    const int dim = displacement.domainDim();
    const int threadId = threadIdx.x;

    int blockId = blockIdx.x;
    const int j = blockId % N_D; blockId /= N_D;
    const int i = blockId % N_D; blockId /= N_D;
    const int elementGlobal = blockId;
    if (elementGlobal >= numElements)
        return;

    if (threadId == 0)
        localValue[0] = 0.0;
    __syncthreads();

    int patchIdx = 0;
    displacement.threadPatch_element(elementGlobal, patchIdx);
    TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
    TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patchIdx);
    const int numGPsInElement = dispBasis.numGPsInElement();
    const FlexoGPOffsets offsets = flexoMakeGPOffsets(dim);
    const FlexoBasisOffsets basisOffsets = flexoMakeBasisOffsets(dim);

    for (int q = threadId; q < numGPsInElement; q += blockDim.x)
    {
        const int GPIdx = elementGlobal * numGPsInElement + q;
        double* gp = flexoGPData.data() + GPIdx * stride;
        const double* basisI =
            flexoBasisData.data() + (GPIdx * N_D + i) * basisStride;
        const double* basisJ =
            flexoBasisData.data() + (GPIdx * N_D + j) * basisStride;
        const double* gradP_i = basisI + basisOffsets.elecGradOffset;
        const double* gradP_j = basisJ + basisOffsets.elecGradOffset;
        double* KBlock = gp + offsets.KOffset;

        double tangent = 0.0;
        for (int A = 0; A < dim; ++A)
            for (int B = 0; B < dim; ++B)
                tangent -= gradP_i[A] * KBlock[A * dim + B] * gradP_j[B];
        atomicAdd(&localValue[0], gp[offsets.weightBodyOffset] * tangent);
    }
    __syncthreads();

    double ptForIndexData[3] = {0.0};
    DeviceVectorView<double> ptForIndex(ptForIndexData, dim);
    for (int a = 0; a < dim; ++a)
        ptForIndex[a] = pts(a, elementGlobal * numGPsInElement);
    const int activeIndexI = elecBasis.activeIndex(ptForIndex, i);
    const int activeIndexJ = elecBasis.activeIndex(ptForIndex, j);
    const int globalIndexI = system.mapColIndex(activeIndexI, patchIdx, dim);
    const int globalIndexJ = system.mapColIndex(activeIndexJ, patchIdx, dim);
    if (threadId == 0)
        system.pushToMatrix(localValue[0], globalIndexI, globalIndexJ,
                            eliminatedDofs, dim, dim);
}

__global__
void assembleFlexoRHSKernel(int numDerivatives, int numElements, int N_D,
                            int stride, int basisStride, int materialLaw, double youngsModulus,
                            double poissonsRatio, double lengthScale,
                            double dielectricPermittivity,
                            double vacuumPermittivity,
                            double muL, double muT, double muS,
                            double forceScaling,
                            MultiPatchDeviceView displacement,
                            MultiPatchDeviceView electricPotential,
                            SparseSystemDeviceView system,
                            DeviceMatrixView<double> pts,
                            DeviceMatrixView<double> dispValuesAndDerss,
                            DeviceMatrixView<double> elecValuesAndDerss,
                            DeviceVectorView<double> bodyForce,
                            DeviceVectorView<double> flexoGPData,
                            DeviceVectorView<double> flexoBasisData)
{
    extern __shared__ double localRHS[];

    const int dim = displacement.domainDim();
    const int dim2 = dim * dim;
    const int dim3 = dim2 * dim;
    const int threadId = threadIdx.x;

    int blockId = blockIdx.x;
    const int i = blockId % N_D;
    const int elementGlobal = blockId / N_D;
    if (elementGlobal >= numElements)
        return;

    for (int idx = threadId; idx < dim + 1; idx += blockDim.x)
        localRHS[idx] = 0.0;
    __syncthreads();

    int patchIdx = 0;
    displacement.threadPatch_element(elementGlobal, patchIdx);
    TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
    TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patchIdx);
    const int numGPsInElement = dispBasis.numGPsInElement();
    const int dispP1 = dispBasis.knotsOrder(0) + 1;
    const int elecP1 = elecBasis.knotsOrder(0) + 1;

    const FlexoGPOffsets offsets = flexoMakeGPOffsets(dim);
    const FlexoBasisOffsets basisOffsets = flexoMakeBasisOffsets(dim);

    for (int q = threadId; q < numGPsInElement; q += blockDim.x)
    {
        const int GPIdx = elementGlobal * numGPsInElement + q;
        double* gp = flexoGPData.data() + GPIdx * stride;
        DeviceMatrixView<double> dispValuesAndDers(
            dispValuesAndDerss.data() + GPIdx * dispP1 * (numDerivatives + 1) * dim,
            dispP1, (numDerivatives + 1) * dim);

        const double* basisI =
            flexoBasisData.data() + (GPIdx * N_D + i) * basisStride;
        const double* grad_i = basisI + basisOffsets.dispGradOffset;
        const double* hess_i = basisI + basisOffsets.dispHessOffset;
        const double* gradP_i = basisI + basisOffsets.elecGradOffset;

        double* S = gp + offsets.SOffset;
        double* Sgrad = gp + offsets.SgradOffset;
        double* D = gp + offsets.DOffset;

        for (int di = 0; di < dim; ++di)
        {
            double testE[9] = {0.0};
            double testEgrad[27] = {0.0};
            flexoGreenVariations(dim, di, grad_i, hess_i, gp + offsets.FOffset,
                                 gp + offsets.gradFOffset, testE, testEgrad);
            double residual = 0.0;
            for (int a = 0; a < dim2; ++a)
                residual += testE[a] * S[a];
            for (int a = 0; a < dim3; ++a)
                residual += testEgrad[a] * Sgrad[a];
            double rhs = -gp[offsets.weightBodyOffset] * residual;
            rhs += gp[offsets.weightForceOffset] * forceScaling * bodyForce[di] *
                   tensorBasisValueFlexo(i, dispP1, dim, numDerivatives,
                                          dispValuesAndDers);
            atomicAdd(&localRHS[di], rhs);
        }

        double potentialResidual = 0.0;
        for (int A = 0; A < dim; ++A)
            potentialResidual += gradP_i[A] * D[A];
        atomicAdd(&localRHS[dim], -gp[offsets.weightBodyOffset] * potentialResidual);
    }
    __syncthreads();

    double ptForIndexData[3] = {0.0};
    DeviceVectorView<double> ptForIndex(ptForIndexData, dim);
    for (int a = 0; a < dim; ++a)
        ptForIndex[a] = pts(a, elementGlobal * numGPsInElement);

    for (int row = threadId; row < dim + 1; row += blockDim.x)
    {
        const int activeIndex = row < dim
            ? dispBasis.activeIndex(ptForIndex, i)
            : elecBasis.activeIndex(ptForIndex, i);
        const int globalIndex = system.mapColIndex(activeIndex, patchIdx, row);
        system.pushToRhs(localRHS[row], globalIndex, row);
    }
}

__global__
void countFlexoEntriesKernel(int numElements, int N_D,
                             MultiPatchDeviceView displacement,
                             MultiPatchDeviceView electricPotential,
                             SparseSystemDeviceView system,
                             DeviceMatrixView<double> pts,
                             int* counter)
{
    const int dim = displacement.domainDim();
    const int numFields = dim + 1;
    const int numGPsInElement =
        displacement.basis(0).totalNumGPs() /
        displacement.basis(0).totalNumElements();
    const int entriesPerElement = N_D * N_D * numFields * numFields;
    const int totalEntries = numElements * entriesPerElement;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalEntries; idx += blockDim.x * gridDim.x)
    {
        int entry = idx % entriesPerElement;
        const int elementGlobal = idx / entriesPerElement;
        const int j = entry % N_D; entry /= N_D;
        const int col = entry % numFields; entry /= numFields;
        const int i = entry % N_D; entry /= N_D;
        const int row = entry % numFields;

        int patchIdx = 0;
        displacement.threadPatch_element(elementGlobal, patchIdx);
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
        TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patchIdx);

        double ptForIndexData[3] = {0.0};
        DeviceVectorView<double> ptForIndex(ptForIndexData, dim);
        for (int a = 0; a < dim; ++a)
            ptForIndex[a] = pts(a, elementGlobal * numGPsInElement);

        const int activeIndexI = row < dim
            ? dispBasis.activeIndex(ptForIndex, i)
            : elecBasis.activeIndex(ptForIndex, i);
        const int activeIndexJ = col < dim
            ? dispBasis.activeIndex(ptForIndex, j)
            : elecBasis.activeIndex(ptForIndex, j);

        const int globalIndexI = system.mapColIndex(activeIndexI, patchIdx, row);
        const int globalIndexJ = system.mapColIndex(activeIndexJ, patchIdx, col);
        if (system.isEntry(globalIndexI, globalIndexJ, row, col))
            atomicAdd(counter, 1);
    }
}

__global__
void computeFlexoCOOKernel(int numElements, int N_D,
                           MultiPatchDeviceView displacement,
                           MultiPatchDeviceView electricPotential,
                           SparseSystemDeviceView system,
                           DeviceMatrixView<double> pts,
                           int* counter,
                           DeviceVectorView<int> cooRows,
                           DeviceVectorView<int> cooCols)
{
    const int dim = displacement.domainDim();
    const int numFields = dim + 1;
    const int numGPsInElement =
        displacement.basis(0).totalNumGPs() /
        displacement.basis(0).totalNumElements();
    const int entriesPerElement = N_D * N_D * numFields * numFields;
    const int totalEntries = numElements * entriesPerElement;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalEntries; idx += blockDim.x * gridDim.x)
    {
        int entry = idx % entriesPerElement;
        const int elementGlobal = idx / entriesPerElement;
        const int j = entry % N_D; entry /= N_D;
        const int col = entry % numFields; entry /= numFields;
        const int i = entry % N_D; entry /= N_D;
        const int row = entry % numFields;

        int patchIdx = 0;
        displacement.threadPatch_element(elementGlobal, patchIdx);
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
        TensorBsplineBasisDeviceView elecBasis = electricPotential.basis(patchIdx);

        double ptForIndexData[3] = {0.0};
        DeviceVectorView<double> ptForIndex(ptForIndexData, dim);
        for (int a = 0; a < dim; ++a)
            ptForIndex[a] = pts(a, elementGlobal * numGPsInElement);

        const int activeIndexI = row < dim
            ? dispBasis.activeIndex(ptForIndex, i)
            : elecBasis.activeIndex(ptForIndex, i);
        const int activeIndexJ = col < dim
            ? dispBasis.activeIndex(ptForIndex, j)
            : elecBasis.activeIndex(ptForIndex, j);

        const int globalIndexI = system.mapColIndex(activeIndexI, patchIdx, row);
        const int globalIndexJ = system.mapColIndex(activeIndexJ, patchIdx, col);
        if (system.isEntry(globalIndexI, globalIndexJ, row, col))
        {
            system.pushToEntryIndex(globalIndexI, globalIndexJ, row, col,
                                    counter, cooRows, cooCols);
        }
    }
}

} // namespace

GPUFlexoelectriciyAssembler::GPUFlexoelectriciyAssembler(
    const MultiPatch& multiPatch,
    const MultiBasis& displacementBasis,
    const MultiBasis& electricPotentialBasis,
    const BoundaryConditions& bc,
    const Eigen::VectorXd& bodyForce)
    : GPUAssembler(multiPatch, displacementBasis, bc, bodyForce, true, 2),
      m_electricPotentialBasisHost(electricPotentialBasis)
{
    m_N_P = electricPotentialBasis.numActive();
    m_elePotentialP1 = electricPotentialBasis.knotOrder() + 1;
    if (m_N_P != N_D())
        throw std::invalid_argument("GPUFlexoelectriciyAssembler currently requires matching displacement and electric-potential active counts for the combined electroelastic assembly strategy");

    setDisplacementPatches(displacementBasis);
    electricPotentialBasis.giveBasis(m_electricPotentialPatchHost,
                                     m_electricPotentialTargetDim);
    m_electricPotentialPatch = MultiPatchDeviceData(m_electricPotentialPatchHost);

    std::vector<DofMapper> dofMappers(targetDim() + m_electricPotentialTargetDim);
    displacementBasis.getMappers(true, boundaryConditions(), dofMappers, true);
    electricPotentialBasis.getMapper(true, boundaryConditions(), targetDim(),
                                    dofMappers.back(), true);
    SparseSystem sparseSystem(dofMappers,
        Eigen::VectorXi::Ones(targetDim() + m_electricPotentialTargetDim));
    setupSparseSystem(sparseSystem);

    std::vector<Eigen::VectorXd> ddof(targetDim() + m_electricPotentialTargetDim);
    std::vector<Eigen::VectorXd> ddofZero(targetDim() + m_electricPotentialTargetDim);
    for (int unk = 0; unk < targetDim(); ++unk)
    {
        computeDirichletDofs(unk, dofMappers, ddof, displacementBasis);
        ddofZero[unk] = Eigen::VectorXd::Zero(ddof[unk].size());
    }
    for (int unk = targetDim(); unk < targetDim() + m_electricPotentialTargetDim; ++unk)
    {
        computeDirichletDofs(unk, dofMappers, ddof, electricPotentialBasis);
        ddofZero[unk] = Eigen::VectorXd::Zero(ddof[unk].size());
    }
    setDdof(ddof);
    setDdofZero(ddofZero);

    setBasisPatches();
    setElecBasisPatches();
    computeGPTable();

    const int numFields = targetDim() + m_electricPotentialTargetDim;
    const int entriesPerElement = N_D() * N_D() * numFields * numFields;
    const int totalEntries = numElements() * entriesPerElement;
    int* entryCounter = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&entryCounter),
                                 sizeof(int));
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA malloc failed while counting flexoelectric COO: ") +
                                 cudaGetErrorString(err));
    err = cudaMemset(entryCounter, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        cudaFree(entryCounter);
        throw std::runtime_error(std::string("CUDA memset failed while counting flexoelectric COO: ") +
                                 cudaGetErrorString(err));
    }
    int minGrid = 0;
    int blockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        countFlexoEntriesKernel, 0, totalEntries);
    if (blockSize <= 0)
        blockSize = 128;
    const int gridSize = (totalEntries + blockSize - 1) / blockSize;
    countFlexoEntriesKernel<<<gridSize, blockSize>>>(
        numElements(), N_D(), displacementView(),
        m_electricPotentialPatch.deviceView(), sparseSystemDeviceView(),
        gpTable(), entryCounter);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFree(entryCounter);
        throw std::runtime_error(std::string("CUDA launch failed while counting flexoelectric COO: ") +
                                 cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        cudaFree(entryCounter);
        throw std::runtime_error(std::string("CUDA synchronize failed while counting flexoelectric COO: ") +
                                 cudaGetErrorString(err));
    }
    int entryCountHost = 0;
    err = cudaMemcpy(&entryCountHost, entryCounter, sizeof(int),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cudaFree(entryCounter);
        throw std::runtime_error(std::string("CUDA memcpy failed while counting flexoelectric COO: ") +
                                 cudaGetErrorString(err));
    }

    DeviceArray<int> cooRows(entryCountHost);
    DeviceArray<int> cooCols(entryCountHost);
    err = cudaMemset(entryCounter, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        cudaFree(entryCounter);
        throw std::runtime_error(std::string("CUDA memset failed while computing flexoelectric COO: ") +
                                 cudaGetErrorString(err));
    }
    computeFlexoCOOKernel<<<gridSize, blockSize>>>(
        numElements(), N_D(), displacementView(),
        m_electricPotentialPatch.deviceView(), sparseSystemDeviceView(),
        gpTable(), entryCounter, cooRows.vectorView(), cooCols.vectorView());
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFree(entryCounter);
        throw std::runtime_error(std::string("CUDA launch failed while computing flexoelectric COO: ") +
                                 cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        cudaFree(entryCounter);
        throw std::runtime_error(std::string("CUDA synchronize failed while computing flexoelectric COO: ") +
                                 cudaGetErrorString(err));
    }
    cudaFree(entryCounter);
    setCSRMatrixFromCOO(sparseSystem.matrixRows(), sparseSystem.matrixCols(),
                        cooRows.vectorView(), cooCols.vectorView());

    evaluateBasisValuesAndDerivativesAtGPs();
    evaluateElecBasisValuesAndDerivativesAtGPs();
    allocateGPData();
    setDefaultOptions();
}

void GPUFlexoelectriciyAssembler::setDefaultOptions()
{
    OptionList opt;
    opt.addReal("youngs_modulus", "Young's modulus", 1.0);
    opt.addReal("poissons_ratio", "Poisson's ratio", 0.3);
    opt.addReal("dielectric_permittivity", "Dielectric permittivity", 1.0);
    opt.addReal("vacuum_permittivity", "Vacuum electric permittivity", 0.0);
    opt.addReal("flexoelectric_mu_L", "Longitudinal flexoelectric coefficient", 0.0);
    opt.addReal("flexoelectric_mu_T", "Transversal flexoelectric coefficient", 0.0);
    opt.addReal("flexoelectric_mu_S", "Shear flexoelectric coefficient", 0.0);
    opt.addReal("length_scale", "Strain-gradient length scale", 0.0);
    opt.addReal("force_scaling", "Body-force scaling", 1.0);
    opt.addReal("neumann_load_scaling", "Multiplier for Neumann boundary and corner loads", 1.0);
    opt.addReal("local_stiffening", "Local stiffening exponent", 0.0);
    opt.addInt("material_law", "0: StVK, 1: neo-Hookean", 1);
    opt.addInt("include_hbar_flexo_correction",
               "Include eliminated-polarization flexoelectric correction in hbar", 0);
    GPUAssembler::setDefaultOptions(opt);
}

void GPUFlexoelectriciyAssembler::evaluateElecBasisValuesAndDerivativesAtGPs()
{
    m_elecValuesAndDerss.resize(m_elePotentialP1 * numGPs() *
                                (numDerivatives() + 1) * domainDim());
    DeviceArray<double> elecWorkingSpaces(numGPs() * m_elePotentialP1 *
                                          (m_elePotentialP1 + 4) * domainDim());
    int minGrid = 0;
    int blockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        evaluateFlexoEleBasisValuesAndDerivativesAtGPsKernel, 0,
        numGPs() * domainDim());
    int gridSize = (numGPs() * domainDim() + blockSize - 1) / blockSize;
    evaluateFlexoEleBasisValuesAndDerivativesAtGPsKernel<<<gridSize, blockSize>>>(
        numDerivatives(), numGPs(), domainDim(), displacementView(),
        m_electricPotentialPatch.deviceView(), gpTable(),
        elecWorkingSpaces.vectorView(),
        m_elecValuesAndDerss.matrixView(m_elePotentialP1,
            numGPs() * (numDerivatives() + 1) * domainDim()));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA launch failed while evaluating flexoelectric potential basis: ") +
                                 cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA synchronize failed while evaluating flexoelectric potential basis: ") +
                                 cudaGetErrorString(err));
}

void GPUFlexoelectriciyAssembler::constructElecSolution(
    const DeviceVectorView<double>& solVector,
    const DeviceNestedArrayView<double>& fixedDoFs) const
{
    int minGrid = 0;
    int blockSize = 0;
    const int CPSize = totalNumControlPoints();
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        constructFlexoElecSolutionKernel, 0, CPSize);
    int gridSize = (CPSize + blockSize - 1) / blockSize;
    constructFlexoElecSolutionKernel<<<gridSize, blockSize>>>(
        solVector, fixedDoFs, multiBasisDeviceView(), sparseSystemDeviceView(),
        m_electricPotentialPatch.deviceView(), CPSize);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA synchronize failed while constructing electric potential solution");
}

void GPUFlexoelectriciyAssembler::constructElecSolution(
    const DeviceVectorView<double>& solVector,
    const DeviceNestedArrayView<double>& fixedDoFs,
    GPUFunction& electricPotentialFunction) const
{
    int minGrid = 0;
    int blockSize = 0;
    const int CPSize = totalNumControlPoints();
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        constructFlexoElecSolutionKernel, 0, CPSize);
    int gridSize = (CPSize + blockSize - 1) / blockSize;
    constructFlexoElecSolutionKernel<<<gridSize, blockSize>>>(
        solVector, fixedDoFs, multiBasisDeviceView(), sparseSystemDeviceView(),
        electricPotentialFunction.multiPatchDeviceView(), CPSize);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA synchronize failed while constructing electric potential function");
}

void GPUFlexoelectriciyAssembler::constructElectricFieldFunction(
    const DeviceVectorView<double>& solVector,
    const DeviceNestedArrayView<double>& fixedDoFs,
    GPUFunction& electricFieldFunction) const
{
    constructElecSolution(solVector, fixedDoFs);
    constructElectricFieldFunctionFromPotential(
        m_electricPotentialPatch.deviceView(), electricFieldFunction);
}

void GPUFlexoelectriciyAssembler::constructElectricFieldFunction(
    GPUFunction& electricPotentialFunction,
    GPUFunction& electricFieldFunction) const
{
    if (electricPotentialFunction.domainDim() != domainDim())
        throw std::invalid_argument(
            "Electric potential function domain dimension must match assembler domain dimension");
    if (electricPotentialFunction.targetDim() != 1)
        throw std::invalid_argument(
            "Electric potential function target dimension must be 1");
    constructElectricFieldFunctionFromPotential(
        electricPotentialFunction.multiPatchDeviceView(), electricFieldFunction);
}

void GPUFlexoelectriciyAssembler::constructElectricFieldFunctionFromPotential(
    MultiPatchDeviceView electricPotentialView,
    GPUFunction& electricFieldFunction) const
{
    if (electricFieldFunction.domainDim() != domainDim())
        throw std::invalid_argument(
            "Electric field function domain dimension must match assembler domain dimension");
    if (electricFieldFunction.targetDim() != domainDim())
        throw std::invalid_argument(
            "Electric field function target dimension must match assembler domain dimension");

    int minGrid = 0;
    int blockSize = 0;
    int gridSize = 0;
    cudaError_t err = cudaSuccess;
    const int totalControlPoints = totalNumControlPoints();
    const int totalEntries = totalControlPoints * domainDim();

    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        zeroFlexoFunctionControlPointsKernel, 0, totalEntries);
    gridSize = (totalEntries + blockSize - 1) / blockSize;
    zeroFlexoFunctionControlPointsKernel<<<gridSize, blockSize>>>(
        electricFieldFunction.multiPatchDeviceView(), totalEntries);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error(
            "CUDA synchronize failed while zeroing electric field function");

    DeviceArray<double> nodalWeights(totalControlPoints);
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        recoverElectricFieldAtNodesKernel, 0, numGPs());
    gridSize = (numGPs() + blockSize - 1) / blockSize;
    recoverElectricFieldAtNodesKernel<<<gridSize, blockSize>>>(
        numDerivatives(), numGPs(), geometryView(), electricPotentialView,
        electricFieldFunction.multiPatchDeviceView(), gpTable(),
        wts().vectorView(), geoValuesAndDerssView(),
        m_elecValuesAndDerss.matrixView(m_elePotentialP1,
            numGPs() * (numDerivatives() + 1) * domainDim()),
        nodalWeights.vectorView());
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error(
            "CUDA synchronize failed while recovering electric field function");

    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        normalizeRecoveredFlexoFunctionKernel, 0, totalEntries);
    gridSize = (totalEntries + blockSize - 1) / blockSize;
    normalizeRecoveredFlexoFunctionKernel<<<gridSize, blockSize>>>(
        electricFieldFunction.multiPatchDeviceView(),
        nodalWeights.vectorView(), totalControlPoints);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error(
            "CUDA synchronize failed while normalizing electric field function");
}

void GPUFlexoelectriciyAssembler::checkNumericalJacobian(
    const Eigen::VectorXd& solVector,
    const DeviceNestedArrayView<double>& fixedDoFs,
    double relativeStep,
    int numIter)
{
    if (solVector.size() != numDofs())
        throw std::invalid_argument("checkNumericalJacobian: solution size does not match assembler dofs");
    if (relativeStep <= 0.0)
        throw std::invalid_argument("checkNumericalJacobian: relativeStep must be positive");

    DeviceArray<double> solDevice(solVector);
    assemble(solDevice.vectorView(), numIter, fixedDoFs);
    const Eigen::SparseMatrix<double, Eigen::RowMajor, int> analyticSparse =
        csrMatrix().toEigenCSR();
    const Eigen::MatrixXd analytic = Eigen::MatrixXd(analyticSparse);

    Eigen::MatrixXd numeric(numDofs(), numDofs());
    numeric.setZero();

    for (int d = 0; d < numDofs(); ++d)
    {
        const double eps = relativeStep *
            std::max(1.0, std::abs(solVector[d]));

        Eigen::VectorXd up = solVector;
        Eigen::VectorXd um = solVector;
        up[d] += eps;
        um[d] -= eps;

        DeviceArray<double> upDevice(up);
        assemble(upDevice.vectorView(), numIter, fixedDoFs);
        const Eigen::VectorXd Rp = hostRHS();

        DeviceArray<double> umDevice(um);
        assemble(umDevice.vectorView(), numIter, fixedDoFs);
        const Eigen::VectorXd Rm = hostRHS();

        numeric.col(d) = -((Rp - Rm) / (2.0 * eps));
    }

    DeviceArray<double> restoreDevice(solVector);
    assemble(restoreDevice.vectorView(), numIter, fixedDoFs);

    const Eigen::MatrixXd diff = analytic - numeric;
    const double analyticNorm = analytic.norm();
    const double numericNorm = numeric.norm();
    const double diffNorm = diff.norm();
    const double denom = std::max({1.0, analyticNorm, numericNorm});

    double maxAbs = 0.0;
    int worstRow = 0;
    int worstCol = 0;
    for (int r = 0; r < diff.rows(); ++r)
        for (int c = 0; c < diff.cols(); ++c)
        {
            const double absDiff = std::abs(diff(r, c));
            if (absDiff > maxAbs)
            {
                maxAbs = absDiff;
                worstRow = r;
                worstCol = c;
            }
        }

    std::cout << std::scientific << std::setprecision(6)
              << "Flexoelectric numerical Jacobian check\n"
              << "  relative step: " << relativeStep << "\n"
              << "  analytic Frobenius norm: " << analyticNorm << "\n"
              << "  numeric Frobenius norm:  " << numericNorm << "\n"
              << "  difference Frobenius:    " << diffNorm << "\n"
              << "  relative Frobenius:      " << diffNorm / denom << "\n"
              << "  max abs difference:      " << maxAbs
              << " at (" << worstRow << ", " << worstCol << ")\n"
              << "  analytic worst entry:    " << analytic(worstRow, worstCol) << "\n"
              << "  numeric worst entry:     " << numeric(worstRow, worstCol) << "\n";
}

void GPUFlexoelectriciyAssembler::assemble(
    const DeviceVectorView<double>& solVector,
    int numIter,
    const DeviceNestedArrayView<double>& fixedDoFs)
{
    setMatrixAndRHSZeros();
    constructDispSolution(solVector, fixedDoFs);
    constructElecSolution(solVector, fixedDoFs);

    DeviceNestedArrayView<double> fixedDofsAssemble;
    getFixedDofsForAssemble(numIter, fixedDofsAssemble);

    const int stride = flexoDoublesPerGP(domainDim());
    const int dataSize = stride * numGPs();
    if (m_flexoGPData.size() != dataSize)
        m_flexoGPData.resize(dataSize);
    else
        m_flexoGPData.setZero();

    const int basisStride = flexoDoublesPerGPBasis(domainDim());
    const int basisDataSize = basisStride * numGPs() * N_D();
    if (m_flexoBasisData.size() != basisDataSize)
        m_flexoBasisData.resize(basisDataSize);

    const double localStiffening = options().getReal("local_stiffening");
    const int materialLaw = options().getInt("material_law");
    const double youngsModulus = options().getReal("youngs_modulus");
    const double poissonsRatio = options().getReal("poissons_ratio");
    const double lengthScale = options().getReal("length_scale");
    const double dielectricPermittivity = options().getReal("dielectric_permittivity");
    const double vacuumPermittivity = options().getReal("vacuum_permittivity");
    const double muL = options().getReal("flexoelectric_mu_L");
    const double muT = options().getReal("flexoelectric_mu_T");
    const double muS = options().getReal("flexoelectric_mu_S");
    const int includeHbarFlexoCorrection =
        options().getInt("include_hbar_flexo_correction");
    const double forceScaling = options().getReal("force_scaling");

    const std::vector<int> patchMaterialLaws =
        patchIntOptionValues("material_law");
    const std::vector<double> patchYoungsModuli =
        patchRealOptionValues("youngs_modulus");
    const std::vector<double> patchPoissonsRatios =
        patchRealOptionValues("poissons_ratio");
    const std::vector<double> patchLengthScales =
        patchRealOptionValues("length_scale");
    const std::vector<double> patchDielectricPermittivities =
        patchRealOptionValues("dielectric_permittivity");
    const std::vector<double> patchVacuumPermittivities =
        patchRealOptionValues("vacuum_permittivity");
    const std::vector<double> patchMuL =
        patchRealOptionValues("flexoelectric_mu_L");
    const std::vector<double> patchMuT =
        patchRealOptionValues("flexoelectric_mu_T");
    const std::vector<double> patchMuS =
        patchRealOptionValues("flexoelectric_mu_S");
    const std::vector<int> patchHbarFlexoCorrections =
        patchIntOptionValues("include_hbar_flexo_correction");

    std::vector<double> flexoMaterialParameters;
    flexoMaterialParameters.reserve(static_cast<std::size_t>(10 * numPatches()));
    for (int p = 0; p < numPatches(); ++p)
    {
        const std::size_t patch = static_cast<std::size_t>(p);
        flexoMaterialParameters.push_back(
            static_cast<double>(patchMaterialLaws[patch]));
        flexoMaterialParameters.push_back(patchYoungsModuli[patch]);
        flexoMaterialParameters.push_back(patchPoissonsRatios[patch]);
        flexoMaterialParameters.push_back(patchLengthScales[patch]);
        flexoMaterialParameters.push_back(
            patchDielectricPermittivities[patch]);
        flexoMaterialParameters.push_back(patchVacuumPermittivities[patch]);
        flexoMaterialParameters.push_back(patchMuL[patch]);
        flexoMaterialParameters.push_back(patchMuT[patch]);
        flexoMaterialParameters.push_back(patchMuS[patch]);
        flexoMaterialParameters.push_back(
            static_cast<double>(patchHbarFlexoCorrections[patch]));
    }
    DeviceArray<double> materialParameterValues(flexoMaterialParameters);

    int minGrid = 0;
    int blockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        evaluateFlexoGPKernel, 0, numGPs());
    int gridSize = (numGPs() + blockSize - 1) / blockSize;
    evaluateFlexoGPKernel<<<gridSize, blockSize>>>(
        numDerivatives(), numGPs(), stride, materialParameterValues.vectorView(),
        localStiffening, displacementView(),
        m_electricPotentialPatch.deviceView(), geometryView(), gpTable(),
        wts().vectorView(), geoValuesAndDerssView(), dispValuesAndDerssView(),
        m_elecValuesAndDerss.matrixView(m_elePotentialP1,
            numGPs() * (numDerivatives() + 1) * domainDim()),
        m_flexoGPData.vectorView());
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA synchronize failed in evaluateFlexoGPKernel");

    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        evaluateFlexoBasisDataKernel, 0, numGPs() * N_D());
    gridSize = (numGPs() * N_D() + blockSize - 1) / blockSize;
    evaluateFlexoBasisDataKernel<<<gridSize, blockSize>>>(
        numDerivatives(), numGPs(), N_D(), stride, basisStride,
        displacementView(), m_electricPotentialPatch.deviceView(),
        dispValuesAndDerssView(),
        m_elecValuesAndDerss.matrixView(m_elePotentialP1,
            numGPs() * (numDerivatives() + 1) * domainDim()),
        m_flexoGPData.vectorView(), m_flexoBasisData.vectorView());
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA synchronize failed in evaluateFlexoBasisDataKernel");

    const bool useSplitMatrixKernels =
        flexoEnvFlag("SIGA_FLEXO_SPLIT_MATRIX_KERNELS", false);
    const bool printMatrixTiming =
        flexoEnvFlag("SIGA_FLEXO_MATRIX_TIMING", false);
    cudaEvent_t matrixStart = nullptr;
    cudaEvent_t matrixStop = nullptr;
    if (printMatrixTiming)
    {
        cudaEventCreate(&matrixStart);
        cudaEventCreate(&matrixStop);
        cudaEventRecord(matrixStart);
    }

    blockSize = N_D();
    if (useSplitMatrixKernels)
    {
        const size_t matrixSharedBytes = sizeof(double);
        gridSize = N_D() * N_D() * numElements() * domainDim() * domainDim();
        assembleFlexoMatrixUUKernel<<<gridSize, blockSize, matrixSharedBytes>>>(
            numElements(), N_D(), stride, basisStride, displacementView(),
            m_electricPotentialPatch.deviceView(), sparseSystemDeviceView(),
            fixedDofsAssemble, gpTable(), m_flexoGPData.vectorView(),
            m_flexoBasisData.vectorView());
        gridSize = N_D() * N_D() * numElements() * domainDim();
        assembleFlexoMatrixUPhiKernel<<<gridSize, blockSize, matrixSharedBytes>>>(
            numElements(), N_D(), stride, basisStride, displacementView(),
            m_electricPotentialPatch.deviceView(), sparseSystemDeviceView(),
            fixedDofsAssemble, gpTable(), m_flexoGPData.vectorView(),
            m_flexoBasisData.vectorView());
        assembleFlexoMatrixPhiUKernel<<<gridSize, blockSize, matrixSharedBytes>>>(
            numElements(), N_D(), stride, basisStride, displacementView(),
            m_electricPotentialPatch.deviceView(), sparseSystemDeviceView(),
            fixedDofsAssemble, gpTable(), m_flexoGPData.vectorView(),
            m_flexoBasisData.vectorView());
        gridSize = N_D() * N_D() * numElements();
        assembleFlexoMatrixPhiPhiKernel<<<gridSize, blockSize, matrixSharedBytes>>>(
            numElements(), N_D(), stride, basisStride, displacementView(),
            m_electricPotentialPatch.deviceView(), sparseSystemDeviceView(),
            fixedDofsAssemble, gpTable(), m_flexoGPData.vectorView(),
            m_flexoBasisData.vectorView());
    }
    else
    {
        gridSize = N_D() * N_D() * numElements();
        const size_t matrixSharedBytes =
            (domainDim() + 1) * (domainDim() + 1) * sizeof(double);
        assembleFlexoMatrixKernel<<<gridSize, blockSize, matrixSharedBytes>>>(
            numDerivatives(), numElements(), N_D(), stride, basisStride,
            materialLaw, youngsModulus, poissonsRatio, lengthScale,
            dielectricPermittivity, vacuumPermittivity, muL, muT, muS,
            displacementView(), m_electricPotentialPatch.deviceView(),
            sparseSystemDeviceView(), fixedDofsAssemble, gpTable(),
            dispValuesAndDerssView(),
            m_elecValuesAndDerss.matrixView(m_elePotentialP1,
                numGPs() * (numDerivatives() + 1) * domainDim()),
            m_flexoGPData.vectorView(), m_flexoBasisData.vectorView());
    }

    if (printMatrixTiming)
        cudaEventRecord(matrixStop);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA synchronize failed in flexoelectric matrix kernels");
    if (printMatrixTiming)
    {
        float matrixMilliseconds = 0.0f;
        cudaEventElapsedTime(&matrixMilliseconds, matrixStart, matrixStop);
        cudaEventDestroy(matrixStart);
        cudaEventDestroy(matrixStop);
        std::cout << "Flexo matrix assembly path: "
                  << (useSplitMatrixKernels ? "split4" : "single")
                  << ", CUDA time: " << matrixMilliseconds << " ms\n";
    }

    gridSize = N_D() * numElements();
    const size_t rhsSharedBytes = (domainDim() + 1) * sizeof(double);
    assembleFlexoRHSKernel<<<gridSize, blockSize, rhsSharedBytes>>>(
        numDerivatives(), numElements(), N_D(), stride, basisStride, materialLaw,
        youngsModulus, poissonsRatio, lengthScale, dielectricPermittivity,
        vacuumPermittivity, muL, muT, muS, forceScaling, displacementView(),
        m_electricPotentialPatch.deviceView(), sparseSystemDeviceView(), gpTable(),
        dispValuesAndDerssView(),
        m_elecValuesAndDerss.matrixView(m_elePotentialP1,
            numGPs() * (numDerivatives() + 1) * domainDim()),
        bodyForce().vectorView(), m_flexoGPData.vectorView(),
        m_flexoBasisData.vectorView());
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA synchronize failed in assembleFlexoRHSKernel");

    assembleNeumannBoundaryCondition();
    assembleNeumannCornerPointLoads();
}

void GPUFlexoelectriciyAssembler::setElecBasisPatches()
{
    m_electricPotentialBasisHost.giveBasis(m_electricPotentialPatchHost,
                                           m_electricPotentialTargetDim);
    m_electricPotentialPatch = MultiPatchDeviceData(m_electricPotentialPatchHost);
}

void GPUFlexoelectriciyAssembler::refreshFixedDofs()
{
    std::vector<DofMapper> dofMappers(targetDim() + m_electricPotentialTargetDim);
    basisHost().getMappers(true, boundaryConditions(), dofMappers, true);
    m_electricPotentialBasisHost.getMapper(true, boundaryConditions(), targetDim(),
                                          dofMappers.back(), true);
    std::vector<Eigen::VectorXd> ddof(targetDim() + m_electricPotentialTargetDim);
    for (int unk = 0; unk < targetDim(); ++unk)
        computeDirichletDofs(unk, dofMappers, ddof, basisHost());
    for (int unk = targetDim(); unk < targetDim() + m_electricPotentialTargetDim; ++unk)
        computeDirichletDofs(unk, dofMappers, ddof, m_electricPotentialBasisHost);
    setDdof(ddof);
}
