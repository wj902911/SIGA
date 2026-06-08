#pragma once

#include <GPUAssembler.h>

class GPUFlexoelectriciyAssembler : public GPUAssembler
{
private:
    MultiPatchDeviceData m_electricPotentialPatch;
    MultiPatch m_electricPotentialPatchHost;
    const MultiBasis& m_electricPotentialBasisHost;
    int m_electricPotentialTargetDim = 1;
    int m_N_P = 0;
    int m_elePotentialP1 = 1;
    DeviceArray<double> m_elecValuesAndDerss;
    DeviceArray<double> m_flexoGPData;
    DeviceArray<double> m_flexoBasisData;

    __host__
    void constructElectricFieldFunctionFromPotential(
        MultiPatchDeviceView electricPotentialView,
        GPUFunction& electricFieldFunction) const;

public:
    __host__
    GPUFlexoelectriciyAssembler(const MultiPatch& multiPatch,
                                const MultiBasis& displacementBasis,
                                const MultiBasis& electricPotentialBasis,
                                const BoundaryConditions& bc,
                                const Eigen::VectorXd& bodyForce);

    __host__
    void setDefaultOptions();

    __host__
    void evaluateElecBasisValuesAndDerivativesAtGPs();

    __host__
    void constructElecSolution(const DeviceVectorView<double>& solVector,
                               const DeviceNestedArrayView<double>& fixedDoFs) const;

    __host__
    void constructElecSolution(const DeviceVectorView<double>& solVector,
                               const DeviceNestedArrayView<double>& fixedDoFs,
                               GPUFunction& electricPotentialFunction) const;

    __host__
    void constructElectricFieldFunction(const DeviceVectorView<double>& solVector,
                                        const DeviceNestedArrayView<double>& fixedDoFs,
                                        GPUFunction& electricFieldFunction) const;

    __host__
    void constructElectricFieldFunction(GPUFunction& electricPotentialFunction,
                                        GPUFunction& electricFieldFunction) const;

    __host__
    void assemble(const DeviceVectorView<double>& solVector,
                  int numIter,
                  const DeviceNestedArrayView<double>& fixedDoFs) override;

    __host__
    void checkNumericalJacobian(const Eigen::VectorXd& solVector,
                                const DeviceNestedArrayView<double>& fixedDoFs,
                                double relativeStep = 1.0e-6,
                                int numIter = 1);

    __host__
    void setElecBasisPatches();

    __host__
    void refreshFixedDofs() override;
};
