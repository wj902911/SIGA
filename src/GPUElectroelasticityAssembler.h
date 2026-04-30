#pragma once

#include <GPUAssembler.h>

class GPUElectroelasticityAssembler : public GPUAssembler
{
private:
    MultiPatchDeviceData m_electricPotentialPatch;
    MultiPatch m_electricPotentialPatchHost;
    const MultiBasis& m_electricPotentialBasisHost;
    int m_electricPotentialTargetDim = 1;
    int m_numElementsElPatch = 0;
    int m_N_P = 0;
    int m_elePotentialP1 = 1;
    DeviceArray<double> m_elecValuesAndDerss;
    DeviceArray<double> m_GPData_As;
public:
    __host__
    GPUElectroelasticityAssembler(const MultiPatch& multiPatch,
                                const MultiBasis& displacementBasis,
                                const MultiBasis& electricPotentialBasis,
                                const BoundaryConditions& bc,
                                const Eigen::VectorXd& bodyForce);

    __host__
    int numMatrixEntries() const;
    __host__
    int numMixMatrixEntries() const;
    __host__
    int numEleMatrixEntries() const;

    __host__
    void computeWholeMatrixCOO(DeviceVectorView<int> cooRows,
                               DeviceVectorView<int> cooCols) const;

    __host__
    void computeMixMatrixCOO(DeviceVectorView<int> cooRows,
                             DeviceVectorView<int> cooCols) const;
    __host__
    void computeEleMatrixCOO(DeviceVectorView<int> cooRows,
                             DeviceVectorView<int> cooCols) const;

    __host__
    void setDefaultOptions();

    __host__
    void evaluateElecBasisValuesAndDerivativesAtGPs();

    __host__
    void allocateGPElecData();

    __host__
    void constructElecSolution(const DeviceVectorView<double>& solVector,
                               const DeviceNestedArrayView<double>& fixedDoFs) const;

    __host__
    virtual void assemble(const DeviceVectorView<double>& solVector, int numIter,
                  const DeviceNestedArrayView<double>& fixedDoFs) override;

    __host__
    void setElecBasisPatches();

    __host__
    int N_P() const { return m_N_P; }

    __host__
    void refreshFixedDofs() override;
};
