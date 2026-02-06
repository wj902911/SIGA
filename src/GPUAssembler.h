#pragma once

#include <MultiPatchDeviceData.h>
#include <MultiBasisDeviceData.h>
#include <SparseSystemDeviceData.h>
#include <BoundaryCondition.h>
#include <MultiGaussPointsDeviceData.h>

class GPUAssembler
{
private:
    MultiPatchDeviceData m_multiPatch;
    MultiPatchDeviceData m_displacement;
    const MultiPatch& m_multiPatchHost;
    MultiBasisDeviceData m_multiBasis;
    const MultiBasis& m_multiBasisHost;
    BoundaryConditions m_boundaryConditions;
    DeviceNestedArray<double> m_ddof;
    DeviceNestedArray<double> m_ddof_zero;
    SparseSystemDeviceData m_sparseSystem;
    DeviceArray<double> m_bodyForce;
    MultiGaussPointsDeviceData m_multiGaussPoints;
    bool m_initialAssemble = true;
public:
    __host__
    GPUAssembler(const MultiPatch& multiPatch,
                  const MultiBasis& multiBasis,
                  const BoundaryConditions& bc,
                  const Eigen::VectorXd& bodyForce);
        

    __host__
    void computeDirichletDofs(int unk_, 
                              const std::vector<DofMapper> &mappers,
                              std::vector<Eigen::VectorXd> &ddof,
                              const MultiBasis &multiBasis);

    __host__
    void print() const;

    __host__
    int numDofs() const;

    __host__
    const DeviceNestedArray<double>& allFixedDofs() const { return m_ddof; }

    __host__
    void constructSolution(const DeviceVectorView<double>& solVector,
                           const DeviceNestedArrayView<double>& fixedDoFs,
                           MultiPatchDeviceView& displacementDeviceView) const;
    
    __host__
    void assemble(const DeviceVectorView<double>& solVector, int numIter,
                  const DeviceNestedArrayView<double>& fixedDoFs);

};