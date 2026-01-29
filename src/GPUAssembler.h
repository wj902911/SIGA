#pragma once

#include <MultiPatchDeviceData.h>
#include <MultiBasisDeviceData.h>
#include <SparseSystemDeviceData.h>
#include <BoundaryCondition.h>

class GPUAssembler
{
private:
    MultiPatchDeviceData m_multiPatch;
    MultiBasisDeviceData m_multiBasis;
    BoundaryConditions m_boundaryConditions;
    DeviceNestedArray<double> m_ddof;
    SparseSystemDeviceData m_sparseSystem;
    DeviceArray<double> m_bodyForce;
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

};