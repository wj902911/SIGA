#pragma once

#include <cuda_runtime.h>
#include <DeviceObjectArray.h>
#include <BoundaryCondition.h>
#include <Boundary_d.h>


struct boundary_condition_d
{
public:
    __host__ __device__
    boundary_condition_d() = default;

    __host__
    boundary_condition_d(const boundary_condition& bc)
    :m_ps(bc.patchSide()), m_unknown(bc.unknown()),
     m_unkcomp(bc.unkcomp()), m_values(bc.values()) { }

    __device__
    const BoxSide_d& side() const { return m_ps.side(); }

    __device__
    const DeviceObjectArray<double>& values() const { return m_values; }
private:
    PatchSide_d m_ps;
    DeviceObjectArray<double> m_values;
    int m_unknown = 0;
    int m_unkcomp = -1;
};