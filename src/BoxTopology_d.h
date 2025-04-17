#pragma once

#include <BoxTopology.h>
#include <DeviceObjectArray.h>
#include <Boundary_d.h>


class BoxTopology_d
{
private:
    int m_dim;
    int m_nboxes;
    DeviceObjectArray<PatchSide_d> m_boundary;
    DeviceObjectArray<BoundaryInterface_d> m_interfaces;
public:
    __host__
    BoxTopology_d(const BoxTopology& topology);

    __host__ __device__
    int dim() const { return m_dim; }
    __host__ __device__
    int nboxes() const { return m_nboxes; }
    __host__ __device__
    size_t nInterfaces() const { return m_interfaces.size(); }
    __host__ __device__
    size_t nBoundary() const   { return m_boundary.size(); }

    __device__
    const BoundaryInterface_d& interface(int i) const
    {
        assert(i >= 0 && i < nInterfaces());
        return m_interfaces[i];
    }

    __device__
    const PatchSide_d& boundary(int i) const
    {
        assert(i >= 0 && i < nBoundary());
        return m_boundary[i];
    }
};