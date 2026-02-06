#pragma once

#include <cstdio>
#include <DeviceVectorView.h>

#define MAPPER_PATCH_DOF2(a,b) m_dofs[m_offset[b]+a]

class DofMapperDeviceView
{
private:
    DeviceVectorView<int> m_dofs;
    DeviceVectorView<int> m_offset;
    DeviceVectorView<int> m_numFreeDofs;
    DeviceVectorView<int> m_numElimDofs;
    DeviceVectorView<int> m_numCpldDofs;
    DeviceVectorView<int> m_tagged;
    int m_shift = 0;
    int m_bshift = 0;
    int m_curElimId = -1;

public:
    __device__
    DofMapperDeviceView(DeviceVectorView<int> dofs,
                        DeviceVectorView<int> offset,
                        DeviceVectorView<int> numFreeDofs,
                        DeviceVectorView<int> numElimDofs,
                        DeviceVectorView<int> numCpldDofs,
                        DeviceVectorView<int> tagged,
                        int shift, int bshift, int curElimId)
                      : m_dofs(dofs),
                        m_offset(offset),
                        m_numFreeDofs(numFreeDofs),
                        m_numElimDofs(numElimDofs),
                        m_numCpldDofs(numCpldDofs),
                        m_tagged(tagged),
                        m_shift(shift), m_bshift(bshift), m_curElimId(curElimId)
    {
    }

    __device__
    void print() const
    {
        printf("DofMapperDeviceView:\n");
        printf("Dofs:\n");
        m_dofs.print();
        printf("Offset:\n");
        m_offset.print();
        printf("Num Free Dofs:\n");
        m_numFreeDofs.print();
        printf("Num Elim Dofs:\n");
        m_numElimDofs.print();
        printf("Num Cpld Dofs:\n");
        m_numCpldDofs.print();
        printf("Tagged:\n");
        m_tagged.print();
        printf("Shift: %d\n", m_shift);
        printf("BShift: %d\n", m_bshift);
        printf("CurElimId: %d\n", m_curElimId);
    }

    __device__
    bool is_free_index(int gl) const 
    { return gl < m_curElimId + m_shift; }

    __device__
    int index(int dof, int patch, int comp = 0) const
    { return MAPPER_PATCH_DOF2(dof, patch) + m_shift; }

    __device__
    bool is_free(int i, int k = 0) const
    { return is_free_index(index(i, k)); }

    __device__
    int bindex(int i, int k = 0, int c = 0) const
    { return MAPPER_PATCH_DOF2(i, k) - m_numFreeDofs.back() + m_bshift; }
};