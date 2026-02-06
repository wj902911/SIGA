#pragma once

#include <DeviceVectorView.h>
#include <DofMapperDeviceView.h>

class SparseSystemDeviceView
{
private:
    DeviceVectorView<int> m_mappersData;
    DeviceVectorView<int> m_row;
    DeviceVectorView<int> m_col;
    DeviceVectorView<int> m_rstr;
    DeviceVectorView<int> m_cstr;
    DeviceVectorView<int> m_cvar;
    DeviceVectorView<int> m_dims;

    DeviceMatrixView<double> m_matrix;
    DeviceVectorView<double> m_RHS;

public:
    __host__ __device__
    SparseSystemDeviceView(DeviceVectorView<int> mappersData,
                           DeviceVectorView<int> row,
                           DeviceVectorView<int> col,
                           DeviceVectorView<int> rstr,
                           DeviceVectorView<int> cstr,
                           DeviceVectorView<int> cvar,
                           DeviceVectorView<int> dims,
                           DeviceMatrixView<double> matrix,
                           DeviceVectorView<double> rhs)
                         : m_mappersData(mappersData),
                           m_row(row),
                           m_col(col),
                           m_rstr(rstr),
                           m_cstr(cstr),
                           m_cvar(cvar),
                           m_dims(dims),
                           m_matrix(matrix),
                           m_RHS(rhs)
    {
    }

    __device__
    DofMapperDeviceView mapper(int index) const
    {
        int start = 0;
        for (int i = 0; i< index; i++)
        {
            int mapperDataSize = m_mappersData(start);
            start += mapperDataSize;
        }
        int mapperDataSize = m_mappersData(start++);
        int numComponents = m_mappersData(start);
        start += numComponents + 1;
        int dofsSize = m_mappersData(2);
        DeviceVectorView<int> dofs(m_mappersData.data() + start, dofsSize);
        start += dofsSize;
        int offsetSize = m_mappersData(start++);
        DeviceVectorView<int> offset(m_mappersData.data() + start, offsetSize);
        start += offsetSize;
        int shift = m_mappersData(start++);
        int bshift = m_mappersData(start++);
        int numFreeDofsSize = m_mappersData(start++);
        DeviceVectorView<int> numFreeDofs(m_mappersData.data() + start, numFreeDofsSize);
        start += numFreeDofsSize;
        int numElimDofsSize = m_mappersData(start++);
        DeviceVectorView<int> numElimDofs(m_mappersData.data() + start, numElimDofsSize);
        start += numElimDofsSize;
        int numCpldDofsSize = m_mappersData(start++);
        DeviceVectorView<int> numCpldDofs(m_mappersData.data() + start, numCpldDofsSize);
        start += numCpldDofsSize;
        int curElimId = m_mappersData(start++);
        int taggedSize = m_mappersData(start++);
        DeviceVectorView<int> tagged(m_mappersData.data() + start, taggedSize);
        return DofMapperDeviceView(dofs,
                                   offset,
                                   numFreeDofs,
                                   numElimDofs,
                                   numCpldDofs,
                                   tagged,
                                   shift,
                                   bshift,
                                   curElimId);
    }

    __device__
    void print() const
    {
        printf("SparseSystemDeviceView:\n");
        printf("Number of mappers: %d\n", m_dims.size());
        for (int i = 0; i < m_dims.size(); i++)
        {
            printf("Mapper %d:\n", i);
            mapper(i).print();
        }
        printf("m_row:\n");
        m_row.print();
        printf("m_col:\n");
        m_col.print();
        printf("m_rstr:\n");
        m_rstr.print();
        printf("m_cstr:\n");
        m_cstr.print();
        printf("m_cvar:\n");
        m_cvar.print();
        printf("m_dims:\n");
        m_dims.print();
        printf("Matrix (%d x %d):\n", m_matrix.rows(), m_matrix.cols());
        m_matrix.print();
        printf("RHS (%d):\n", m_RHS.size());
        m_RHS.print();
    }

    __device__
    int mapToGlobalColIndex(int active, int patchIndex, int c = 0) const
    { return mapper(m_col(c)).index(active, patchIndex) + m_cstr(c); }
};