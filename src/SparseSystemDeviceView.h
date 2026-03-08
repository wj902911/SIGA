#pragma once

//#include <cuda_runtime.h>
#include <DeviceVectorView.h>
#include <DofMapperDeviceView.h>
#include <DeviceCSRMatrix.h>

#define USE_PERMUTATION

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

    //DeviceMatrixView<double> m_matrix;

    //DeviceVectorView<int> m_rows;
    //DeviceVectorView<int> m_cols;
    //DeviceVectorView<double> m_values;
    DeviceVectorView<double> m_RHS;
    //DeviceVectorView<double> m_RHS_coo;

    DeviceCSRMatrixView m_csrMatrix;

    DeviceVectorView<int> m_perm_old2new;
    DeviceVectorView<int> m_perm_new2old;


public:
    __host__ __device__
    SparseSystemDeviceView(DeviceVectorView<int> mappersData,
                           DeviceVectorView<int> row,
                           DeviceVectorView<int> col,
                           DeviceVectorView<int> rstr,
                           DeviceVectorView<int> cstr,
                           DeviceVectorView<int> cvar,
                           DeviceVectorView<int> dims,
                           //DeviceMatrixView<double> matrix,
                           //DeviceVectorView<double> rhs,
                           //DeviceVectorView<int> rows,
                           //DeviceVectorView<int> cols,
                           //DeviceVectorView<double> values,
                           DeviceVectorView<double> rhs,
                           DeviceCSRMatrixView csrMatrix,
                           DeviceVectorView<int> permOld2New,
                           DeviceVectorView<int> permNew2Old)
                         : m_mappersData(mappersData),
                           m_row(row),
                           m_col(col),
                           m_rstr(rstr),
                           m_cstr(cstr),
                           m_cvar(cvar),
                           m_dims(dims),
                           //m_matrix(matrix),
                           //m_RHS(rhs),
                           //m_rows(rows),
                           //m_cols(cols),
                           //m_values(values),
                           m_RHS(rhs),
                           m_csrMatrix(csrMatrix),
                           m_perm_old2new(permOld2New),
                           m_perm_new2old(permNew2Old)
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
        //printf("Matrix (%d x %d):\n", m_matrix.rows(), m_matrix.cols());
        //m_matrix.print();
        //printf("cols:\n");
        //m_cols.print();
        //printf("rows:\n");
        //m_rows.print();
        //printf("values:\n");
        //m_values.print();
        printf("RHS (%d):\n", m_RHS.size());
        m_RHS.print();
        printf("Permutation old2new:\n");
        m_perm_old2new.print();
        printf("Permutation new2old:\n");
        m_perm_new2old.print();
    }

    __device__
    int mapToGlobalColIndex(int active, int patchIndex, int c = 0) const
    {
        int index = mapper(m_col(c)).index(active, patchIndex) + m_cstr(c);
#if defined(USE_PERMUTATION)
        index = m_perm_old2new(index);
#endif
        return index;
    }

    __device__
    int mapColIndex(int active, int patchIndex, int c = 0) const
    { return mapper(m_col(c)).globalIndex(active, patchIndex); }

#if 0
    __device__
    void pushToMatrix(double value, int activeRow, int activeCol,
                      DeviceNestedArrayView<double> eliminatedDofs,
                      int r, int c)
    {
        int rstrLocal = 0;
        int cstrLocal = 0;

        DofMapperDeviceView rowMap = mapper(m_row(r));
        DofMapperDeviceView colMap = mapper(m_col(c));
        const int ii = m_rstr(r) + activeRow;
        const int iiLocal = rstrLocal + activeRow;
        if (rowMap.is_free_index(activeRow))
        {
            const int jj = m_cstr(c) + activeCol;
            const int jjLocal = cstrLocal + activeCol;
            if (colMap.is_free_index(activeCol))
            {
                atomicAdd(&m_matrix(ii, jj), value);
            }
            else
            {
                DeviceVectorView<double> eliminatedDofs_j = eliminatedDofs[c];
                atomicAdd(&m_RHS(ii), -value * eliminatedDofs_j
                    (colMap.global_to_bindex(activeCol)));
            }
        }
    }
#endif

    __device__
    void pushToMatrix(double value, int activeRow, int activeCol,
                          DeviceNestedArrayView<double> eliminatedDofs,
                          int r, int c)
    {
        int rstrLocal = 0;
        int cstrLocal = 0;

        DofMapperDeviceView rowMap = mapper(m_row(r));
        DofMapperDeviceView colMap = mapper(m_col(c));
        int ii = m_rstr(r) + activeRow;
        //const int iiLocal = rstrLocal + activeRow;
        //printf("activeRow=%d, activeCol=%d\n", activeRow, activeCol);
        if (rowMap.is_free_index(activeRow))
        {
#if defined(USE_PERMUTATION)
            ii = m_perm_old2new(ii);
#endif
            int jj = m_cstr(c) + activeCol;
            //const int jjLocal = cstrLocal + activeCol;
            if (colMap.is_free_index(activeCol))
            {
                //int out = atomicAdd(counter, 1);
                //m_rows[out] = ii;
                //m_cols[out] = jj;
                //m_values[out] = value;
                //printf("counter=%d, ii=%d, jj=%d, value=%f\n", out, ii, jj, value);
#if defined(USE_PERMUTATION)
                jj = m_perm_old2new(jj);
#endif
                //printf("after permutation: ii=%d, jj=%d, value=%f\n", ii, jj, value);
                atomicAdd(&m_csrMatrix(ii, jj), value);
            }
            else
            {
                DeviceVectorView<double> eliminatedDofs_j = eliminatedDofs[c];
                atomicAdd(&m_RHS(ii), -value * eliminatedDofs_j
                    (colMap.global_to_bindex(activeCol)));
                //printf("ii=%d, jj=%d\n", ii, jj);
            }
        }
    }

    __device__
    void pushToEntryIndex(int activeRow, int activeCol,
                          int r, int c, int* counter,
                          DeviceVectorView<int> rows,
                          DeviceVectorView<int> cols) const
    {
        int rstrLocal = 0;
        int cstrLocal = 0;

        DofMapperDeviceView rowMap = mapper(m_row(r));
        DofMapperDeviceView colMap = mapper(m_col(c));
        int ii = m_rstr(r) + activeRow;
        const int iiLocal = rstrLocal + activeRow;
        //printf("activeRow=%d, activeCol=%d\n", activeRow, activeCol);
        if (rowMap.is_free_index(activeRow))
        {
            int jj = m_cstr(c) + activeCol;
            const int jjLocal = cstrLocal + activeCol;
            if (colMap.is_free_index(activeCol))
            {
#if defined(USE_PERMUTATION)
                ii = m_perm_old2new(ii);
                jj = m_perm_old2new(jj);
#endif
                int out = atomicAdd(counter, 1);
                rows[out] = ii;
                cols[out] = jj;
                //printf("counter=%d, ii=%d, jj=%d, value=%f\n", out, ii, jj, value);
            }
        }
    }


    __device__
    bool isEntry(int activeRow, int activeCol, int r, int c) const
    {
        DofMapperDeviceView rowMap = mapper(m_row(r));
        DofMapperDeviceView colMap = mapper(m_col(c));
        return rowMap.is_free_index(activeRow) && 
               colMap.is_free_index(activeCol);
    }

    __device__
    void pushToRhs(double value, int activeRow, int r)
    {
        int rstrLocal = 0;
        DofMapperDeviceView rowMap = mapper(m_row(r));
        int ii = m_rstr(r) + activeRow;
        const int iiLocal = rstrLocal + activeRow;
        if (rowMap.is_free_index(activeRow))
        {
#if defined(USE_PERMUTATION)
            ii = m_perm_old2new(ii);
#endif
            atomicAdd(&m_RHS(ii), value);
        }
    }

#if 0
     __device__
    void pushToRhs_coo(double value, int activeRow, int r)
    {
        int rstrLocal = 0;
        DofMapperDeviceView rowMap = mapper(m_row(r));
        const int ii = m_rstr(r) + activeRow;
        const int iiLocal = rstrLocal + activeRow;
        if (rowMap.is_free_index(activeRow))
            atomicAdd(&m_RHS_coo(ii), value);
    }
#endif

    //__host__ __device__
    //DeviceMatrixView<double> matrix() const { return m_matrix; }

    __host__ __device__
    DeviceVectorView<double> rhs() const { return m_RHS; }

    //__host__ __device__
    //DeviceVectorView<double> rhs_coo() const { return m_RHS_coo; }

    //__host__ __device__
    //DeviceVectorView<int> cols() const { return m_cols; }

    //__host__ __device__
    //DeviceVectorView<int> rows() const { return m_rows; }

    //__host__ __device__
    //DeviceVectorView<double> values() const { return m_values; }
};