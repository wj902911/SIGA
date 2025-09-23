#pragma once

//#include <cuda_runtime.h>
//#include <thrust/device_vector.h>
//#include <iostream>
#include <DeviceVector.h>
#include <DeviceObjectArray.h>
#include <DofMapper_d.h>
#include <Eigen/Core>

__global__
void getDofsSizeKernel(int* d_size, DeviceObjectArray<DofMapper_d>* d_mappers);

__global__
void getDofsKernel(int index, int size, int* d_dofs, 
                   DeviceObjectArray<DofMapper_d>* d_mappers);

class SparseSystem
{
private:
    DeviceObjectArray<DofMapper_d> m_mappers;
    
    DeviceVector<int> m_row;
    DeviceVector<int> m_col;
    DeviceVector<int> m_rstr;
    DeviceVector<int> m_cstr;
    DeviceVector<int> m_cvar;
    DeviceVector<int> m_dims;

    DeviceMatrix<double> m_matrix;
    DeviceVector<double> m_RHS;

public:
    __host__ __device__
    SparseSystem() = default;

    __host__
    SparseSystem(const std::vector<DofMapper>& mappers, 
                 const Eigen::VectorXi& dims);

    __device__
    DeviceVector<int> mapColIndices(const DeviceVector<int>& actives,
                                    const int patchIndex,
                                    const int c = 0) const
    { return m_mappers[m_col(c)].getGlobalIndices(actives, patchIndex); }

    __device__
    int mapToGlobalColIndex(int active, int patchIndex, int c = 0) const
    { return m_mappers[m_col(c)].index(active, patchIndex) + m_cstr(c); }
    

    __device__
    const DofMapper_d& colMapper(int c) const { return m_mappers[m_col(c)]; }

    #if 0
    __host__
    const DofMapper_d& colMapper_h(int c) const;
    #endif

    __host__
    DeviceObjectArray<int> getDofs(int c) const;

    __host__
    int numColBlocks() const {return m_col.size();}

    __host__
    int numDofs() const { return m_RHS.size(); }

    __device__
    void pushToRhs(const DeviceVector<double>& localRhs, 
                   const DeviceObjectArray<DeviceVector<int>>& actives_vec, 
                   const DeviceVector<int>& r_vec)
    {
    int rstrLocal = 0;
        for (int r_ind = 0; r_ind != r_vec.size(); r_ind++)
        {
            int r = r_vec(r_ind);
            const DofMapper_d& rowMap = m_mappers[m_row(r)];
            const int numActive_i = actives_vec[r].rows();

            for (int i = 0; i != numActive_i; i++)
            {
                const int ii = m_rstr(r) + actives_vec[r](i);
                const int iiLocal = rstrLocal + i;

                if (rowMap.is_free_index(actives_vec[r](i)))
                    atomicAdd(&m_RHS(ii), localRhs(iiLocal));
            }
            rstrLocal += numActive_i;
        }
    }

    __device__
    void pushToMatrix(const DeviceMatrix<double>& localMat, 
                      const DeviceObjectArray<DeviceVector<int>>& actives_vec, 
                      const DeviceObjectArray<DeviceVector<double>>& eliminatedDofs,
                      const DeviceVector<int>& r_vec,
                      const DeviceVector<int>& c_vec)
    {
        int rstrLocal = 0;
        int cstrLocal = 0;

        for (int r_ind = 0; r_ind != r_vec.size(); r_ind++)
        {
            int r = r_vec(r_ind);
            const DofMapper_d& rowMap = m_mappers[m_row(r)];
            const int numActive_i = actives_vec[r].rows();

            for (int c_ind = 0; c_ind != c_vec.size(); c_ind++)
            {
                int c = c_vec(c_ind);
                const DofMapper_d& colMap = m_mappers[m_col(c)];
                const int numActive_j = actives_vec[c].rows();
                const DeviceVector<double>& eliminatedDofs_j = eliminatedDofs[c];

                for (int i = 0; i != numActive_i; i++)
                {
                    const int ii = m_rstr(r) + actives_vec[r](i);
                    const int iiLocal = rstrLocal + i;

                    if (rowMap.is_free_index(actives_vec[r](i)))
                    {
                        for (int j = 0; j != numActive_j; j++)
                        {
                            const int jj = m_cstr(c) + actives_vec[c](j);
                            const int jjLocal = cstrLocal + j;

                            if (colMap.is_free_index(actives_vec[c](j)))
                            {
                                atomicAdd(&m_matrix(ii, jj), localMat(iiLocal, jjLocal));
                            }
                            else
                            {
                                m_RHS(ii) -= localMat(iiLocal, jjLocal) * eliminatedDofs_j(colMap.global_to_bindex(actives_vec[c](j)));
                            }
                        }
                    }
                }
                cstrLocal += numActive_j;
            }
            cstrLocal = 0;
            rstrLocal += numActive_i;   
        }
    }

    __host__ __device__
    DeviceVector<double>& rhs() { return m_RHS; }

    __host__ __device__
    const DeviceVector<double>& rhs() const { return m_RHS; }

    __host__ __device__
    DeviceMatrix<double>& matrix() { return m_matrix; }

    __host__ __device__
    const DeviceMatrix<double>& matrix() const { return m_matrix; }
};