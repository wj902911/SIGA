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
    
    DeviceMatrix<double> m_matrix;
    DeviceVector<double> m_RHS;

    DeviceVector<int> m_row;
    DeviceVector<int> m_col;
    DeviceVector<int> m_rstr;
    DeviceVector<int> m_cstr;
    DeviceVector<int> m_cvar;
    DeviceVector<int> m_dims;

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
    
};