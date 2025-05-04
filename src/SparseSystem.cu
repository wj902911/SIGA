#include "SparseSystem.h"
#include <iostream>

//template __global__ void parallPlus<int>(int* a, int* b, int* c, int n);
//template __global__ void parallPlus<int>(int* a, int b, int* c, int n);
//template __global__ void destructKernel<DeviceObjectArray<double>>(DeviceObjectArray<double>* a, size_t n);

#if 1
__global__
void dofMapperTestKernel(DeviceObjectArray<DofMapper_d>* dofMappers)
{
    (*dofMappers)[1].getDofs().print();
}
#endif

__global__
void checkMapperKernel(DofMapper_d* d_mapper)
{
    d_mapper->getDofs().print();
}

__global__
void constructDofMapperOnDeviceKernel(DofMapper_d* d_mapper, const int* dofs, int dofsSize, 
                                      const int* offset, int offsetSize, const  int* numFreeDofs, 
                                      int numFreeDofsSize, const int* numElimDofs, int numElimDofsSize, 
                                      const int* numCpldDofs, int numCpldDofsSize, const int* tagged, 
                                      int taggedSize, int m_shift, int bshift, int curElimId)
{
    printArray(dofs, dofsSize, "dofs:");
    new (d_mapper) DofMapper_d(dofs, dofsSize, offset, offsetSize, numFreeDofs,
                              numFreeDofsSize, numElimDofs, numElimDofsSize, numCpldDofs,
                              numCpldDofsSize, tagged, taggedSize, m_shift, bshift, curElimId);
}

SparseSystem::SparseSystem(const std::vector<DofMapper> &mappers, 
                           const Eigen::VectorXi &dims)
                           : m_row(dims.sum()),
                             m_col(dims.sum()),
                             m_rstr(dims.sum()),
                             m_cstr(dims.sum()),
                             //m_cvar(dims.sum()),
                             m_dims(dims),
                             m_mappers(mappers.size())
{
    const int d = dims.size();
    const int s = dims.sum();
    const int ms = mappers.size();

    //m_mappers.swap(mappers);
    //DofMapper_d* h_mappers = new DofMapper_d[ms];
    for (int i = 0; i < ms; ++i)
    {
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!need check~~~
        // Copy each DofMapper to the DeviceObjectArray
        m_mappers.at(i) = DofMapper_d(mappers[i]);
        //h_mappers[i] = DofMapper_d(mappers[i]);
    #if 0
        DofMapper_d mapperTemp(mappers[i]);
        constructDofMapperOnDeviceKernel<<<1,1>>>(m_mappers.data() + i, 
            mapperTemp.getDofs().data(), mapperTemp.getDofs().size(), mapperTemp.getOffset().data(), 
            mapperTemp.getOffset().size(), mapperTemp.getNumFreeDofs().data(), 
            mapperTemp.getNumFreeDofs().size(), mapperTemp.getNumElimDofs().data(), 
            mapperTemp.getNumElimDofs().size(), mapperTemp.getNumCpldDofs().data(), 
            mapperTemp.getNumCpldDofs().size(), mapperTemp.getTagged().data(), 
            mapperTemp.getTagged().size(), mapperTemp.getShift(), mapperTemp.getBShift(),
            mapperTemp.getCurElimId());

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
    #endif
    }
    //m_mappers.parallelDataSetting(h_mappers, ms);
    //delete[] h_mappers;
#if 0
    DeviceObjectArray<DofMapper_d>* d_mappers = nullptr;
    cudaMalloc((void**)&d_mappers, sizeof(DeviceObjectArray<DofMapper_d>));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(d_mappers, &m_mappers, sizeof(DeviceObjectArray<DofMapper_d>), 
               cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    dofMapperTestKernel<<<1, 1>>>(d_mappers);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaFree(d_mappers);
#endif

    int k=0;
    for(int i=0;i<d;++i)
        for(int j=0; j<dims[i];++j)
        {
            m_row[k] = i;
            ++k;
        }
        m_col = m_row;
        //m_col.print("m_col:");
    if (ms == 1 )
    {
        m_row.setZero();
        m_col.setZero();
        m_cvar.setZero(1);
    }
    else if ( ms == 2*s )
        m_col = m_col + s;

    m_cvar = m_row;
    m_rstr[0] = m_cstr[0] = 0;
    for (int r = 1; r < d; ++r)
        m_rstr[r] = int(m_rstr[r-1]) + mappers[m_row[r-1]].freeSize(); // Use the original mappers to get freeSize
    for (int c = 1; c < d; ++c)
        m_cstr[c] = int(m_cstr[c-1]) + mappers[m_col[c-1]].freeSize(); // Use the original mappers to get freeSize

    m_matrix.resize(int(m_rstr[d-1]) + mappers[m_row[d-1]].freeSize(),
                    int(m_cstr[d-1]) + mappers[m_col[d-1]].freeSize());
                    
    m_RHS.setZero(m_matrix.rows());
}

#if 0
__host__ 
const DofMapper_d &SparseSystem::colMapper_h(int c) const
{
    DeviceObjectArray<int> dofs = getDofs(c);
    dofs.print();
    return m_mappers[c];
}
#endif


__host__ 
DeviceObjectArray<int> SparseSystem::getDofs(int c) const
{ 
    DeviceObjectArray<DofMapper_d>* d_mappers = nullptr;
    cudaMalloc((void**)&d_mappers, sizeof(DeviceObjectArray<DofMapper_d>));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(d_mappers, &m_mappers, sizeof(DeviceObjectArray<DofMapper_d>), 
               cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    int* dofsSizes = new int[m_mappers.size()];
    int * d_dofsSizes = nullptr;
    cudaMalloc((void**)&d_dofsSizes, sizeof(int) * m_mappers.size());
    int blockSize = 256;
    int numBlocks = (m_mappers.size() + blockSize - 1) / blockSize;
    getDofsSizeKernel<<<numBlocks, blockSize>>>(d_dofsSizes, d_mappers);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(dofsSizes, d_dofsSizes, sizeof(int) * m_mappers.size(), 
               cudaMemcpyDeviceToHost);
    DeviceObjectArray<int> dofs(dofsSizes[c]);
    numBlocks = (dofsSizes[c] + blockSize - 1) / blockSize;
    getDofsKernel<<<numBlocks, blockSize>>>(c, dofsSizes[c], dofs.data(), d_mappers);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaFree(d_dofsSizes);
    cudaFree(d_mappers);
    delete[] dofsSizes;
    return dofs;
}
