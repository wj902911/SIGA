#include "SparseSystem_d.h"
#include <iostream>

//template __global__ void parallPlus<int>(int* a, int* b, int* c, int n);
//template __global__ void parallPlus<int>(int* a, int b, int* c, int n);
//template __global__ void destructKernel<DeviceObjectArray<double>>(DeviceObjectArray<double>* a, size_t n);

__global__
void getDofsSizeKernel(int* d_size, DeviceObjectArray<DofMapper_d>* d_mappers)
{ 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_mappers->size())
    {
        //printf("getDofsSizeKernel: %d\n", idx);
        //(*d_mappers)[idx].getDofs().print();
        d_size[idx] = (*d_mappers)[idx].getDofs().size();
        //printf("getDofsSizeKernel: %d, %d\n", idx, d_size[idx]);
    }
}

__global__
void getDofsKernel(int index, int size, int* d_dofs, 
                   DeviceObjectArray<DofMapper_d>* d_mappers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) 
        d_dofs[idx] = (*d_mappers)[index].getDofs().data()[idx];
}

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

SparseSystem_d::SparseSystem_d(const std::vector<DofMapper> &mappers, 
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
#if 0
    DeviceObjectArray<DofMapper_d>* d_mappers = nullptr;
    cudaMalloc((void**)&d_mappers, sizeof(DeviceObjectArray<DofMapper_d>));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(d_mappers, &m_mappers, sizeof(DeviceObjectArray<DofMapper_d>), 
               cudaMemcpyHostToDevice);
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

    m_matrix.setZero(int(m_rstr[d-1]) + mappers[m_row[d-1]].freeSize(),
                    int(m_cstr[d-1]) + mappers[m_col[d-1]].freeSize());
                    
    m_RHS.setZero(m_matrix.rows());
}

__device__
SparseSystem_d::SparseSystem_d(const int *intData, const double *doubleData)
    : m_mappers(intData[0])
{
    int offset = 0;
    int numMappers = intData[offset++];
    // Initialize m_mappers
    for (int i = 0; i < numMappers; ++i)
    {
        // Each mapper's data size is stored in intData[offset]
        int mapperDataSize = intData[offset++];
        m_mappers[i] = DofMapper_d(intData + offset);
        offset += mapperDataSize;
    }
    // Initialize m_row
    int numRows = intData[offset++];
    m_row = DeviceVector<int>(numRows, intData + offset);
    offset += numRows;
    // Initialize m_col
    int numCols = intData[offset++];
    m_col = DeviceVector<int>(numCols, intData + offset);
    offset += numCols;
    // Initialize m_rstr
    int numRstr = intData[offset++];
    m_rstr = DeviceVector<int>(numRstr, intData + offset);
    offset += numRstr;
    // Initialize m_cstr
    int numCstr = intData[offset++];
    m_cstr = DeviceVector<int>(numCstr, intData + offset);
    offset += numCstr;
    // Initialize m_cvar
    int numCvar = intData[offset++];
    m_cvar = DeviceVector<int>(numCvar, intData + offset);
    offset += numCvar;
    // Initialize m_dims
    int numDims = intData[offset++];
    m_dims = DeviceVector<int>(numDims, intData + offset);
    offset += numDims;

    // Initialize m_matrix and m_RHS
    int numMatrixRows = intData[offset++];
    int numMatrixCols = intData[offset++];
    int numRowsRHS = intData[offset++];
    m_matrix = DeviceMatrix<double>(numMatrixRows, numMatrixCols, doubleData);
    m_RHS = DeviceVector<double>(numRowsRHS, doubleData + numMatrixRows * numMatrixCols);


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
DeviceObjectArray<int> SparseSystem_d::getDofs(int c) const
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

#if 0
__device__
void SparseSystem::pushToRhs(const DeviceVector<double> &localRhs, 
                             const DeviceObjectArray<DeviceVector<int>> &actives_vec, 
                             const DeviceVector<int> &r_vec)
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
#endif
