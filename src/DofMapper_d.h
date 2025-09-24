#pragma once

#include "DofMapper.h"
#include "DeviceMatrix.h"
#include "DeviceObjectArray.h"

#define singleDofs 1

#if !singleDofs
    #define MAPPER_PATCH_DOF2(a,b,c) m_dofs[c][m_offset[b]+a]
#else
    #define MAPPER_PATCH_DOF2(a,b) m_dofs[m_offset[b]+a]
#endif

class DofMapper_d
{
public:
    __host__ __device__
    DofMapper_d() : m_shift(0), m_bshift(0), m_curElimId(-1) {}

#if !singleDofs
    __device__
DofMapper_d(int* mapperData)
    {
        int start = 1;
        int numComponents = mapperData[start];
        start += numComponents + 1;
        m_dofs.resize(numComponents);
        for (int i = 0; i < numComponents; ++i)
        {
            m_dofs[i] = DeviceObjectArray<int>(mapperData[2 + i], mapperData + start);
            start += mapperData[2 + i];
        }
        int numOffset = mapperData[start++];
        m_offset = DeviceObjectArray<int>(numOffset, mapperData + start);
        start += numOffset;
        m_shift = mapperData[start++];
        m_bshift = mapperData[start++];
        int numFreeDofs = mapperData[start++];
        m_numFreeDofs = DeviceObjectArray<int>(numFreeDofs, mapperData + start);
        start += numFreeDofs;
        int numElimDofs = mapperData[start++];
        m_numElimDofs = DeviceObjectArray<int>(numElimDofs, mapperData + start);
        start += numElimDofs;
        int numCpldDofs = mapperData[start++];
        m_numCpldDofs = DeviceObjectArray<int>(numCpldDofs, mapperData + start);
        start += numCpldDofs;
        m_curElimId = mapperData[start++];
        int numTagged = mapperData[start++];
        m_tagged = DeviceObjectArray<int>(numTagged, mapperData + start);
    }
#endif

    __device__
    DofMapper_d(const int* dofs, int dofsSize, const int* offset, int offsetSize, 
                const int* numFreeDofs, int numFreeDofsSize, const int* numElimDofs, 
                int numElimDofsSize, const int* numCpldDofs, int numCpldDofsSize, 
                const int* tagged, int taggedSize, int m_shift, int bshift, int curElimId)
        : m_dofs(dofsSize, dofs), m_offset(offsetSize, offset),
          m_numFreeDofs(numFreeDofsSize, numFreeDofs), m_numElimDofs(numElimDofsSize, numElimDofs),
          m_numCpldDofs(numCpldDofsSize, numCpldDofs), m_tagged(taggedSize, tagged),
          m_shift(m_shift), m_bshift(bshift), m_curElimId(curElimId)
    {}

    __host__
    DofMapper_d(const DofMapper & mapper)
#if !singleDofs
        : m_dofs(mapper.componentsSize()), 
#else
        : m_dofs(mapper.getDofs(0).size(),mapper.getDofs(0).data()),
#endif
          m_offset(mapper.getOffset().size(), mapper.getOffset().data()),
          m_numFreeDofs(mapper.getNumFreeDofs().size(), mapper.getNumFreeDofs().data()),
          m_numElimDofs(mapper.getNumElimDofs().size(), mapper.getNumElimDofs().data()),
          m_numCpldDofs(mapper.getNumCpldDofs().size(), mapper.getNumCpldDofs().data()),
          m_tagged(mapper.getTagged().size(), mapper.getTagged().data()),
          m_shift(mapper.getShift()), m_bshift(mapper.getBshift()), m_curElimId(mapper.getCurElimId())
    {
#if 0
        DeviceObjectArray<int>* h_dofs = new DeviceObjectArray<int>[numCommponents];
        for (int i = 0; i < numCommponents; ++i)
            h_dofs[i] = DeviceObjectArray<int>(mapper.getDofs(i).size(), mapper.getDofs(i).data());
        m_dofs.parallelDataSetting(h_dofs, numCommponents);
#endif
#if !singleDofs
        for (int i = 0; i < m_dofs.size(); ++i)
            m_dofs.at(i) = DeviceObjectArray<int>(mapper.getDofs(i).size(), mapper.getDofs(i).data());
#endif
#if 0        
        DeviceObjectArray<DeviceObjectArray<int>>* d_dofs = nullptr;
        cudaMalloc((void**)&d_dofs, sizeof(DeviceObjectArray<int>) * m_dofs.size());
        cudaMemcpy(d_dofs, m_dofs.data(), sizeof(DeviceObjectArray<int>) * m_dofs.size(), cudaMemcpyHostToDevice);
        DeviceObjectArray<int> dofsTest = m_dofs.at(0);
        dofsTest.print();
#endif
    }

    __host__
    DofMapper_d(const DeviceObjectArray<int>& dofs, const DeviceObjectArray<int>& offset,
                const DeviceObjectArray<int>& numFreeDofs, const DeviceObjectArray<int>& numElimDofs,
                const DeviceObjectArray<int>& numCpldDofs, const DeviceObjectArray<int>& tagged,
                int shift, int bshift, int curElimId)
        : m_dofs(dofs), m_offset(offset), m_numFreeDofs(numFreeDofs),
          m_numElimDofs(numElimDofs), m_numCpldDofs(numCpldDofs),
          m_tagged(tagged), m_shift(shift), m_bshift(bshift), m_curElimId(curElimId)
    {

    }

    // Copy constructor
    __host__ __device__
    DofMapper_d(const DofMapper_d& other)
#if !singleDofs
        : m_dofs(other.m_dofs.size()), 
#else
        : m_dofs(other.m_dofs), // Directly copy the DeviceObjectArray<int> objects
#endif
          m_offset(other.m_offset), m_numFreeDofs(other.m_numFreeDofs),
          m_numElimDofs(other.m_numElimDofs), m_numCpldDofs(other.m_numCpldDofs),
          m_tagged(other.m_tagged), m_shift(other.m_shift), m_bshift(other.m_bshift),
          m_curElimId(other.m_curElimId) 
    {
#if !singleDofs
    #if defined(__CUDA_ARCH__)
        for (int i = 0; i < m_dofs.size(); ++i)
            m_dofs[i] = other.m_dofs[i];
    #else
        DeviceObjectArray<DeviceObjectArray<int>>* d_otherDofs = nullptr;
        cudaMalloc((void**)&d_otherDofs, sizeof(DeviceObjectArray<int>) * other.m_dofs.size());
        cudaMemcpy(d_otherDofs, &other.m_dofs, sizeof(DeviceObjectArray<int>) * m_dofs.size(), cudaMemcpyHostToDevice);
        // Copy the dof arrays from the other object
        for (int i = 0; i < m_dofs.size(); ++i)
        {
            // Create a new DeviceObjectArray for each component
            DeviceObjectArray<int> tempArray = other.m_dofs.at(i);
            m_dofs.at(i) = tempArray;
        }
        cudaFree(d_otherDofs);
    #endif
#endif
        //printf("DofMapper_d copy constructor used on device!\n");
        //m_dofs.print();
    }

    // Move constructor
    __host__ __device__
    DofMapper_d(DofMapper_d&& other) noexcept
        : m_dofs(std::move(other.m_dofs)), m_offset(std::move(other.m_offset)),
          m_numFreeDofs(std::move(other.m_numFreeDofs)), m_numElimDofs(std::move(other.m_numElimDofs)),
          m_numCpldDofs(std::move(other.m_numCpldDofs)), m_tagged(std::move(other.m_tagged)),
          m_shift(other.m_shift), m_bshift(other.m_bshift), m_curElimId(other.m_curElimId)
    {
        other.m_shift = 0;
        other.m_bshift = 0;
        other.m_curElimId = -1;
    }

    // Copy assignment operator
    __host__ __device__
    DofMapper_d& operator=(const DofMapper_d& other)
    {
        if (this != &other)
        {
            m_dofs = other.m_dofs;
            m_offset = other.m_offset;
            m_numFreeDofs = other.m_numFreeDofs;
            m_numElimDofs = other.m_numElimDofs;
            m_numCpldDofs = other.m_numCpldDofs;
            m_tagged = other.m_tagged;
            m_shift = other.m_shift;
            m_bshift = other.m_bshift;
            m_curElimId = other.m_curElimId;
        }
        return *this;
    }

    // Move assignment operator
    __host__ __device__
    DofMapper_d& operator=(DofMapper_d&& other) noexcept
    {
        if (this != &other)
        {
            m_dofs = std::move(other.m_dofs);
            m_offset = std::move(other.m_offset);
            m_numFreeDofs = std::move(other.m_numFreeDofs);
            m_numElimDofs = std::move(other.m_numElimDofs);
            m_numCpldDofs = std::move(other.m_numCpldDofs);
            m_tagged = std::move(other.m_tagged);
            m_shift = other.m_shift;
            m_bshift = other.m_bshift;
            m_curElimId = other.m_curElimId;
            other.m_shift = 0;
            other.m_bshift = 0;
            other.m_curElimId = -1;
        }
        return *this;
    }

    __host__
    DofMapper_d clone() const 
    {
        return DofMapper_d(*this);
    }

#if !singleDofs
    __device__
    const DeviceObjectArray<int>& getDofs(int comp) const
    {
        if (comp < 0 || comp >= static_cast<int>(m_dofs.size()))
            return DeviceObjectArray<int>();
        return m_dofs[comp];
    }

    __host__
    const DeviceObjectArray<int>& getDofs_h(int comp) const
    {

    }
#else
    __host__ __device__
    const DeviceObjectArray<int>& getDofs() const
    {
        return m_dofs; 
    }

#endif

    __host__ __device__
    DeviceObjectArray<int> getOffset() const { return m_offset; }

    __host__ __device__
    DeviceObjectArray<int> getNumFreeDofs() const { return m_numFreeDofs; }

    __host__ __device__
    DeviceObjectArray<int> getNumElimDofs() const { return m_numElimDofs; }

    __host__ __device__
    DeviceObjectArray<int> getNumCpldDofs() const { return m_numCpldDofs; }

    __host__ __device__
    DeviceObjectArray<int> getTagged() const { return m_tagged; }

    __host__ __device__
    int getShift() const { return m_shift; }

    __host__ __device__
    int getBShift() const { return m_bshift; }

    __host__ __device__
    int getCurElimId() const { return m_curElimId; }

    __device__
    void localToGlobal(const DeviceVector<int>& locals, int patchIndex, 
                       DeviceVector<int>& globals, int comp = 0) const
    {
        const int numActive = locals.rows();
        globals.resize(numActive);
        for (int i = 0; i < numActive; ++i)
        {
            globals(i) = index(locals(i), patchIndex, comp);
        }
    }

    __device__
    DeviceVector<int> getGlobalIndices(const DeviceVector<int>& locals, 
                                       int patchIndex, int comp = 0) const
    {
        DeviceVector<int> globals;
        localToGlobal(locals, patchIndex, globals, comp);
        return globals;
    }

    __device__
    int index(int dof, int patch, int comp = 0) const
    {
        assert(m_curElimId>=0 && "DofMapper_d::index() called before finalize()");
#if !singleDofs 
        return MAPPER_PATCH_DOF2(dof, patch, comp)+m_shift;
#else
        return MAPPER_PATCH_DOF2(dof, patch) + m_shift;
#endif
    }

    __host__ __device__
    int freeSize() const { return m_curElimId; }

    __host__ __device__
    int boundarySize() const
    {
        assert(m_curElimId>=0 && "DofMapper_d::boundarySize() called before finalize()");
        return m_numElimDofs.back();
    }

    __host__ __device__
    int bindex(int i, int k = 0, int c = 0) const
    {
        assert(m_curElimId>=0 && "DofMapper_d::bindex() called before finalize()");
#if !singleDofs
        return MAPPER_PATCH_DOF2(i,k,c) - m_numFreeDofs.back() + m_bshift;
#else
        return MAPPER_PATCH_DOF2(i,k) - m_numFreeDofs.back() + m_bshift;
#endif
    }

    __device__
    bool is_free_index(int gl) const
    { 
        printf("gl:%d, m_curElimId:%d\n", gl, m_curElimId);
        return gl < m_curElimId + m_shift; 
    }

    __device__
    bool is_free(int i, int k = 0) const
    { return is_free_index(index(i, k)); }

    __device__
    int global_to_bindex(int gl) const
    {
        assert(!is_free_index(gl) && "DofMapper_d::global_to_bindex() called with free dof");
        return gl - m_numFreeDofs.back() + m_bshift - m_shift;
    }

private:
#if !singleDofs
    DeviceObjectArray<DeviceObjectArray<int>> m_dofs;
#else
    DeviceObjectArray<int> m_dofs;
#endif
    DeviceObjectArray<int> m_offset;
    DeviceObjectArray<int> m_numFreeDofs;
    DeviceObjectArray<int> m_numElimDofs;
    DeviceObjectArray<int> m_numCpldDofs;
    DeviceObjectArray<int> m_tagged;
    int m_shift;
    int m_bshift;
    int m_curElimId;
};