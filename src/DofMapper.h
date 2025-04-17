#pragma once

#include <vector>
#include "TensorBsplineBasis.h"
#include "BoundaryCondition.h"

#define singleDofs_DofMapper 0

#if !singleDofs_DofMapper
#define MAPPER_PATCH_DOF(a,b,c) m_dofs[c][m_offset[b]+a]
#else
#define MAPPER_PATCH_DOF(a,b) m_dofs[m_offset[b]+a]
#endif


class DofMapper
{
public:
    DofMapper() = default;

    DofMapper(const std::vector<TensorBsplineBasis>& bases, 
              const BoundaryConditions &dirichlet,
              int unk = 0) : m_shift(0), m_bshift(0)
    {
        init(bases, dirichlet, unk);
    }

    void init(const std::vector<TensorBsplineBasis>& bases, 
              int nComp = 1);

    void init(const std::vector<TensorBsplineBasis>& bases, 
              const BoundaryConditions &dirichlet,
              int unk = 0);

    void markBoundary(int k, 
                      const Eigen::VectorXi & boundaryDofs, 
                      int comp = 0);

    void eliminateDof(int i, int k, int comp = 0);

    size_t numPatches() const {return m_offset.size();}
    size_t componentsSize() const {return m_dofs.size();}

    int boundarySize() const
    {
        if (m_curElimId < 0)
            std::cerr << "Error: DofMapper::boundarySize() called before finalize()." << std::endl;
        return m_numElimDofs.back();
    }

    int bindex(int i, int k = 0, int c = 0) const
    {
        if (m_curElimId < 0)
        {
            std::cerr << "Error: DofMapper::bindex() called before finalize()." << std::endl;
            return -1;
        }
#if !singleDofs_DofMapper
        return MAPPER_PATCH_DOF(i,k,c) - m_numFreeDofs.back() + m_bshift;
#else
        return MAPPER_PATCH_DOF(i,k) - m_numFreeDofs.back() + m_bshift;
#endif
    }

    void finalize();
    void finalizeComp(const int comp);

    void matchDof( int u, int i, int v, int j, int comp = 0);

    void matchDofs(int u, const Eigen::MatrixXi & b1,
                   int v, const Eigen::MatrixXi & b2,
                   int comp = 0);

    int getDataSize() const
    {
        int totalSize = 1; // for total size
        totalSize += 1; // for m_dofs.size()
#if !singleDofs_DofMapper
        for (size_t i = 0; i < m_dofs.size(); ++i)
        {
            totalSize += 1; // for size of each dof vector
            totalSize += m_dofs[i].size();
        }
#else
        totalSize += m_dofs.size(); // singleDofs implies m_dofs is a 1D vector
#endif
        totalSize += m_offset.size() + m_numFreeDofs.size() + m_numElimDofs.size() + m_numCpldDofs.size() + m_tagged.size() + 5 + 3;

        return totalSize;
    }

    //std::vector<int> getDofMapperDataVec() const
    std::vector<int> getDofMapperDataVec() const
    {
        int totalSize = getDataSize();
        std::vector<int> data;
        data.reserve(totalSize);
        data.push_back(totalSize); // total size [0]
        data.push_back(m_dofs.size()); // number of components [1]
#if singleDofs_DofMapper
        // singleDofs implies m_dofs is a 1D vector
        data.insert(data.end(), m_dofs.begin(), m_dofs.end()); // directly copy the 1D vector
#else
        for (size_t i = 0; i < m_dofs.size(); ++i)
        {
            data.push_back(m_dofs[i].size()); // size of each dof vector [2 + [1]]
        }
        for (size_t i = 0; i < m_dofs.size(); ++i)
        {
            data.insert(data.end(), m_dofs[i].begin(), m_dofs[i].end());
        }
#endif
        data.push_back(m_offset.size());
        data.insert(data.end(), m_offset.begin(), m_offset.end());
        data.push_back(m_shift);
        data.push_back(m_bshift);
        data.push_back(m_numFreeDofs.size());
        data.insert(data.end(), m_numFreeDofs.begin(), m_numFreeDofs.end());
        data.push_back(m_numElimDofs.size());
        data.insert(data.end(), m_numElimDofs.begin(), m_numElimDofs.end());
        data.push_back(m_numCpldDofs.size());
        data.insert(data.end(), m_numCpldDofs.begin(), m_numCpldDofs.end());
        data.push_back(m_curElimId);
        data.push_back(m_tagged.size());
        data.insert(data.end(), m_tagged.begin(), m_tagged.end());

        return data;
    }

#if singleDofs_DofMapper
    std::vector<int> getDofs() const
    {
        return m_dofs; // singleDofs implies m_dofs is a 1D vector
    }
#else
    //get m_dofs
    std::vector<int> getDofs(int comp) const
    {
        if (comp < 0 || comp >= static_cast<int>(m_dofs.size()))
            return std::vector<int>();
        return m_dofs[comp];
    }
#endif

    //get m_offset
    std::vector<int> getOffset() const { return m_offset; }

    //get m_shift
    int getShift() const { return m_shift; }

    //get m_bshift
    int getBshift() const { return m_bshift; }

    //get m_numFreeDofs
    std::vector<int> getNumFreeDofs() const { return m_numFreeDofs; }

    //get m_numElimDofs
    std::vector<int> getNumElimDofs() const { return m_numElimDofs; }

    //get m_numCpldDofs
    std::vector<int> getNumCpldDofs() const { return m_numCpldDofs; }

    //get m_curElimId
    int getCurElimId() const { return m_curElimId; }

    int freeSize() const { return m_curElimId; }

    //get m_tagged
    std::vector<int> getTagged() const { return m_tagged; }
    
    
private:
    void replaceDofGlobally(int oldIdx, int newIdx);
    void replaceDofGlobally(int oldIdx, int newIdx, int comp);

    void mergeDofsGlobally(int dof1, int dof2);
    void mergeDofsGlobally(int dof1, int dof2, int comp);

private:
#if 1
#if singleDofs_DofMapper
    std::vector<int> m_dofs;
#else
    std::vector<std::vector<int>> m_dofs;
#endif
    std::vector<int> m_offset;
    int m_shift;
    int m_bshift;
    std::vector<int> m_numFreeDofs;
    std::vector<int> m_numElimDofs;
    std::vector<int> m_numCpldDofs;
    int m_curElimId;
    std::vector<int> m_tagged;
#else
    std::vector<thrust::device_vector<int>> m_dofs;
    thrust::device_vector<size_t> m_offset;
    int m_shift;
    int m_bshift;
    thrust::device_vector<int> m_numFreeDofs;
    thrust::device_vector<int> m_numElimDofs;
    thrust::device_vector<int> m_numCpldDofs;
    int m_curElimId;
    thrust::device_vector<int> m_tagged;
#endif
};