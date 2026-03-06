#pragma once

#include <vector>
#include "TensorBsplineBasis.h"
#include "BoundaryCondition.h"

#define MAPPER_PATCH_DOF(a,b,c) m_dofs[c][m_offset[b]+a]

class DofMapperInterleaved
{
private:
    std::vector<std::vector<int>> m_dofs;
    std::vector<int> m_offset;
    int m_shift;
    int m_bshift;
    std::vector<int> m_numFreeDofs;
    std::vector<int> m_numElimDofs;
    std::vector<int> m_numCpldDofs;
    int m_curElimId;
    std::vector<int> m_tagged;
public:
    DofMapperInterleaved() = default;

    void init(const std::vector<TensorBsplineBasis>& bases, 
              int nComp = 1);

    void init(const std::vector<TensorBsplineBasis>& bases, 
              const BoundaryConditions &dirichlet,
              int unk = 0);
};
