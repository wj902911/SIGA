#include "MultiBasis.h"

MultiBasis::MultiBasis(const MultiPatch &multiPatch)
: m_topology(multiPatch.topology())
{
    m_bases.reserve(multiPatch.getNumPatches());
    for (int i = 0; i < multiPatch.getNumPatches(); i++)
    {
        const Patch& patch = multiPatch.patch(i);
        m_bases.push_back(patch.getBasis());
    }
}

void MultiBasis::uniformRefine(int patchIndex, int direction, int numKnots)
{ m_bases[patchIndex].uniformRefine(direction, numKnots); }

void MultiBasis::uniformRefine(int direction, int numKnots)
{
    for (int i = 0; i < m_bases.size(); i++)
    {
        m_bases[i].uniformRefine(direction, numKnots);
    }
}

void MultiBasis::uniformRefine(int numKnots)
{
    for (int i = 0; i < m_bases.size(); i++)
    {
        m_bases[i].uniformRefine(numKnots);
    }
}

void MultiBasis::getMapper(bool conforming, const BoundaryConditions &bc, 
                           int unk, DofMapper &dofMapper, bool finalize) const
{
    dofMapper= DofMapper(m_bases, bc, unk);

    if ( conforming )
    {
        for (std::vector<BoundaryInterface>::const_iterator it = m_topology.iBegin(); 
             it != m_topology.iEnd(); ++it)
        {
            matchInterface(*it, dofMapper);
        }
    }

    if (finalize)
    {
        dofMapper.finalize();
    }
}

void MultiBasis::getMappers(bool conforming, const BoundaryConditions &bc, 
                            std::vector<DofMapper> &dofMappers, bool finalize) const
{
    dofMappers = std::vector<DofMapper>(m_bases.size());
    for (int d = 0; d < m_bases.size(); d++)
    {
        getMapper(conforming, bc, d, dofMappers[d], finalize);
    }
}

void MultiBasis::matchInterface(const BoundaryInterface &bi, DofMapper &mapper) const
{
    Eigen::MatrixXi b1, b2;
    m_bases[bi.first().patchIndex()].matchWith(bi, m_bases[bi.second().patchIndex()],
                                         b1, b2);

    for (size_t i = 0; i!=mapper.componentsSize(); ++i)
        mapper.matchDofs(bi.first().patchIndex(), b1, bi.second().patchIndex(), b2, i );
}

int MultiBasis::totalNumGPs() const
{
    int totalNumGaussPoints = 0;
    for (int i = 0; i < m_bases.size(); i++)
    {
        totalNumGaussPoints += m_bases[i].getTotalNumGaussPoints();
    }
    return totalNumGaussPoints;
}

int MultiBasis::totalNumBdGPs() const
{
    int totalNumBoundaryGaussPoints = 0;
    for (int i = 0; i < m_bases.size(); i++)
    {
        totalNumBoundaryGaussPoints += m_bases[i].getTotalNumBoundaryGaussPoints();
    }
    return totalNumBoundaryGaussPoints;
}

int MultiBasis::totalNumGPsOnBdries(const std::deque<boundary_condition> &bcs) const
{
    int totalNumGaussPoints = 0;
    int dim = m_bases[0].getDim();
    for (auto it = bcs.begin(); it != bcs.end(); ++it)
    {
        BoxSide side = it -> side();
        int fixDir = side.direction();
        bool param = side.parameter();

    }
    return 0;
}

void MultiBasis::getData(std::vector<int> &intData, 
                         std::vector<double> &knotsPools) const
{
    intData.clear();
    knotsPools.clear();

    int numPatches = static_cast<int>(m_bases.size());
    intData.reserve(2 * numPatches + 2);
    int patchIntDataOffsets = 0;
    intData.push_back(patchIntDataOffsets);
    for (int i = 0; i < numPatches; i++)
    {
        patchIntDataOffsets += m_bases[i].getIntDataSize();
        intData.push_back(patchIntDataOffsets);
    }
    int patchKnotsPoolOffsets = 0;
    intData.push_back(patchKnotsPoolOffsets);
    for (int i = 0; i < numPatches; i++)
    {
        patchKnotsPoolOffsets += m_bases[i].getTotalNumKnots();
        intData.push_back(patchKnotsPoolOffsets);
    }
    knotsPools.reserve(patchKnotsPoolOffsets);
    for (int i = 0; i < numPatches; i++)
    {
        std::vector<int> patchIntData;
        std::vector<double> patchKnotsPool;
        m_bases[i].getData(patchIntData, patchKnotsPool);
        intData.reserve(intData.size() + patchIntData.size());
        intData.insert(intData.end(), patchIntData.begin(), patchIntData.end());
        knotsPools.insert(knotsPools.end(), patchKnotsPool.begin(), patchKnotsPool.end());
    }
}

void MultiBasis::giveBasis(MultiPatch &multiPatch, int targetDim) const
{
    multiPatch.clear();
    for (int i = 0; i < m_bases.size(); i++)
    {
        Patch patch(m_bases[i], targetDim);
        multiPatch.addPatch(patch);
    }
}
