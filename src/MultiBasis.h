#pragma once

#include <vector>
#include "TensorBsplineBasis.h"
#include "MultiPatch.h"
#include <DeviceArray.h>

class MultiBasis
{
private:
    std::vector<TensorBsplineBasis> m_bases;
    BoxTopology m_topology;
public:
    MultiBasis() = default;

    MultiBasis(const MultiPatch& multiPatch);

    ~MultiBasis() = default;

    void addBasis(const TensorBsplineBasis& basis) { m_bases.push_back(basis); }
    void clear() { m_bases.clear(); }

    int getDim() const { return m_bases[0].getDim(); }
    int getNumBases() const { return static_cast<int>(m_bases.size()); }
    const TensorBsplineBasis& basis(int index) const { return m_bases[index]; }

    void uniformRefine(int patchIndex, int direction, int numKnots);
    void uniformRefine(int direction, int numKnots);
    void uniformRefine(int numKnots = 1);

    void getMapper(bool conforming, const BoundaryConditions& bc,
                   int unk, DofMapper& dofMapper, bool finalize = true) const;

    void getMappers(bool conforming, const BoundaryConditions& bc,
                    std::vector<DofMapper>& dofMappers, bool finalize = true) const;

    void matchInterface(const BoundaryInterface & bi, DofMapper & mapper) const;

    int numGPs() const;
    int totalNumGPs() const;
    int totalNumBdGPs() const;
    int totalNumElements() const;

    int totalNumGPsOnBdries(const std::deque<boundary_condition>& bcs) const;

    void getData(std::vector<int>& intData,
                 std::vector<double>& knotsPools) const;

    void getData(std::vector<int>& intData,
                 std::vector<double>& knotsPools,
                 std::vector<std::vector<int>>& multSumsOffsets,
                 std::vector<std::vector<int>>& multSums) const;

    void getData(DeviceArray<int>& intData,
                 DeviceArray<double>& knotsPools) const
    {
        std::vector<int> intDataVec;
        std::vector<double> knotsPoolsVec;
        getData(intDataVec, knotsPoolsVec);
        intData = intDataVec;
        knotsPools = knotsPoolsVec;
    }

    void getData(DeviceArray<int>& intData,
                 DeviceArray<double>& knotsPools,
                 DeviceNestedArray<int>& multSumsOffsets,
                 DeviceNestedArray<int>& multSums) const
    {
        std::vector<int> intDataVec;
        std::vector<double> knotsPoolsVec;
        std::vector<std::vector<int>> multSumsOffsetsVec;
        std::vector<std::vector<int>> multSumsVec;
        getData(intDataVec, knotsPoolsVec, 
                multSumsOffsetsVec, multSumsVec);
        intData = intDataVec;
        knotsPools = knotsPoolsVec;
        multSumsOffsets = DeviceNestedArray<int>(multSumsOffsetsVec);
        multSums = DeviceNestedArray<int>(multSumsVec);
    }

    void giveBasis(MultiPatch& multiPatch, int targetDim) const;

    void degreeElevate(bool eleInternal = true, int const i = 1, int const dir = -1)
    {
        for (int k = 0; k < m_bases.size(); k++)
            m_bases[k].degreeElevate(i, dir, eleInternal);
    }

    int numActive(int patchIndex, int direction) const
    { return m_bases[patchIndex].numActive(direction); }

    int numActive(int patchIndex) const
    { return m_bases[patchIndex].numActive(); }

    int numActive() const
    { return m_bases[0].numActive(); }

    int knotOrder() const { return m_bases[0].getOrder(0); }

};