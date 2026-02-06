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

    int totalNumGPs() const;
    int totalNumBdGPs() const;

    int totalNumGPsOnBdries(const std::deque<boundary_condition>& bcs) const;

    void getData(std::vector<int>& intData,
                 std::vector<double>& knotsPools) const;
    void getData(DeviceArray<int>& intData,
                 DeviceArray<double>& knotsPools) const
    {
        std::vector<int> intDataVec;
        std::vector<double> knotsPoolsVec;
        getData(intDataVec, knotsPoolsVec);
        intData = intDataVec;
        knotsPools = knotsPoolsVec;
    }

    void giveBasis(MultiPatch& multiPatch, int targetDim) const;
};