#pragma once

//#include <cuda_runtime.h>
//#include <thrust/device_vector.h>
#include "Patch.h"
#include "DofMapper.h"
#include "BoxTopology.h"
#include <DeviceArray.h>

class MultiPatch
{
public:
    MultiPatch() = default;
    ~MultiPatch() = default;

    void addPatch(Patch& patch);
    //void update();

    int getBasisDim() const;
    int getCPDim() const;
    int getNumPatches() const;
	Eigen::VectorXi coefSlice(int patchIndex, int dir, int k) const;
    void getMapper(bool conforming,
                   const BoundaryConditions& bc,
                   int unk,
                   DofMapper& dofMapper,
                   bool finalize = true) const;

    void getMappers(bool conforming,
                    const BoundaryConditions& bc,
                    std::vector<DofMapper>& dofMappers,
                    bool finalize = true) const;

    void matchInterface(const BoundaryInterface & bi,
                        DofMapper & mapper) const;

#if 0
    const thrust::device_vector<int>& getNumKnots() const;
    const thrust::device_vector<double>& getKnots() const;
    const thrust::device_vector<int>& getOrders() const;
    const thrust::device_vector<double>& getControlPoints() const;
    const thrust::device_vector<int>& getNumControlPoints() const;
    const thrust::device_vector<int>& getNumGpAndEle() const;
    const thrust::device_vector<int>& getNumKnots_ref() const;
    const thrust::device_vector<double>& getKnots_ref() const;
    const thrust::device_vector<int>& getOrders_ref() const;
#endif

    std::vector<int> getBasisNumKnots(int patchIndex) const;
    std::vector<double> getBasisKnots(int patchIndex) const;
    std::vector<int> getBasisOrders(int patchIndex) const;
    Eigen::MatrixXd getControlPoints(int patchIndex) const;
    int getNumControlPoints(int patchIndex) const;
    int getTotalNumControlPoints() const;
    std::vector<int> getNumGpAndEle(int patchIndex) const;
    std::vector<int> getGeoNumKnots(int patchIndex) const;
    std::vector<double> getGeoKnots(int patchIndex) const;
    std::vector<int> getGeoOrders(int patchIndex) const;

    std::vector<int> getBasisNumKnots() const;
    std::vector<double> getBasisKnots() const;
    std::vector<int> getBasisOrders() const;
    std::vector<double> getControlPoints() const;
    std::vector<int> getNumControlPoints() const;
    std::vector<int> getNumGpAndEle() const;
    std::vector<int> getGeoNumKnots() const;
    std::vector<double> getGeoKnots() const;
    std::vector<int> getGeoOrders() const;

    int getTotalNumKnots() const;

    int getTotalNumGaussPoints() const;

    const std::vector<double>& getGeoKnots(int patchIndex, int direction) const;
    const std::vector<double>& getBasisKnots(int patchIndex, int direction) const;
    //const thrust::device_vector<double>& getControlPoints_ref() const;
    //const thrust::device_vector<int>& getNumControlPoints_ref() const;

#if 0
    int* getNumKnots_ptr();
    double* getKnots_ptr();
    int* getOrders_ptr();
    double* getControlPoints_ptr();
    int* getNumControlPoints_ptr();
    int* getNumGpAndEle_ptr();
    int* getNumKnots_ref_ptr();
    double* getKnots_ref_ptr();
    int* getOrders_ref_ptr();
    //double* getControlPoints_ref_ptr();
    //int* getNumControlPoints_ref_ptr();
#endif

    //int getNumKnots(int patchIndex, int direction) const;
    //int getKnotOrder(int patchIndex, int direction) const;
    //thrust::device_vector<int>::iterator patchNumKnotBegin(int patchIndex);
    //thrust::device_vector<double>::iterator knotBegin(int patchIndex, int direction) const;
    //thrust::device_vector<double>::iterator knotUBegin(int patchIndex, int direction) const;
    //thrust::device_vector<double>::iterator knotEnd(int patchIndex, int direction) const;
    //thrust::device_vector<double>::iterator knotUEnd(int patchIndex, int direction) const;

    //void uniformRefineWithoutUpdate(int patchIndex, int direction, int numKnots = 1);
    void uniformRefine(int patchIndex, int direction, int numKnots);
    void uniformRefine(int direction, int numKnots);
    void uniformRefine(int numKnots = 1);

    const TensorBsplineBasis& basis(int patchIndex) const;
    const Patch& patch(int patchIndex) const;
    Patch& patch(int patchIndex);
    const BoxTopology& topology() const { return m_topology; }
    bool computeTopology( double tol = 1e-4, bool cornersOnly = false, bool tjunctions = false);
    static bool matchVerticesOnSide (const Eigen::MatrixXd& cc1, const std::vector<BoxCorner> &ci1, int start, 
                                     const Eigen::MatrixXd& cc2, const std::vector<BoxCorner> &ci2,
                                     const Eigen::Vector<bool, -1>& matched, Eigen::VectorXi &dirMap, 
                                     Eigen::Vector<bool, -1>& dirO,
                                     double tol, int reference = 0);
#if 0
    void getData(std::vector<int>& intData,
                 std::vector<double>& doubleData) const;
#endif

    void getData(std::vector<int>& intData,
                 std::vector<double>& knotsPools,
                 std::vector<int>& patchControlPointsPoolOffsets,
                 std::vector<double>& controlPointsPools) const;
    void getData(DeviceArray<int>& intData,
                 DeviceArray<double>& knotsPools,
                 DeviceArray<int>& patchControlPointsPoolOffsets,
                 DeviceArray<double>& controlPointsPools) const
    {
        std::vector<int> intData_vec;
        std::vector<double> knotsPools_vec;
        std::vector<int> patchControlPointsPoolOffsets_vec;
        std::vector<double> controlPointsPools_vec;
        getData(intData_vec, knotsPools_vec, patchControlPointsPoolOffsets_vec, controlPointsPools_vec);
        intData = intData_vec;
        knotsPools = knotsPools_vec;
        patchControlPointsPoolOffsets = patchControlPointsPoolOffsets_vec;
        controlPointsPools = controlPointsPools_vec;
    }

    void clear()
    {
        m_bases.clear();
        m_patches.clear();
        m_topology.clearAll();
    }

private:
    int m_basisDim = 0;
    int m_CPDim = 0;
#if 0
    //int m_numPatches = 0;
    thrust::device_vector<double> m_knots;
    thrust::device_vector<int> m_numKnots;
    thrust::device_vector<int> m_orders;
    thrust::device_vector<int> m_numGpAndEle;

    thrust::device_vector<double> m_knots_ref;
    thrust::device_vector<int> m_numKnots_ref;
    thrust::device_vector<int> m_orders_ref;
    //thrust::device_vector<double> m_controlPoints_ref;
    //thrust::device_vector<int> m_numcontrolPoints_ref;

    thrust::device_vector<double> m_controlPoints;
    thrust::device_vector<int> m_numcontrolPoints;
#endif
    std::vector<TensorBsplineBasis> m_bases;
    std::vector<Patch> m_patches;
    BoxTopology m_topology;
};
