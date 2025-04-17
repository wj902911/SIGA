#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "MultiPatch.h"

class MultiPatchData
{
public:
    MultiPatchData() = default;

    MultiPatchData(const MultiPatch& multiPatch);

    ~MultiPatchData() = default;

    int getBasisDim() const { return m_basisDim; }
    int getCPDim() const { return m_CPDim; }
    int getNumPatches() const { return m_numPatches; }

    int* getNumKnots_ptr();
    double* getKnots_ptr();
    int* getOrders_ptr();
    double* getControlPoints_ptr();
    int* getNumControlPoints_ptr();
    int* getNumGpAndEle_ptr();
    int* getNumKnots_ref_ptr();
    double* getKnots_ref_ptr();
    int* getOrders_ref_ptr();

private:
    int m_basisDim = 0;
    int m_CPDim = 0;
    int m_numPatches = 0;

    thrust::device_vector<double> m_knots;
    thrust::device_vector<int> m_numKnots;
    thrust::device_vector<int> m_orders;
    thrust::device_vector<int> m_numGpAndEle;

    thrust::device_vector<double> m_knots_ref;
    thrust::device_vector<int> m_numKnots_ref;
    thrust::device_vector<int> m_orders_ref;

    thrust::device_vector<double> m_controlPoints;
    thrust::device_vector<int> m_numControlPoints;
};