#include "MultiPatchData.h"

MultiPatchData::MultiPatchData(const MultiPatch &multiPatch)
{
    m_basisDim = multiPatch.getBasisDim();
    m_CPDim = multiPatch.getCPDim();
    m_numPatches = multiPatch.getNumPatches();

    m_orders = multiPatch.getBasisOrders();
    m_numKnots = multiPatch.getBasisNumKnots();
    m_knots = multiPatch.getBasisKnots();
    m_numGpAndEle = multiPatch.getNumGpAndEle();

    m_orders_ref = multiPatch.getGeoOrders();
    m_numKnots_ref = multiPatch.getGeoNumKnots();
    m_knots_ref = multiPatch.getGeoKnots();

    m_controlPoints = multiPatch.getControlPoints();
    m_numControlPoints = multiPatch.getNumControlPoints();
}

int *MultiPatchData::getNumKnots_ptr()
{
    return thrust::raw_pointer_cast(m_numKnots.data());
}

double *MultiPatchData::getKnots_ptr()
{
    return thrust::raw_pointer_cast(m_knots.data());
}

int *MultiPatchData::getOrders_ptr()
{
    return thrust::raw_pointer_cast(m_orders.data());
}

double *MultiPatchData::getControlPoints_ptr()
{
    return thrust::raw_pointer_cast(m_controlPoints.data());
}

int *MultiPatchData::getNumControlPoints_ptr()
{
    return thrust::raw_pointer_cast(m_numControlPoints.data());
}

int *MultiPatchData::getNumGpAndEle_ptr()
{
    return thrust::raw_pointer_cast(m_numGpAndEle.data());
}

int *MultiPatchData::getNumKnots_ref_ptr()
{
    return thrust::raw_pointer_cast(m_numKnots_ref.data());
}

double *MultiPatchData::getKnots_ref_ptr()
{
    return thrust::raw_pointer_cast(m_knots_ref.data());
}

int *MultiPatchData::getOrders_ref_ptr()
{
    return thrust::raw_pointer_cast(m_orders_ref.data());
}
