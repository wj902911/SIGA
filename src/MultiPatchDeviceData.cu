#include <MultiPatchDeviceData.h>
#include <vector>

#if 0
__host__
MultiPatchDeviceData::MultiPatchDeviceData(const MultiPatch& multiPatch)
    : m_numPatches(multiPatch.getNumPatches()),
      m_domainDim(multiPatch.getBasisDim()),
      m_targetDim(multiPatch.getCPDim()),
      m_intData(),
      m_doubleData()
{
    std::vector<int> intData;
    std::vector<double> doubleData;
    multiPatch.getData(intData, doubleData);

    m_intData = DeviceArray<int>(intData);
    m_doubleData = DeviceArray<double>(doubleData);
}
#endif

__host__ 
MultiPatchDeviceView MultiPatchDeviceData::deviceView() const
{
    //const int* intDataPtr = m_intData.data();
    //const double* doubleDataPtr = m_doubleData.data();

    //int* knotsOffset = const_cast<int*>(intDataPtr);
    //int knotsOrdersStartIdx = m_numPatches * m_domainDim + 1;
    //int* knotsOrders = const_cast<int*>(intDataPtr + m_intDataOffsets[1]);
    //double* knotsPool = const_cast<double*>(doubleDataPtr);

    //int controlPointsOffsetStartIdx = m_intData.size() - (m_numPatches + 1);
    //int* controlPointsOffset = const_cast<int*>(intDataPtr + m_intDataOffsets[2]);
    //double* controlPointsPool = const_cast<double*>(doubleDataPtr + m_doubleDataOffsets[1]);

    int patchIntDataOffsetsSize = m_numPatches + 1;
    int* dataStart = m_intData.data();
    DeviceVectorView<int> patchIntDataOffsets(dataStart, patchIntDataOffsetsSize);
    dataStart += patchIntDataOffsetsSize;
    DeviceVectorView<int> patchKnotsPoolOffsets(dataStart, patchIntDataOffsetsSize);
    dataStart += patchIntDataOffsetsSize;
    DeviceVectorView<int> intData(dataStart,
                                  m_intData.size() - (patchIntDataOffsetsSize * 2));
    DeviceVectorView<double> knotsPools(m_knotsPools.data(), m_knotsPools.size());
    DeviceVectorView<int> patchControlPointsPoolOffsets(m_patchControlPointsPoolOffsets.data(), 
                                                        m_patchControlPointsPoolOffsets.size());
    DeviceVectorView<double> controlPointsPools(m_controlPointsPools.data(), 
                                                m_controlPointsPools.size());
    return MultiPatchDeviceView(m_numPatches,
                                m_domainDim,
                                m_targetDim,
                                patchIntDataOffsets,
                                patchKnotsPoolOffsets,
                                patchControlPointsPoolOffsets,
                                intData,
                                knotsPools,
                                controlPointsPools);                                
}
