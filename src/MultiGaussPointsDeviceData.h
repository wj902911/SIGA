#pragma once

#include <MultiGaussPointsDeviceView.h>
#include <MultiBasis.h>

class MultiGaussPointsDeviceData
{
private:
    int m_dim = 0;
    std::vector<int> m_intDataOffsets;//m_patchGaussPointsOffsets start, m_offsetsPool start, m_offsetsPool end
    std::vector<int> m_doubleDataOffsets;//m_gaussWeightsPool start, m_gaussWeightsPool end
    DeviceArray<int> m_intData; //m_patchOffsetsOffsets, m_patchGaussPointsOffsets, m_offsetsPool
    DeviceArray<double> m_doubleData; //m_gaussPointsPool, m_gaussWeightsPool
public:
    __host__
    MultiGaussPointsDeviceData(const std::vector<std::vector<int>>& numGPs)
    {
        m_dim = static_cast<int>(numGPs[0].size());
        m_intDataOffsets.resize(3);
        m_doubleDataOffsets.resize(2);
        std::vector<int> patchOffsetsOffsetsVec;
        std::vector<int> patchGaussPointsOffsetsVec;
        std::vector<int> offsetsPoolVec;
        std::vector<double> gaussPointsPoolVec;
        std::vector<double> gaussWeightsPoolVec;
        std::vector<int> intDataVec;
        std::vector<double> doubleDataVec;
        int totalGPs = 0;
        int totalOffsets = 0;
        int numPatches = static_cast<int>(numGPs.size());
        for (int p = 0; p < numPatches; ++p)
        {
            totalOffsets += m_dim;
            for (int d = 0; d < m_dim; ++d)
            {
                totalGPs += numGPs[p][d];
            }
        }
        patchOffsetsOffsetsVec.reserve(numPatches);
        patchGaussPointsOffsetsVec.reserve(numPatches);
        offsetsPoolVec.reserve(totalOffsets);
        gaussPointsPoolVec.reserve(totalGPs);
        gaussWeightsPoolVec.reserve(totalGPs);
        doubleDataVec.reserve(2 * totalGPs);
        intDataVec.reserve(2 * numPatches + totalOffsets);
        for (int p = 0; p < numPatches; ++p)
        {
            int offset = 0;
            for (int d = 0; d < m_dim; ++d)
            {
                std::vector<double> pts;
                std::vector<double> wts;
                setPoints(numGPs[p][d], pts, wts);
                gaussPointsPoolVec.insert(gaussPointsPoolVec.end(), pts.begin(), pts.end());
                gaussWeightsPoolVec.insert(gaussWeightsPoolVec.end(), wts.begin(), wts.end());
                offset += numGPs[p][d];
                offsetsPoolVec.push_back(offset);
            }
            patchOffsetsOffsetsVec.push_back(offsetsPoolVec.size());
            patchGaussPointsOffsetsVec.push_back(gaussPointsPoolVec.size());
        }
        intDataVec.insert(intDataVec.end(), patchOffsetsOffsetsVec.begin(), patchOffsetsOffsetsVec.end());
        intDataVec.insert(intDataVec.end(), patchGaussPointsOffsetsVec.begin(), patchGaussPointsOffsetsVec.end());
        intDataVec.insert(intDataVec.end(), offsetsPoolVec.begin(), offsetsPoolVec.end());
        doubleDataVec.insert(doubleDataVec.end(), gaussPointsPoolVec.begin(), gaussPointsPoolVec.end());
        doubleDataVec.insert(doubleDataVec.end(), gaussWeightsPoolVec.begin(), gaussWeightsPoolVec.end());
        m_intDataOffsets[0] = static_cast<int>(patchOffsetsOffsetsVec.size());
        m_intDataOffsets[1] = m_intDataOffsets[0] + static_cast<int>(patchGaussPointsOffsetsVec.size());
        m_intDataOffsets[2] = m_intDataOffsets[1] + static_cast<int>(offsetsPoolVec.size());
        m_doubleDataOffsets[0] = static_cast<int>(gaussPointsPoolVec.size());
        m_doubleDataOffsets[1] = m_doubleDataOffsets[0] + static_cast<int>(gaussWeightsPoolVec.size());
        m_intData = intDataVec;
        m_doubleData = doubleDataVec;
    }

    __host__
    MultiGaussPointsDeviceData(const MultiBasis& multiBasis)
    {
        int domainDim = multiBasis.getDim();
        int numPatches = multiBasis.getNumBases();
        std::vector<std::vector<int>> numGPs;
        numGPs.resize(numPatches);
        for (int p = 0; p < numPatches; ++p)
        {
            numGPs[p].resize(domainDim);
            for (int d = 0; d < domainDim; ++d)
            {
                numGPs[p][d] = multiBasis.basis(p).getNumGaussPoints(d);
            }
        }
        *this = MultiGaussPointsDeviceData(numGPs);
    }

    __host__
    MultiGaussPointsDeviceView view() const
    {
        return MultiGaussPointsDeviceView(m_dim,
               DeviceVectorView(m_intData.data(), m_intDataOffsets[0]),
               DeviceVectorView(m_intData.data() + m_intDataOffsets[0], 
                                m_intDataOffsets[1] - m_intDataOffsets[0]),
               DeviceVectorView(m_intData.data() + m_intDataOffsets[1], 
                                m_intDataOffsets[2] - m_intDataOffsets[1]),
               DeviceVectorView(m_doubleData.data(), m_doubleDataOffsets[0]),
               DeviceVectorView(m_doubleData.data() + m_doubleDataOffsets[0], 
                                m_doubleDataOffsets[1] - m_doubleDataOffsets[0]));
    }

    __host__
    void setPoints(int numPts, std::vector<double>& pts, 
                  std::vector<double>& wts)
    {
        pts.resize(numPts);
        wts.resize(numPts);
        switch (numPts)
        {
        case 1:
            pts[0] = 0.0;
            wts[0] = 2.0;
            break;
        case 2:
            pts[0] = -0.577350269189625764509148780502;
            pts[1] =  0.577350269189625764509148780502;
            wts[0] =  1.0;
            wts[1] =  1.0;
            break;
        case 3:
            pts[0] = -0.774596669241483377035853079956;
            pts[1] =  0.000000000000000000000000000000;
            pts[2] =  0.774596669241483377035853079956;
            wts[0] =  0.555555555555555555555555555556;
            wts[1] =  0.888888888888888888888888888889;
            wts[2] =  0.555555555555555555555555555556;
            break;
        case 4:
            pts[0] = -0.861136311594052575223946488893;
            pts[1] = -0.339981043584856264802665759103;
            pts[2] =  0.339981043584856264802665759103;
            pts[3] =  0.861136311594052575223946488893;
            wts[0] =  0.347854845137453857373063949222;
            wts[1] =  0.652145154862546142626936050778;
            wts[2] =  0.652145154862546142626936050778;
            wts[3] =  0.347854845137453857373063949222;
            break;
        default:
            throw std::runtime_error("Gauss points not implemented for " + std::to_string(numPts) + " points");
            break;
        }
    }
};