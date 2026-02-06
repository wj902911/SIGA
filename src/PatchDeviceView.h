#pragma once

#include <cstdio>
#include <TensorBsplineBasisDeviceView.h>
#include <Utility_d.h>

class PatchDeviceView
{
private:
    int m_domainDim;
    int m_targetDim;
    TensorBsplineBasisDeviceView m_basis;
    DeviceMatrixView<double> m_controlPoints;

public:
    __host__ __device__
    PatchDeviceView(int domainDim, int targetDim,
                    TensorBsplineBasisDeviceView basis,
                    DeviceMatrixView<double> controlPoints)
                  : m_domainDim(domainDim), m_targetDim(targetDim), 
                    m_basis(basis), m_controlPoints(controlPoints)
    {
    }

    __host__ __device__
    PatchDeviceView(int domainDim, int targetDim,
                    DeviceVectorView<int> knotsOffset,
                    DeviceVectorView<int> knotsOrders,
                    DeviceVectorView<double> knotsPool,
                    DeviceMatrixView<double> controlPoints)
                  : m_domainDim(domainDim), m_targetDim(targetDim),
                    m_basis(domainDim, knotsOffset, knotsOrders, knotsPool), 
                    m_controlPoints(controlPoints)
    {
    }

    __device__
    PatchDeviceView(int domainDim, int targetDim,
                    DeviceVectorView<int> intData,
                    DeviceVectorView<double> knotsPool,
                    DeviceMatrixView<double> controlPoints)
                  : m_domainDim(domainDim), m_targetDim(targetDim),
                    m_basis(domainDim, 
                            DeviceVectorView<int>(intData.data(), 2 * domainDim + 1),
                            knotsPool), 
                    m_controlPoints(controlPoints)
    {
    }

    __device__
    int domainDim() const { return m_basis.dim(); }

    __device__
    int targetDim() const { return m_controlPoints.cols(); }

    __device__
    TensorBsplineBasisDeviceView basis() const { return m_basis; }

    __device__
    void print() const
    {
            printf("Knot vectors:\n");
            for (int d = 0; d < m_basis.dim(); d++)
            {
                KnotVectorDeviceView kv = m_basis.knotVector(d);
                printf("Direction %d:\nOrder %d\nKnots:\n", d, kv.order());
                kv.knots().print();
            }
            printf("Control points:\n");
            m_controlPoints.print();
    }
    
    __device__
    int numControlPoints() const { return m_controlPoints.rows(); }

    __device__
    void setCoefficients(int row, int col, double value)
    { m_controlPoints(row, col) = value; }

    __device__
    TensorBsplineBasisDeviceView basis() 
    { return m_basis; }

    __device__
    void activeControlPoints(DeviceVectorView<double> pt,
                             DeviceMatrixView<double> activeCPs) const
    {
      	int activeIndexesData[125]; //max 5^3=125
      	DeviceVectorView<int> activeIndexes(activeIndexesData, 
      	                  m_basis.numActiveControlPoints());
	  	m_basis.activeIndexes(pt, activeIndexes);
	  	int CPDim = targetDim();
        for (int i = 0; i < activeIndexes.size(); i++)
			for (int j = 0; j < CPDim; j++)
				activeCPs(i, j) = m_controlPoints(activeIndexes[i], j);
    }

	__device__
	double activeControlPointComponent(DeviceVectorView<double> pt, int r, int c) const
	{
		int activeIndex = m_basis.activeIndex(pt, r);
		return m_controlPoints(activeIndex, c);
	}

	__device__
	void jacobian(DeviceVectorView<double> pt,
				  DeviceMatrixView<double> jacobian) const
	{
		int numDerivatives = 1;
		int P = m_basis.knotsOrder(0); //assume same order in all directions
		int dim = m_basis.dim();
		double valuesAndDersData[5*2*3]; //max 4th order, 3D, first derivatives
		DeviceMatrixView<double> basisValuesAndDers(valuesAndDersData, 
		                                            P+1, 
		                                            (numDerivatives+1)*domainDim());
		m_basis.evalAllDers_into(pt, numDerivatives, basisValuesAndDers);
		int numActiveCPs = m_basis.numActiveControlPoints();
        for (int r = 0; r < numActiveCPs; r++)
		{
            int tensorCoordData[3]; //max 3D
			DeviceVectorView<int> tensorCoord(tensorCoordData, m_basis.dim());
			getTensorCoordinate(dim, P+1, r, tensorCoordData);
			for (int j = 0; j < dim; j++)
			{
                double dN_rj = 1.0;
                for (int d = 0; d < dim; d++)
				{
					DeviceMatrixView<double> oneDimGeoValuesAndDers(
						valuesAndDersData+(P+1)*(numDerivatives+1)*d, P+1, numDerivatives+1);
					if (d == j)
						dN_rj *= oneDimGeoValuesAndDers(tensorCoord[d], 1);
					else
						dN_rj *= oneDimGeoValuesAndDers(tensorCoord[d], 0);
				}
                for (int i = 0; i < dim; i++)
					jacobian(i, j) += activeControlPointComponent(pt, r, i) * dN_rj;
			}
		}
	}
};