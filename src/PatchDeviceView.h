#pragma once

#include <cstdio>
#include <TensorBsplineBasisDeviceView.h>
#include <Utility_d.h>

class PatchDeviceView
{
private:
    int m_domainDim = 0;
    int m_targetDim = 0;
    TensorBsplineBasisDeviceView m_basis;
    DeviceMatrixView<double> m_controlPoints;

public:
    __host__ __device__
    PatchDeviceView() = default;
    
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
                printf("Direction %d:\n", d);
                kv.print();
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
	double boundaryActiveControlPointComponent(BoxSide_d const& s, 
                                               DeviceVectorView<double> pt, 
                                               int r, int c) const
    {
        int activeIndex = m_basis.boundaryActiveIndex(s, pt, r);
        //printf("Boundary active index: %d\n", activeIndex);
        return m_controlPoints(activeIndex, c);
    }

    __device__
	double boundaryActiveControlPointComponent(BoxSide_d const& s1, 
                                               BoxSide_d const& s2,
                                               double pt, 
                                               int r, int c) const
    {
         int activeIndex = m_basis.boundaryActiveIndex_3D_edge(s1, s2, pt, r);
         return m_controlPoints(activeIndex, c);
    }

    __device__
	void jacobian(DeviceVectorView<double> pt,
                  DeviceMatrixView<double> basisValuesAndDers,
                  int numDerivatives,
				  DeviceMatrixView<double> result) const
    {
		int dim = m_basis.dim();
		int P = m_basis.knotsOrder(0); //assume same order in all directions
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
					// DeviceMatrixView<double> oneDimGeoValuesAndDers(
					//	valuesAndDersData+(P+1)*(numDerivatives+1)*d, P+1, numDerivatives+1);
					if (d == j)
						dN_rj *= basisValuesAndDers(tensorCoord[d], (numDerivatives + 1)* d + 1);
					else
						dN_rj *= basisValuesAndDers(tensorCoord[d], (numDerivatives + 1) * d);
				}
                for (int i = 0; i < m_targetDim; i++)
					result(i, j) += activeControlPointComponent(pt, r, i) * dN_rj;
			}
		}
    }

    __device__
	void jacobian(int tid, int numThreads,
                  DeviceVectorView<double> pt,
                  DeviceMatrixView<double> basisValuesAndDers,
                  int numDerivatives,
				  DeviceMatrixView<double> result) const
    {
		int dim = m_basis.dim();
		int P = m_basis.knotsOrder(0); //assume same order in all directions
        int numActiveCPs = m_basis.numActiveControlPoints();
        for (int r = tid; r < numActiveCPs; r += numThreads)
		{
            int tensorCoordData[3]; //max 3D
			DeviceVectorView<int> tensorCoord(tensorCoordData, m_basis.dim());
			getTensorCoordinate(dim, P+1, r, tensorCoordData);
			for (int j = 0; j < dim; j++)
			{
                double dN_rj = 1.0;
                for (int d = 0; d < dim; d++)
				{
					// DeviceMatrixView<double> oneDimGeoValuesAndDers(
					//	valuesAndDersData+(P+1)*(numDerivatives+1)*d, P+1, numDerivatives+1);
					if (d == j)
						dN_rj *= basisValuesAndDers(tensorCoord[d], (numDerivatives + 1)* d + 1);
					else
						dN_rj *= basisValuesAndDers(tensorCoord[d], (numDerivatives + 1) * d);
				}
                for (int i = 0; i < m_targetDim; i++)
					atomicAdd(&result(i, j), activeControlPointComponent(pt, r, i) * dN_rj);
			}
		}
    }

     __device__
    void boundaryJacobian(DeviceVectorView<double> pt,
                          DeviceMatrixView<double> basisValuesAndDers,
                          DeviceMatrixView<double> result) const
    {
        
    }

	__device__
	void jacobian(DeviceVectorView<double> pt,
                  int numDerivatives,
				  DeviceMatrixView<double> result) const
	{
		int P = m_basis.knotsOrder(0); //assume same order in all directions
		int dim = m_basis.dim();
		double valuesAndDersData[5*2*3]; //max 4th order, 3D, first derivatives
		DeviceMatrixView<double> basisValuesAndDers(valuesAndDersData, 
		                                            P+1, 
		                                            (numDerivatives+1)*domainDim());
		m_basis.evalAllDers_into(pt, numDerivatives, basisValuesAndDers);
#if 1
        jacobian(pt, basisValuesAndDers, numDerivatives, result);
#else
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
					result(i, j) += activeControlPointComponent(pt, r, i) * dN_rj;
			}
		}
#endif
	}

    __device__
	void jacobian(DeviceVectorView<double> pt,
                  int numDerivatives, double* workingSpace,
				  DeviceMatrixView<double> result) const
	{
        int P1 = m_basis.knotsOrder(0) + 1;
        DeviceMatrixView<double> basisValuesAndDers(workingSpace, P1, (numDerivatives+1)*domainDim());
		m_basis.evalAllDers_into(pt, numDerivatives, workingSpace + P1*(numDerivatives+1)*domainDim(), basisValuesAndDers);
        jacobian(pt, basisValuesAndDers, numDerivatives, result);
	}

    __device__
	void jacobian(int tid, int numThreads,
                  DeviceVectorView<double> pt,
                  int numDerivatives, 
                  DeviceMatrixView<double> basisValuesAndDersTempSpace,
				  DeviceMatrixView<double> result) const
	{
		int P = m_basis.knotsOrder(0); //assume same order in all directions
		int dim = m_basis.dim();
		m_basis.evalAllDers_into(tid, numThreads, pt, numDerivatives, basisValuesAndDersTempSpace);
        __syncthreads();
        jacobian(tid, numThreads, pt, basisValuesAndDersTempSpace, numDerivatives, result);
	}

    __device__
	void jacobian(int tid, int numThreads,
                  DeviceVectorView<double> pt,
                  int numDerivatives, double* workingSpace,
                  DeviceMatrixView<double> basisValuesAndDersTempSpace,
				  DeviceMatrixView<double> result) const
	{
		int P = m_basis.knotsOrder(0); //assume same order in all directions
		int dim = m_basis.dim();
		m_basis.evalAllDers_into(tid, numThreads, pt, numDerivatives, workingSpace, basisValuesAndDersTempSpace);
        __syncthreads();
        jacobian(tid, numThreads, pt, basisValuesAndDersTempSpace, numDerivatives, result);
	}

    __device__
    void boundaryJacobian(BoxSide_d const& s, DeviceVectorView<double> pt,
                          DeviceMatrixView<double> result) const
    {
        int dir = s.direction();
		int P = m_basis.knotsOrder(0); //assume same order in all directions
        int dim = m_basis.dim();
		double valuesAndDersData[5*2*2]; //max 4th order, 2D, first derivatives
        DeviceMatrixView<double> basisValuesAndDers(valuesAndDersData, 
		                                            P+1, 2*(dim-1));
        for (int d = 0, i = 0; d < dim; d++)
        {
            if (d != dir)
            {
                DeviceMatrixView<double> oneDimResults(
                    basisValuesAndDers.data() + i * (basisValuesAndDers.rows() * 2),
                    basisValuesAndDers.rows(), 2);
                m_basis.evalAllDers_into(d, pt[i], 1, oneDimResults);
                i++;
            }
        }
        //printf("Boundary basis values and ders:\n");
        //basisValuesAndDers.print();
        int numActiveCPs = m_basis.numActiveControlPointsWithoutDir(dir);
        for (int r = 0; r < numActiveCPs; r++)
        {
            int tensorCoordData[2] = {0}; //max dim 2
            DeviceVectorView<int> tensorCoord(tensorCoordData, dim - 1);
            getTensorCoordinate(dim - 1, P + 1, r, tensorCoordData);
            //printf("tensor coord for r=%d: ", r);
            //tensorCoord.print();
			for (int j = 0; j < dim - 1; j++)
            {
                double dN_rj = 1.0;
                for (int d = 0; d < dim - 1; d++)
                {
                    DeviceMatrixView<double> oneDimGeoValuesAndDers(
						valuesAndDersData+(P+1)*2*d, P+1, 2);
                    if (d == j)
						dN_rj *= oneDimGeoValuesAndDers(tensorCoord[d], 1);
					else
						dN_rj *= oneDimGeoValuesAndDers(tensorCoord[d], 0);
                }
                //printf("dN_rj for j=%d: %f\n", j, dN_rj);
                for (int i = 0; i < m_targetDim; i++)
					result(i, j) += 
                        boundaryActiveControlPointComponent(s, pt, r, i) 
                        * dN_rj;
            }
        }
    }

    __device__
    void boundaryJacobian(BoxSide_d const& s1, BoxSide_d const& s2, 
                          double pt,
                          DeviceMatrixView<double> result) const
    {
        int dir1 = s1.direction();
        int dir2 = s2.direction();
        if (dir1 == dir2)
        {
            assert("boundaryJacobian: directions are the same");
            return;
        }
        int d = 3 - dir1 - dir2;
        int P = m_basis.knotsOrder(d);
        int dim = m_basis.dim();
		double valuesAndDersData[5*2];
        DeviceMatrixView<double> basisValuesAndDers(valuesAndDersData, 
		                                            P+1, 2);
        m_basis.evalAllDers_into(d, pt, 1, basisValuesAndDers);
        int numActiveCPs = m_basis.numActiveControlPoints(d);
        for (int r = 0; r < numActiveCPs; r++)
        {
            double dN_rj = basisValuesAndDers(r, 1);
            for (int i = 0; i < m_targetDim; i++)
                result(i, 0) += 
                    boundaryActiveControlPointComponent(s1, s2, pt, r, i) 
                    * dN_rj;
        }
    }

    __device__
    void evaluate(DeviceVectorView<double> pt,
                      DeviceVectorView<double> result) const
    {
        int P = m_basis.knotsOrder(0); //assume same order in all directions
		int dim = m_basis.dim();
        double valuesAndDersData[5*1*3]; //max 4th order, 3D, no derivatives
        DeviceMatrixView<double> basisValuesAndDers(valuesAndDersData, 
                                                    P+1, 
                                                    1 * dim);
		m_basis.evalAllDers_into(pt, 0, basisValuesAndDers);
        //printf("Basis values at point:\n");
        //basisValuesAndDers.print();
        int numActiveCPs = m_basis.numActiveControlPoints();
        for (int r = 0; r < numActiveCPs; r++)
        {
            int tensorCoordData[3] = {0}; //max 3D
            DeviceVectorView<int> tensorCoord(tensorCoordData, m_basis.dim());
            getTensorCoordinate(dim, P + 1, r, tensorCoordData);
            double N_r = 1.0;
            for (int d = 0; d < dim; d++)
                N_r *= basisValuesAndDers(tensorCoord[d], d);
            //printf("N_r for r=%d: %f\n", r, N_r);
            for (int i = 0; i < m_targetDim; i++)
            {
                //printf("Adding to result[%d]: %f * %f\n", i, 
                //       activeControlPointComponent(pt, r, i), N_r);
                result(i) += activeControlPointComponent(pt, r, i) * N_r;
            }
        }
    }

    __device__
    void evaluate(int tid, int numThreads, int blockCoord, int numActivePerBlock,
                  DeviceMatrixView<double> basisValuesAndDers,
                  DeviceVectorView<double> pt,
                  DeviceVectorView<double> result) const
    {
        int P = m_basis.knotsOrder(0); //assume same order in all directions
		int dim = m_basis.dim();

        //printf("Basis values at point:\n");
        //basisValuesAndDers.print();
        int numActiveCPs = m_basis.numActiveControlPoints();
        for (int r = blockCoord * numActivePerBlock + tid; 
            r < (blockCoord + 1) * numActivePerBlock && r < numActiveCPs; 
            r += numThreads)
        {
            int tensorCoordData[3] = {0}; //max 3D
            DeviceVectorView<int> tensorCoord(tensorCoordData, m_basis.dim());
            getTensorCoordinate(dim, P + 1, r, tensorCoordData);
            double N_r = 1.0;
            for (int d = 0; d < dim; d++)
                N_r *= basisValuesAndDers(tensorCoord[d], d);
            //printf("N_r for r=%d: %f\n", r, N_r);
            for (int i = 0; i < m_targetDim; i++)
            {
                //printf("Adding to result[%d]: %f * %f\n", i, 
                //       activeControlPointComponent(pt, r, i), N_r);
                atomicAdd(&result(i), activeControlPointComponent(pt, r, i) * N_r);
            }
        }
    }
};