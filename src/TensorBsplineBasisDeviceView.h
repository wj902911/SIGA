#pragma once

#include <cassert>
#include <cstdio>
#include <KnotVectorDeviceView.h>
#include <GaussPointsDeviceView.h>
#include <Utility_d.h>

class TensorBsplineBasisDeviceView
{
private:
    int m_dim;
    DeviceVectorView<int> m_knotsOffset;
    DeviceVectorView<int> m_knotsOrders;
    //DeviceVectorView<int> m_intData; // combined storage for offsets and orders
    DeviceVectorView<double> m_knotsPool;

public:
    __device__
    TensorBsplineBasisDeviceView(int dim,
                                 DeviceVectorView<int> intData,
                                 DeviceVectorView<double> knotsPool)
                               : m_dim(dim),
                                 m_knotsOffset(intData.data() + dim, dim + 1),
                                 m_knotsOrders(intData.data(), dim),
                                 m_knotsPool(knotsPool)
    {
    }

    __device__
    TensorBsplineBasisDeviceView(int dim,
                                 DeviceVectorView<int> knotsOffset,
                                 DeviceVectorView<int> knotsOrders,
                                 DeviceVectorView<double> knotsPool)
                               : m_dim(dim),
                                 m_knotsOffset(knotsOffset),
                                 m_knotsOrders(knotsOrders),
                                 m_knotsPool(knotsPool)
    {
    }

    __device__
    KnotVectorDeviceView knotVector(int dir) const
    {
        return KnotVectorDeviceView(m_knotsOrders[dir],
                                    m_knotsOffset[dir + 1] - m_knotsOffset[dir],
                                    m_knotsPool.data() + m_knotsOffset[dir] - m_knotsOffset[0]);
    }

    __device__
    int dim() const { return m_dim; }

    __device__
    DeviceVectorView<int> knotsOffsets() const
    {
        return m_knotsOffset;
    }

    __device__
    int knotsOrder(int dir) const { return m_knotsOrders[dir]; }

    __device__
    DeviceVectorView<int> knotsOrders() const
    {
        return m_knotsOrders;
    }
    __device__
    DeviceVectorView<double> knotsPool() const
    {
        return m_knotsPool;
    }

    __device__
    void print() const
    {
        for (int d = 0; d < m_dim; d++)
        {
            KnotVectorDeviceView kv = knotVector(d);
            kv.print();
        }
    }

    __device__
    int totalNumGPsInDir(int d) const { return knotVector(d).totalNumGaussPoints(); }

    __device__
    int totalNumGPs() const
    {
        int numGPS = 1;

        for (int d = 0; d < m_dim; d++)
            numGPS *= totalNumGPsInDir(d);

        return numGPS;
    }

    __device__
    void ptCoords(int idx, DeviceVectorView<int> coords) const
    {
        if (coords.size() != m_dim * 2)
        {
            assert("ptCoords: coords size mismatch");
            return;
        }
        for (int d = 0; d < m_dim; d++)
        {
            int numGPs = knotVector(d).numGaussPoints();
            coords[d] = idx % numGPs;
            idx /= numGPs;
        }
        for (int d = m_dim; d < 2 * m_dim; d++)
        {
            int numElements = knotVector(d - m_dim).numElements();
            coords[d] = idx % numElements;
            idx /= numElements;
        }
    }

    __device__
    void elementSupport(DeviceVectorView<int> coords,
                        DeviceVectorView<double> lower,
                        DeviceVectorView<double> upper) const
    {
        if (coords.size() != m_dim * 2 )
        {
            assert("elementSupport: coords size mismatch");
            return;
        }
        if (lower.size() != m_dim || upper.size() != m_dim)
        {
            assert("elementSupport: lower/upper size mismatch");
            return;
        }

        for (int d = 0; d < m_dim; d++)
        {
            int order = knotVector(d).order();
            lower[d] = knotVector(d).knots()[coords[m_dim + d] + order];
            upper[d] = knotVector(d).knots()[coords[m_dim + d] + order + 1];
        }
    }

    __device__
    double gsPoint(int idx,
                   GaussPointsDeviceView gspts,
                   DeviceVectorView<double> result) const
    {
        if (result.size() != m_dim)
        {
            assert("gsPoint: result size mismatch");
            return 0.0;
        }
        int coordsData[6]; //max dim 3
        double lowerData[3], upperData[3]; //max dim 3
        DeviceVectorView<int> coords(coordsData, m_dim * 2);
        ptCoords(idx, coords);
        DeviceVectorView<double> lower(lowerData, m_dim);
        DeviceVectorView<double> upper(upperData, m_dim);
        elementSupport(coords, lower, upper);
        return gspts.threadGaussPoint(lower, upper, coords, result);
    }

    __device__
    int upperBound(int direction, double value) const
    { return knotVector(direction).upperBound(value); }

    __device__
    int numActiveControlPoints() const
    {
        int numAct = 1;
        for (int d = 0; d < m_dim; d++)
            numAct *= knotVector(d).order() + 1;
        return numAct;
    }

    __device__
    int size(int d) const { return  knotVector(d).numControlPoints(); }

    __device__
    void activeIndexes(DeviceVectorView<double> pt,
                       DeviceVectorView<int> activeIndexes) const
    {
        int numAct = numActiveControlPoints();
        int firstActData[3]; //max dim 3
        DeviceVectorView<int> firstAct(firstActData, m_dim);
        int sizesData[3]; //max dim 3
        DeviceVectorView<int> sizes(sizesData, m_dim);
        for (int d = 0; d < m_dim; ++d)
        {
            int order = knotVector(d).order();
            firstAct[d] = pt(d) == 
            knotVector(d).domainEnd() ?
            knotVector(d).numKnots() - order - 2 - order :
            upperBound(d, pt[d]) - order - 1;
            sizes[d] = order + 1;
        }
        for (int r = 0; r < numAct; r++)
        {
            int indexData[3]; //max dim 3
            DeviceVectorView<int> index(indexData, m_dim);
            getTensorCoordinate(m_dim, sizes.data(), r, index.data());
            int gidx = firstAct[m_dim - 1] + index[m_dim - 1];
            for (int d = m_dim - 2; d >= 0; d--)
                gidx = gidx * size(d) + firstAct[d] + index[d];
            activeIndexes[r] = gidx;
        }
    }

    __device__
    int activeIndex(DeviceVectorView<double> pt, int r) const
    {
        int firstActData[3]; //max dim 3
        DeviceVectorView<int> firstAct(firstActData, m_dim);
        int sizesData[3]; //max dim 3
        DeviceVectorView<int> sizes(sizesData, m_dim);
        for (int d = 0; d < m_dim; ++d)
        {
            int order = knotVector(d).order();
            firstAct[d] = pt(d) == 
            knotVector(d).domainEnd() ?
            knotVector(d).numKnots() - order - 2 - order :
            upperBound(d, pt[d]) - order - 1;
            sizes[d] = order + 1;
        }
        int indexData[3]; //max dim 3
        DeviceVectorView<int> index(indexData, m_dim);
        getTensorCoordinate(m_dim, sizes.data(), r, index.data());
        int gidx = firstAct[m_dim - 1] + index[m_dim - 1];
        for (int d = m_dim - 2; d >= 0; d--)
            gidx = gidx * size(d) + firstAct[d] + index[d];
        return gidx;
    }

    __device__
    void evalAllDers_into(int dir, double u, int n,
                          DeviceMatrixView<double> results) const
    {
        int p = knotVector(dir).order();
        int p1 = p + 1;
        double nduData[5 * 5]; //max order 4
        DeviceVectorView<double> ndu(nduData, p1 * p1);
        double leftData[5]; //max order 4
        DeviceVectorView<double> left(leftData, p1);
        double rightData[5]; //max order 4
        DeviceVectorView<double> right(rightData, p1);
        double aData[2 * 5]; //max order 4
        DeviceVectorView<double> a(aData, 2 * p1);

        if (!knotVector(dir).inDomain(u))
        {
            for(int k=0; k<=n; k++)
                for (int j=0; j<=p; j++)
                    results(j, k) = 0.0;
            return;
        }

        int span = 0;
        if (u == knotVector(dir).domainEnd())
            span = knotVector(dir).numKnots() - knotVector(dir).order() - 2;
        else
            span = upperBound(dir, u) - 1;

        ndu[0] = 1.0;
        for (int j = 1; j <= p; ++j)
        {
            left[j] = u - knotVector(dir).knots()[span + 1 - j];
            right[j] = knotVector(dir).knots()[span + j] - u;
            double saved = 0.0;
            for (int r = 0; r < j; ++r) {
                ndu[j * p1 + r] = right[r + 1] + left[j - r];
                double temp = ndu[r * p1 + j - 1] / ndu[j * p1 + r];
                ndu[r * p1 + j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            ndu[j * p1 + j] = saved;
        }

        for(int j = 0; j <= p; j++)
            results(j, 0) = ndu[j * p1 + p];

        for (int r = 0; r <= p; r++)
        {
            double* a1 = &a[0];
            double* a2 = &a[p1];

            a1[0] = 1.0;

            for(int k = 1; k <= n; k++)
            {
                int rk,pk,j1,j2 ;
                double der = 0.0;
                rk = r-k ; pk = p-k ;
                if(r >= k)
                {
                    a2[0] = a1[0] / ndu[ (pk+1)*p1 + rk] ;
                    der = a2[0] * ndu[rk*p1 + pk] ;
                }
                j1 = ( rk >= -1  ? 1   : -rk   );
                j2 = ( r-1 <= pk ? k-1 : p - r );
                for(int j = j1; j <= j2; j++)
                {
                    a2[j] = (a1[j] - a1[j-1]) / ndu[ (pk+1)*p1 + rk+j ] ;
                    der += a2[j] * ndu[ (rk+j)*p1 + pk ] ;
                }
                if(r <= pk)
                {
                    a2[k] = -a1[k-1] / ndu[ (pk+1)*p1 + r ] ;
                    der += a2[k] * ndu[ r*p1 + pk ] ;
                }
                results(r, k) = der;
                double* temp = a1;
                a1 = a2;
                a2 = temp;
            }
        }

        int r = p;
        for (int k = 1; k <= n; k++)
        {
            for (int j = 0; j <= p; j++)
                results(j, k) = results(j, k) * double(r);
            r *= p - k;
        }
    }

    __device__
    void evalAllDers_into(DeviceVectorView<double> pt,
                          int n,
                          DeviceMatrixView<double> results) const
    {
        for (int d = 0; d < m_dim; d++)
        {
            DeviceMatrixView<double> oneDimResults(
                results.data() + d * (results.rows() * (n + 1)),
                results.rows(), n + 1);
            evalAllDers_into(d, pt[d], n, oneDimResults);
        }
    }

};