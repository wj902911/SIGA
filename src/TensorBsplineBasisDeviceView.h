#pragma once

#include <cassert>
#include <cstdio>
#include <KnotVectorDeviceView.h>
#include <GaussPointsDeviceView.h>
#include <Utility_d.h>
#include <Boundary_d.h>

class TensorBsplineBasisDeviceView
{
private:
    int m_dim = 0;
    DeviceVectorView<int> m_knotsOffset;
    DeviceVectorView<int> m_knotsOrders;
    //DeviceVectorView<int> m_intData; // combined storage for offsets and orders
    DeviceVectorView<double> m_knotsPool;
    DeviceNestedArrayView<int> m_multSums;

public:
    __device__
    TensorBsplineBasisDeviceView() = default;
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
                                 DeviceVectorView<int> intData,
                                 DeviceVectorView<double> knotsPool,
                                 DeviceVectorView<int> multSumsOffsets,
                                 DeviceVectorView<int> multSums)
                               : m_dim(dim),
                                 m_knotsOffset(intData.data() + dim, dim + 1),
                                 m_knotsOrders(intData.data(), dim),
                                 m_knotsPool(knotsPool),
                                 m_multSums(multSumsOffsets, multSums)
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
                                    m_knotsPool.data() + m_knotsOffset[dir] - m_knotsOffset[0],
                                    m_multSums[dir]);
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
    int numGPsInElement() const 
    { 
        int numGPs = 1;
        for (int d = 0; d < m_dim; d++)
            numGPs *= knotVector(d).numGaussPoints();
        return numGPs;
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
    int totalNumBdGPs() const
    {
        int numGPS = 0;

        for (int d = 0; d < m_dim; d++)
            numGPS += totalNumGPsInDir(d) * pow(2, m_dim - 1);

        return numGPS;
    }

    __device__
    int totalNumBdGPsInDir(int d) const
    { return totalNumGPsInDir(d) * pow(2, m_dim - 1); }

    __device__
    int numElementsInDir(int d) const { return knotVector(d).numElements(); }
    __device__
    int totalNumElements() const
    {
        int numElems = 1;

        for (int d = 0; d < m_dim; d++)
            numElems *= numElementsInDir(d);

        return numElems;
    }

    __device__
    int numEdgesInEachDir() const { return pow(2, m_dim - 1); }
    __device__
    int numEdges() const { return m_dim * pow(2, m_dim - 1); }

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
    void ptCoords_sepId(int gpid, int eleid, DeviceVectorView<int> coords) const
    {
        if (coords.size() != m_dim * 2)
        {
            assert("ptCoords: coords size mismatch");
            return;
        }
        for (int d = 0; d < m_dim; d++)
        {
            int numGPs = knotVector(d).numGaussPoints();
            coords[d] = gpid % numGPs;
            gpid /= numGPs;
        }
        for (int d = m_dim; d < 2 * m_dim; d++)
        {
            int numElements = knotVector(d - m_dim).numElements();
            coords[d] = eleid % numElements;
            eleid /= numElements;
        }
    }

    __device__
    void ptCoords(int idx, int d, DeviceVectorView<int> coords) const
    {
        if (coords.size() != 1 * 2)
        {
            assert("ptCoords: coords size mismatch");
            return;
        }
        int numGPs = knotVector(d).numGaussPoints();
            int numElements = knotVector(d).numElements();
        coords[0] = idx % numGPs;
        idx /= numGPs;
        coords[1] = idx % numElements;
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
            //lower[d] = knotVector(d).knots()[coords[m_dim + d] + order];
            //upper[d] = knotVector(d).knots()[coords[m_dim + d] + order + 1];
            lower[d] = *(knotVector(d).domainUBegin() + coords[m_dim + d]);
            upper[d] = *(knotVector(d).domainUBegin() + coords[m_dim + d] + 1);
        }
    }

    __device__
    void elementSupport(int d, DeviceVectorView<int> coords,
                        double& lower, double& upper) const
    {
        if (coords.size() != 2)
        {
            assert("elementSupport: coords size mismatch");
            return;
        }
        int order = knotVector(d).order();
        //printf("knotVector:\n");
        //knotVector(d).print();
        //printf("ElementSupport dim %d, elem %d, order %d\n", d, coords[1], order);
        //lower = knotVector(d).knots()[coords[1] + order];
        lower = *(knotVector(d).domainUBegin() + coords[1]);
        //printf("Lower knot: %f\n", lower);
        //upper = knotVector(d).knots()[coords[1] + order + 1];
        upper = *(knotVector(d).domainUBegin() + coords[1] + 1);
        //printf("Upper knot: %f\n", upper);
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
    double gsPoint(int gpid, int eleid,
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
        ptCoords_sepId(gpid, eleid, coords);
        DeviceVectorView<double> lower(lowerData, m_dim);
        DeviceVectorView<double> upper(upperData, m_dim);
        elementSupport(coords, lower, upper);
        return gspts.threadGaussPoint(lower, upper, coords, result);
    }

    __device__
    double gsPoint(int idx, int d, GaussPointsDeviceView gspts,
                   double& result) const
    {
        if (d < 0 || d >= m_dim)
        {
            assert("gsPoint: dimension out of range");
            return 0.0;
        }
        int coordsData[2]; //size 2 for single direction
        DeviceVectorView<int> coords(coordsData, 2);
        ptCoords(idx, d, coords);
        //printf("idx=%d coords for dim %d: gp %d, elem %d\n", idx, d, coords[0], coords[1]);
        double lower, upper;
        elementSupport(d, coords, lower, upper);
        //printf("Lower: %f, Upper: %f\n", lower, upper);
        return gspts.threadGaussPoint(d, lower, upper, coords[0], result);
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
    int numActiveControlPointsWithoutDir(int dir) const
    {
        int numAct = 1;
        for (int d = 0; d < m_dim; d++)
            if (d != dir)
                numAct *= knotVector(d).order() + 1;
        return numAct;
    }

    __device__
    int numActiveControlPoints(int d) const
    { return knotVector(d).order() + 1; }

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
    int activeIndex(int d, double pt, int r) const
    {
        int order = knotVector(d).order();
        int firstAct = pt == 
            knotVector(d).domainEnd() ?
            knotVector(d).numKnots() - order - 2 - order :
            upperBound(d, pt) - order - 1;
        //int idx = r % (order + 1);
        return firstAct + r;
        //return firstAct + idx;
    }

    __device__
    int boundaryActiveIndex(BoxSide_d const& s, DeviceVectorView<double>  pt, 
                            int r) const
    {
        int dir = s.direction();
        //printf("dir: %d\n", dir);
        int firstActData[2] = {0}; //max dim 2
        DeviceVectorView<int> firstAct(firstActData, m_dim - 1);
        int sizesData[2] = {0}; //max dim 2
        DeviceVectorView<int> sizes(sizesData, m_dim - 1);
        for (int d = 0, i = 0; d < m_dim; ++d)
        {
            if (d != dir)
            {
                int order = knotVector(d).order();
                //printf("order for dim %d: %d\n", d, order);
                firstAct[i] = pt(i) == 
                knotVector(d).domainEnd() ?
                knotVector(d).numKnots() - order - 2 - order :
                upperBound(d, pt[i]) - order - 1;
                sizes[i] = order + 1;
                i++;
            }
        }
        //printf("firstAct: ");
        //firstAct.print();
        //printf("sizes: ");
        //sizes.print();
        int indexData[2] = {0}; //max dim 2
        DeviceVectorView<int> index(indexData, m_dim - 1);
        getTensorCoordinate(m_dim - 1, sizes.data(), r, index.data());
        //printf("index: ");
        //index.print();
        int idx = firstAct[m_dim - 2] + index[m_dim - 2];
        //printf("idx after last dim: %d\n", idx);
        if (m_dim == 3)
            idx = idx * size(0) + firstAct[0] + index[0];
        return boundaryCoeffIndex(s, 0, idx);
    }

    __device__
    int boundaryActiveIndex_2D(BoxSide_d const& s, double pt, int r) const
    {
        int dir = s.direction();
        int d = (dir + 1) % 2;
        int idx = activeIndex(d, pt, r);
        return boundaryCoeffIndex(s, 0, idx);
    }

    __device__
    int boundaryActiveIndex_3D_edge(BoxSide_d const& s1, BoxSide_d const& s2,
                                    double pt, int r) const
    {
        int dir1 = s1.direction();
        int dir2 = s2.direction();
        if (dir1 == dir2)
        {
            assert("boundaryActiveIndex_3D_edge: directions are the same");
            return -1;
        }
        int d = 3 - dir1 - dir2;
        int idx = activeIndex(d, pt, r);
        return boundaryCoeffIndex_3D_edge(s1, s2, idx);
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

#if 0
        int span = 0;
        if (u == knotVector(dir).domainEnd())
            span = knotVector(dir).numKnots() - knotVector(dir).order() - 2;
        else
            span = upperBound(dir, u) - 1;
#else
        const double* span = knotVector(dir).iFind(u);
#endif

        ndu[0] = 1.0;
        for (int j = 1; j <= p; ++j)
        {
#if 0
            left[j] = u - knotVector(dir).knots()[span + 1 - j];
            right[j] = knotVector(dir).knots()[span + j] - u;
#else
            left[j] = u  - *(span+1-j);
            right[j] = *(span+j) - u;
#endif
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

        if (n == 0)
            return;
            
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
    void evalAllDers_into(int dir, double u, int n, double* workingSpace,
                          DeviceMatrixView<double> results) const
    {
        int p = knotVector(dir).order();
        int p1 = p + 1;
#if 1
        int dataStart = 0;
        DeviceVectorView<double> ndu(workingSpace + dataStart, p1 * p1);
        dataStart += p1 * p1;
        DeviceVectorView<double> left(workingSpace + dataStart, p1);
        dataStart += p1;
        DeviceVectorView<double> right(workingSpace + dataStart, p1);
        dataStart += p1;
        DeviceVectorView<double> a(workingSpace + dataStart, 2 * p1);
        dataStart += 2 * p1;
#else
        double nduData[5 * 5]; //max order 4
        DeviceVectorView<double> ndu(nduData, p1 * p1);
        double leftData[5]; //max order 4
        DeviceVectorView<double> left(leftData, p1);
        double rightData[5]; //max order 4
        DeviceVectorView<double> right(rightData, p1);
        double aData[2 * 5]; //max order 4
        DeviceVectorView<double> a(aData, 2 * p1);
#endif

        if (!knotVector(dir).inDomain(u))
        {
            for(int k=0; k<=n; k++)
                for (int j=0; j<=p; j++)
                    results(j, k) = 0.0;
            return;
        }


        const double* span = knotVector(dir).iFind(u);

        ndu[0] = 1.0;
        for (int j = 1; j <= p; ++j)
        {
            left[j] = u  - *(span+1-j);
            right[j] = *(span+j) - u;
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

        if (n == 0)
            return;
            
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

    __device__
    void evalAllDers_into(DeviceVectorView<double> pt,
                          int n, double* workingSpace,
                          DeviceMatrixView<double> results) const
    {
        for (int d = 0; d < m_dim; d++)
        {
            DeviceMatrixView<double> oneDimResults(
                results.data() + d * (results.rows() * (n + 1)),
                results.rows(), n + 1);
            evalAllDers_into(d, pt[d], n, workingSpace, oneDimResults);
        }
    }

    __device__
    void evalAllDers_into(int tid, int numThreads,
                          DeviceVectorView<double> pt,
                          int n,
                          DeviceMatrixView<double> results) const
    {
        for (int d = tid; d < m_dim; d += numThreads)
        {
            //printf("Thread %d evaluating all derivatives for dimension %d.\n", tid, d);
            DeviceMatrixView<double> oneDimResults(
                results.data() + d * (results.rows() * (n + 1)),
                results.rows(), n + 1);
            //printf("%d\n", oneDimResults.size());
            evalAllDers_into(d, pt[d], n, oneDimResults);
            //printf("Thread %d finished evaluating all derivatives for dimension %d.\n", tid, d);
        }
    }

    __device__
    void evalAllDers_into(int tid, int numThreads,
                          DeviceVectorView<double> pt,
                          int n, double* workingSpace,
                          DeviceMatrixView<double> results) const
    {
        for (int d = tid; d < m_dim; d += numThreads)
        {
            DeviceMatrixView<double> oneDimResults(
                results.data() + d * (results.rows() * (n + 1)),
                results.rows(), n + 1);
            int p1 = knotVector(d).order() + 1;
            int dataStride = p1 * p1 + p1 * 4;
            evalAllDers_into(d, pt[d], n, workingSpace + d * dataStride, oneDimResults);
        }
    }

    __device__
    void componentsForSide(BoxSide_d const& s,
                           TensorBsplineBasisDeviceView result) const
    {
        int dir = s.direction();
        int targetDim = m_dim - 1;
        if (result.dim() != targetDim || 
            result.knotsOrders().size() != targetDim || 
            result.knotsOffsets().size() != m_dim)
        {
            assert("ComponentsForSide: result dim mismatch");
            return;
        }
        result.knotsOffsets()[0] = 0;
        for (int i = 0, n = 0; i < m_dim; i++)
            if (i != dir)
            {
                KnotVectorDeviceView kv = knotVector(i);
                int numKnots = kv.numKnots();
                int order = kv.order();
                result.knotsOrders()[n] = order;
                result.knotsOffsets()[n + 1] = 
                    result.knotsOffsets()[n] + numKnots;
                for (int j = 0; j < numKnots; j++)
                    result.knotsPool()[result.knotsOffsets()[n] + j] = kv.knots()[j];
                n++;
            }
    }

    __device__
    int coefSliceSize(BoxSide_d const& s) const
    {
        int dir = s.direction();
        int dim = m_dim;
        int sliceSize = 1;
        for(int d = 0; d < dim; ++d)
            sliceSize *= size(d);
        sliceSize /= size(dir);
        return sliceSize;
    }

    __device__
    void coefSlice(int dir, int k, DeviceVectorView<int> res) const
    {
        int dim = m_dim;
        if(dir < 0 || dir >= dim)
            printf("Error: dir is out of range in coefSlice.\n");
        if(k < 0 || k >= size(dir))
            printf("Error: k is out of range in coefSlice.\n");

        int sliceSize = 1;
        int lowData[3] = {0}, uppData[3] = {0}; //max dim 3
        DeviceVectorView<int> low(lowData, m_dim);
        DeviceVectorView<int> upp(uppData, m_dim);
        for(int d = 0; d < dim; ++d)
        {
            sliceSize *= size(d);
            low[d] = 0;
            upp[d] = size(d);
        }
        sliceSize /= upp[dir];
        low[dir] = k;
        upp[dir] = k + 1;

        if (res.size() != sliceSize)
        {
            assert("coefSlice: res size mismatch");
            return;
        }
        int vData[3] = {0}; //max dim 3
        DeviceVectorView<int> v(vData, m_dim);
        v.copyFrom(low);
        int i = 0;
        do
        {
            res(i++) = index(v);
        } while (nextLexicographic_d(v, low, upp));

    }

    __device__
    int index(const DeviceVectorView<int>& coords) const
    {
        int index = 0;
        int dim = m_dim;
        index = coords(dim - 1);
        for (int d = dim - 2; d >= 0; --d)
        {
            index = index * size(d) + coords(d);
        }
        return index;
    }

    __device__
    void boundaryCoeffCoords(BoxSide_d const& s, int offset, int idx, 
                             DeviceVectorView<int> res) const
    {
        int dir = s.direction();
        bool r = s.parameter();
        int k = r ? size(dir) - 1 - offset : offset;
        res[dir] = k;
        for (int d = 0; d < m_dim; ++d)
        {
            if (d != dir)
            {
                int numCPs = size(d);
                res[d] = idx % numCPs;
                idx /= numCPs;
            }
        }
    }

    __device__
    void boundaryCoeffCoords_3D_edge(BoxSide_d const& s1, BoxSide_d const& s2, 
                                    int idx, DeviceVectorView<int> res) const
    {
        int dir1 = s1.direction();
        int dir2 = s2.direction();
        int d = 3 - dir1 - dir2;
        bool r1 = s1.parameter();
        bool r2 = s2.parameter();
        int k1 = r1 ? size(dir1) - 1 : 0;
        int k2 = r2 ? size(dir2) - 1 : 0;
        res[dir1] = k1;
        res[dir2] = k2;
        //int numCPs = size(d);
        //res[d] = idx % numCPs;
        res[d] = idx;
    }

    __device__
    int boundaryCoeffIndex(BoxSide_d const& s, int offset, int idx) const
    {
        int coordsData[3]; //max dim 3
        DeviceVectorView<int> coords(coordsData, m_dim);
        boundaryCoeffCoords(s, offset, idx, coords);
        return index(coords);
    }

    __device__
    int boundaryCoeffIndex_3D_edge(BoxSide_d const& s1, BoxSide_d const& s2, int idx) const
    {
        int coordsData[3];
        DeviceVectorView<int> coords(coordsData, m_dim);
        boundaryCoeffCoords_3D_edge(s1, s2, idx, coords);
        return index(coords);
    }

    __device__
    void boundaryOffset(BoxSide_d const& s, int offset, 
                        DeviceVectorView<int> res) const
    {
        int k = s.direction();
        bool r = s.parameter();
        if (!(offset < size(k))) 
            printf("Offset cannot be bigger than the amount of basis functions orthogonal to Boxside s!\n");
        coefSlice(k, (r?size(k)-1-offset : offset), res);
    }

    __device__
    void boundary(BoxSide_d const& s, DeviceVectorView<int> res) const
    { boundaryOffset(s, 0, res); }
};