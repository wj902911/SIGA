#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <MultiPatch.h>
#include <DeviceObjectArray.h>
//#include <BoxTopology_d.h>
#include <Patch_d.h>

class MultiPatch_d
{
public:
#if 0
    __device__
    MultiPatch_d(int basisDim, int CPDim, int numPatches, double *knots, 
                 int *numKnots, int *orders, double *controlPoints, 
                 int* numControlPoints, int *numGpAndEle)
                 : m_basisDim(basisDim), m_CPDim(CPDim), m_numPatches(numPatches),
                   m_knots(knots), m_numKnots(numKnots), m_orders(orders),
                   m_controlPoints(controlPoints), m_numControlPoints(numControlPoints),
                   m_numGpAndEle(numGpAndEle) { }
#endif
    __host__
    MultiPatch_d() = default;

    __host__ 
    MultiPatch_d(const MultiPatch& mp);
#if 0
    : m_patches(mp.getNumPatches())//, m_topology(mp.topology())
    {
        for (int i = 0; i < mp.getNumPatches(); i++)
            m_patches.at(i) = mp.patch(i);
    }
#endif

    __host__
    MultiPatch_d(int numPatches);

    ~MultiPatch_d()=default;

    __device__
    int getBasisDim() const
    { return m_patches[0].getDim(); }

    __host__
    int getBasisDim_host() const;

    __device__
    int getCPDim() const
    { return m_patches[0].getCPDim(); }

    __host__ __device__
    int getNumPatches() const
    { return m_patches.size(); }

    __host__
    int getNumPatches_host() const;

    __device__
    int getNumControlPoints(int patchIndex) const
    { return m_patches[patchIndex].getNumControlPoints(); }

    __device__
    int getTotalNumControlPoints() const
    { 
        int total = 0;
        for (int i = 0; i < m_patches.size(); i++)
            total += getNumControlPoints(i);
        return total;
    }

    __device__
    int threadPatchAndDof(int idx, int& patch, int& unk) const
    {
        int dim = getCPDim();
        int point_idx_patch = idx;
        int point_idx_dof = idx;
        int total_points = getTotalNumControlPoints();
        for (int d = 0; d < dim; d++)
        {
            if (point_idx_dof < total_points)
            {
                unk = d;
                break;
            }
            point_idx_dof -= total_points;
            point_idx_patch = point_idx_dof;
        }
        for (int i = 0; i < getNumPatches(); i++)
        {
            int patch_points = m_patches[i].getNumControlPoints();
            if (point_idx_patch < patch_points) 
            {
            	patch = i;
            	break;
            }
            point_idx_patch -= patch_points;
        }
        return point_idx_patch;
    }

    __device__ 
    DeviceObjectArray<int> dofCoords(int idx) const
    {
        DeviceObjectArray<int> coords(2);
        int numCPs = getTotalNumControlPoints();
        coords[1] = idx % numCPs;
        idx /= numCPs;
        coords[0] = idx;
        return coords;
    }

    __device__
    const Patch_d& patch(int patchIndex) const
    { return m_patches[patchIndex]; }

    __device__
    void setCoefficients(int patchIndex, int row, int col, double value)
    { m_patches[patchIndex].setCoefficients(row, col, value); }

    __device__
    int threadPatch(int idx, int& patch) const
    {
        int point_idx = idx;
        for (int i = 0; i < m_patches.size(); i++)
        {
            int patch_points = m_patches[i].basis().totalNumGPs();
            if (point_idx < patch_points) 
            {
                patch = i;
                break;
            }
            point_idx -= patch_points;
        }
        return point_idx;
    }

    __device__
    int threadPatch_edge(int idx, int& patch) const
    {
        int point_idx = idx;
        for (int i = 0; i < m_patches.size(); i++)
        {
            int patch_points = totalNumBdGPsInPatch(i);
            if (point_idx < patch_points) 
            {
                patch = i;
                break;
            }
            point_idx -= patch_points;
        }
        return point_idx;
    }

    __device__
    int threadEdgeDir(int idx, int patch, int& dir) const
    {
        int point_idx = idx;
        int dim = getBasisDim();
        for (int d = 0; d < dim; d++)
        {
            int dir_points = m_patches[patch].totalNumBdGPsInDir(d);
            if (point_idx < dir_points) 
            {
                dir = d;
                break;
            }
            point_idx -= dir_points;
        }
        return point_idx;
    }

    __device__
    int threadEdge(int idx, int patch, int dir, int& edge) const
    {
        int point_idx = idx;
        int numEdges = m_patches[patch].getNumEdgesInEachDir();
        for (int e = 0; e < numEdges; e++)
        {
            int edge_points = m_patches[patch].totalNumGPsInDir(dir);
            if (point_idx < edge_points) 
            {
                edge = e + dir * numEdges;
                break;
            }
            point_idx -= edge_points;
        }
        return point_idx;
    }

#if 0
    __device__
    int getTotalNumEdges() const
    {
        int totalNumEdges = 0;
        for (int i = 0; i < m_patches.size(); i++)
        {
            totalNumEdges += m_patches[i].getNumEdges();
        }
        return totalNumEdges;
    }
#endif

    __device__
    double gsPoint(int idx, int patch, const GaussPoints_d& gps,
                   DeviceVector<double>& result) const
    { return m_patches[patch].basis().gsPoint(idx, gps, result); }

    __device__
    double gsPoint(int idx, const DeviceObjectArray<GaussPoints_d>& gps,
                   DeviceVector<double>& result) const
    {
        int patch = 0;
        int point_idx = threadPatch(idx, patch);
        return gsPoint(point_idx, patch, gps[patch], result);
    }

    __device__
    void evalAllDers_into(int patch, int dir, double u, int n, 
                          DeviceObjectArray<DeviceVector<double>>& result) const
    { m_patches[patch].basis().evalAllDers_into(u, dir, n, result); }

    __device__
    void evalAllDers_into(int patch, const DeviceVector<double>& u, int n, 
                          DeviceObjectArray<DeviceVector<double>>& result) const
    { m_patches[patch].basis().evalAllDers_into(u, n, result); }

    __device__
    DeviceMatrix<int> getActiveIndexes(int patchIndex, DeviceVector<double> pt)
    {
        return m_patches[patchIndex].getActiveIndexes(pt);
    }
    
    __device__
    DeviceMatrix<double> getActiveControlPoints(int patchIndex, DeviceVector<double> pt) const
    {
        return m_patches[patchIndex].getActiveControlPoints(pt);
    }

#if 0
    __device__
    DeviceMatrix<double> getActiveControlPoints(int patchIndex, DeviceVector<double> pt) const
    {
        return m_patches[patchIndex].getActiveControlPoints(pt);
    }
#endif

    __device__
    DeviceVector<int> coefSlice(int patch, int dir, int k) const
    {
        return m_patches[patch].coefSlice(dir, k);
    }

    __device__
    Patch_d boundary(int patch, BoxSide_d const& s) const
    { return m_patches[patch].boundary(s); }

    __device__
    void setPatch(int patchIndex, const Patch_d& patch)
    { 
        printf("Setting patch %d in MultiPatch_d\n", patchIndex);
        new (&m_patches[patchIndex]) Patch_d(patch);
        printf("After setting patch %d in MultiPatch_d:\n", patchIndex);
    }

    __device__
    int getNumEdgesInEachDir(int patch) const
    {
        return m_patches[patch].getNumEdgesInEachDir();
    }

    __device__
    int getNumEdgesInPatch(int patch) const
    {
        
        return m_patches[patch].getNumEdgesInEachDir() * getBasisDim();
    }

    __device__
    int getTotalNumEdges() const
    {
        int totalNumEdges = 0;
        for (int i = 0; i < m_patches.size(); i++)
        {
            totalNumEdges += getNumEdgesInPatch(i);
        }
        return totalNumEdges;
    }

    __host__
    int getTotalNumEdges_host() const;

    __device__
    int totalNumGPsInPatch(int patch) const
    {
        return m_patches[patch].totalNumGPs();
    }

    __host__ __device__
    int totalNumBdGPsInPatch(int patch) const
    {
        return m_patches[patch].totalNumBdGPs();
    }

    __device__
    int totalNumGPs() const
    {
        int totalNumGaussPoints = 0;
        for (int i = 0; i < m_patches.size(); i++)
        {
            totalNumGaussPoints += totalNumGPsInPatch(i);
        }
        return totalNumGaussPoints;
    }

    __device__
    int totalNumBdGPs() const
    {
        int totalNumBdGPs = 0;
        for (int i = 0; i < m_patches.size(); i++)
        {
            totalNumBdGPs += totalNumBdGPsInPatch(i);
        }
        return totalNumBdGPs;
    }

    __host__
    void getEdgeLengthes(DeviceVector<double>& lengths) const;

    __host__
    void getPatchLengthes(DeviceVector<double>& lengths) const;

    __host__
    int totalNumBdGPs_host() const;

    __host__
    void getUpperSupports(DeviceMatrix<double>& upperSupports) const;

    __host__
    void getLowerSupports(DeviceMatrix<double>& lowerSupports) const;

    __host__
    void retrieveControlPoints(MultiPatch& mp) const;

    __host__
    void eval_into(const Eigen::MatrixXi &numPointsPerDir, Eigen::MatrixXd& values) const;

#if 0
    __device__
    const int *getPatchNumKnots(int patchIndex) const
    {
        return m_numKnots + patchIndex * m_basisDim;
    }

    __device__
    const double* getPatchKnots(int patchIndex) const
    {
        int patchKnotsOffset = 0;
        for (int i = 0; i < patchIndex * m_basisDim; i++)
        {
            patchKnotsOffset += m_numKnots[i];
        }
        return m_knots + patchKnotsOffset;
    }

    __device__
    const int* getPatchKnotOrders(int patchIndex) const
    {
        return m_orders + patchIndex * m_basisDim;
    }

    __device__
    const int* getPatchNumGpAndEle(int patchIndex) const
    {
        return m_numGpAndEle + patchIndex * m_basisDim * 2;
    }

    __device__
    int getPatchNumActiveControlPoints(int patchIndex) const
    {
        const int* patchOrders = getPatchKnotOrders(patchIndex);
        int patchNumActiveControlPoints = 1;
        for (int j = 0; j < m_basisDim; j++)
        {
            patchNumActiveControlPoints *= patchOrders[j] + 1;
        }
        return patchNumActiveControlPoints;
    }

    __device__
    int getPatchNumControlPoints(int patchIndex) const
    {
        return m_numControlPoints[patchIndex];
    }

    __device__
    const double* getPatchControlPoints(int patchIndex) const
    {
        int patchControlPointsOffset = 0;
        for (int i = 0; i < patchIndex; i++)
        {
            int numControlPoints = getPatchNumControlPoints(i);
            patchControlPointsOffset += numControlPoints;
        }
        return m_controlPoints + patchControlPointsOffset * m_CPDim;
    }

    __device__
    void getPatchActiveControlPoints(int patchIndex, int* activeIndexes,  int numActiveCPs, double* activeCPs)
    {
        const double* controlPoints_patch = getPatchControlPoints(patchIndex);
        for (int i = 0; i < numActiveCPs; i++)
        {
            for (int j = 0; j < m_CPDim; j++)
            {
                activeCPs[i * m_CPDim + j] = controlPoints_patch[activeIndexes[i] * m_CPDim + j];
            }
        }
    }

    __device__
    void getGPFirstOrderGradients(double* firstDers, int numActiveCPs, double* activeCPs, double* firstGrads)
    {
        for (int i = 0; i < m_CPDim; i++)
        {
            for (int j = 0; j < m_basisDim; j++)
            {
                firstGrads[i * m_basisDim + j] = 0.0;
                for (int k = 0; k < numActiveCPs; k++)
                {
                    firstGrads[i * m_basisDim + j] += firstDers[k * m_basisDim + j] * activeCPs[k * m_CPDim + i];
                }
            }
        }
    }

    __device__
    int getTotalNumGaussPoints() const
    {
        int totalNumGaussPoints = 0;
        for (int i = 0; i < m_numPatches; i++)
        {
            int patchNumGaussPoints = 1;
            const int* numGpAndEle = getPatchNumGpAndEle(i); 
            for (int j = 0; j < m_basisDim * 2; j++)
            {
                patchNumGaussPoints *= numGpAndEle[j];
            }
            totalNumGaussPoints += patchNumGaussPoints;
        }
        return totalNumGaussPoints;
    }
#endif

#if 0
    __device__
    void getThreadElementSupport(
        int patchIndex,
        int* threadEleCoords, 
        double* lower, 
        double* upper)
    {
        const double* patch_knots = getPatchKnots(patchIndex);
        const int* patchKnotDegrees = getPatchKnotOrders(patchIndex);
        const int* patchNumKnots = getPatchNumKnots(patchIndex);
        int knot_Start = 0;
        for (int i = 0; i < m_dim; i++)
        {
            lower[i] = patch_knots[knot_Start + patchKnotDegrees[i] + threadEleCoords[m_dim + i]];
            upper[i] = patch_knots[knot_Start + patchKnotDegrees[i] + threadEleCoords[m_dim + i] + 1];
            knot_Start += patchNumKnots[i];
            //printf("dim %d:lower %f upper %f\n", i, lower[i], upper[i]);
        }

    }
#endif

private:
    DeviceObjectArray<Patch_d> m_patches;
    //BoxTopology_d m_topology;
#if 0
    int m_basisDim = 0;
    int m_CPDim = 0;
    int m_numPatches = 0;
    double* m_knots = nullptr;
    int* m_numKnots = nullptr;
    int* m_orders = nullptr;
    double* m_controlPoints = nullptr;
    int* m_numControlPoints = nullptr;
    int* m_numGpAndEle = nullptr;

    double* m_knots_ref = nullptr;
    int* m_numKnots_ref = nullptr;
    int* m_orders_ref = nullptr;
    double* m_controlPoints_ref = nullptr;
#endif
};