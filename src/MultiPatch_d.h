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
    MultiPatch_d(const MultiPatch& mp);
#if 0
    : m_patches(mp.getNumPatches())//, m_topology(mp.topology())
    {
        for (int i = 0; i < mp.getNumPatches(); i++)
            m_patches.at(i) = mp.patch(i);
    }
#endif

    ~MultiPatch_d()=default;

    __device__
    int getBasisDim() const
    { return m_patches[0].getDim(); }

    __device__
    int getCPDim() const
    { return m_patches[0].getCPDim(); }

    __device__
    int getNumPatches() const
    { return m_patches.size(); }

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
    DeviceMatrix<double> getActiveControlPoints(int patchIndex, DeviceVector<double> pt)
    {
        return m_patches[patchIndex].getActiveControlPoints(pt);
    }

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