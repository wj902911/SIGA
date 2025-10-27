#include "MultiPatch_d.h"

//template __global__ void testKernel<int>(int);

#if 0
MultiPatch_d::MultiPatch_d(int basisDim, int CPDim, int numPatches, double *knots, 
                           int *numKnots, int *orders, double *controlPoints, 
                           int *numControlPoints, int *numGpAndEle)
    : m_basisDim(basisDim), m_CPDim(CPDim), m_numPatches(numPatches),
      m_knots(knots), m_numKnots(numKnots), m_orders(orders),
      m_controlPoints(controlPoints), m_numControlPoints(numControlPoints),
      m_numGpAndEle(numGpAndEle) { }
#endif


MultiPatch_d::MultiPatch_d(const MultiPatch &mp)
: m_patches(mp.getNumPatches())//, m_topology(mp.topology())
{
#if 1
    for (int i = 0; i < mp.getNumPatches(); i++)
        m_patches.at(i) = Patch_d(mp.patch(i));
#else
    Patch_d* h_patches_d = new Patch_d[mp.getNumPatches()];
    for (int i = 0; i < mp.getNumPatches(); i++)
    {
        Patch_d temp(mp.patch(i));
        //temp.getControlPoints().print();
        h_patches_d[i] = temp;
        //h_patches_d[i].getControlPoints().print();
    }
    m_patches.parallelDataSetting(h_patches_d, mp.getNumPatches());
    delete[] h_patches_d;
#endif
}

#if 0
int MultiPatch_d::threadPatch(int idx, int &patch) const
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

double MultiPatch_d::gsPoint(int idx, int patch, const GaussPoints_d &gps, 
                             DeviceVector<double> &result) const
{
    return m_patches[patch].basis().gsPoint(idx, gps, result);
}

double MultiPatch_d::gsPoint(int idx, const DeviceObjectArray<GaussPoints_d> &gps, 
                             DeviceVector<double> &result) const
{
    int patch = 0;
    int point_idx = threadPatch(idx, patch);
    return gsPoint(point_idx, patch, gps[patch], result);
}

void MultiPatch_d::evalAllDers_into(int patch, int dir, double u, int n, 
                                    DeviceObjectArray<DeviceVector<double>> &result) const
{
    m_patches[patch].basis().evalAllDers_into(u, dir, n, result);
}

void MultiPatch_d::evalAllDers_into(int patch, const DeviceVector<double> &u, int n, 
                                    DeviceObjectArray<DeviceVector<double>> &result) const
{
    m_patches[patch].basis().evalAllDers_into(u, n, result);
}
#endif

MultiPatch_d::MultiPatch_d(int numPatches)
: m_patches(numPatches) {}
