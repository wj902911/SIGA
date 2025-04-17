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
