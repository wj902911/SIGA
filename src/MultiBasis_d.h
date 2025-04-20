#pragma once

#include <cuda_runtime.h>
#include <MultiBasis.h>
#include <TensorBsplineBasis_d.h>

#if 0
__global__
void MultiBasis_dTestKenel(TensorBsplineBasis_d* d_bases_d)
{
    d_bases_d[1].getKnotVector(0).getKnots().print();
    printf("MultiBasis_dTestKenel done\n");
}
#endif


class MultiBasis_d
{
public:
#if 0
    __device__
    MultiBasis_d(int dim, int numBasis, double* knots, int *numKnots, int *orders)
        : m_bases(numBasis)
    {
    }
#endif

    __host__
    MultiBasis_d(const MultiBasis& multiBasis)
        : m_bases(multiBasis.getNumBases())
    {
    #if 1
        for (int i = 0; i < multiBasis.getNumBases(); i++)
        {
            //TensorBsplineBasis_d basis = multiBasis.getBasis(i);
            m_bases.at(i) = TensorBsplineBasis_d(multiBasis.basis(i));
        }
        #if 0
        MultiBasis_dTestKenel<<<1, 1>>>(m_bases.data());
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error in MultiBasis_dTestKenel: %s\n", cudaGetErrorString(err));
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) 
            printf("Error in MultiBasis_dTestKenel: %s\n", cudaGetErrorString(err));
        #endif
    #else
        TensorBsplineBasis_d* h_bases_d = new TensorBsplineBasis_d[multiBasis.getNumBases()];
        for (int i = 0; i < multiBasis.getNumBases(); i++)
        {
            TensorBsplineBasis_d basis = multiBasis.getBasis(i);
            h_bases_d[i] = basis;
        }
        m_bases.parallelDataSetting(h_bases_d, multiBasis.getNumBases());
        delete[] h_bases_d;
    #endif
    }

    __device__
    int getDim() const
    {
        return m_bases[0].getDim();
    }

    __device__
    int getNumBasis() const
    {
        return m_bases.size();
    }

    __device__
    const TensorBsplineBasis_d& basis(int index) const
    {
        return m_bases[index];
    }

    __host__
    const TensorBsplineBasis_d* getBasisPtr() const
    {
        return m_bases.data();
    }

    __device__
    int threadPatch(int idx, int& patch) const
    {
        int dim = getDim();
        int point_idx = idx;
        for (int i = 0; i < m_bases.size(); i++)
        {
            int patch_points = m_bases[i].totalNumGPs();
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
    DeviceObjectArray<int> ptCoords(int idx) const
    {
        int patch = 0;
        int point_idx = threadPatch(idx, patch);
        return m_bases[patch].ptCoords(point_idx);
    }

    __device__
    DeviceObjectArray<int> ptCoords(int idx, BoxSide_d side) const
    {
        int patch = 0;
        int point_idx = threadPatch(idx, patch);
        return m_bases[patch].ptCoords(point_idx, side);
    }

    __device__
    DeviceVector<double> lowerElementSupport(int patch, int idx) const
    { return m_bases[patch].lowerElementSupport(idx); }

    __device__
    DeviceVector<double> upperElementSupport(int patch, int idx) const
    { return m_bases[patch].upperElementSupport(idx); }

    __device__
    void elementSupport(int idx, DeviceVector<double>& lower, 
                        DeviceVector<double>& upper) const
    {
        int patch = 0;
        int point_idx = threadPatch(idx, patch);
        m_bases[patch].elementSupport(point_idx, lower, upper);
    }

    __device__
    double gsPoint(int idx, int patch, const GaussPoints_d& gps,
                      DeviceVector<double>& result) const
    { return m_bases[patch].gsPoint(idx, gps, result); }

    __device__
    double gsPoint(int idx, const DeviceObjectArray<GaussPoints_d>& gps,
                      DeviceVector<double>& result) const
    {
        int patch = 0;
        int point_idx = threadPatch(idx, patch);
        return gsPoint(point_idx, patch, gps[patch], result);
    }

    __device__
#if 1
    void evalAllDers_into(int patch, int dir, double u, int n, 
                          DeviceObjectArray<DeviceVector<double>>& result) const
    { m_bases[patch].evalAllDers_into(dir, u, n, result); }
#else
    DeviceObjectArray<DeviceVector<double>> 
    evalAllDers_into(int patch, int dir, double u, int n) const
    { return m_bases[patch].evalAllDers_into(dir, u, n); }
#endif

    __device__
    void evalAllDers_into(int patch, const DeviceVector<double>& u, int n, 
                          DeviceObjectArray<DeviceVector<double>>& result) const
    { m_bases[patch].evalAllDers_into(u, n, result); }

#if 0
    __device__
    const int *getPatchNumKnots(int patchIndex) const
    {
        return m_numKnots + patchIndex * m_dim;
    }

    __device__
    const double* getPatchKnots(int patchIndex) const
    {
        int patchKnotsOffset = 0;
        for (int i = 0; i < patchIndex * m_dim; i++)
        {
            patchKnotsOffset += m_numKnots[i];
        }
        return m_knots + patchKnotsOffset;
    }

    __device__
    const int* getPatchKnotOrders(int patchIndex) const
    {
        return m_orders + patchIndex * m_dim;
    }

    __device__
    const int* getPatchNumGpAndEle(int patchIndex) const
    {
        return m_numGpAndEle + patchIndex * m_dim * 2;
    }

    __device__
    int getPatchNumActiveFuncs(int patchIndex) const
    {
        const int* patchOrders = getPatchKnotOrders(patchIndex);
        int patchNumActiveFuncs = 1;
        for (int j = 0; j < m_dim; j++)
        {
            patchNumActiveFuncs *= patchOrders[j] + 1;
        }
        return patchNumActiveFuncs;
    }
#endif


private:
    DeviceObjectArray<TensorBsplineBasis_d> m_bases;
#if 0
    int m_dim = 0;
    int m_numBasis = 0;
    double* m_knots = nullptr;
    int* m_numKnots = nullptr;
    int* m_orders = nullptr;
    int* m_numGpAndEle = nullptr;
#endif
};