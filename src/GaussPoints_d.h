#pragma once

#include <cuda_runtime.h>
#include <DeviceObjectArray.h>
#include <vector>

__global__ inline
void retrieveSizes(int numArrays, const DeviceObjectArray<double>* arrays, 
                  int* sizes)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < numArrays; idx += blockDim.x * gridDim.x)
        sizes[idx] = arrays[idx].size();
}

__global__ inline
void retrieveData(int dir, const DeviceObjectArray<double>* array, 
                  double* data)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < array->size(); idx += blockDim.x * gridDim.x)
        data[idx] = (*array)[idx];
}

class GaussPoints_d
{
public:
#if 0
    __device__
    GaussPoints_d(
        int dim,
        const int* numGspts,
        int fixDir = -1)
        : m_dim(dim)
    {
        m_numGspts = new int[dim];

        if (-1==fixDir)
            fixDir = m_dim;
        else
            m_numGspts[fixDir] = 1;

        int i = 0;
        for ( ; i != fixDir; ++i)
            m_numGspts[i] = numGspts[i];
        for ( ++i; i < m_dim; ++i)
            m_numGspts[i] = numGspts[i];

        int gaussPointsSize = 0;
        for (int i = 0; i < m_dim; i++)
        {
            gaussPointsSize += m_numGspts[i];
        }
        m_gaussPoints = new double[gaussPointsSize];
        m_gaussWeights = new double[gaussPointsSize];
        int start = 0;
        for(int i = 0; i < m_dim; i++) 
        {
			getGaussPts_1D(m_numGspts[i], &m_gaussPoints[start], &m_gaussWeights[start]);
            start += m_numGspts[i];
		}

    }

    __device__
    ~GaussPoints_d()
    {
        delete[] m_numGspts;
        delete[] m_gaussPoints;
        delete[] m_gaussWeights;
    }

    __device__
    void getGaussPts_1D(int num_gausspts, double* gausspts, double* weight)
    {
    	switch (num_gausspts)
    	{
        case 1:
        {
            gausspts[0] = 0.0;
            weight[0] = 2.0;
            break;
        }
    	case 2:
    	{
    		gausspts[0] = -0.577350269189625764509148780502;
    		gausspts[1] =  0.577350269189625764509148780502;
    		  weight[0] =  1.0;
    		  weight[1] =  1.0;
    		break;
    	}
    	case 3:
    	{
    		gausspts[0] = -0.774596669241483377035853079956;
    		gausspts[1] =  0.000000000000000000000000000000;
    		gausspts[2] =  0.774596669241483377035853079956;
    		  weight[0] =  0.555555555555555555555555555556;
    		  weight[1] =  0.888888888888888888888888888889;
    		  weight[2] =  0.555555555555555555555555555556;
    		break;
    	}
    	case 4:
    	{
    		gausspts[0] = -0.861136311594052575223946488893;
    		gausspts[1] = -0.339981043584856264802665759103;
    		gausspts[2] =  0.339981043584856264802665759103;
    		gausspts[3] =  0.861136311594052575223946488893;
    		  weight[0] =  0.347854845137453857373063949222;
    		  weight[1] =  0.652145154862546142626936050778;
    		  weight[2] =  0.652145154862546142626936050778;
    		  weight[3] =  0.347854845137453857373063949222;
    		break;
    	}
    	}
    }

    __device__
    const double* getGaussPointVecs() const
    {
        return m_gaussPoints;
    }

    __device__
    const int* getNumGaussPoints() const
    {
        return m_numGspts;
    }

    __device__
    void getThreadGaussPoint(
        double* lower, 
        double* upper,
        int* coords,
        double* gausspt, 
        double& weight)
    {
		int gusptVec_start = 0;
        for (int d = 0; d < m_dim; ++d)
        {
            gausspt[d] = m_gaussPoints[gusptVec_start + coords[d]];
            weight *= m_gaussWeights[gusptVec_start + coords[d]];
            //printf("%f = %f * %f\n", weight, weight, m_gaussWeights[gusptVec_start + coords[d]]);
            gusptVec_start += m_numGspts[d];
        }
		double hprod = 1.0;
        for (int d = 0; d < m_dim; ++d) {
			double h = (upper[d] - lower[d]) / 2.0;
			gausspt[d] = h * (gausspt[d] + 1.0) + lower[d];
			hprod *= (h == 0.0 ? 0.5 : h);
		}
        //printf("hprod = %f\n", hprod);
		weight *= hprod;
    }

    __device__
    const double* getGaussWeights() const
    {
        return m_gaussWeights;
    }
#else
    __host__ __device__
    GaussPoints_d() = default;

    __device__
    GaussPoints_d(int dim, DeviceObjectArray<int> numGspts, int fixDir = -1)
        : m_gaussPoints(dim), m_gaussWeights(dim)
    {
        if (-1 != fixDir)
            numGspts[fixDir] = 1;

        for (int i = 0; i < dim; ++i)
        {
            DeviceObjectArray<double> pts;
            DeviceObjectArray<double> wts;
            setPoints_1D(numGspts[i], pts, wts);
            m_gaussPoints[i] = pts;
            m_gaussWeights[i] = wts;
        }
    }
    __host__
    GaussPoints_d(int dim, std::vector<int> numGspts, int fixDir = -1)
        : m_gaussPoints(dim), m_gaussWeights(dim)
    {
        if (-1 != fixDir)
            numGspts[fixDir] = 1;

        for (int i = 0; i < dim; ++i)
        {
            DeviceObjectArray<double> pts;
            DeviceObjectArray<double> wts;
            setPoints_1D(numGspts[i], pts, wts);
            m_gaussPoints.at(i) = pts;
            m_gaussWeights.at(i) = wts;
        }
    }
    
    // copy constructor
    __host__ __device__
    GaussPoints_d(const GaussPoints_d& other)
        : m_gaussPoints(other.m_gaussPoints.size()), 
          m_gaussWeights(other.m_gaussWeights.size())
    {
        //printf("GaussPoints_d copy constructor called\n");
        for (int i = 0; i < m_gaussPoints.size(); ++i)
        {
            #ifdef __CUDA_ARCH__
            m_gaussPoints[i] = other.m_gaussPoints[i];
            m_gaussWeights[i] = other.m_gaussWeights[i];
            #else
            m_gaussPoints.at(i) = other.gaussPointsOnDir_h(i);
            m_gaussWeights.at(i) = other.gaussWeightsOnDir_h(i);
            #endif
        }
        //printf("GaussPoints_d copy constructor done\n");
    }

    // move constructor
    __host__ __device__
    GaussPoints_d(GaussPoints_d&& other) noexcept
        : m_gaussPoints(other.m_gaussPoints.size()),
          m_gaussWeights(other.m_gaussWeights.size())
    {
        for (int i = 0; i < m_gaussPoints.size(); ++i)
        {
            #ifdef __CUDA_ARCH__
            m_gaussPoints[i] = std::move(other.m_gaussPoints[i]);
            m_gaussWeights[i] = std::move(other.m_gaussWeights[i]);
            #else
            m_gaussPoints.at(i) = std::move(other.gaussPointsOnDir_h(i));
            m_gaussWeights.at(i) = std::move(other.gaussWeightsOnDir_h(i));
            #endif
        }
        other.m_gaussPoints.clear();
        other.m_gaussWeights.clear();
    }

    __host__ __device__
    ~GaussPoints_d() = default;

    __host__ __device__
    void setPoints_1D(int numPts, DeviceObjectArray<double>& pts, 
                      DeviceObjectArray<double>& wts)
    {
        pts.resize(numPts);
        wts.resize(numPts);
        switch (numPts)
        {
        case 1:
        #ifdef __CUDA_ARCH__
            pts[0] = 0.0;
            wts[0] = 2.0;
        #else
            pts.at(0) = 0.0;
            wts.at(0) = 2.0;
        #endif
            break;
        case 2:
        #ifdef __CUDA_ARCH__
            pts[0] = -0.577350269189625764509148780502;
            pts[1] =  0.577350269189625764509148780502;
            wts[0] =  1.0;
            wts[1] =  1.0;
        #else
            pts.at(0) = -0.577350269189625764509148780502;
            pts.at(1) =  0.577350269189625764509148780502;
            wts.at(0) =  1.0;
            wts.at(1) =  1.0;
        #endif
            break;
        case 3:
        #ifdef __CUDA_ARCH__
            pts[0] = -0.774596669241483377035853079956;
            pts[1] =  0.000000000000000000000000000000;
            pts[2] =  0.774596669241483377035853079956;
            wts[0] =  0.555555555555555555555555555556;
            wts[1] =  0.888888888888888888888888888889;
            wts[2] =  0.555555555555555555555555555556;
        #else
            pts.at(0) = -0.774596669241483377035853079956;
            pts.at(1) =  0.000000000000000000000000000000;
            pts.at(2) =  0.774596669241483377035853079956;
            wts.at(0) =  0.555555555555555555555555555556;
            wts.at(1) =  0.888888888888888888888888888889;
            wts.at(2) =  0.555555555555555555555555555556;
        #endif
            break;
        case 4:
        #ifdef __CUDA_ARCH__
            pts[0] = -0.861136311594052575223946488893;
            pts[1] = -0.339981043584856264802665759103;
            pts[2] =  0.339981043584856264802665759103;
            pts[3] =  0.861136311594052575223946488893;
            wts[0] =  0.347854845137453857373063949222;
            wts[1] =  0.652145154862546142626936050778;
            wts[2] =  0.652145154862546142626936050778;
            wts[3] =  0.347854845137453857373063949222;
        #else
            pts.at(0) = -0.861136311594052575223946488893;
            pts.at(1) = -0.339981043584856264802665759103;
            pts.at(2) =  0.339981043584856264802665759103;
            pts.at(3) =  0.861136311594052575223946488893;
            wts.at(0) =  0.347854845137453857373063949222;
            wts.at(1) =  0.652145154862546142626936050778;
            wts.at(2) =  0.652145154862546142626936050778;
            wts.at(3) =  0.347854845137453857373063949222;
        #endif
            break;
        default:
            printf("Error: Gauss points not implemented for %d points\n", numPts);
            break;
        }
    }

    __host__ __device__
    int numGPs(int dir) const
    {
        #ifdef __CUDA_ARCH__
            return m_gaussPoints[dir].size();
        #else
            int dim = m_gaussPoints.size();
            int* sizes = nullptr;
            cudaMalloc((void**)&sizes, dim * sizeof(int));
            int blockSize = 256;
            int numBlocks = (dim + blockSize - 1) / blockSize;
            retrieveSizes<<<numBlocks, blockSize>>>(dim, m_gaussPoints.data(), sizes);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) 
                printf("Error in retrieveSizes: %s\n", cudaGetErrorString(err));
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) 
                printf("Error in cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
            int size = 0;
            cudaMemcpy(&size, sizes + dir, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(sizes);
            return size;
        #endif
    }

    __device__
    const DeviceObjectArray<double>& gaussPointsOnDir(int dir) const
    {
        return m_gaussPoints[dir];
    }

    __host__
    DeviceObjectArray<double> gaussPointsOnDir_h(int dir) const
    {
        DeviceObjectArray<double> pts(numGPs(dir));
        int blockSize = 256;
        int numBlocks = (pts.size() + blockSize - 1) / blockSize;
        retrieveData<<<numBlocks, blockSize>>>(dir, m_gaussPoints.data() + dir, 
                                               pts.data());
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error in retrieveData: %s\n", cudaGetErrorString(err));
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) 
            printf("Error in cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
        return pts;
    }

    __host__
    DeviceObjectArray<double> gaussWeightsOnDir_h(int dir) const
    {
        DeviceObjectArray<double> wts(numGPs(dir));
        int blockSize = 256;
        int numBlocks = (wts.size() + blockSize - 1) / blockSize;
        retrieveData<<<numBlocks, blockSize>>>(dir, m_gaussWeights.data() + dir, 
                                               wts.data());
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error in retrieveData: %s\n", cudaGetErrorString(err));
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) 
            printf("Error in cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
        return wts;
    }

    __device__
    double threadGaussPoint(const DeviceVector<double>& lower,
                            const DeviceVector<double>& upper,
                            const DeviceObjectArray<int>& coords,
                            DeviceVector<double>& gausspt) const
    {
        gausspt.resize(m_gaussPoints.size());
        double weight = 1.0;
        for (int d = 0; d < m_gaussPoints.size(); ++d)
        {
            gausspt(d) = m_gaussPoints[d][coords[d]];
            weight *= m_gaussWeights[d][coords[d]];
        }
        double hprod = 1.0;
        for (int d = 0; d < m_gaussPoints.size(); ++d) {
            double h = (upper(d) - lower(d)) / 2.0;
            gausspt(d) = h * (gausspt(d) + 1.0) + lower(d);
            hprod *= (h == 0.0 ? 0.5 : h);
        }
        //gausspt.print();
        return weight * hprod;
    }

#endif
private:
#if 0
    int m_dim;
    int* m_numGspts;
    double* m_gaussPoints;
    double* m_gaussWeights;
#else
    DeviceObjectArray<DeviceObjectArray<double>> m_gaussPoints;
    DeviceObjectArray<DeviceObjectArray<double>> m_gaussWeights;
#endif
};