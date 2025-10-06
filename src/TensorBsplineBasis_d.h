#pragma once

#include <KnotVector_d.h>
#include <Utility_d.h>
#include <Boundary_d.h>
#include <DeviceVector.h>
#include <GaussPoints_d.h>
#include <TensorBsplineBasis.h>

#if 0
__device__
void getTensorCoordinate(int dim, int* numVlues, int index, int* out)
{
    for (int i = 0; i < dim; i++)
    {
        out[i] = index % numVlues[i];
        index /= numVlues[i];
    }
}

__device__
int getTotalNumber(int dim, int* numVlues)
{
    int total = 1;
    for (int i = 0; i < dim; i++)
    {
        total *= numVlues[i];
    }
    return total;
}

__device__
void getTensorProduct(int dim, int* numVlues, double* in, double* out)
{
    int numVluesTotal = getTotalNumber(dim, numVlues);
    for (int i = 0; i < numVluesTotal; i++)
    {
        int index[3] = { 0 };
        getTensorCoordinate(dim, numVlues, i, index);
        out[i] = in[index[0]];
        for (int j = 1, valuesStart = numVlues[0]; j < dim; valuesStart += numVlues[j], j++)
        {
            out[i] *= in[valuesStart+index[j]];
        }
    }
}
#endif

__global__
void retrieveKnotData(int dir, DeviceObjectArray<KnotVector_d>* d_knots_d, double* d_result);

__global__
void retrieveKnotSize(DeviceObjectArray<KnotVector_d>* d_knots_d, int* d_result);

__global__
void retrieveKnotSizeAndOrder(DeviceObjectArray<KnotVector_d>* d_knots_d, 
                              int* d_order, int* d_size);

__global__
void retrieveKnotOrder(DeviceObjectArray<KnotVector_d>* d_knots_d, int* d_result);

__global__
void deviceConstructKnotVector(KnotVector_d* knotVector, KnotVector_d* input);

class TensorBsplineBasis_d
{
public:
    // Default constructor
    __host__ __device__
    TensorBsplineBasis_d()
    : m_dim(0) { }

    __host__ __device__
    TensorBsplineBasis_d(int dim, const int* numKnots, const double* knots,
                         const int* orders)
        : m_dim(dim), m_knotVectors(dim)
    {
        
        //m_knotVectors = new KnotVector_d[m_dim];
        int KnotStart = 0;
    #if !defined(__CUDA_ARCH__)
        KnotVector_d* h_knotVectors = new KnotVector_d[m_dim];
    #endif
        for (int i = 0; i < m_dim; i++)
        {
    #if defined(__CUDA_ARCH__)
            m_knotVectors[i] = KnotVector_d(orders[i], numKnots[i], knots + KnotStart);
    #else
            h_knotVectors[i] = KnotVector_d(orders[i], numKnots[i], knots + KnotStart);
            //m_knotVectors.at(i) = KnotVector_d(orders[i], numKnots[i], knots + KnotStart);
    #endif
            KnotStart += numKnots[i];
        }
    #if !defined(__CUDA_ARCH__)
        m_knotVectors.parallelDataSetting(h_knotVectors, m_dim);
        for (int i = 0; i < m_dim; i++)
        {
            h_knotVectors[i].~KnotVector_d();
        }
        delete[] h_knotVectors;
    #endif
    }

    __device__
    TensorBsplineBasis_d(const KnotVector_d& u, const KnotVector_d& v)
        : m_dim(2), m_knotVectors(2)
    {
        m_knotVectors[0] = u;
        m_knotVectors[1] = v;
    }

    __host__
    TensorBsplineBasis_d(const KnotVector& u, const KnotVector& v)
        : m_dim(2), m_knotVectors(2)
    {
#if 0
    #if 1
        KnotVector_d* h_knotVectors = new KnotVector_d[m_dim];
        h_knotVectors[0] = u;
        h_knotVectors[1] = v;
        m_knotVectors.parallelDataSetting(h_knotVectors, m_dim);
        for (int i = 0; i < m_dim; i++)
        {
            h_knotVectors[i].~KnotVector_d();
        }
        delete[] h_knotVectors;
    #else
        KnotVector_d h_u_d(u);
        KnotVector_d* d_u_d = nullptr;
        cudaMalloc((void**)&d_u_d, sizeof(KnotVector_d));
        cudaMemcpy(d_u_d, &h_u_d, sizeof(KnotVector_d), cudaMemcpyHostToDevice);
        KnotVector_d h_v_d(v);
        KnotVector_d* d_v_d = nullptr;
        cudaMalloc((void**)&d_v_d, sizeof(KnotVector_d));
        cudaMemcpy(d_v_d, &h_v_d, sizeof(KnotVector_d), cudaMemcpyHostToDevice);
        deviceConstructKnotVector<<<1, 1>>>(m_knotVectors.data(), d_u_d);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error in deviceConstructKnotVector: %s\n", cudaGetErrorString(err));
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            printf("Error in deviceConstructKnotVector: %s\n", cudaGetErrorString(err));
        deviceConstructKnotVector<<<1, 1>>>(m_knotVectors.data() + 1, d_v_d);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error in deviceConstructKnotVector: %s\n", cudaGetErrorString(err));
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            printf("Error in deviceConstructKnotVector: %s\n", cudaGetErrorString(err));
        cudaFree(d_u_d);
        cudaFree(d_v_d);
    #endif
#else
        m_knotVectors.at(0) = KnotVector_d(u);
        m_knotVectors.at(1) = KnotVector_d(v);
#endif
    }

    __host__
    TensorBsplineBasis_d(const KnotVector& u, const KnotVector& v, const KnotVector& w)
        : m_dim(3), m_knotVectors(3)
    {
    #if 1
        m_knotVectors.at(0) = KnotVector_d(u);
        m_knotVectors.at(1) = KnotVector_d(v);
        m_knotVectors.at(2) = KnotVector_d(w);
    #else
        KnotVector_d* h_knotVectors = new KnotVector_d[m_dim];
        h_knotVectors[0] = u;
        h_knotVectors[1] = v;
        h_knotVectors[2] = w;
        m_knotVectors.parallelDataSetting(h_knotVectors, m_dim);
        for (int i = 0; i < m_dim; i++)
        {
            h_knotVectors[i].~KnotVector_d();
        }
        delete[] h_knotVectors;
    #endif
    }
        

    __host__
    TensorBsplineBasis_d(const TensorBsplineBasis& host_other)
        : m_dim(host_other.getDim()), m_knotVectors(host_other.getDim())
    {
    #if 1
        #if 1
        for (int i = 0; i < m_dim; i++)
            m_knotVectors.at(i) = KnotVector_d(host_other.getKnotVector(i));
        #else
        KnotVector_d* h_knotVectors = new KnotVector_d[m_dim];
        for (int i = 0; i < m_dim; i++)
        {
            KnotVector_d temp(host_other.getKnotVector(i));
            //m_knotVectors.at(i) = temp;
            //m_knotVectors.at(i) = KnotVector_d(host_other.getKnotVector(i));
            h_knotVectors[i] = temp;
        }
        m_knotVectors.parallelDataSetting(h_knotVectors, m_dim);
        for (int i = 0; i < m_dim; i++)
        {
            h_knotVectors[i].~KnotVector_d();
        }
        delete[] h_knotVectors;
        #endif
    #else
        for (int i = 0; i < m_dim; i++)
        {
            KnotVector_d h_u_d(host_other.getKnotVector(i));
            KnotVector_d* d_u_d = nullptr;
            cudaMalloc((void**)&d_u_d, sizeof(KnotVector_d));
            cudaMemcpy(d_u_d, &h_u_d, sizeof(KnotVector_d), cudaMemcpyHostToDevice);
            deviceConstructKnotVector<<<1, 1>>>(m_knotVectors.data() + i, d_u_d);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("Error in deviceConstructKnotVector: %s\n", cudaGetErrorString(err));
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
                printf("Error in deviceConstructKnotVector: %s\n", cudaGetErrorString(err));
            cudaFree(d_u_d);
        }
    #endif
    }

    // Copy constructor
    __host__ __device__
    TensorBsplineBasis_d(const TensorBsplineBasis_d& other)
        : m_dim(other.m_dim), m_knotVectors(other.m_knotVectors.size()) 
    {
    #if 1
        for (int i = 0; i < m_dim; i++)
        #if defined(__CUDA_ARCH__)
            m_knotVectors[i] = other.m_knotVectors[i];
        #else
            m_knotVectors.at(i) = other.getKnotVector_h(i);
        #endif
    #else
        #if !defined(__CUDA_ARCH__)
        KnotVector_d* h_knotVectors = new KnotVector_d[m_dim];
        #endif
        for (int i = 0; i < m_dim; i++)
        {
        #if defined(__CUDA_ARCH__)
            m_knotVectors[i] = other.m_knotVectors[i];
        #else
            //KnotVector_d temp(other.m_knotVectors.at(i));
            //m_knotVectors.at(i) = temp;
            //m_knotVectors.at(i) = KnotVector_d(other.m_knotVectors.at(i));
            h_knotVectors[i] = other.getKnotVector_h(i);
        #endif
        }
        #if !defined(__CUDA_ARCH__)
        m_knotVectors.parallelDataSetting(h_knotVectors, m_dim);
        for (int i = 0; i < m_dim; i++)
        {
            h_knotVectors[i].~KnotVector_d();
        }
        delete[] h_knotVectors;
        #endif
    #endif
    }

    // Copy assignment operator
    __host__ __device__
    TensorBsplineBasis_d& operator=(const TensorBsplineBasis_d& other)
    {
        if (this != &other)
        {
            m_dim = other.m_dim;
            m_knotVectors.resize(m_dim);
        #if 1
            for (int i = 0; i < m_dim; i++)
            #if defined(__CUDA_ARCH__)
                m_knotVectors[i] = other.m_knotVectors[i];
            #else
                m_knotVectors.at(i) = other.getKnotVector_h(i);
            #endif
        #else
            #if !defined(__CUDA_ARCH__)
            KnotVector_d* h_knotVectors = new KnotVector_d[m_dim];
            #endif
            for (int i = 0; i < m_dim; i++)
            {
            #if defined(__CUDA_ARCH__)
                m_knotVectors[i] = other.m_knotVectors[i];
            #else
                //KnotVector_d temp(other.m_knotVectors.at(i));
                //m_knotVectors.at(i) = temp;
                //m_knotVectors.at(i) = KnotVector_d(other.m_knotVectors.at(i));
                KnotVector_d temp(other.getKnotVector_h(i));
                h_knotVectors[i] = temp;
                //h_knotVectors[i] = other.getKnotVector_h(i);
            #endif
            }
            #if !defined(__CUDA_ARCH__)
            m_knotVectors.parallelDataSetting(h_knotVectors, m_dim);
            for (int i = 0; i < m_dim; i++)
            {
                h_knotVectors[i].~KnotVector_d();
            }
            delete[] h_knotVectors;
            #endif
        #endif
        }
        return *this;
    }

    // Move constructor
    __host__ __device__
    TensorBsplineBasis_d(TensorBsplineBasis_d&& other) noexcept
        : m_dim(other.m_dim)
    {
        other.m_dim = 0;
    #if 0
        for (int i = 0; i < m_dim; i++)
        {
        #if defined(__CUDA_ARCH__)
            m_knotVectors[i] = std::move(other.m_knotVectors[i]);
        #else
            m_knotVectors.at(i) = std::move(other.getKnotVector_h(i));
        #endif
        }
    #else
        m_knotVectors = std::move(other.m_knotVectors);
    #endif
    }

    // Move assignment operator
    __host__ __device__
    TensorBsplineBasis_d& operator=(TensorBsplineBasis_d&& other) noexcept
    {
        if (this != &other)
        {
            m_dim = other.m_dim;
#if 0
            for (int i = 0; i < m_dim; i++)
            {
            #if defined(__CUDA_ARCH__)
                m_knotVectors[i] = std::move(other.m_knotVectors[i]);
            #else
                m_knotVectors.at(i) = 
                std::move(KnotVector_d(other.m_knotVectors.at(i)));
            #endif
            }
#else
            m_knotVectors = std::move(other.m_knotVectors);
#endif
            other.m_dim = 0;
        }
        return *this;
    }

#if 0
    __device__
    ~TensorBsplineBasis_d()
    {
        delete[] m_knotVectors;
    }
#else
    __host__ __device__
    ~TensorBsplineBasis_d() = default;
#endif

    __device__
    int getDim() const
    {
        return m_dim;
    }

    __device__
    int upperBound(int direction, double value) const
    {
        return m_knotVectors[direction].upperBound(value);
    }

    __device__
    void getThreadElementSupport(int* threadEleCoords, double* lower, double* upper)
    {
        for (int i = 0; i < m_dim; i++)
        {
            int order = m_knotVectors[i].getOrder();
            lower[i] = m_knotVectors[i].getKnots()[threadEleCoords[m_dim + i] + order];
            upper[i] = m_knotVectors[i].getKnots()[threadEleCoords[m_dim + i] + order + 1];
        }
    }

    __device__
    void getTensoredDerivatives(int maxDer, int* numDers, double* in_values, 
                                double* in_ders, double* out)
    {
        double* values[3] = { nullptr };
        values[0] = in_values;
        for (int i = 1; i < m_dim; i++)
        {
            values[i] = values[i - 1] + numDers[i - 1];
        }
        double* firstDers[3] = { nullptr };
        int numFirstDersTotal = 0;
        if (maxDer >= 1)
        {
            firstDers[0] = in_ders;
            numFirstDersTotal = getTotalNumber(m_dim, numDers) * m_dim;
            for (int i = 1; i < m_dim; i++)
            {
                firstDers[i] = firstDers[i - 1] + numDers[i - 1];
            }
            for (int r = 0, w = 0; r < numFirstDersTotal; w++)
            {
                int index[3] = { 0 };
                getTensorCoordinate(m_dim, numDers, w, index);
                for (int k = 0; k < m_dim; k++)
                {
                    out[r] = firstDers[k][index[k]];
                    for (int i = 0; i < k; i++)
                        out[r] *= values[i][index[i]];
                    for (int i = k + 1; i < m_dim; i++)
                        out[r] *= values[i][index[i]];
                    r++;
                }
            }
        }
        double* secondDers[3] = { nullptr };
        if (maxDer > 1)
        {
            double* out_secondders = out + numFirstDersTotal;
            secondDers[0] = firstDers[m_dim - 1] + numDers[m_dim - 1];
            for (int i = 1; i < m_dim; i++)
            {
                secondDers[i] = secondDers[i - 1] + numDers[i - 1];
            }
            int stride = m_dim + m_dim * (m_dim - 1) / 2;
            int numSecondDersTotal = stride * getTotalNumber(m_dim, numDers);
            for (int r = 0, w = 0; r < numSecondDersTotal; w++, r+=stride)
            {
                int m = m_dim;
                int index[3] = { 0 };
                getTensorCoordinate(m_dim, numDers, w, index);
                for (int k = 0; k < m_dim; k++)
                {
                    int cur = r + k;
                    out_secondders[cur] = secondDers[k][index[k]];
                    for (int i = 0; i < k; i++)
                        out_secondders[cur] *= values[i][index[i]];
                    for (int i = k + 1; i < m_dim; i++)
                        out_secondders[cur] *= values[i][index[i]];
                    for (int l = k + 1; l < m_dim; l++)
                    {
                        cur = r + m;
                        out_secondders[cur] = firstDers[k][index[k]] * firstDers[l][index[l]];
                        for (int i = 0; i < k; i++)
                            out_secondders[cur] *= values[i][index[i]];
                        for (int i = k + 1; i < l; i++)
                            out_secondders[cur] *= values[i][index[i]];
                        for (int i = l + 1; i < m_dim; i++)
                            out_secondders[cur] *= values[i][index[i]];
                        m++;
                    }
                }
            }
        }
    }

    __device__
    void evalAllDers_into(int dir, double u, int n, 
                          DeviceObjectArray<DeviceVector<double>>& result) const
    //DeviceObjectArray<DeviceVector<double>> 
    //evalAllDers_into(int dir, double u, int n) const
    {
        int p = m_knotVectors[dir].getOrder();
        int p1 = p + 1;
        DeviceObjectArray<double> ndu(p1 * p1);
        DeviceObjectArray<double> left(p1);
        DeviceObjectArray<double> right(p1);
        DeviceObjectArray<double> a(2 * p1);
        result.resize(n+1);
        //printf("1111\n");
        //DeviceObjectArray<DeviceVector<double>> result(n+1);
        for(int k=0; k<=n; k++)
            result[k].resize(p1);
        
        if (!m_knotVectors[dir].inDomain(u))
        {
            for(int k=0; k<=n; k++)
                result[k].setZero();
            return;
        }
        int span = upperBound(dir, u) - 1;
        ndu[0] = 1.0;
        for (int j = 1; j <= p; ++j)
        {
            left[j] = u - m_knotVectors[dir][span + 1 - j];
            right[j] = m_knotVectors[dir][span + j] - u;
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
            result[0](j) = ndu[j * p1 + p];

        for (int r = 0; r <= p; r++)
        {
            double* a1 = &a[0];
            double* a2 = &a[p1];

            a1[0] = 1.0;

            for(int k = 1; k <= n; k++)
            {
                int rk,pk,j1,j2 ;
                double der(0);
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
                result[k](r) = der;
                double* temp = a1;
                a1 = a2;
                a2 = temp;
            }
        }

        int r = p;
        for(int k=1; k<=n; k++)
        {
            result[k] = result[k] * double(r);
            r *= p - k;
        }

        //return result;
    }

    __device__
    void deriv2_tp(const DeviceObjectArray<DeviceVector<double>> values[],
    const DeviceVector<int>& nb_cwise, DeviceVector<double>& result) const
    {
        const unsigned nb = nb_cwise.prod();
        const unsigned stride = m_dim + m_dim*(m_dim-1)/2;

        result.resize(stride*nb);

        DeviceVector<int> v(m_dim);
        v.setZero();
        unsigned r = 0;
        do
        {
            unsigned m = m_dim;
            for ( int k=0; k<m_dim; ++k)
            {
                int cur = r + k;
                result(cur) = values[k][2](v(k));
                for ( int i=0; i<k; ++i)
                    result(cur) *= values[i][0](v(i));
                for ( int i=k+1; i<m_dim; ++i)
                    result(cur) *= values[i][0](v(i));
                for ( int l=k+1; l<m_dim; ++l)
                {
                    cur = r + m;
                    result(cur) = values[k][1](v(k)) * values[l][1](v(l));
                    for ( int i=0; i<k; ++i)
                        result(cur) *= values[i][0](v(i));
                    for ( int i=k+1; i<l; ++i)
                        result(cur) *= values[i][0](v(i));
                    for ( int i=l+1; i<m_dim; ++i)
                        result(cur) *= values[i][0](v(i));
                    m++;
                }
            }

            r+= stride;
        } while (nextLexicographic_d(v, nb_cwise));
        
    }

    __device__
    void evalAllDers_into(const DeviceVector<double>& u, int n, 
                          DeviceObjectArray<DeviceVector<double>>& result) const
    {
        //DeviceObjectArray<DeviceVector<double>>* values = 
        //new DeviceObjectArray<DeviceVector<double>>[m_dim];
        //DeviceObjectArray<DeviceVector<double>> values[3];
        DeviceObjectArray<DeviceObjectArray<DeviceVector<double>>> values(m_dim);
        DeviceVector<int> v(m_dim), nb_cwise(m_dim);
        result.resize(n+1);
        int nb = 1;
        for(int i = 0; i < m_dim; i++)
        {
            evalAllDers_into(i, u(i), n, values[i]);
            //printf("evalAllDers_into: %d, %f\n", i, u(i));
            //values[i] = evalAllDers_into(i, u(i), n);
            const int num_i = values[i][0].size();
            nb_cwise(i) = num_i;
            nb *= num_i;
        }
        //values[0][0].print();
        v.setZero();
        result[0].resize(nb);
        int r = 0;
        do
        {
            result[0](r) = values[0][0](v(0));
            for ( int i=1; i!=m_dim; ++i)
                result[0](r) *= values[i][0](v(i));
            r++;
        } while (nextLexicographic_d(v, nb_cwise));

        if ( n>=1)
        {
            result[1].resize(m_dim*nb);
            v.setZero();
            r = 0;
            do
            {
                for ( int k=0; k<m_dim; ++k)
                {
                    result[1](r) = values[k][1](v(k));
                    for ( int i=0; i<k; ++i)
                        result[1](r) *= values[i][0](v(i));
                    for ( int i=k+1; i<m_dim; ++i)
                        result[1](r) *= values[i][0](v(i));
                    r++;
                }
            } while (nextLexicographic_d(v, nb_cwise));
        }

        if (n>1)
        {
            deriv2_tp( values.data(), nb_cwise, result[2] );
            DeviceVector<int> cc(m_dim);
            for (int i = 3; i <=n; ++i)
            {
                result[i].resize( nb*numCompositions(i, m_dim) );
                v.setZero();
            
                r = 0;
                do
                {
                    firstComposition(i, m_dim, cc);
                    do
                    {
                        result[i](r) = values[0][cc(0)](v(0));
                        for (int k = 1; k!=m_dim; ++k)
                            result[i](r) = values[k][cc(k)](v(k));
                        ++r;
                    } while (nextComposition(cc));
                } while (nextLexicographic_d(v, nb_cwise));
            }
        }
    }

    __device__
    void getValuesAnddDerivatives(double* pt, int maxDer, double* values,
                                  double* derivatives)
    {
        //temp results size calculation and memory allocation
        int temp_values_size = 0;
        int temp_derivatives_size = 0;
        int numValues[3] = { 0 };
        int stride[3] = { 0 };
        for (int i = 0; i < m_dim; i++)
        {
            int order = m_knotVectors[i].getOrder();
            stride[i] = order + 1;
            numValues[i] = stride[i];
            temp_values_size += stride[i];
            temp_derivatives_size += stride[i] * maxDer;
        }
        double* temp_values = new double[temp_values_size];
        double* temp_derivatives = new double[temp_derivatives_size];

        for (int d = 0; d < m_dim; ++d)
        {
            int p = m_knotVectors[d].getOrder();
            //const double* knots = m_knotVectors[d].getKnots();
            DeviceObjectArray<double> knots = m_knotVectors[d].getKnots();
            int p1 = p + 1;
            double* ndu = new double[p1 * p1];
			double* left = new double[p1];
			double* right = new double[p1];
			double* a = new double[2 * p1];

            int span = upperBound(d, pt[d]) - 1;
			ndu[0] = 1.0;
            for (int j = 1; j <= p; ++j)
            {
                left[j] = pt[d] - knots[span + 1 - j];
				right[j] = knots[span + j] - pt[d];
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
            {
                temp_values[d * stride[d] + j] = ndu[j * p1 + p];
            }
            
            for (int r = 0; r <= p; r++)
            {
                double* a1 = &a[0];
                double* a2 = &a[p1];

                a1[0] = 1.0;

                for(int k = 1; k <= maxDer; k++)
                {
                    int rk,pk,j1,j2 ;
                    double der(0);
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
                    temp_derivatives[(k - 1) * m_dim * stride[d] + d * stride[d] + r] = der;
                    double* temp = a1;
                    a1 = a2;
                    a2 = temp;
                }
            }
			delete[] ndu, left, right, a;
        }
#if 0
        printf("temp_values:");
        for (int i = 0; i < temp_values_size; ++i) 
        {
            printf("%f ", temp_values[i]);
        }
        printf("\n");
        if (maxDer > 0)
        {
            printf("temp_derivatives:");
            for (int i = 0; i < temp_derivatives_size; ++i)
            {
                printf("%f ", temp_derivatives[i]);
            }
            printf("\n");
        }
#endif
        getTensorProduct(m_dim, numValues, temp_values, values);
        getTensoredDerivatives(maxDer, numValues, temp_values, temp_derivatives, derivatives);
        delete[] temp_values, temp_derivatives;
    }

    __device__
    void getActiveIndexes(double* pt, int* activeIndexes, int numAct)
    {
        int firstAct[3] = { 0 };
        int sizes[3] = { 0 };
        for (int d = 0; d < m_dim; ++d)
        {
            int order = m_knotVectors[d].getOrder();
            firstAct[d] = upperBound(d, pt[d]) - order - 1;
            sizes[d] =order + 1;
        }
        for (int r = 0; r < numAct; r++)
        {
            int index[3] = { 0 };
            getTensorCoordinate(m_dim, sizes, r, index);
            int gidx = firstAct[m_dim - 1] + index[m_dim - 1];
            for (int d = m_dim - 2; d >= 0; d--)
                gidx = gidx * size(d) + firstAct[d] + index[d];
            activeIndexes[r] = gidx;
        }
    }

    __device__
    DeviceVector<int> getActiveIndexes(DeviceVector<double> pt)
    {
        int numAct = getNumActiveControlPoints();
        DeviceVector<int> activeIndexes(numAct);
        DeviceVector<int> firstAct(m_dim);
        DeviceVector<int> sizes(m_dim);
        for (int d = 0; d < m_dim; ++d)
        {
            int order = m_knotVectors[d].getOrder();
            firstAct(d) = upperBound(d, pt(d)) - order - 1;
            sizes(d) = order + 1;
        }
        for (int r = 0; r < numAct; r++)
        {
            DeviceVector<int> index(m_dim);
            getTensorCoordinate(m_dim, sizes.data(), r, index.data());
            int gidx = firstAct(m_dim - 1) + index(m_dim - 1);
            for (int d = m_dim - 2; d >= 0; d--)
                gidx = gidx * size(d) + firstAct(d) + index(d);
            activeIndexes(r) = gidx;
        }
        return activeIndexes;
    }

    __device__
    int size(int d)
    {
        return m_knotVectors[d].getNumControlPoints();
    }

    #if 1
    __device__
    int size(int d) const
    {
        //printf("size const called\n");
        return m_knotVectors[d].getNumControlPoints();
    }
    #endif

    __device__
    int totalNumGPsInDir(int d) const { return m_knotVectors[d].totalNumGaussPoints(); }

    __device__
    int totalNumGPs() const
    {
        int numGPS = 1;

        for (int d = 0; d < m_dim; d++)
            numGPS *= totalNumGPsInDir(d);

        return numGPS;
    }

    __device__
    int numCPsInDir(int d) const { return m_knotVectors[d].getNumControlPoints(); }

    __device__
    int numCPs() const
    {
        int numCPs = 1;

        for (int d = 0; d < m_dim; d++)
            numCPs *= numCPsInDir(d);

        return numCPs;
    }

    __device__
    int getNumActiveControlPoints()
    {
        int numAct = 1;
        for (int d = 0; d < m_dim; d++)
        {
            numAct *= m_knotVectors[d].getOrder() + 1;
        }
        return numAct;
    }

    __device__
    int getNumValues(int dir) const
    {
        return m_knotVectors[dir].getOrder() + 1;
    }

    __device__
    int getNumValues() const
    {
        int numValues = 1;
        for (int d = 0; d < m_dim; d++)
        {  
            numValues *= getNumValues(d);
        }
        return numValues;
    }

    __device__
    int getNumDerivatives(int maxDer) const
    {
        int numValues = getNumValues();
        if (maxDer < 1)
            return 0;
        else if (maxDer == 1)
            return numValues * m_dim;
        else if (maxDer == 2)
            return numValues * m_dim + numValues * (m_dim + m_dim * (m_dim - 1) / 2);
        else
            printf("Error: maxDer > 2 is not supported yet.\n");
    }

    __device__
    static int getNumValues(int dim, const int* numValuesInDirs)
    {
        int numValues = 1;
        for (int d = 0; d < dim; d++)
        {
            numValues *= numValuesInDirs[d];
        }
        return numValues;
    }

    __device__
    static int getNumDerivatives(int dim, int maxDer, int numValues) 
    {
        if (maxDer < 1)
            return 0;
        else if (maxDer == 1)
            return numValues * dim;
        else if (maxDer == 2)
            return numValues * dim + numValues * (dim + dim * (dim - 1) / 2);
        else
            printf("Error: maxDer > 2 is not supported yet.\n");
    }

    __device__
    const KnotVector_d& getKnotVector(int d) const
    {
        return m_knotVectors[d];
    }

    __device__
    const DeviceObjectArray<double>& getKnotData(int d) const
    {
        return m_knotVectors[d].getKnots();
    }

    __host__
    int knotSize(int d) const
    {
        int* h_sizes = new int[m_dim];
        int* d_sizes = nullptr;
        cudaMalloc((void**)&d_sizes, m_dim * sizeof(int));
        DeviceObjectArray<KnotVector_d>* d_knots_d = nullptr;
	    cudaMalloc((void**)&d_knots_d, sizeof(DeviceObjectArray<KnotVector_d>));
	    cudaMemcpy(d_knots_d, &m_knotVectors, sizeof(DeviceObjectArray<KnotVector_d>), 
                   cudaMemcpyHostToDevice);
        int blockSize = 256;
        int numBlocks = (m_dim + blockSize - 1) / blockSize;
        retrieveKnotSize<<<numBlocks, blockSize>>>(d_knots_d, d_sizes);
        if (cudaGetLastError() != cudaSuccess) 
            printf("Error in retrieveKnotSize: %s\n", cudaGetErrorString(cudaGetLastError()));
        if (cudaDeviceSynchronize() != cudaSuccess)
            printf("Error in cudaDeviceSynchronize: %s\n", cudaGetErrorString(cudaGetLastError()));
        cudaMemcpy(h_sizes, d_sizes, m_dim * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_sizes);
        cudaFree(d_knots_d);
        int size = h_sizes[d];
        delete[] h_sizes;
        return size;
    }

    __host__
    int knotOrder(int d) const
    {
        int* h_orders = new int[m_dim];
        int* d_orders = nullptr;
        cudaMalloc((void**)&d_orders, m_dim * sizeof(int));
        DeviceObjectArray<KnotVector_d>* d_knots_d = nullptr;
        cudaMalloc((void**)&d_knots_d, sizeof(DeviceObjectArray<KnotVector_d>));
        cudaMemcpy(d_knots_d, &m_knotVectors, sizeof(DeviceObjectArray<KnotVector_d>), 
                   cudaMemcpyHostToDevice);
        int blockSize = 256;
        int numBlocks = (m_dim + blockSize - 1) / blockSize;
        retrieveKnotOrder<<<numBlocks, blockSize>>>(d_knots_d, d_orders);
        if (cudaGetLastError() != cudaSuccess) 
            printf("Error in retrieveKnotOrder: %s\n", cudaGetErrorString(cudaGetLastError()));
        if (cudaDeviceSynchronize() != cudaSuccess) 
            printf("Error in cudaDeviceSynchronize: %s\n", cudaGetErrorString(cudaGetLastError()));
        cudaMemcpy(h_orders, d_orders, m_dim * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_orders);
        cudaFree(d_knots_d);
        int order = h_orders[d];
        delete[] h_orders;
        return order;
    }

    __host__
    void knotSizesAndOrders(int* sizes, int* orders) const
    {
        int* h_orders = new int[m_dim];
        int* h_sizes = new int[m_dim];
        int* d_orders = nullptr;
        int* d_sizes = nullptr;
        cudaMalloc((void**)&d_orders, m_dim * sizeof(int));
        cudaMalloc((void**)&d_sizes, m_dim * sizeof(int));
        DeviceObjectArray<KnotVector_d>* d_knots_d = nullptr;
        cudaMalloc((void**)&d_knots_d, sizeof(DeviceObjectArray<KnotVector_d>));
        cudaMemcpy(d_knots_d, &m_knotVectors, sizeof(DeviceObjectArray<KnotVector_d>), 
                   cudaMemcpyHostToDevice);
        int blockSize = 256;
        int numBlocks = (m_dim + blockSize - 1) / blockSize;
        retrieveKnotSizeAndOrder<<<numBlocks, blockSize>>>(d_knots_d, d_orders, d_sizes);
        if (cudaGetLastError() != cudaSuccess) 
            printf("Error in retrieveKnotSizeAndOrder: %s\n", cudaGetErrorString(cudaGetLastError()));
        if (cudaDeviceSynchronize() != cudaSuccess) 
            printf("Error in cudaDeviceSynchronize: %s\n", cudaGetErrorString(cudaGetLastError()));
        cudaMemcpy(h_orders, d_orders, m_dim * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sizes, d_sizes, m_dim * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_orders);
        cudaFree(d_sizes);
        cudaFree(d_knots_d);
        for (int i = 0; i < m_dim; ++i)
        {
            orders[i] = h_orders[i];
            sizes[i] = h_sizes[i];
        }
        delete[] h_orders;
        delete[] h_sizes;
    }

    __host__
    void knotSizeAndOrder(int d, int& size, int& order) const
    {
        int * h_orders = new int[m_dim];
        int * h_sizes = new int[m_dim];
        knotSizesAndOrders(h_sizes, h_orders);
        order = h_orders[d];
        size = h_sizes[d];
        delete[] h_orders;
        delete[] h_sizes;
    }

    __host__
    KnotVector_d getKnotVector_h(int d) const
    {
#if 0
        KnotVector_d* h_knotVectors_sh = new KnotVector_d[m_dim];
        m_knotVectors.shallowCopyToHost(h_knotVectors_sh);
        int size = h_knotVectors_sh[d].getNumKnots();
        int order = h_knotVectors_sh[d].getOrder();
        delete[] h_knotVectors_sh;
#else
        int size = 0, order = 0;
        knotSizeAndOrder(d, size, order);
#endif
        double* data = new double[size];
        double* d_data = nullptr;
        cudaMalloc((void**)&d_data, size * sizeof(double));
        DeviceObjectArray<KnotVector_d>* d_knots_d = nullptr;
	    cudaMalloc((void**)&d_knots_d, sizeof(DeviceObjectArray<KnotVector_d>));
	    cudaMemcpy(d_knots_d, &m_knotVectors, sizeof(DeviceObjectArray<KnotVector_d>), 
                   cudaMemcpyHostToDevice);
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        //retrieveKnotData<<<1, 1>>>(d, d_knots_d, d_data);
        retrieveKnotData<<<numBlocks, blockSize>>>(d, d_knots_d, d_data);
        if (cudaGetLastError() != cudaSuccess) 
            printf("Error in retrieveKnotData: %s\n", cudaGetErrorString(cudaGetLastError()));
        if (cudaDeviceSynchronize() != cudaSuccess) 
            printf("Error in cudaDeviceSynchronize: %s\n", cudaGetErrorString(cudaGetLastError()));
        cudaMemcpy(data, d_data, size * sizeof(double), cudaMemcpyDeviceToHost);
        //printArray(data, size);
        cudaFree(d_data);
        cudaFree(d_knots_d);
        KnotVector_d knotVector(order, size, data);
        delete[] data;
        return knotVector;
    }

    __device__ 
    DeviceObjectArray<int> dofCoords(int idx) const
    {
        DeviceObjectArray<int> coords(2);
        coords[0] = idx % numCPs();
        idx /= numCPs();
        coords[1] = idx;
        return coords;
    }

    __device__
    DeviceObjectArray<int> ptCoords(int idx) const
    {
        DeviceObjectArray<int> coords(m_dim * 2);
        for (int d = 0; d < m_dim; d++)
        {
            coords[d] = idx % m_knotVectors[d].numGaussPoints();
            idx /= m_knotVectors[d].numGaussPoints();
        }
        for (int d = m_dim; d < 2 * m_dim; d++)
        {
            coords[d] = idx % m_knotVectors[d - m_dim].numElements();
            idx /= m_knotVectors[d - m_dim].numElements();
        }
        return coords;
    }

    __device__
    DeviceObjectArray<int> ptCoords(int idx , BoxSide_d side) const
    {
        int fixDir = side.direction();
        bool param = side.parameter();
        DeviceObjectArray<int> coords(m_dim * 2);
        for (int d = 0; d < m_dim; d++)
        {
            if (d == fixDir)
            {
                if (param)
                    coords[d] = m_knotVectors[d].numGaussPoints();
                else
                    coords[d] = 0;
            }
            else
            {
                coords[d] = idx % m_knotVectors[d].numGaussPoints();
                idx /= m_knotVectors[d].numGaussPoints();
            }
        }
        for (int d = m_dim; d < 2 * m_dim; d++)
        {
            if (d == fixDir + m_dim)
            {
                if (param)
                    coords[d] = m_knotVectors[d - m_dim].numElements();
                else
                    coords[d] = 0;
            }
            else
            {
                coords[d] = idx % m_knotVectors[d - m_dim].numElements();
                idx /= m_knotVectors[d - m_dim].numElements();
            }
        }
        return coords;
    }

    __device__
    DeviceVector<double> lowerElementSupport(int idx) const
    {
        DeviceVector<double> lower(m_dim);
        DeviceObjectArray<int> coords = ptCoords(idx);
        for (int d = 0; d < m_dim; d++)
        {
            int order = m_knotVectors[d].getOrder();
            lower(d) = m_knotVectors[d].getKnots()[coords[m_dim + d] + order];
        }
        return lower;
    }

    __device__
    DeviceVector<double> upperElementSupport(int idx) const
    {
        DeviceVector<double> upper(m_dim);
        DeviceObjectArray<int> coords = ptCoords(idx);
        for (int d = 0; d < m_dim; d++)
        {
            int order = m_knotVectors[d].getOrder();
            upper(d) = m_knotVectors[d].getKnots()[coords[m_dim + d] + order + 1];
        }
        return upper;
    }

    __device__
    void elementSupport(int idx, const DeviceObjectArray<int>& coords, 
                        DeviceVector<double>& lower, 
                        DeviceVector<double>& upper) const
    {
        lower.resize(m_dim);
        upper.resize(m_dim);
        for (int d = 0; d < m_dim; d++)
        {
            int order = m_knotVectors[d].getOrder();
            lower(d) = m_knotVectors[d].getKnots()[coords[m_dim + d] + order];
            upper(d) = m_knotVectors[d].getKnots()[coords[m_dim + d] + order + 1];
        }
    }

    __device__
    void elementSupport(int idx, DeviceVector<double>& lower, 
                        DeviceVector<double>& upper) const
    {
        elementSupport(idx, ptCoords(idx), lower, upper);
    }

    __device__
    double gsPoint(int idx, const GaussPoints_d& gspts,
                   DeviceVector<double>& result) const
    {
        DeviceVector<double> gsPoint(m_dim);
        DeviceObjectArray<int> coords = ptCoords(idx);
        DeviceVector<double> lower, upper;
        elementSupport(idx, coords, lower, upper);
        return gspts.threadGaussPoint(lower, upper, coords, result);
    }

    __device__
    int index(const DeviceVector<int>& coords) const
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
    DeviceVector<int> coefSlice(int dir, int k) const
    {
        printf("coefSlice: dir = %d, k = %d\n", dir, k);
        int dim = m_dim;
        if(dir < 0 || dir >= dim)
            printf("Error: dir is out of range in coefSlice.\n");
        if(k < 0 || k >= size(dir))
            printf("Error: k is out of range in coefSlice.\n");

        int sliceSize = 1;
        DeviceVector<int> low(dim), upp(dim);
        //printf("here!\n");
        for(int d = 0; d < dim; ++d)
        {
            sliceSize *= size(d);
            low(d) = 0;
            upp(d) = size(d);
            printf("%d", d);
        }
        //low.print();
        //upp.print();
        sliceSize /= upp(dir);
        low(dir) = k;
	    upp(dir) = k + 1;

        DeviceVector<int> res(sliceSize);
        DeviceVector<int> v = low;
        int i = 0;
        do
        {
            res(i++) = index(v);
        } while (nextLexicographic_d(v, low, upp));

        //printf("res[0]: %d\n", res(0));
        //printf("res[1]: %d\n", res(1));
        return res;
    }

private:
    int m_dim;
    DeviceObjectArray<KnotVector_d> m_knotVectors;
};