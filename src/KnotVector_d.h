#pragma once

#include <DeviceObjectArray.h>
#include <KnotVector.h>

class KnotVector_d
{
public:
    KnotVector_d() = default;

    __host__ __device__
    KnotVector_d(int order, int numKnots, const double* knots)
        : m_order(order), m_knots(numKnots, knots) { }

    __host__ __device__
    KnotVector_d(int order, const DeviceObjectArray<double>& knots)
        : m_order(order), m_knots(knots) { }
    
    __host__
    KnotVector_d(const KnotVector& host_other)
        : m_order(host_other.getOrder()), 
          m_knots(host_other.getKnots().size(), 
                  host_other.getKnots().data()) { }

    // Copy constructor
    __host__ __device__
    KnotVector_d(const KnotVector_d& other)
        : m_order(other.m_order), m_knots(other.m_knots) 
    {
#if 0 
        printf("From KnotVector_d copy constructor:\n");
        printf("From:");
        other.m_knots.print();
        printf("To:");
        m_knots.print();
#endif
    }

    // Copy assignment operator
    __host__ __device__
    KnotVector_d& operator=(const KnotVector_d& other)
    {
        if (this != &other)
        {
            m_order = other.m_order;
            m_knots = other.m_knots;
#if 0
            printf("From KnotVector_d copy assignment operator:\n");
            printf("From:");
            other.m_knots.print();
            printf("To:");
            m_knots.print();
#endif
        }
        return *this;
    }

    // Move constructor
    __host__ __device__
    KnotVector_d(KnotVector_d&& other) noexcept
        : m_order(other.m_order), m_knots(std::move(other.m_knots))
    { 
        other.m_order = 0; 
        //printf("From KnotVector_d move constructor:\n");
        //printf("From:");
        //other.m_knots.print();
        //printf("To:");
        //m_knots.print();
    }

    // Move assignment operator
    __host__ __device__
    KnotVector_d& operator=(KnotVector_d&& other) noexcept
    {
        if (this != &other)
        {
            m_order = other.m_order;
            m_knots = std::move(other.m_knots);
            other.m_order = 0;
        }
        return *this;
    }
    
    __host__ __device__
    ~KnotVector_d() = default;

    __host__ __device__
    const DeviceObjectArray<double>& getKnots() const { return m_knots; }

    __host__ __device__
    KnotVector_d clone() const
    {
        return KnotVector_d(m_order, m_knots);
    }

    __host__ __device__
    int numGaussPoints() const { return m_order + 1; }

    __host__ __device__
    int numElements() const { return m_knots.size() - m_order * 2 - 1; }

    __host__ __device__
    int totalNumGaussPoints() const { return numElements() * numGaussPoints(); }

    __host__ __device__
    int getNumControlPoints() const { return m_knots.size() - m_order - 1; }

    __device__
    int upperBound(double value) const
    {
        int low = m_order - 1;
        int high = m_knots.size();;
        while (low < high)
        {
            int mid = low + (high - low) / 2;
            if (m_knots[mid] < value)
            {
                low = mid + 1;
            }
            else
            {
                high = mid;
            }
        }
        return low;
    }

    __host__ __device__
    int getOrder() const { return m_order; }

    __host__ __device__
    int getNumKnots() const { return m_knots.size(); }

    __device__
    double domainBegin() const { return m_knots[m_order]; }
    
    __device__
    double domainEnd() const { return m_knots[m_knots.size() - m_order - 1]; }

    __device__
    bool inDomain(double u) const
    { return u >= domainBegin() && u <= domainEnd(); }

    __device__
	const double& operator[](int i) const
    {
        if (i < 0 || i >= m_knots.size())
        {
            printf("Error: KnotVector_d::operator[] - index out of range.\n");
            return m_knots[0];
        }
        return m_knots[i];
    }

    __device__
    void print() const
    {
        printf("KnotVector_d: order = %d, knots = ", m_order);
        for (int i = 0; i < m_knots.size(); ++i)
        {
            printf("%f ", m_knots[i]);
        }
        printf("\n");
    }

private:
    int m_order = 0;
    //int m_numKnots = 0;
    DeviceObjectArray<double> m_knots;
};