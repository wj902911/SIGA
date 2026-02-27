#pragma once

#include <cstdio>
#include <DeviceVectorView.h>
#include <KnotGPUIterator.h>
#include <Utility_d.h>

class KnotVectorDeviceView
{
private:
    int m_order = 0;
    DeviceVectorView<double> m_knots;
    DeviceVectorView<int> m_multSum;

public:
	typedef UKnotGPUIterator uiterator;
    typedef KnotGPUIterator  smart_iterator;


    __host__ __device__
    KnotVectorDeviceView(int order, int numKnots, double* knots)
    : m_order(order), m_knots(knots, numKnots) { }

    __host__ __device__
    KnotVectorDeviceView(int order, int numKnots, double* knots, 
                         DeviceVectorView<int> multSum)
    : m_order(order), m_knots(knots, numKnots), m_multSum(multSum) { }

    __device__
    DeviceVectorView<double> knots() const { return m_knots; }

    __device__
    int order() const { return m_order; }

    __device__
    void print() const
    {
        printf("Knot Vector (order %d):\n", m_order);
        printf("knots:\n");
        m_knots.print();
        printf("multiplicities:\n");
        m_multSum.print();
    }

    __host__ __device__
    int numElements() const { return m_knots.size() - m_order * 2 - 1; }

    __host__ __device__
    int numGaussPoints() const { return m_order + 1; }

    __host__ __device__
    int totalNumGaussPoints() const { return numElements() * numGaussPoints(); }

    __device__
    double domainBegin() const { return m_knots[m_order]; }
    
    __device__
    double domainEnd() const { return m_knots[m_knots.size() - m_order - 1]; }

    __device__
    bool inDomain(double u) const
    { return u >= domainBegin() && u <= domainEnd(); }

    __host__ __device__
    int numKnots() const { return m_knots.size(); }

     __device__
    int upperBound(double value) const
    {
        int low = m_order - 1;
        int high = m_knots.size();
        while (low < high)
        {
            int mid = low + (high - low) / 2;
            if (m_knots[mid] <= value)
                low = mid + 1;
            else
                high = mid;
        }
        return low;
    }

    __host__ __device__
    int numControlPoints() const { return m_knots.size() - m_order - 1; }

    __device__
    double* begin() const { return m_knots.data(); }

    __device__
    const int* multSumData() const { return m_multSum.data(); }
    
    __device__
    const double* data() const { return m_knots.data(); }

    __device__
    int uSize() const { return m_multSum.size(); }

    __device__
    int size() const { return m_knots.size(); }

    __device__
    int numLeftGhosts() const
    {
        smart_iterator it(*this,0,0);
        it += dmin( (int)m_order, size() );
        return it.uIterator() - uiterator(*this, 0, 0);
    }

    __device__
    smart_iterator sbegin()  const
    { return smart_iterator(*this,0,numLeftGhosts()); }

    __device__
    smart_iterator send() const
    { return smart_iterator::End(*this); }

    __device__
    smart_iterator domainSBegin() const
	{ return sbegin() + m_order; }  

    __device__
    smart_iterator domainSEnd() const
	{ return send() - (m_order + 1); }

    __device__
    uiterator domainUBegin() const
	{ return domainSBegin().uIterator(); }

    __device__
    uiterator domainUEnd() const
	{ return domainSEnd().uIterator(); }

    __device__
    uiterator uFind( const double u ) const
    {
        uiterator dend = domainUEnd();
	    if (u==*dend) // knot at domain end ?
    	    return --dend;
        else
            return upper_bound_it(domainUBegin(), dend, u) - 1;
    }

    __device__
    const double* iFind( const double u ) const
    { return begin() + uFind(u).lastAppearance(); }
};

#include "UKnotGPUIterator_impl.h"