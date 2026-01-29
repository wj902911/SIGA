#pragma once

#include <cstdio>
#include <DeviceVectorView.h>

class KnotVectorDeviceView
{
private:
    int m_order = 0;
    DeviceVectorView<double> m_knots;

public:
    __host__ __device__
    KnotVectorDeviceView(int order, int numKnots, double* knots)
    : m_order(order), m_knots(knots, numKnots) { }

    __device__
    DeviceVectorView<double> knots() const { return m_knots; }

    __device__
    int order() const { return m_order; }

    __device__
    void print() const
    {
        printf("Knot Vector (order %d):\n", m_order);
        m_knots.print();
    }

};

