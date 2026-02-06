#pragma once

#include <DeviceArray.h>

class GaussPointsDeviceView
{
private:
    int m_dim = 0;
    DeviceNestedArrayView<double> m_gaussPoints;
    DeviceNestedArrayView<double> m_gaussWeights;

public:
    __device__
    GaussPointsDeviceView(int dim,
                          const DeviceNestedArrayView<double>& gaussPoints,
                          const DeviceNestedArrayView<double>& gaussWeights)
                        : m_dim(dim),
                          m_gaussPoints(gaussPoints),
                          m_gaussWeights(gaussWeights)
    {
    }

    __device__
    void print() const
    {
        printf("Gauss Points:\n");
        m_gaussPoints.print();
        printf("Gauss Weights:\n");
        m_gaussWeights.print();
    }

    __host__ __device__
    int dim() const { return m_dim; }

    __device__
    double threadGaussPoint(DeviceVectorView<double> lower,
                            DeviceVectorView<double> upper,
                            DeviceVectorView<int> coords,
                            DeviceVectorView<double> gausspt) const
    {
        if (gausspt.size() != m_dim)
        {
            assert("threadGaussPoint: gausspt size mismatch");
            return 0.0;
        }
        double weight = 1.0;
        for (int d = 0; d < m_dim; ++d)
        {
            gausspt[d] = m_gaussPoints[d][coords[d]];
            weight *= m_gaussWeights[d][coords[d]];
        }
        double hprod = 1.0;
        for (int d = 0; d < m_dim; ++d) {
            double h = (upper[d] - lower[d]) / 2.0;
            gausspt[d] = h * (gausspt[d] + 1.0) + lower[d];
            hprod *= (h == 0.0 ? 0.5 : h);
        }
        return weight * hprod;
    }
};