#pragma once

#include <TensorBsplineBasis_d.h>
//#include <DeviceMatrix.h>
#include <DeviceVector.h>
#include <Patch.h>

class Patch_d
{
public:
    // Default constructor
    __host__ __device__
    Patch_d() {};

    __device__
    Patch_d(int basisDim, int CPDim, const int* numKnots, const double* knots, 
            const int* orders, const double* controlPoints, int numControlPoints)
        : m_controlPoints(numControlPoints, CPDim, controlPoints),
        //m_CPDim(CPDim),
          m_basis(TensorBsplineBasis_d(basisDim, numKnots, knots, orders))
        //m_numControlPoints(numControlPoints,CPDim, controlPoints)
    {}

    __host__ __device__
    Patch_d(const TensorBsplineBasis_d& basis, const DeviceMatrix<double>& controlPoints)
        : m_controlPoints(controlPoints), m_basis(basis) {}

#if 0
    __host__
    Patch_d(const TensorBsplineBasis_d& basis, int CPDim)
        : m_basis(basis), m_controlPoints(basis.numCPs(), CPDim)
    { m_controlPoints.setZero(); }
#endif

    __host__
    Patch_d(const Patch& host_other)
        : m_controlPoints(host_other.getControlPoints()),
          m_basis(host_other.getBasis()) {}

    // Copy constructor
    __host__ __device__
    Patch_d(const Patch_d& other)
        : m_controlPoints(other.m_controlPoints), m_basis(other.m_basis) 
    {
        //printf("From Patch_d copy constructor:\n");
        //m_controlPoints.print();
        //printf("\n");
    }
    
    // Copy assignment operator
    __host__ __device__
    Patch_d& operator=(const Patch_d& other)
    {
        if (this != &other) // self-assignment check
        {
            m_controlPoints = other.m_controlPoints;
            //printf("From Patch_d copy assignment operator:\n");
            //m_controlPoints.print();
            //printf("\n");
            m_basis = other.m_basis;
        }
        return *this;
    }

    // Move constructor
    __host__ __device__
    Patch_d(Patch_d&& other) noexcept
        : m_controlPoints(std::move(other.m_controlPoints)), 
          m_basis(std::move(other.m_basis)) {}

    // Move assignment operator
    __host__ __device__
    Patch_d& operator=(Patch_d&& other) noexcept
    {
        if (this != &other) // self-assignment check
        {
            m_controlPoints = std::move(other.m_controlPoints);
            m_basis = std::move(other.m_basis);
        }
        return *this;
    }
    
    ~Patch_d() = default;

    __host__
    Patch_d clone() const
    {
        return Patch_d(m_basis, m_controlPoints);
    }

    __device__
    int getDim() const
    {
        return m_basis.getDim();
    }

    __device__
    const TensorBsplineBasis_d& basis() const
    {
        return m_basis;
    }

    __device__
    int getNumControlPoints() const
    {
        return m_controlPoints.rows();
    }

    __host__ __device__
    const DeviceMatrix<double>& controlPoints() const
    {
        //printf("From Patch_d getControlPoints:\n");
        //m_controlPoints.print();
        //printf("\n");
        return m_controlPoints;
    }

    __device__
    void setCoefficients(int row, int col, double value)
    { m_controlPoints(row, col) = value; }

    __device__
    int getCPDim() const
    {
        return m_controlPoints.cols();
    }

    __device__
    int getNumBasisFunctions() const
    {
        return m_basis.getNumValues();
    }

    __device__
    int getNumDerivatives(int maxDer) const
    {
        return m_basis.getNumDerivatives(maxDer);
    }

    __device__
    void getThreadElementSupport(
        int* threadEleCoords, 
        double* lower,
        double* upper)
    {
        m_basis.getThreadElementSupport(threadEleCoords, lower, upper);
    }

    __device__
    void getActiveControlPoints(double* pt, int numActiveCPs, double* activeCPs)
    {
        int* activeIndexes = new int[numActiveCPs];
        m_basis.getActiveIndexes(pt, activeIndexes, numActiveCPs);
        //int basisDim = m_basis.getDim();
        int CPDim = getCPDim();
        for (int i = 0; i < numActiveCPs; i++)
        {
            for (int j = 0; j < CPDim; j++)
            {
                //activeCPs[i * CPDim + j] = m_controlPoints[activeIndexes[i] * CPDim + j];
                activeCPs[i * CPDim + j] = m_controlPoints(activeIndexes[i], j);
            }
        }
        delete[] activeIndexes;
    }

    __device__
    DeviceMatrix<int> getActiveIndexes(DeviceVector<double> pt)
    {
        return m_basis.getActiveIndexes(pt);
    }

    __device__
    DeviceMatrix<double> getActiveControlPoints(DeviceVector<double> pt) const
    {
        DeviceVector<int> activeIndexes = m_basis.getActiveIndexes(pt);
        int CPDim = getCPDim();
        DeviceMatrix<double> activeCPs(activeIndexes.size(), CPDim);
        for (int i = 0; i < activeIndexes.size(); i++)
        {
            //for (int j = 0; j < CPDim; j++)
            //{
            //    activeCPs(i, j) = m_controlPoints(activeIndexes[i], j);
            //}
            activeCPs.row(i) = m_controlPoints.row(activeIndexes(i));
        }
        return activeCPs;
    }

#if 0
    __device__
    DeviceMatrix<double> getActiveControlPoints(DeviceVector<double> pt) const
    {
        DeviceVector<int> activeIndexes = m_basis.getActiveIndexes(pt);
        int CPDim = getCPDim();
        DeviceMatrix<double> activeCPs(activeIndexes.size(), CPDim);
        for (int i = 0; i < activeIndexes.size(); i++)
        {
            //for (int j = 0; j < CPDim; j++)
            //{
            //    activeCPs(i, j) = m_controlPoints(activeIndexes[i], j);
            //}
            activeCPs.row(i) = m_controlPoints.row(activeIndexes(i));
        }
        return activeCPs;
    }
#endif

    __device__
    void getValuesAnddDerivatives(
        double* pt, 
        int maxDer, 
        double* values,
        double* derivatives)
    {
        m_basis.getValuesAnddDerivatives(pt, maxDer, values, derivatives);
    }

    __device__
    int getNumActiveControlPoints()
    {
        return m_basis.getNumActiveControlPoints();
    }

    __device__
    void getGPFirstOrderGradients(double* firstDers, int numActiveCPs, double* activeCPs, double* firstGrads)
    {
        int basisDim = m_basis.getDim();
        int CPDim = getCPDim();
        for (int i = 0; i < CPDim; i++)
        {
            for (int j = 0; j < basisDim; j++)
            {
                firstGrads[i * basisDim + j] = 0.0;
                for (int k = 0; k < numActiveCPs; k++)
                {
                        //      du_i/dxi_j                 dN_k/dxi_j          C_k        X_i
                    firstGrads[i * basisDim + j] += firstDers[k * basisDim + j] * activeCPs[k * CPDim + i];
                }
            }
        }
    }

    __device__
    void getGPValuesAndFirstOrderGradients(double* pt, double* firstGrads)
    {
        double* values = new double[m_basis.getNumValues()];
        int numActiveCPs = m_basis.getNumActiveControlPoints();
        double* activeCPs = new double[numActiveCPs * getCPDim()];
        getActiveControlPoints(pt, numActiveCPs, activeCPs);
        double* firstDers = new double[getCPDim() * m_basis.getDim()];
        getValuesAnddDerivatives(pt, 1, values, firstDers);
        getGPFirstOrderGradients(firstDers, numActiveCPs, activeCPs, firstGrads);
        delete[] activeCPs, values, firstDers;
    }

    __device__
    DeviceVector<int> coefSlice(int dir, int k) const
    {
        return m_basis.coefSlice(dir, k);
    }

    __device__
    Patch_d boundary(BoxSide_d const& s) const
    {
        DeviceVector<int> ind = m_basis.boundary(s);
        DeviceMatrix<double> coeffs (ind.size(), getCPDim());
        coeffs.setZero();
        for (int i = 0; i != ind.size(); i++)
            coeffs.row(i) = m_controlPoints.row(ind(i));

        //coeffs.print();

        return Patch_d(m_basis.getComponentsForSide(s), coeffs);
    }

    __device__
    double curveLengthPerGSPt(const DeviceVector<double>& pt, double wt)
    {
        int CPdim = getCPDim();
        DeviceMatrix<double> geoActiveCPs = getActiveControlPoints(pt);
        DeviceObjectArray<DeviceVector<double>> geoValues;
        m_basis.evalAllDers_into(pt, 1, geoValues);
        DeviceMatrix<double> md = geoValues[1].reshape(CPdim, geoActiveCPs.rows()) * geoActiveCPs;
        return wt*md.transpose().norm();
    }

    __device__
    int getNumEdgesInEachDir() const
    {
        return m_basis.getNumEdgesInEachDir();
    }

    __device__
    int getNumEdges() const
    {
        return m_basis.getNumEdges();
    }

    __device__
    int numGPsInDir(int d) const
    {
        return m_basis.numGPsInDir(d);
    }

    __device__
    int totalNumGPsInDir(int d) const
    {
        return m_basis.totalNumGPsInDir(d);
    }

    __device__
    int totalNumGPs() const
    {
        return m_basis.totalNumGPs();
    }

    __host__ __device__
    int totalNumBdGPs() const
    {
        return m_basis.totalNumBdGPs();
    }

    __device__
    int totalNumBdGPsInDir(int d) const
    {
        return m_basis.totalNumBdGPsInDir(d);
    }

    __device__
    double upperSupportsInDir(int d) const
    {
        return m_basis.upperSupportInDir(d);
    }

    __device__
    double lowerSupportsInDir(int d) const
    {
        return m_basis.lowerSupportInDir(d);
    }

    __device__
    DeviceVector<double> upperSupports() const
    {
        return m_basis.upperSupports();
    }

    __device__
    DeviceVector<double> lowerSupports() const
    {
        return m_basis.lowerSupports();
    }

    __host__
    void retrieveControlPoints(Eigen::MatrixXd& host_controlPoints) const
    {
        Eigen::MatrixXd host_controlPoints_temp;
        cudaError_t err = cudaMemcpy(host_controlPoints_temp.data(), 
                                         m_controlPoints.data(), 
                                         m_controlPoints.rows() * m_controlPoints.cols() * sizeof(double), 
                                         cudaMemcpyDeviceToHost);
        host_controlPoints = host_controlPoints_temp.transpose();
    }

    __device__
    const DeviceMatrix<double> &getControlPoints() const
    {
        return m_controlPoints;
    }

private:
    //int m_CPDim;
    TensorBsplineBasis_d m_basis;
    //const double* m_controlPoints;
    //int m_numControlPoints = 0;
    DeviceMatrix<double> m_controlPoints;
};