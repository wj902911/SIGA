#pragma once

#include "TensorBsplineBasis.h"

class Patch
{
public:
    Patch(
        const TensorBsplineBasis& basis,
        //thrust::device_vector<double>& controlPoints);
        const Eigen::MatrixXd& controlPoints);

    Patch(
        const KnotVector& ku,
        const KnotVector& kv,
        //thrust::device_vector<double>& controlPoints);
        const Eigen::MatrixXd& controlPoints);

    Patch(
        const KnotVector& ku,
        const KnotVector& kv,
        const KnotVector& kw,
        //thrust::device_vector<double>& controlPoints);
        const Eigen::MatrixXd& controlPoints);

#if 0
    // Copy constructor.
    Patch(const Patch& other) = default;

	// Move constructor.
    Patch(Patch&& other) noexcept;
    Patch& operator=(Patch&& other) noexcept;
#endif

    ~Patch() = default;

    int getBasisDim() const;
    int getCPDim() const;
    int getTotalNumKnots() const;
	//thrust::device_vector<int> coefSlice(int dir, int k) const;
	Eigen::VectorXi coefSlice(int dir, int k) const;

    int getOrder(int direction) const;
    int getNumKnots(int direction) const;
    int getNumGaussPoints(int direction) const;
    int getTotalNumGaussPoints(int direction) const;
    int getTotalNumGaussPoints() const;
    int getNumElements(int direction) const;
    //const thrust::device_vector<double>& getKnots(int direction) const;
    const std::vector<double>& getKnots(int direction) const;

    //void getOrders(thrust::device_vector<int>& orders) const;
    void getOrders(std::vector<int>& orders) const;
    std::vector<int> getOrders() const;
    //void getNumKnots(thrust::device_vector<int>& numKnots) const;
    void getNumKnots(std::vector<int>& numKnots) const;
    std::vector<int> getNumKnots() const;
    //void getKnots(thrust::device_vector<double>& knots) const;
    void getKnots(std::vector<double>& knots) const;
    std::vector<double> getKnots() const;
    //void getNumGpAndEle(thrust::device_vector<int>& numGpAndEle) const;
    void getNumGpAndEle(std::vector<int>& numGpAndEle) const;
    std::vector<int> getNumGpAndEle() const;

    const TensorBsplineBasis& getBasis() const;
    //const thrust::device_vector<double>& getControlPoints() const;
    const Eigen::MatrixXd& getControlPoints() const;
    int getNumControlPoints() const;
    const KnotVector& getKnotVector(int direction) const;

    Eigen::MatrixXd parameterRange() const;

    void eval_into(const Eigen::MatrixXd& u, Eigen::MatrixXd& result) const
    {m_basis.evalFunc_into(u, m_controlPoints, result);}

private:
    TensorBsplineBasis m_basis;
    Eigen::MatrixXd m_controlPoints;
};