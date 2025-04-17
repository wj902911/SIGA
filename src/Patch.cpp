#include "Patch.h"

Patch::Patch(
    const TensorBsplineBasis& basis, 
    const Eigen::MatrixXd& controlPoints)
    : m_basis(basis), 
      m_controlPoints(controlPoints)
{
    if(getNumControlPoints() != m_basis.getNumControlPoints()) 
    {
        std::cerr << "Error: Number of control points ("<< getNumControlPoints() <<") does not match the basis ("<< m_basis.getNumControlPoints() <<")!" << std::endl;
    }
}

Patch::Patch(
    const KnotVector& ku, 
    const KnotVector& kv, 
    const Eigen::MatrixXd& controlPoints)
    : m_basis(TensorBsplineBasis(ku, kv)), 
      m_controlPoints(controlPoints)
{
    if(getNumControlPoints() != m_basis.getNumControlPoints()) 
    {
        std::cerr << "Error: Number of control points ("<< getNumControlPoints() <<") does not match the basis ("<< m_basis.getNumControlPoints() <<")!" << std::endl;
    }
}

Patch::Patch(
    const KnotVector& ku, 
    const KnotVector& kv, 
    const KnotVector& kw, 
    const Eigen::MatrixXd& controlPoints)
    : m_basis(TensorBsplineBasis(ku, kv, kw)), 
      m_controlPoints(std::move(controlPoints))
{
    if(getNumControlPoints() != m_basis.getNumControlPoints()) 
    {
        std::cerr << "Error: Number of control points ("<< getNumControlPoints() <<") does not match the basis ("<< m_basis.getNumControlPoints() <<")!" << std::endl;
    }
}
#if 0
Patch::Patch(Patch &&other) noexcept
    : m_basis(std::move(other.m_basis)), 
      m_controlPoints(std::move(other.m_controlPoints))
{
}

Patch &Patch::operator=(Patch &&other) noexcept
{
    if (this != &other)
    {
        m_basis = std::move(other.m_basis);
        m_controlPoints = std::move(other.m_controlPoints);
    }
    return *this;
}
#endif
int Patch::getBasisDim() const
{
    return m_basis.getDim();
}

int Patch::getCPDim() const
{
    return m_controlPoints.cols();
}

int Patch::getTotalNumKnots() const
{
    return m_basis.getTotalNumKnots();
}

//thrust::device_vector<int> Patch::coefSlice(int dir, int k) const
Eigen::VectorXi Patch::coefSlice(int dir, int k) const
{
    return m_basis.coefSlice(dir, k);
}

int Patch::getOrder(int direction) const
{
    return m_basis.getOrder(direction);
}

int Patch::getNumKnots(int direction) const
{
    return m_basis.getNumKnots(direction);
}

int Patch::getNumGaussPoints(int direction) const
{
    return m_basis.getNumGaussPoints(direction);
}

int Patch::getTotalNumGaussPoints(int direction) const
{
    return m_basis.getTotalNumGaussPoints(direction);
}

int Patch::getTotalNumGaussPoints() const
{
    return m_basis.getTotalNumGaussPoints();
}

int Patch::getNumElements(int direction) const
{
    return m_basis.getNumElements(direction);
}

//const thrust::device_vector<double> &Patch::getKnots(int direction) const
const std::vector<double> &Patch::getKnots(int direction) const
{
    return m_basis.getKnots(direction);
}

//void Patch::getOrders(thrust::device_vector<int> &orders) const
void Patch::getOrders(std::vector<int> &orders) const
{
    m_basis.getOrders(orders);
}

std::vector<int> Patch::getOrders() const
{
    return m_basis.getOrders();
}

//void Patch::getNumKnots(thrust::device_vector<int> &numKnots) const
void Patch::getNumKnots(std::vector<int> &numKnots) const
{
    m_basis.getNumKnots(numKnots);
}

std::vector<int> Patch::getNumKnots() const
{
    return m_basis.getNumKnots();
}

//void Patch::getKnots(thrust::device_vector<double> &knots) const
void Patch::getKnots(std::vector<double> &knots) const
{
    m_basis.getKnots(knots);
}

std::vector<double> Patch::getKnots() const
{
    return m_basis.getKnots();
}

//void Patch::getNumGpAndEle(thrust::device_vector<int> &numGpAndEle) const
void Patch::getNumGpAndEle(std::vector<int> &numGpAndEle) const
{
    m_basis.getNumGpAndEle(numGpAndEle);
}

std::vector<int> Patch::getNumGpAndEle() const
{
    return m_basis.getNumGpAndEle();
}

const TensorBsplineBasis &Patch::getBasis() const
{
    return m_basis;
}

//const thrust::device_vector<double> &Patch::getControlPoints() const
const Eigen::MatrixXd &Patch::getControlPoints() const
{
    return m_controlPoints;
}

int Patch::getNumControlPoints() const
{
    return m_controlPoints.size() / m_basis.getDim();
}

const KnotVector& Patch::getKnotVector(int direction) const
{
    return m_basis.getKnotVector(direction);
}

Eigen::MatrixXd Patch::parameterRange() const
{
    return m_basis.support();
}
