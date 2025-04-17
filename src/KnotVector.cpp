#include "KnotVector.h"
#include <algorithm>
//#include <thrust/merge.h>

KnotVector::KnotVector(
	int order,
	//thrust::device_vector<double>& knots)
	const std::vector<double>& knots)
	: m_order(order),
	  //m_knots(std::move(knots)),
	  m_knots(knots),
	  m_numElements(knots.size()-order*2-1),
	  m_numGaussPoints(order+1)
{
}

#if 0
KnotVector::KnotVector(KnotVector &&other) noexcept
	: m_order(other.m_order),
	  m_knots(std::move(other.m_knots)),
	  m_numElements(other.m_numElements),
	  m_numGaussPoints(other.m_numGaussPoints)
{
}

KnotVector &KnotVector::operator=(KnotVector &&other) noexcept
{
	if (this != &other)
	{4
		m_order = other.m_order;
		m_knots = std::move(other.m_knots);
		m_numElements = other.m_numElements;
		m_numGaussPoints = other.m_numGaussPoints;
	}
	return *this;
}
#endif

KnotVector::~KnotVector()
{
}


int KnotVector::getOrder() const
{
	return m_order;
}

int KnotVector::size() const
{
	return m_knots.size();
}

int KnotVector::getNumElements() const
{
	return m_numElements;
}

int KnotVector::getNumGaussPoints() const
{
	return m_numGaussPoints;
}

int KnotVector::getTotalNumGaussPoints() const
{
	return m_numElements * m_numGaussPoints;
}

//const thrust::device_vector<double>& KnotVector::getKnots() const
const std::vector<double>& KnotVector::getKnots() const
{
	return m_knots;
}

int KnotVector::uSize() const
{
	return m_knots.size() - 2 * m_order;
}

KnotVector::const_iterator KnotVector::uBegin() const
{
	return m_knots.begin() + m_order;
}

KnotVector::const_iterator KnotVector::uEnd() const
{
	return m_knots.end() - m_order;
}

KnotVector::const_iterator KnotVector::Find(double u) const
{
	const_iterator dend = domainiEnd();
	if (u == *dend)
		return dend - 1;
	else
		return std::upper_bound(uBegin(), uEnd(), u) - 1;
}

int KnotVector::getNumControlPoints() const
{
	return m_knots.size() - m_order - 1;
}

//void KnotVector::getUniformRefinementKnots(int numKnots, thrust::device_vector<double> &knots) const
void KnotVector::getUniformRefinementKnots(int numKnots, std::vector<double> &knots) const
{
	knots.clear();

	double prev = *uBegin();
	for (const_iterator it = uBegin() + 1; it != uEnd(); ++it)
	{
		double next = *it;
		double step = (next - prev) / double(numKnots + 1);
		for (int i = 1; i <= numKnots; ++i)
		{
			knots.push_back(prev + step * i);
		}
		prev = next;
	}
}

void KnotVector::insert(const_iterator ibeg, const_iterator iend)
{
	//thrust::device_vector<double> temp(m_knots.size() + (iend - ibeg));
	std::vector<double> temp(m_knots.size() + (iend - ibeg));
	std::merge(begin(), end(), ibeg, iend, temp.begin());
	m_knots.swap(temp);
}

void KnotVector::uniformRefine(int numKnots)
{
	//thrust::device_vector<double> newKnots;
	std::vector<double> newKnots;
	getUniformRefinementKnots(numKnots, newKnots);
	insert(newKnots.begin(), newKnots.end());

	m_numElements = m_knots.size() - m_order * 2 - 1;
}

KnotVector::const_iterator KnotVector::begin() const
{
    return m_knots.begin();
}

KnotVector::const_iterator KnotVector::end() const
{
	return m_knots.end();
}

Eigen::MatrixXd KnotVector::support() const
{
	Eigen::MatrixXd result(1, 2);
	result << domainBegin(), domainEnd();
	return result;
}
