#include "KnotVector.h"
#include <algorithm>
//#include <thrust/merge.h>

# define GS_BIND1ST(_op,_arg) std::bind(_op, _arg, std::placeholders::_1)
# define GS_BIND2ND(_op,_arg) std::bind(_op, std::placeholders::_1, _arg)

KnotVector::KnotVector(
	int order,
	//thrust::device_vector<double>& knots)
	const std::vector<double>& knots)
	: m_order(order),
	  //m_knots(std::move(knots)),
	  m_knots(knots),
	  m_numGaussPoints(order+1)
{ 
	rebuildMultSum(); 
	m_numElements = numElements();
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

void KnotVector::rebuildMultSum()
{
	m_multSum.clear();
	
	const_iterator bb=begin();
	const_iterator it=bb;
	const_iterator ee=end();

	while(it!=ee)
    {
        it = std::find_if(it,ee,GS_BIND1ST(std::not_equal_to<double>(),*it));
        m_multSum.push_back(it-bb);
    }
}

void KnotVector::increaseMultiplicity(const int i, bool boundary)
{
    int newSize = size() + i*(uSize()-2);
	std::vector<double> tmp;
    tmp.reserve(newSize);
	tmp.insert(tmp.end(), m_multSum.front() + (boundary ? i : 0),
               m_knots.front());

	uiterator uit = ubegin()+1;
    for (; uit != uend()-1; ++uit)
        tmp.insert(tmp.end(), i + uit.multiplicity(), *uit);
	tmp.insert(tmp.end(), uit.multiplicity() + (boundary ? i : 0) , *uit);
	m_knots.swap(tmp);
	int r = ( boundary ?  1 : 0 );
	for (std::vector<int>::iterator m = m_multSum.begin(); m != m_multSum.end()-1; ++m)
		*m += i*(r++);
	m_multSum.back() += i * (boundary ? r : r-1 );
}

void KnotVector::increaseEndMultiplicity(const int i, bool boundary)
{
	int newSize = size() + i*(uSize()-2);
	std::vector<double> tmp;
    tmp.reserve(newSize);
	tmp.insert(tmp.end(), m_multSum.front() + (boundary ? i : 0),
               m_knots.front());
	uiterator uit = ubegin()+1;
	for (; uit != uend()-1; ++uit)
		tmp.insert(tmp.end(), uit.multiplicity(), *uit);
	tmp.insert(tmp.end(), uit.multiplicity() + (boundary ? i : 0) , *uit);
	m_knots.swap(tmp);
	int r = ( boundary ?  1 : 0 );
	for (std::vector<int>::iterator m = m_multSum.begin(); m != m_multSum.end()-1; ++m)
		if (m == m_multSum.begin())
			*m += i * r;
		else
			*m += r;
	r++;
	m_multSum.back() += i * (boundary ? r : r-1 );
}

void KnotVector::degreeElevate(const int &i, bool eleInternal)
{
	if (eleInternal)
		increaseMultiplicity(i,true);
	else
		increaseEndMultiplicity(i,true);
	m_order += i;
	m_numElements = numElements();
	m_numGaussPoints = m_order + 1;
}

int KnotVector::getOrder() const { return m_order; }

int KnotVector::size() const { return m_knots.size(); }

int KnotVector::getNumElements() const { return m_numElements; }

int KnotVector::getNumGaussPoints() const { return m_numGaussPoints; }

int KnotVector::getTotalNumGaussPoints() const
{ return m_numElements * m_numGaussPoints; }

//const thrust::device_vector<double>& KnotVector::getKnots() const
const std::vector<double>& KnotVector::getKnots() const
{ return m_knots; }

int KnotVector::uSize() const
{ return m_multSum.size(); }

KnotVector::const_iterator KnotVector::uBegin() const
{ return m_knots.begin() + m_order; }

KnotVector::const_iterator KnotVector::uEnd() const
{ return m_knots.end() - m_order; }

KnotVector::uiterator KnotVector::ubegin() const
{ return uiterator(*this,0,numLeftGhosts()); }

KnotVector::uiterator KnotVector::uend() const
{
    return uiterator::End(*this);
}

KnotVector::smart_iterator KnotVector::sbegin() const
{ return smart_iterator(*this,0,numLeftGhosts()); }

KnotVector::smart_iterator KnotVector::send() const
{ return smart_iterator::End(*this); }

KnotVector::uiterator KnotVector::uFind(const double u) const
{
    uiterator dend = domainUEnd();
	if (u==*dend) // knot at domain end ?
    	return --dend;
	else
    	return std::upper_bound( domainUBegin(), dend, u ) - 1;
}

KnotVector::const_iterator KnotVector::iFind(const double u) const
{
    return begin() + uFind(u).lastAppearance();
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
{ return m_knots.size() - m_order - 1; }

//void KnotVector::getUniformRefinementKnots(int numKnots, thrust::device_vector<double> &knots) const
void KnotVector::getUniformRefinementKnots(int numKnots, std::vector<double> &knots, int mult) const
{
	knots.clear();
	knots.reserve( (uSize()-1) * numKnots * mult );

	double prev = m_knots.front();
	for (uiterator uit = ubegin() + 1; uit != uend(); ++uit)
	{
		const double step = (*uit - prev)/(double)(numKnots+1);
		for (int i = 1; i <= numKnots; ++i)
			knots.insert( knots.end(), mult, prev + (double)(i)*step );
		prev = *uit;
	}
}

void KnotVector::insert(const_iterator ibeg, const_iterator iend)
{
	//thrust::device_vector<double> temp(m_knots.size() + (iend - ibeg));
	std::vector<double> temp(m_knots.size() + (iend - ibeg));
	std::merge(begin(), end(), ibeg, iend, temp.begin());
	m_knots.swap(temp);
	rebuildMultSum();
}

void KnotVector::uniformRefine(int numKnots, int mult)
{
	const int l = ( domainUBegin() - ubegin() ) * numKnots * mult;
	const int r = ( uend() - domainUEnd() - 1 ) * numKnots * mult;
	//thrust::device_vector<double> newKnots;
	std::vector<double> newKnots;
	getUniformRefinementKnots(numKnots, newKnots);
	insert(newKnots.begin(), newKnots.end());

	if (0!=l) trimLeft (l);
	if (0!=r) trimRight(r);

	//m_numElements = m_knots.size() - m_order * 2 - 1;
	m_numElements = numElements();
}

KnotVector::const_iterator KnotVector::begin() const
{ return m_knots.begin(); }

KnotVector::const_iterator KnotVector::end() const
{ return m_knots.end(); }

Eigen::MatrixXd KnotVector::support() const
{
	Eigen::MatrixXd result(1, 2);
	result << domainBegin(), domainEnd();
	return result;
}

void KnotVector::trimLeft(const int numKnots)
{
	m_knots.erase(m_knots.begin(), m_knots.begin()+numKnots);
	std::vector<int>::iterator upos = 
		std::upper_bound(m_multSum.begin(), m_multSum.end(), numKnots);
	std::transform(upos, m_multSum.end(), upos, GS_BIND2ND(std::minus<int>(),numKnots));
}   


void KnotVector::trimRight(const int numKnots)
{
	m_knots.resize(m_knots.size()-numKnots);
	const int newSum = m_multSum.back()-numKnots;
	std::vector<int>::iterator upos = 
		std::lower_bound(m_multSum.begin(), m_multSum.end(), newSum) + 1;
	m_multSum.erase(upos, m_multSum.end() );
	m_multSum.back() = newSum;
}
