#pragma once

#include <vector>
#include <iostream>
//#include <cuda_runtime.h>
//#include <thrust/device_vector.h>
#include <utility>
//#include "Matrix.h"
#include <Eigen/Core>
#include <KnotIterator.h>
#include <math.h>

class KnotVector {
public:
	//typedef thrust::device_vector<double>::iterator iterator;
	//typedef thrust::device_vector<double>::const_iterator const_iterator;
	typedef std::vector<double>::iterator iterator;
	typedef std::vector<double>::const_iterator const_iterator;
	typedef UKnotIterator uiterator;
    typedef KnotIterator  smart_iterator;
	typedef std::reverse_iterator<uiterator> reverse_uiterator;
	typedef std::reverse_iterator<smart_iterator>  reverse_smart_iterator;

	KnotVector(
		int order, 
		//thrust::device_vector<double>& knots);
		const std::vector<double>& knots);

#if 0
	// Copy constructor.
	KnotVector(const KnotVector& other) = default;

	// Move constructor.
    KnotVector(KnotVector&& other) noexcept;
	KnotVector& operator=(KnotVector&& other) noexcept;
#endif

	~KnotVector();

	void rebuildMultSum();
	void increaseMultiplicity(const int i = 1, bool boundary = false);
	void increaseEndMultiplicity(const int i = 1, bool boundary = false);
	const int* multSumData() const { return m_multSum.data(); }
	const double* data() const { return m_knots.data(); }

	void degreeElevate(const int & i = 1, bool eleInternal = true);

	int getOrder() const;
	int size() const;
	int getNumElements() const;
	int numElements() const { return (domainUEnd() - domainUBegin()); }
	int getNumGaussPoints() const;
	int getTotalNumGaussPoints() const;
	//const thrust::device_vector<double>& getKnots() const;
	const std::vector<double>& getKnots() const;
	int uSize() const;
	const_iterator uBegin() const;
	const_iterator uEnd() const;
	uiterator ubegin() const;
    uiterator uend() const;
    smart_iterator sbegin()  const;
    smart_iterator send()    const;  
	uiterator uFind( const double u ) const;
	const_iterator iFind( const double u ) const;

	smart_iterator domainSBegin() const
	{ return sbegin() + m_order; }  

	smart_iterator domainSEnd() const
	{ return send() - (m_order + 1); }

	uiterator domainUBegin() const
	{ return domainSBegin().uIterator(); }

	uiterator domainUEnd() const
	{ return domainSEnd().uIterator(); }

	std::vector<double> breaks() const
	{ return std::vector<double>(domainUBegin(), domainUEnd() + 1); }

	const_iterator Find(double u) const;
	int getNumControlPoints() const;

	//void getUniformRefinementKnots(int numKnots, thrust::device_vector<double>& knots) const;
	void getUniformRefinementKnots(int numKnots, std::vector<double>& knots, int mult = 1) const;
	void insert(const_iterator begin, const_iterator end);
	void uniformRefine(int numKnots = 1, int mult = 1);

	const_iterator begin() const;
	const_iterator end() const;

	const_iterator domainiBegin() const {return begin()+m_order;}
	const_iterator domainiEnd() const {return end()-m_order-1;}

	double domainBegin() const {return *(begin()+m_order);}
	double domainEnd() const {return *(end()-m_order-1);}

	bool inDomain(double u) const
	{ return u >= domainBegin() && u <= domainEnd(); }

	Eigen::MatrixXd support() const;

	int numActive() const { return m_order + 1; }

	int firstActive(double u) const
	{ return inDomain(u) ? Find(u) - begin() - m_order : 0; }

	const double& operator[](int i) const
	{
		if (i < 0 || i >= size())
		{
			std::cerr << "Error: KnotVector::operator[] - index out of range." << std::endl;
			return m_knots[0];
		}
		return m_knots[i];
	}

	int numLeftGhosts() const
    {
        smart_iterator it(*this,0,0);
        it += std::min( (int)m_order, size() );
        return std::distance( uiterator(*this,0,0), it.uIterator() );
    }

	void trimLeft (const int numKnots);
	void trimRight(const int numKnots);
	
private:
	int m_order;
	int m_numElements;
	int m_numGaussPoints;
	//thrust::device_vector<double> m_knots;
	std::vector<double> m_knots;
	std::vector<int> m_multSum;
};