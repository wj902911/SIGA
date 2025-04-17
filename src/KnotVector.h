#pragma once

#include <vector>
#include <iostream>
//#include <cuda_runtime.h>
//#include <thrust/device_vector.h>
#include <utility>
//#include "Matrix.h"
#include <Eigen/Core>

class KnotVector {
public:
	//typedef thrust::device_vector<double>::iterator iterator;
	//typedef thrust::device_vector<double>::const_iterator const_iterator;
	typedef std::vector<double>::iterator iterator;
	typedef std::vector<double>::const_iterator const_iterator;

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

	int getOrder() const;
	int size() const;
	int getNumElements() const;
	int getNumGaussPoints() const;
	int getTotalNumGaussPoints() const;
	//const thrust::device_vector<double>& getKnots() const;
	const std::vector<double>& getKnots() const;
	int uSize() const;
	const_iterator uBegin() const;
	const_iterator uEnd() const;
	const_iterator Find(double u) const;
	int getNumControlPoints() const;

	//void getUniformRefinementKnots(int numKnots, thrust::device_vector<double>& knots) const;
	void getUniformRefinementKnots(int numKnots, std::vector<double>& knots) const;
	void insert(const_iterator begin, const_iterator end);
	void uniformRefine(int numKnots = 1);

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
	
private:
	int m_order;
	int m_numElements;
	int m_numGaussPoints;
	//thrust::device_vector<double> m_knots;
	std::vector<double> m_knots;
};