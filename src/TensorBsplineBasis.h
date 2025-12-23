#pragma once
#include "KnotVector.h" 
#include "Boundary.h"
//#include "Matrix.h"
//#include <vector>
#include <functional>
//#include <Eigen/core>

class TensorBsplineBasis
{
public:
	

	TensorBsplineBasis();

	TensorBsplineBasis(
		const KnotVector& ku, 
		const KnotVector& kv);

	TensorBsplineBasis(
		const KnotVector& ku, 
		const KnotVector& kv, 
		const KnotVector& kw);

	
#if 0
	// Copy constructor.
	TensorBsplineBasis(const TensorBsplineBasis& other) = default;

	// Move constructor.
	TensorBsplineBasis(TensorBsplineBasis&& other) noexcept;
	TensorBsplineBasis& operator=(TensorBsplineBasis&& other) noexcept;
#endif

	~TensorBsplineBasis();

	int getDim() const;
	int getTotalNumKnots() const;
	int size(int direction) const;
	int size() const;
	int index(const Eigen::VectorXi& coords) const;
	Eigen::VectorXi coefSlice(int dir, int k) const;
	Eigen::VectorXi boundaryOffset(BoxSide const & s, int offset) const;

	Eigen::MatrixXd support() const;

	Eigen::VectorXi boundary(BoxSide const & s) const
	{ return boundaryOffset(s, 0); }

	int corner(BoxCorner const & c) const;


	int getOrder(int direction) const;
	int getNumKnots(int direction) const;
	//const thrust::device_vector<double>& getKnots(int direction) const;
	const std::vector<double>& getKnots(int direction) const;
	int getNumGaussPoints(int direction) const;
	int getTotalNumGaussPoints(int direction) const;
	int getNumElements(int direction) const;

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
	int getNumControlPoints() const;

	int getTotalNumGaussPoints() const;
	int getTotalNumBoundaryGaussPoints() const;

	const KnotVector& getKnotVector(int direction) const;

	void uniformRefine(int direction, int numKnots);
	void uniformRefine(int numKnots = 1);

	void eval_into(int dir, 
		           const Eigen::MatrixXd& u, 
				   Eigen::MatrixXd& result) const;

	void eval_into(const Eigen::MatrixXd& u, 
		           Eigen::MatrixXd& result) const;

	void evalFunc_into(const Eigen::MatrixXd& u,
		               const Eigen::MatrixXd& coefs, 
					   Eigen::MatrixXd& result) const;

	void active_into(const Eigen::MatrixXd& u, 
		             Eigen::MatrixXi& result) const;

	static void linearCombination_into(const Eigen::MatrixXd& coefs,
		                               const Eigen::MatrixXi& actives, 
		                               const Eigen::MatrixXd& values, 
							           Eigen::MatrixXd& result);

	void matchWith(const BoundaryInterface & bi, const TensorBsplineBasis & other,
                   Eigen::MatrixXi & bndThis, Eigen::MatrixXi & bndOther) const;

#if 0
	template <typename T, int d>
	void flipTensorVector(const int dir, 
		                  const Eigen::Vector<int, d> & sz, 
						  Eigen::Matrix<T, -1, -1>& coefs) const
	{
		assert(sz.prod()  == coefs.rows());

		Eigen::VectorXi perstr = sz;
		perstr[dir] /= 2;
	}

	template <typename T, int d>
	void permuteTensorVector(const Eigen::Vector<int, d>& perm,
		                    Eigen::Vector<int, d> & sz,
							Eigen::Matrix<T, -1, -1>& coefs) const
	{
		assert(sz.prod() == coefs.rows());
		assert(perm.sum() == sz.size()*(sz.size()-1)/2);

		if ( perm == Eigen::VectorXi::LinSpaced(sz.size(),0,sz.size()-1) )
			return;

		Eigen::PermutationWrapper<Eigen::VectorXi> p(perm);
	}
#endif

private:
	std::vector<KnotVector> m_knotVectors;
};