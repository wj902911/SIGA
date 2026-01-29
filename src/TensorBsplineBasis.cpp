#include "TensorBsplineBasis.h"
#include "Utility_h.h"
#include <cassert>

TensorBsplineBasis::TensorBsplineBasis()
{
}

TensorBsplineBasis::TensorBsplineBasis(
	const KnotVector& ku, 
	const KnotVector& kv)
#if 0
	:m_dim(2),
	 m_orders(2),
	 m_numKnots(2),
	 m_numGpAndEle(4),
	 m_knots(ku.size() + kv.size())
#endif
{
#if 0
m_orders[0] = ku.getOrder();
    m_orders[1] = kv.getOrder();
    m_numKnots[0] = ku.size();
    m_numKnots[1] = kv.size();
	m_numGpAndEle[0] = ku.getNumGaussPoints();
	m_numGpAndEle[1] = kv.getNumGaussPoints();
	m_numGpAndEle[2] = ku.getNumElements();
	m_numGpAndEle[3] = kv.getNumElements();
	thrust::copy(ku.begin(), ku.end(), m_knots.begin());
	thrust::copy(kv.begin(), kv.end(), m_knots.begin() + ku.size());
#endif
	m_knotVectors.reserve(2);
	m_knotVectors.push_back(ku);
	m_knotVectors.push_back(kv);
}

TensorBsplineBasis::TensorBsplineBasis(
	const KnotVector& ku, 
	const KnotVector& kv, 
	const KnotVector& kw)
#if 0
	:m_dim(3),
	 m_orders(3),
	 m_numKnots(3),
	 m_numGpAndEle(6),
	 m_knots(ku.size() + kv.size() + kw.size())
#endif
{
#if 0
	m_orders[0] = ku.getOrder();
	m_orders[1] = kv.getOrder();
	m_orders[2] = kw.getOrder();
	m_numKnots[0] = ku.size();
	m_numKnots[1] = kv.size();
	m_numKnots[2] = kw.size();
	m_numGpAndEle[0] = ku.getNumGaussPoints();
	m_numGpAndEle[1] = kv.getNumGaussPoints();
	m_numGpAndEle[2] = kw.getNumGaussPoints();
	m_numGpAndEle[3] = ku.getNumElements();
	m_numGpAndEle[4] = kv.getNumElements();
	m_numGpAndEle[5] = kw.getNumElements();
	thrust::copy(ku.begin(), ku.end(), m_knots.begin());
	thrust::copy(kv.begin(), kv.end(), m_knots.begin() + ku.size());
	thrust::copy(kw.begin(), kw.end(), m_knots.begin() + ku.size() + kv.size());
#endif
	m_knotVectors.reserve(3);
	m_knotVectors.push_back(std::move(ku));
	m_knotVectors.push_back(std::move(kv));
	m_knotVectors.push_back(std::move(kw));
}

#if 0
TensorBsplineBasis::TensorBsplineBasis(TensorBsplineBasis &&other) noexcept
	:m_knotVectors(std::move(other.m_knotVectors))
{
}

TensorBsplineBasis &TensorBsplineBasis::operator=(TensorBsplineBasis &&other) noexcept
{
	if (this != &other)
	{
		m_knotVectors = std::move(other.m_knotVectors);
	}
	return *this;
}
#endif
TensorBsplineBasis::~TensorBsplineBasis()
{
}

int TensorBsplineBasis::corner(BoxCorner const &c) const
{
	int d = getDim();
    Eigen::Vector<bool, -1> position(d);
	c.parameters_into(d, position);

	int index = 0;
	int str = 1;
	for (int i = 0; i != d; ++i)
	{
		const int sz_i = size(i);
		if ( position[i] )
			index+= str * ( sz_i - 1 );
		str *= sz_i;
	}

	return index;
}

int TensorBsplineBasis::getOrder(int direction) const
{
	return m_knotVectors[direction].getOrder();
}

int TensorBsplineBasis::getNumKnots(int direction) const
{
	return m_knotVectors[direction].size();
}

int TensorBsplineBasis::getTotalNumKnots() const
{
	int totalNumKnots = 0;
	for (int d = 0; d < getDim(); ++d)
	{
		totalNumKnots += getNumKnots(d);
	}
	return totalNumKnots;
}

int TensorBsplineBasis::size(int direction) const
{
	return m_knotVectors[direction].getNumControlPoints();
}

int TensorBsplineBasis::size() const
{
	int size = 1;
	for (int d = 0; d < getDim(); ++d)
	{
		size *= TensorBsplineBasis::size(d);
	}
	return size;
}

int TensorBsplineBasis::index(const Eigen::VectorXi& coords) const
{
	int index = 0;
	int dim = getDim();
	index = coords(dim - 1);
	for (int d = dim - 2; d >= 0; --d)
	{
		index = index * size(d) + coords(d);
	}
	return index;
}

Eigen::VectorXi TensorBsplineBasis::coefSlice(int dir, int k) const
{
	int dim = getDim();

	if(dir < 0 || dir >= dim)
	{
		std::cerr << "Error: Invalid slice direction!" << std::endl;
	}
	if(k < 0 || k >= size(dir))
	{
		std::cerr << "Error: Invalid slice position!" << std::endl;
	}

	int sliceSize = 1;
	Eigen::VectorXi low(dim), upp(dim);
	for(int d = 0; d < dim; ++d)
	{
		sliceSize *= size(d);
		low[d] = 0;
		upp[d] = size(d);
	}

    sliceSize /= upp(dir);
	low(dir) = k;
	upp(dir) = k + 1;

	Eigen::VectorXi res(sliceSize);

	Eigen::VectorXi v = low;
	int i = 0;
	do 
	{
		res[i++] = index(v);
	} while (nextLexicographic(v, low, upp));

	return res;
}

Eigen::VectorXi TensorBsplineBasis::boundaryOffset(BoxSide const &s, int offset) const
{
	int k = s.direction();
	bool r = s.parameter();
	if (offset > size(k))
	{
		std::cerr << "Offset cannot be bigger than the amount of basis functions orthogonal to Boxside s!" << std::endl;
	}
	return coefSlice(k, r ? size(k) - 1 - offset : offset);
}

Eigen::MatrixXd TensorBsplineBasis::support() const
{
	int d = getDim();
	Eigen::MatrixXd res(d, 2);
	for (int i = 0; i < d; ++i)
		res.row(i) = m_knotVectors[i].support();
	return res;
}

//const thrust::device_vector<double> &TensorBsplineBasis::getKnots(int direction) const
const std::vector<double> &TensorBsplineBasis::getKnots(int direction) const
{
	return m_knotVectors[direction].getKnots();
}

int TensorBsplineBasis::getNumGaussPoints(int direction) const
{
	return m_knotVectors[direction].getNumGaussPoints();
}

int TensorBsplineBasis::getTotalNumGaussPoints(int direction) const
{
	return m_knotVectors[direction].getTotalNumGaussPoints();
}

int TensorBsplineBasis::getNumElements(int direction) const
{
	return m_knotVectors[direction].getNumElements();
}

//void TensorBsplineBasis::getOrders(thrust::device_vector<int>& orders) const
void TensorBsplineBasis::getOrders(std::vector<int>& orders) const
{
	int dim = getDim();
	orders.reserve(dim);
	for (int d = 0; d < dim; ++d)
	{
		orders.push_back(m_knotVectors[d].getOrder());
	}
}

std::vector<int> TensorBsplineBasis::getOrders() const
{
	int dim = getDim();
	std::vector<int> orders;
	orders.reserve(dim);
	for (int d = 0; d < dim; ++d)
	{
		orders.push_back(m_knotVectors[d].getOrder());
	}
	return orders;
}

int TensorBsplineBasis::getDim() const
{
	return m_knotVectors.size();
}

//void TensorBsplineBasis::getNumKnots(thrust::device_vector<int>& numKnots) const
void TensorBsplineBasis::getNumKnots(std::vector<int>& numKnots) const
{
	int dim = getDim();
	numKnots.reserve(dim);
	for (int d = 0; d < dim; ++d)
	{
		numKnots.push_back(m_knotVectors[d].size());
	}
}

std::vector<int> TensorBsplineBasis::getNumKnots() const
{
	int dim = getDim();
	std::vector<int> numKnots;
	numKnots.reserve(dim);
	for (int d = 0; d < dim; ++d)
	{
		numKnots.push_back(m_knotVectors[d].size());
	}
	return numKnots;
}

//void TensorBsplineBasis::getKnots(thrust::device_vector<double>& knots) const
void TensorBsplineBasis::getKnots(std::vector<double>& knots) const
{
	int dim = getDim();
	knots.reserve(getTotalNumKnots());
	for (int d = 0; d < dim; ++d)
	{
		knots.insert(knots.end(), m_knotVectors[d].begin(), m_knotVectors[d].end());
	}
}

std::vector<double> TensorBsplineBasis::getKnots() const
{
	int dim = getDim();
	std::vector<double> knots;
	knots.reserve(getTotalNumKnots());
	for (int d = 0; d < dim; ++d)
	{
		knots.insert(knots.end(), m_knotVectors[d].begin(), m_knotVectors[d].end());
	}
	return knots;
}

//void TensorBsplineBasis::getNumGpAndEle(thrust::device_vector<int>& numGpAndEle) const
void TensorBsplineBasis::getNumGpAndEle(std::vector<int>& numGpAndEle) const
{
	int dim = getDim();
	numGpAndEle.reserve(dim * 2);
	for (int d = 0; d < dim; ++d)
	{
		numGpAndEle.push_back(m_knotVectors[d].getNumGaussPoints());
	}
	for (int d = 0; d < dim; ++d)
	{
		numGpAndEle.push_back(m_knotVectors[d].getNumElements());
	}
}

std::vector<int> TensorBsplineBasis::getNumGpAndEle() const
{
	int dim = getDim();
	std::vector<int> numGpAndEle;
	numGpAndEle.reserve(dim * 2);
	for (int d = 0; d < dim; ++d)
	{
		numGpAndEle.push_back(m_knotVectors[d].getNumGaussPoints());
	}
	for (int d = 0; d < dim; ++d)
	{
		numGpAndEle.push_back(m_knotVectors[d].getNumElements());
	}
	return numGpAndEle;
}

int TensorBsplineBasis::getNumControlPoints() const
{
	int dim = getDim();
	int numControlPoints = 1;
	for (int d = 0; d < dim; ++d)
	{
		numControlPoints *= getNumKnots(d) - getOrder(d) - 1;
	}
	return numControlPoints;
}

int TensorBsplineBasis::getTotalNumGaussPoints() const
{
	int totalNumGaussPoints = 1;
	for (int d = 0; d < getDim(); ++d)
	{
		totalNumGaussPoints *= getTotalNumGaussPoints(d);
	}
	return totalNumGaussPoints;
}

int TensorBsplineBasis::getTotalNumBoundaryGaussPoints() const
{
	int totalNumGaussPoints = 0;
	int dim = getDim();
	for (int d = 0; d < dim; ++d)
	{
		totalNumGaussPoints += getTotalNumGaussPoints(d)*pow(2, dim - 1);
	}
    return totalNumGaussPoints;
}

const KnotVector& TensorBsplineBasis::getKnotVector(int direction) const
{
    return m_knotVectors[direction];
}

void TensorBsplineBasis::uniformRefine(int direction, int numKnots)
{
	m_knotVectors[direction].uniformRefine(numKnots);
}

void TensorBsplineBasis::uniformRefine(int numKnots)
{
	for (int d = 0; d < getDim(); ++d)
	{
		m_knotVectors[d].uniformRefine(numKnots);
	}
}

void TensorBsplineBasis::eval_into(int dir, 
	                               const Eigen::MatrixXd &u, 
								   Eigen::MatrixXd &result) const
{
	int order = getOrder(dir);
	result.resize(order+1, u.cols());
	std::vector<double> left(order+1), right(order+1);

	for(int v = 0; v < u.cols(); ++v)
	{
		//std::cout << "u(0,v) = " << u(0,v) << "\n\n";
		unsigned span = m_knotVectors[dir].Find(u(0,v)) - m_knotVectors[dir].begin();
		result(0, v) = 1.0;
		for (int j = 1; j <= order; j++)
		{
			left[j] = u(0,v) - m_knotVectors[dir][span + 1 - j];
			//std::cout << left[j] << " = " << u(0,v) << " - " << m_knotVectors[dir][span + 1 - j] << "\n";
			right[j] = m_knotVectors[dir][span + j] - u(0,v);
			//std::cout << right[j] << " = " << m_knotVectors[dir][span + j] << " - " << u(0,v) << "\n";
			double saved = 0.0;
			for (int r = 0; r != j; r++)
			{
				double temp = result(r,v) / (right[r + 1] + left[j - r]);
				//std::cout << temp << " = " << result(r,v) << " / (" << right[r+1] << " + " << left[j-r] << ")\n";
				result(r,v) = saved + right[r + 1] * temp;
				//std::cout << result(r,v) << " = " << saved << " + " << right[r+1] << " * " << temp << "\n";
				saved = left[j - r] * temp;
				//std::cout << saved << " = " << left[j-r] << " * " << temp << "\n";
			}
			result(j,v) = saved;
			//std::cout << result(j,v) << " = " << saved << "\n";
		}
	}
}

void TensorBsplineBasis::eval_into(const Eigen::MatrixXd &u, Eigen::MatrixXd &result) const
{
	int d = getDim();
	assert(u.rows() == d);

	std::vector<Eigen::MatrixXd> ev(d);
	Eigen::VectorXi v(d), size(d);
	unsigned nb = 1;
	for (int i = 0; i < d; ++i)
	{
		eval_into(i, u.row(i), ev[i]);
		//std::cout << "ev[" << i << "]:\n" << ev[i] << "\n\n";
		nb *= ev[i].rows();
		size[i] = ev[i].rows();
	}

	result.resize( nb, u.cols() );

	v.setZero();
	unsigned r = 0;
	do {
		result.row( r )=  ev[0].row( v(0) );
		for ( int i=1; i<d; ++i)
			result.row( r ) = result.row( r ).cwiseProduct( ev[i].row( v(i) ) );
		++r;
	} while(nextLexicographic(v, size));
}

void TensorBsplineBasis::evalFunc_into(const Eigen::MatrixXd &u, 
	                                   const Eigen::MatrixXd &coefs, 
									   Eigen::MatrixXd &result) const
{
	Eigen::MatrixXd B;
	Eigen::MatrixXi actives;

	eval_into(u, B);
	//std::cout << "B:\n" << B << "\n\n";

	active_into(u,actives);
	//std::cout << "actives:\n" << actives << "\n\n";

	//std::cout << "coefs:\n" << coefs << "\n\n";
	linearCombination_into(coefs, actives, B, result);
	//std::cout << "result:\n" << result << "\n\n";
}

void TensorBsplineBasis::active_into(const Eigen::MatrixXd &u, Eigen::MatrixXi &result) const
{
	int d = getDim();
	assert(u.rows() == d);
	Eigen::VectorXi firstAct(d);
	Eigen::VectorXi v(d), size(d);

	unsigned numAct = 1;
	for (unsigned i = 0; i < d; ++i)
	{
		size[i] = m_knotVectors[i].numActive();
		numAct *= size[i];
	}

	result.resize( numAct, u.cols() ); 

	for (int j = 0; j < u.cols(); ++j)
	{
		for (int i = 0; i < d; ++i)
			firstAct[i] = m_knotVectors[i].firstActive(u(i,j));

		unsigned r = 0;
		v.setZero();
		do
		{
			int gidx = firstAct[d-1] + v(d-1);

			for ( int i=d-2; i>=0; --i )
				gidx = gidx * size(i) + firstAct[i] + v(i);

			result(r,j) = gidx;
			++r;
		} while (nextLexicographic(v, size));
	}
}

void TensorBsplineBasis::linearCombination_into(const Eigen::MatrixXd &coefs, 
	                                            const Eigen::MatrixXi &actives, 
												const Eigen::MatrixXd &values, 
												Eigen::MatrixXd &result)
{
	const int numPts = values.cols();
	const int tarDim = coefs.cols();
	const int stride = values.rows() / actives.rows();

	assert(actives.rows() * stride == values.rows());

	result.resize( tarDim * stride, numPts );
	result.setZero();

	for ( int pt = 0; pt < numPts; ++pt )
    	for ( int i = 0; i < actives.rows(); ++i )
        	for ( int c = 0; c < tarDim; ++c )
        	{
        	    result.block( stride * c, pt, stride, 1).noalias() +=
        	    	coefs( actives(i,pt), c) * values.block( stride * i, pt, stride, 1);
        	}
}

void TensorBsplineBasis::matchWith(const BoundaryInterface &bi, 
	                               const TensorBsplineBasis &other, 
								   Eigen::MatrixXi &bndThis, 
								   Eigen::MatrixXi &bndOther) const
{
	int d = getDim();
	bndThis = boundary( bi.first().side() );
	bndOther= other.boundary( bi.second().side() );
	assert(bndThis.rows() == bndOther.rows());

#if 0
	if (bndThis.size() == 1) return;

	const int s1 = bi.first().direction();
	const int s2 = bi.second().direction();
	const Eigen::Vector<bool, -1>& dirOr = bi.dirOrientation();
	const Eigen::VectorXi & bMap = bi.dirMap();

	Eigen::VectorXi  bSize(d-1);
	int c = 0;
	for (int k = 0; k<d; ++k )
	{
		if ( k == s1 )
		    continue;
		bSize[c] = size(k);
		c++;
	}

	Eigen::VectorXi bPerm(d-1);
	c = 0;
	for (int k = 0; k<d; ++k )
	{
		if ( k == s1 )
		    continue;

		if ( ! dirOr[k] )
			flipTensorVector(c, bSize, bndThis);

		bPerm[c] = ( bMap[k] < s2 ? bMap[k] : bMap[k]-1 );
		c++;
	}

	permuteTensorVector(bPerm, bSize, bndThis);
#endif

	return;
}

int TensorBsplineBasis::getIntDataSize() const
{
	return getDim() * 2 + 1;
}

int TensorBsplineBasis::getDoubleDataSize() const
{
	return getTotalNumKnots();
}

// data layout:
// int data: [orders..., knotsOffsets...]
// double data: [knots...]
void TensorBsplineBasis::getData(std::vector<int> &intData, 
	                             std::vector<double> &doubleData) const
{
	intData.clear();
	doubleData.clear();

	int intDataSize = getIntDataSize();
	intData.reserve(intDataSize);
	std::vector<int> orders = getOrders();
	intData.insert(intData.end(), orders.begin(), orders.end());
	int knotsOffset = 0;
	intData.push_back(knotsOffset);
	for (int d = 0; d < getDim(); ++d)
	{
		knotsOffset += getNumKnots(d);
		intData.push_back(knotsOffset);
	}

	int doubleDataSize = getDoubleDataSize();
	doubleData.reserve(doubleDataSize);
	for (int d = 0; d < getDim(); ++d)
	{
		std::vector<double> knots = getKnots(d);
		doubleData.insert(doubleData.end(), knots.begin(), knots.end());
	}
}
