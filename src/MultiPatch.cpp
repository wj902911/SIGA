#include "MultiPatch.h"
#include <set>

#if 0
template <typename T>
T* appendToDeviceArray(
    T* oldArray, 
    int oldSize, 
    T* newData, 
    int newSize) 
{
    T* newArray;
    cudaMalloc(&newArray, (oldSize + newSize) * sizeof(T));

    if (oldArray) 
    {
        cudaMemcpy(newArray, oldArray, oldSize * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaFree(oldArray);
    }

    cudaMemcpy(newArray + oldSize, newData, newSize * sizeof(T), cudaMemcpyDeviceToDevice);
    return newArray;
}
#endif

void MultiPatch::addPatch(Patch& patch)
{
    //const TensorBsplineBasis& basis = patch.getBasis();
    
    // Extract basis properties
    int basisDim = patch.getBasisDim();
    if (m_patches.size() == 0) {
        m_basisDim = basisDim;
    } else if (m_basisDim != basisDim) {
        std::cerr << "Error: All patches must have the same dimension!" << std::endl;
        return;
    }
    int CPDim = patch.getCPDim();
    if (m_patches.size() == 0) {
        m_CPDim = CPDim;
    } else if (m_CPDim != CPDim) {
        std::cerr << "Error: All patches must have the same control point dimension!" << std::endl;
        return;
    }

#if 0
    // Append knots, orders, and numKnots to MultiPatch
    //thrust::device_vector<int> orders;
    std::vector<int> orders;
    patch.getOrders(orders);
    std::vector<int> numKnots;
    patch.getNumKnots(numKnots);
    std::vector<int> numGpAndEle;
    patch.getNumGpAndEle(numGpAndEle);
    std::vector<double> knots;
    patch.getKnots(knots);
    m_orders.insert(m_orders.end(), orders.begin(), orders.end());
    m_orders_ref.insert(m_orders_ref.end(), orders.begin(), orders.end());
    m_numKnots.insert(m_numKnots.end(), numKnots.begin(), numKnots.end());
    m_numKnots_ref.insert(m_numKnots_ref.end(), numKnots.begin(), numKnots.end());
    m_numGpAndEle.insert(m_numGpAndEle.end(), numGpAndEle.begin(), numGpAndEle.end());
    m_knots.insert(m_knots.end(), knots.begin(), knots.end());
    m_knots_ref.insert(m_knots_ref.end(), knots.begin(), knots.end());
    Eigen::VectorXd cps = patch.getControlPoints().reshaped<Eigen::RowMajor>();
    m_controlPoints.insert(m_controlPoints.end(), cps.begin(), cps.end());
    //m_controlPoints_ref.insert(m_controlPoints_ref.end(), patch.getControlPoints().begin(), patch.getControlPoints().end());
    m_numcontrolPoints.push_back(patch.getNumControlPoints());
    //m_numcontrolPoints_ref.push_back(patch.getNumControlPoints());
    // Update the number of patches
    //m_numPatches++;
#endif

    m_bases.push_back(patch.getBasis());
    m_patches.push_back(patch);
}

#if 0
void MultiPatch::update()
{
    m_orders.clear();
    m_numKnots.clear();
    m_numGpAndEle.clear();
    m_knots.clear();

    for (int i = 0; i < m_patches.size(); i++)
    {
        std::vector<int> orders;
        m_bases[i].getOrders(orders);
        std::vector<int> numKnots;
        m_bases[i].getNumKnots(numKnots);
        std::vector<int> numGpAndEle;
        m_bases[i].getNumGpAndEle(numGpAndEle);
        std::vector<double> knots;
        m_bases[i].getKnots(knots);
        m_orders.insert(m_orders.end(), orders.begin(), orders.end());
        m_numKnots.insert(m_numKnots.end(), numKnots.begin(), numKnots.end());
        m_numGpAndEle.insert(m_numGpAndEle.end(), numGpAndEle.begin(), numGpAndEle.end());
        m_knots.insert(m_knots.end(), knots.begin(), knots.end());
    }
}
#endif

int MultiPatch::getBasisDim() const
{
    return m_basisDim;
}

int MultiPatch::getCPDim() const
{
    return m_CPDim;
}

int MultiPatch::getNumPatches() const
{
    return m_patches.size();
}

Eigen::VectorXi MultiPatch::coefSlice(int patchIndex, int dir, int k) const
{
    return m_bases[patchIndex].coefSlice(dir, k);
}

void MultiPatch::getMapper(bool conforming, 
                           const BoundaryConditions &bc,
                           int unk,
                           DofMapper &dofMapper, 
                           bool finalize) const
{
    dofMapper= DofMapper(m_bases, bc, unk);

    if ( conforming )
    {
        for (std::vector<BoundaryInterface>::const_iterator it = m_topology.iBegin(); it != m_topology.iEnd(); ++it)
        {
            matchInterface(*it, dofMapper);
        }
    }

    if (finalize)
    {
        dofMapper.finalize();
    }
}

void MultiPatch::getMappers(bool conforming, 
                            const BoundaryConditions &bc, 
                            std::vector<DofMapper> &dofMappers, 
                            bool finalize) const
{
    dofMappers = std::vector<DofMapper>(m_CPDim);
    for (int d = 0; d < m_CPDim; d++)
    {
        getMapper(conforming, bc, d, dofMappers[d], finalize);
    }
}

void MultiPatch::matchInterface(const BoundaryInterface &bi, DofMapper &mapper) const
{
    Eigen::MatrixXi b1, b2;
    m_bases[bi.first().patchIndex()].matchWith(bi, m_bases[bi.second().patchIndex()],
                                         b1, b2);

    for (size_t i = 0; i!=mapper.componentsSize(); ++i)
        mapper.matchDofs(bi.first().patchIndex(), b1, bi.second().patchIndex(), b2, i );
}

#if 0
const thrust::device_vector<int> &MultiPatch::getNumKnots() const
{
    return m_numKnots;
}

const thrust::device_vector<int> &MultiPatch::getOrders() const
{
    return m_orders;
}

const thrust::device_vector<int> &MultiPatch::getNumGpAndEle() const
{
    return m_numGpAndEle;
}

const thrust::device_vector<int> &MultiPatch::getNumControlPoints() const
{
    return m_numcontrolPoints;
}

const thrust::device_vector<double> &MultiPatch::getKnots() const
{
    return m_knots;
}

const thrust::device_vector<double> &MultiPatch::getControlPoints() const
{
    return m_controlPoints;
}

const thrust::device_vector<int> &MultiPatch::getNumKnots_ref() const
{
    return m_numKnots_ref;
}

const thrust::device_vector<int> &MultiPatch::getOrders_ref() const
{
    return m_orders_ref;
}

const thrust::device_vector<double> &MultiPatch::getKnots_ref() const
{
    return m_knots_ref;
}
#endif

std::vector<int> MultiPatch::getBasisNumKnots(int patchIndex) const
{
    return m_bases[patchIndex].getNumKnots();
}

std::vector<double> MultiPatch::getBasisKnots(int patchIndex) const
{
    return m_bases[patchIndex].getKnots();
}

std::vector<int> MultiPatch::getBasisOrders(int patchIndex) const
{
    return m_bases[patchIndex].getOrders();
}

Eigen::MatrixXd MultiPatch::getControlPoints(int patchIndex) const
{
    return m_patches[patchIndex].getControlPoints();
}

int MultiPatch::getNumControlPoints(int patchIndex) const
{
    return m_patches[patchIndex].getNumControlPoints();
}

int MultiPatch::getTotalNumControlPoints() const
{
    int total = 0;
    for (const auto& patch : m_patches)
    {
        total += patch.getNumControlPoints();
    }
    return total;
}

std::vector<int> MultiPatch::getNumGpAndEle(int patchIndex) const
{
    return m_bases[patchIndex].getNumGpAndEle();
}

std::vector<int> MultiPatch::getGeoNumKnots(int patchIndex) const
{
    return m_patches[patchIndex].getNumKnots();
}

std::vector<double> MultiPatch::getGeoKnots(int patchIndex) const
{
    return m_patches[patchIndex].getKnots();
}

std::vector<int> MultiPatch::getGeoOrders(int patchIndex) const
{
    return m_patches[patchIndex].getOrders();
}

std::vector<int> MultiPatch::getBasisNumKnots() const
{
    std::vector<int> numKnots;
    numKnots.reserve(m_bases.size() * m_basisDim);
    for (int i = 0; i < m_bases.size(); i++)
    {
        std::vector<int> nk = m_bases[i].getNumKnots();
        numKnots.insert(numKnots.end(), nk.begin(), nk.end());
    }
    return numKnots;
}

std::vector<double> MultiPatch::getBasisKnots() const
{
    std::vector<double> knots;
    int numTotalKnots = 0;
    for (int i = 0; i< m_bases.size(); i++)
    {
        numTotalKnots += m_bases[i].getTotalNumKnots();
    }
    knots.reserve(numTotalKnots);
    for (int i = 0; i < m_bases.size(); i++)
    {
        std::vector<double> k = m_bases[i].getKnots();
        knots.insert(knots.end(), k.begin(), k.end());
    }
    return knots;
}

std::vector<int> MultiPatch::getBasisOrders() const
{
    std::vector<int> orders;
    orders.reserve(m_bases.size() * m_basisDim);
    for (int i = 0; i < m_bases.size(); i++)
    {
        std::vector<int> o = m_bases[i].getOrders();
        orders.insert(orders.end(), o.begin(), o.end());
    }
    return orders;
}

std::vector<double> MultiPatch::getControlPoints() const
{
    std::vector<double> cps;
    int numTotalControlPoints = 0;
    for (int i = 0; i < m_patches.size(); i++)
    {
        numTotalControlPoints += m_patches[i].getNumControlPoints();
    }
    cps.reserve(numTotalControlPoints * m_CPDim);
    for (int i = 0; i < m_patches.size(); i++)
    {
        Eigen::VectorXd cp = m_patches[i].getControlPoints().reshaped<Eigen::RowMajor>();
        cps.insert(cps.end(), cp.data(), cp.data() + cp.size());
    }
    return cps;
}

std::vector<int> MultiPatch::getNumControlPoints() const
{
    std::vector<int> numControlPoints;
    numControlPoints.reserve(m_patches.size());
    for (int i = 0; i < m_patches.size(); i++)
    {
        numControlPoints.push_back(m_patches[i].getNumControlPoints());
    }
    return numControlPoints;
}

std::vector<int> MultiPatch::getNumGpAndEle() const
{
    std::vector<int> numGpAndEle;
    numGpAndEle.reserve(m_bases.size() * m_basisDim * 2);
    for (int i = 0; i < m_bases.size(); i++)
    {
        std::vector<int> ng = m_bases[i].getNumGpAndEle();
        numGpAndEle.insert(numGpAndEle.end(), ng.begin(), ng.end());
    }
    return numGpAndEle;
}

std::vector<int> MultiPatch::getGeoNumKnots() const
{
    std::vector<int> numKnots;
    numKnots.reserve(m_patches.size() * m_basisDim);
    for (int i = 0; i < m_patches.size(); i++)
    {
        std::vector<int> nk = m_patches[i].getNumKnots();
        numKnots.insert(numKnots.end(), nk.begin(), nk.end());
    }
    return numKnots;
}

std::vector<double> MultiPatch::getGeoKnots() const
{
    std::vector<double> knots;
    int numTotalKnots = 0;
    for (int i = 0; i< m_patches.size(); i++)
    {
        numTotalKnots += m_patches[i].getTotalNumKnots();
    }
    knots.reserve(numTotalKnots);
    for (int i = 0; i < m_patches.size(); i++)
    {
        std::vector<double> k = m_patches[i].getKnots();
        knots.insert(knots.end(), k.begin(), k.end());
    }
    return knots;
}

std::vector<int> MultiPatch::getGeoOrders() const
{
    std::vector<int> orders;
    orders.reserve(m_patches.size() * m_basisDim);
    for (int i = 0; i < m_patches.size(); i++)
    {
        std::vector<int> o = m_patches[i].getOrders();
        orders.insert(orders.end(), o.begin(),o.end());
    }
    return orders;
}

int MultiPatch::getTotalNumGaussPoints() const
{
    int totalNumGaussPoints = 0;
    for (int i = 0; i < m_bases.size(); i++)
    {
        totalNumGaussPoints += m_bases[i].getTotalNumGaussPoints();
    }
    return totalNumGaussPoints;
}

const std::vector<double> &MultiPatch::getGeoKnots(int patchIndex, int direction) const
{
    return m_patches[patchIndex].getKnots(direction);
}

const std::vector<double> &MultiPatch::getBasisKnots(int patchIndex, int direction) const
{
    return m_bases[patchIndex].getKnots(direction);
}

#if 0
const thrust::device_vector<double> &MultiPatch::getControlPoints_ref() const
{
    return m_controlPoints_ref;
}

const thrust::device_vector<int> &MultiPatch::getNumControlPoints_ref() const
{
    return m_numcontrolPoints_ref;
}
#endif

#if 0
int *MultiPatch::getNumKnots_ptr()
{
    return thrust::raw_pointer_cast(m_numKnots.data());
}

int *MultiPatch::getOrders_ptr()
{
    return thrust::raw_pointer_cast(m_orders.data());
}

int *MultiPatch::getNumGpAndEle_ptr()
{
    return thrust::raw_pointer_cast(m_numGpAndEle.data());
}

double *MultiPatch::getKnots_ptr()
{
#if 0
    thrust::device_vector<double> knots;
    for (int i = 0; i < m_numPatches; i++)
    {
        for (int j = 0; j < m_dim; j++)
        {
            knots.insert(knots.end(), m_patches[i].getKnotVector(j).begin(), m_patches[i].getKnotVector(j).end());
        }
    }
#endif
    return thrust::raw_pointer_cast(m_knots.data());
}

double *MultiPatch::getControlPoints_ptr()
{
    return thrust::raw_pointer_cast(m_controlPoints.data());
}

int *MultiPatch::getNumControlPoints_ptr()
{
    return thrust::raw_pointer_cast(m_numcontrolPoints.data());
}

int *MultiPatch::getNumKnots_ref_ptr()
{
    return thrust::raw_pointer_cast(m_numKnots_ref.data());
}

int *MultiPatch::getOrders_ref_ptr()
{
    return thrust::raw_pointer_cast(m_orders_ref.data());
}

double *MultiPatch::getKnots_ref_ptr()
{
    return thrust::raw_pointer_cast(m_knots_ref.data());
}
#endif

#if 0
double *MultiPatch::getControlPoints_ref_ptr()
{
    return thrust::raw_pointer_cast(m_controlPoints_ref.data());
}

int *MultiPatch::getNumControlPoints_ref_ptr()
{
    return thrust::raw_pointer_cast(m_numcontrolPoints_ref.data());
}

thrust::device_vector<double>::iterator MultiPatch::knotUBegin(int patchIndex, int direction) const
{
    return knotBegin(patchIndex, direction) + getKnotOrder(patchIndex, direction);
}

thrust::device_vector<double>::iterator MultiPatch::knotEnd(int patchIndex, int direction) const
{
    return knotBegin(patchIndex, direction) + getNumKnots(patchIndex, direction) - 1;
}

thrust::device_vector<double>::iterator MultiPatch::knotUEnd(int patchIndex, int direction) const
{
    return knotEnd(patchIndex, direction) - getKnotOrder(patchIndex, direction);
}
#endif

#if 0
void MultiPatch::uniformRefineWithoutUpdate(int patchIndex, int direction, int numKnots)
{
    m_bases[patchIndex].uniformRefine(direction, numKnots);
}
#endif

void MultiPatch::uniformRefine(int patchIndex, int direction, int numKnots)
{
    m_bases[patchIndex].uniformRefine(direction, numKnots);
    //update();
}

void MultiPatch::uniformRefine(int direction, int numKnots)
{
    for (int i = 0; i < m_bases.size(); i++)
    {
        m_bases[i].uniformRefine(direction, numKnots);
    }
    //update();
}

void MultiPatch::uniformRefine(int numKnots)
{
    for (int i = 0; i < m_bases.size(); i++)
    {
        m_bases[i].uniformRefine(numKnots);
    }
    //update();
}

const TensorBsplineBasis &MultiPatch::basis(int patchIndex) const
{
    return m_bases[patchIndex];
}

const Patch &MultiPatch::patch(int patchIndex) const
{
    return m_patches[patchIndex];
}

bool MultiPatch::computeTopology(double tol, bool cornersOnly, bool tjunctions)
{
    m_topology.clearTopology();
    m_topology.setDim(m_basisDim);
    m_topology.setNBoxes(m_patches.size());
    const size_t np = m_patches.size();
    const int nCorP = 1 << m_basisDim;
    const int nCorS = 1 << (m_basisDim-1);

    Eigen::MatrixXd supp, coor;
    if (cornersOnly)
        coor.resize(m_basisDim,nCorP);
    else
        coor.resize(m_basisDim,nCorP + 2*m_basisDim);

    Eigen::Vector<bool, -1> boxPar(m_basisDim);

    std::vector<Eigen::MatrixXd> pCorners(np);

    std::vector<PatchSide> pSide;
    pSide.reserve(np * 2 * m_basisDim);
    for (size_t p=0; p<np; ++p)
    {
        supp = m_patches[p].parameterRange();
        //std::cout << "supp:\n" << supp << "\n\n";
        for (BoxCorner c = BoxCorner::getFirst(m_basisDim); c.index() < BoxCorner::getEnd(m_basisDim).index(); ++c)
        {
            boxPar = c.parameters(m_basisDim);
            for (int i=0; i<m_basisDim;++i)
                coor(i,c.index()-1) = boxPar(i) ? supp(i,1) : supp(i,0);
        }

        if (!cornersOnly)
        {
            int l = nCorP;
            for (BoxSide c=BoxSide::getFirst(m_basisDim); c.index()<BoxSide::getEnd(m_basisDim).index(); ++c)
            {
                const int dir = c.direction();
                const int s = static_cast<int>(c.parameter());

                for (int i=0; i<m_basisDim;++i)
                    coor(i,l) = ( dir==i ? supp(i,s) : (supp(i,1)+supp(i,0))/2.0 );
                l++;
            }
        }
        //std::cout << "coor:\n" << coor << "\n\n";

        m_patches[p].eval_into(coor,pCorners[p]);
        //std::cout << "pCorners[" << p << "]:\n" << pCorners[p] << "\n\n";
        for (BoxSide bs=BoxSide::getFirst(m_basisDim); bs.index()<BoxSide::getEnd(m_basisDim).index(); ++bs)
            pSide.push_back(PatchSide(p,bs));
    }

    Eigen::VectorXi dirMap(m_basisDim);
    Eigen::Vector<bool,-1> matched(nCorS), dirOr(m_basisDim);
    std::vector<BoxCorner> cId1, cId2;
    cId1.reserve(nCorS);
    cId2.reserve(nCorS);

    std::set<int> found;
    for (size_t sideind=0; sideind<pSide.size(); ++sideind)
    {
        const PatchSide & side = pSide[sideind];
        for (size_t other=sideind+1; other<pSide.size(); ++other)
        {
            //std::cout << "sideind: " << sideind << ", other: " << other << "\n\n";
            side        .getContainedCorners(m_basisDim,cId1);
            pSide[other].getContainedCorners(m_basisDim,cId2);
            matched.setConstant(false);
#if 0
            std::cout << "cId1:\n";
            for (const auto &c : cId1)
                std::cout << c.index() << " ";
            std::cout << "\n\n";
            std::cout << "cId2:\n";
            for (const auto &c : cId2)
                std::cout << c.index() << " ";
            std::cout << "\n\n";
#endif            
            if (!cornersOnly)
                if ( ( pCorners[side.patchIndex()        ].col(nCorP+side.index()-1        ) -
                       pCorners[pSide[other].patchIndex()].col(nCorP+pSide[other].index()-1)
                     ).norm() >= tol )
                    continue;
            
            if ( matchVerticesOnSide( pCorners[side.patchIndex()]        , cId1, 0,
                                      pCorners[pSide[other].patchIndex()], cId2,
                                      matched, dirMap, dirOr, tol ) )
            {
#if 0
                std::cout << "matched:\n" << matched << "\n\n";
                std::cout << "dirMap:\n" << dirMap << "\n\n";
                std::cout << "dirOr:\n" << dirOr << "\n\n";
#endif            
                dirMap(side.direction()) = pSide[other].direction();
                dirOr (side.direction()) = !( side.parameter() == pSide[other].parameter() );
                m_topology.addInterface( BoundaryInterface(side, pSide[other], dirMap, dirOr) );
                found.insert(sideind);
                found.insert(other);
            }
        }
    }

    int k = 0;
    found.insert(found.end(), pSide.size());
    for (const auto &s : found)
    {
        for (;k<s;++k)
            m_topology.addBoundary(pSide[k]);
        ++k;
    }

    return true;
}

bool MultiPatch::matchVerticesOnSide(const Eigen::MatrixXd &cc1, const std::vector<BoxCorner> &ci1, int start, 
                                     const Eigen::MatrixXd &cc2, const std::vector<BoxCorner> &ci2, 
                                     const Eigen::Vector<bool, -1> &matched, 
                                     Eigen::VectorXi &dirMap, 
                                     Eigen::Vector<bool, -1> &dirO, double tol, int reference)
{
    const bool computeOrientation = !(start&(start-1)) && (start != 0);
    const bool setReference       = start==0;

    const int dim = static_cast<int>(cc1.rows());

    int o_dir = 0, d_dir = 0;

    Eigen::Vector<bool, -1> refPar, newPar, newMatched;

    if (computeOrientation)
    {
        const Eigen::Vector<bool, -1> parStart = ci1[start].parameters(dim);
        const Eigen::Vector<bool, -1> parRef   = ci1[0].parameters(dim);
        for (; o_dir<dim && parStart(o_dir)==parRef(o_dir)  ;) ++o_dir;
    }

    if (!setReference)
        refPar = ci2[reference].parameters(dim);

    for (size_t j=0;j<ci2.size();++j)
    {
        if( !matched(j) && (cc1.col(ci1[start].index()-1)-cc2.col(ci2[j].index()-1)).norm() < tol )
        {
            int newRef =   (setReference) ? j : reference;
            if (computeOrientation)
            {
                int count=0;
                d_dir = 0;
                newPar =  ci2[j].parameters(dim);
                for (int i=0; i< newPar.rows();++i)
                {
                    if ( newPar(i)!=refPar(i) )
                    {
                        d_dir=i;
                        ++count;
                    }
                }
                if (count != 1)
                {
                    continue;
                }
                dirMap(o_dir) = d_dir;
                dirO  (o_dir) = (static_cast<int>(j) > reference);
            }
            if ( start + 1 == static_cast<int>( ci1.size() ) )
            {
                return true;
            }
            newMatched = matched;
            newMatched(j) = true;
            if ( matchVerticesOnSide(cc1, ci1, start+1, cc2, ci2, newMatched, dirMap, dirO, tol, newRef) )
                return true;
        }
    }

    return false;
}

#if 0
int MultiPatch::getNumKnots(int patchIndex, int direction) const

{
    return m_numKnots[patchIndex * m_dim + direction];
}

int MultiPatch::getKnotOrder(int patchIndex, int direction) const
{
    return m_orders[patchIndex * m_dim + direction];
}

thrust::device_vector<int>::iterator MultiPatch::patchNumKnotBegin(int patchIndex)
{
    return m_numKnots.begin() + patchIndex * m_dim;
}

thrust::device_vector<double>::iterator MultiPatch::knotBegin(int patchIndex, int direction) const
{
    int offset = 0;
    for (int i = 0; i < patchIndex * m_dim + direction; i++)
        offset += m_numKnots[i];
    return m_knots.begin() + offset;
}
#endif