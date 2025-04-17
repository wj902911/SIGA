#include "DofMapper.h"
#include <algorithm>

void DofMapper::init(const std::vector<TensorBsplineBasis> &bases, int nComp)
{
    if (nComp <=0)
    {
        std::cerr << "Number of components must be greater than 0" << std::endl;
        return;
    }

    m_shift = m_bshift = 0;
    m_curElimId   = -1;
    m_numCpldDofs.assign(nComp+1, 1); 
    m_numCpldDofs.front()=0;
    m_numElimDofs.assign(nComp+1,0);
    m_offset.clear();

    const size_t nPatches = bases.size();

    // Initialize offsets and dof holder
    m_offset.reserve( nPatches );
    m_offset.push_back(0);
    for (size_t k = 1; k < nPatches; ++k)
        m_offset.push_back( m_offset.back() + bases[k-1].size() );

    m_numFreeDofs.assign(1+nComp,m_offset.back() + bases.back().size()); 
    m_numFreeDofs.front()=0;

#if !singleDofs_DofMapper
    m_dofs.resize(nComp, std::vector<int>(m_numFreeDofs.back(), 0));
#else
    m_dofs.resize(m_numFreeDofs.back());
#endif
    //m_dofs.resize(nComp, thrust::device_vector<int>(m_numFreeDofs.back(), 0));
}

void DofMapper::init(const std::vector<TensorBsplineBasis> &basis, 
                     const BoundaryConditions &bc, int unk)
{
    init(basis, 1);
    for (std::deque<boundary_condition>::const_iterator
         it = bc.dirichletBegin() ; it != bc.dirichletEnd(); ++it)
    {
        if (unk == -1 || it->unknown() == unk)
        {
            int patchIndex = it->patchIndex();
            if (patchIndex >= numPatches())
            {
                std::cerr << "Problem: a boundary condition is set on a patch id which does not exist." << std::endl;
                return;
            }

            Eigen::VectorXi bnd = basis[patchIndex].boundary(it->side());
            markBoundary(patchIndex, bnd);
        }
    }

    for (std::deque<corner_condition>::const_iterator
         it = bc.dirichletCornerBegin() ; it != bc.dirichletCornerEnd(); ++it)
    {
        if (unk == -1 || it->unknown() == unk)
        {
            int patchIndex = it->patchIndex();
            if (patchIndex < numPatches())
            {
                std::cerr << "Problem: a boundary condition is set on a patch id which does not exist." << std::endl;
                return;
            }

            eliminateDof(basis[patchIndex].corner(it->corner()), patchIndex);
        }
    }
}

void DofMapper::markBoundary(int k, 
                             const Eigen::VectorXi &boundaryDofs, 
                             int comp)
{
    for (int i = 0; i < boundaryDofs.size(); ++i)
        eliminateDof( boundaryDofs(i), k, comp );
}

void DofMapper::eliminateDof(int i, int k, int comp)
{
    if(k > numPatches())
    {
        std::cerr << "Invalid patch index" << std::endl;
        return;
    }

    if(comp >=componentsSize())
    {
        std::cerr << "Invalid component index" << std::endl;
        return;
    }

    if (-1==comp)
    {
        for (int c = 0; static_cast<size_t>(c) != componentsSize(); ++c)
            eliminateDof(i,k,c);
        return;
    }
    const int old = MAPPER_PATCH_DOF(i,k,comp);
    if (old == 0)
    {
        --m_numFreeDofs[comp+1];
        MAPPER_PATCH_DOF(i,k,comp) = m_curElimId--;
    }
    else if (old > 0)
    {
        --m_numFreeDofs[comp+1];
        replaceDofGlobally( old, m_curElimId--, comp);
    }
}

void DofMapper::finalize()
{
    if(m_curElimId >= 0)
    {
        std::cerr << "Error: DofMapper::finalize() called more than once." << std::endl;
        return;
    }

    for (size_t c = 0; c!=m_dofs.size(); ++c)
    {
        finalizeComp(c);
        m_curElimId -= m_numElimDofs[c+1];
        m_numFreeDofs[c+1] += m_numFreeDofs[c];
        m_numElimDofs[c+1] += m_numElimDofs[c];
        m_numCpldDofs[c+1] += m_numCpldDofs[c];
    }
#if !singleDofs_DofMapper
    if ( 1!=m_dofs.size() )
        for (size_t c = 0; c!=m_dofs.size(); ++c)
        {
            std::vector<int> & dofs = m_dofs[c];
            //thrust::device_vector<int> & dofs = m_dofs[c];
            for(std::vector<int>::iterator j = dofs.begin(); j!= dofs.end(); ++j)
            //for(thrust::device_vector<int>::iterator j = dofs.begin(); j!= dofs.end(); ++j)
                *j = (*j<m_numFreeDofs[c+1]+m_numElimDofs[c] ?
                      *j - m_numElimDofs[c]                  :
                      *j - m_numFreeDofs[c+1] + m_numFreeDofs.back());
        }
#endif
    
    m_curElimId = m_numFreeDofs.back();
}

void DofMapper::finalizeComp(const int comp)
{
#if !singleDofs_DofMapper
    std::vector<int> & dofs = m_dofs[comp];
#else
    std::vector<int> & dofs = m_dofs;
#endif
    //thrust::device_vector<int> & dofs = m_dofs[comp];
    std::vector<int> couplingDofs(m_numCpldDofs[comp+1] -1, -1);
    //thrust::device_vector<int> couplingDofs(m_numCpldDofs[comp+1] -1, -1);
    std::map<int,int> elimDofs;

    int curFreeDof = m_numFreeDofs[comp]+m_numElimDofs[comp];
    int curElimDof = m_numFreeDofs[comp+1] + curFreeDof;
    int curCplDof = std::count(dofs.begin(), dofs.end(), 0);
    m_numCpldDofs[comp+1] = m_numFreeDofs[comp+1] - curCplDof;
    curCplDof += curFreeDof;

    for (size_t k = 0; k < dofs.size(); ++k)
    {
        const int dofType = dofs[k];
        if (dofType == 0)
            dofs[k] = curFreeDof++;
        else if (dofType < 0)
        {
            const int id = -dofType-1;
            if (elimDofs.find(id)==elimDofs.end())
                elimDofs[id] = curElimDof++;
            dofs[k] = elimDofs[id];
        }
        else
        {
            const int id = dofType - 1;
            if (couplingDofs[id] < 0)
                couplingDofs[id] = curCplDof++;
            dofs[k] = couplingDofs[id];
        }
    }

    m_numElimDofs[comp+1] = curElimDof - curCplDof;

    curCplDof -= m_numFreeDofs[comp]+m_numElimDofs[comp];
    if (curCplDof != m_numFreeDofs[1+comp])
    {
        std::cerr << "Error: DofMapper::finalizeComp() - computed number of coupling "
                 "dofs does not match allocated number, "<<curCplDof<<"!="<<m_numFreeDofs[comp+1] << std::endl;
        return;
    }

    curFreeDof -= m_numFreeDofs[comp]+m_numElimDofs[comp];
    if (curFreeDof + m_numCpldDofs[comp+1] != m_numFreeDofs[comp+1])
    {
        std::cerr << "Error: DofMapper::finalizeComp() - computed number of free dofs "
                 "does not match allocated number." << std::endl;
        return;
    }
}

void DofMapper::matchDof(int u, int i, int v, int j, int comp)
{
    if (-1==comp)
    {
        for (int c = 0; static_cast<int>(c) != componentsSize(); ++c)
            matchDof(u,i,v,j,c);
        return;
    }

    assert(u < numPatches());
    assert(v < numPatches());

    int d1 = MAPPER_PATCH_DOF(i,u,comp);
    int d2 = MAPPER_PATCH_DOF(j,v,comp);

    if (d1 > d2)
    {
        std::swap(d1, d2);
        std::swap(u, v);
        std::swap(i, j);
    }

    if (d1 < 0)         // first dof is eliminated
    {
        if (d2 < 0)
            mergeDofsGlobally(d1, d2, comp);  // both are eliminated, merge their indices
        else if (d2 == 0)
            MAPPER_PATCH_DOF(j,v, comp) = d1;   // second is free, eliminate it along with first
        else /* d2 > 0*/
            replaceDofGlobally(d2, d1, comp); // second is coupling, eliminate all instances of it
    }
    else if (d1 == 0)   // first dof is a free dof
    {
        if (d2 == 0)
        {
            MAPPER_PATCH_DOF(i,u,comp) = MAPPER_PATCH_DOF(j,v,comp) = m_numCpldDofs[1+comp]++;  // both are free, assign them a new coupling id
            if (u==v && i==j) return;
        }
        else if (d2 > 0)
            MAPPER_PATCH_DOF(i,u,comp) = d2;   // second is coupling, add first to the same coupling group
        else
            std::cerr << "Something went terribly wrong" << std::endl;
    }
    else /* d1 > 0 */   // first dof is a coupling dof
    {
        assert(d2 > 0);
        mergeDofsGlobally( d1, d2, comp);      // both are coupling dofs, merge them
    }

    // if we merged two different non-eliminated dofs, we lost one free dof
    if ( (d1 != d2 && (d1 >= 0 || d2 >= 0) ) || (d1 == 0 && d2 == 0) )
        --m_numFreeDofs[1+comp];
}

void DofMapper::matchDofs(int u, const Eigen::MatrixXi &b1, int v, const Eigen::MatrixXi &b2, int comp)
{
    const int sz = b1.size();
    assert(sz == b2.size());
    for ( int k=0; k<sz; ++k)
        matchDof(u, b1(k,0), v, b2(k,0), comp);
}

void DofMapper::replaceDofGlobally(int oldIdx, int newIdx)
{
    for(size_t i = 0; i!= m_dofs.size(); ++i)
        std::replace(m_dofs[i].begin(), m_dofs[i].end(), oldIdx, newIdx );
}

void DofMapper::replaceDofGlobally(int oldIdx, int newIdx, int comp)
{
    if (comp > -1)
    {
        std::cerr << "Component is invalid" << std::endl;
        return; 
    }

    std::vector<int> & dofs = m_dofs[comp];
    //thrust::device_vector<int> & dofs = m_dofs[comp];
    std::replace(dofs.begin(), dofs.end(), oldIdx, newIdx );
}

void DofMapper::mergeDofsGlobally(int dof1, int dof2)
{
    if (dof1 != dof2)
    {
        if (dof1 < dof2)
            std::swap(dof1, dof2);

        replaceDofGlobally(dof1, dof2);
    }
}

void DofMapper::mergeDofsGlobally(int dof1, int dof2, int comp)
{
    if (dof1 != dof2)
    {
        if (dof1 < dof2)
            std::swap(dof1, dof2);
        replaceDofGlobally(dof1, dof2, comp);
    }
}
