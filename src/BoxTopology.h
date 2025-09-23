#pragma once

#include "Boundary.h"
#include <cassert> 

class BoxTopology
{
public:
    BoxTopology(int d = -1, int n = 0) : m_dim(d), m_nboxes(n) { }

    BoxTopology( int d, int boxes,
                 const std::vector<PatchSide> & boundary,
                 const std::vector<BoundaryInterface> & interfaces )
        : m_dim(d), m_nboxes(boxes), m_boundary(boundary), m_interfaces(interfaces) { }

    int dim() const { return m_dim; }
    int nboxes() const { return m_nboxes; }

    size_t nInterfaces() const { return m_interfaces.size(); }
    size_t nBoundary() const   { return m_boundary.size(); }

    std::vector<BoundaryInterface>::const_iterator iBegin() const 
    { return m_interfaces.begin(); }

    std::vector<BoundaryInterface>::const_iterator iEnd() const 
    { return m_interfaces.end(); }

    std::vector<BoundaryInterface>::iterator iBegin()
    { return m_interfaces.begin(); }

    std::vector<BoundaryInterface>::iterator iEnd()
    { return m_interfaces.end(); }

    const BoundaryInterface& getInterface(int i) const
    {
        assert(i >= 0 && i < nInterfaces());
        return m_interfaces[i];
    }

    std::vector<PatchSide>::const_iterator bBegin() const 
    { return m_boundary.begin(); }

    std::vector<PatchSide>::const_iterator bEnd() const 
    { return m_boundary.end(); }

    std::vector<PatchSide>::iterator bBegin()
    { return m_boundary.begin(); }

    std::vector<PatchSide>::iterator bEnd()
    { return m_boundary.end(); }

    const PatchSide& boundary(int i) const
    {
        assert(i >= 0 && i < nBoundary());
        return m_boundary[i];
    }

    void clearTopology()
    {
        m_boundary  .clear();
        m_interfaces.clear();
    }

    void clearAll()
    {
        clearTopology();
        m_dim  = -1;
        m_nboxes =  0;
    }

    void swap(BoxTopology& other)
    {
        std::swap( m_dim, other.m_dim );
        std::swap( m_nboxes, other.m_nboxes );
        m_boundary.swap( other.m_boundary );
        m_interfaces.swap( other.m_interfaces );
    }

    void addInterface( const BoundaryInterface& bi )
    {
        m_interfaces.push_back( bi );
    }

    void addInterface(int p1, BoxSide s1,
                      int p2, BoxSide s2)
    {
        addInterface(BoundaryInterface(PatchSide(p1, s1), PatchSide(p2, s2), m_dim));
    }

    void addBox(int i = 1)
    {
        m_nboxes += i;
    }

    void addBoundary(const PatchSide& ps)
    {
        m_boundary.push_back( ps );
    }

    void addBoundary(int p, BoxSide s)
    {
        addBoundary(PatchSide(p, s));
    }

    void setDim(int d) { m_dim = d; }
    void setNBoxes(int n) { m_nboxes = n; }

private:
    int m_dim;
    int m_nboxes;
    std::vector<PatchSide> m_boundary;
    std::vector<BoundaryInterface> m_interfaces;
};