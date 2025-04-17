#include <BoxTopology_d.h>

BoxTopology_d::BoxTopology_d(const BoxTopology &topology)
    : m_dim(topology.dim()), m_nboxes(topology.nboxes()),
      m_boundary(topology.nBoundary()), m_interfaces(topology.nInterfaces())
{
    for (int i = 0; i < m_boundary.size(); ++i)
        m_boundary.at(i) = topology.boundary(i);
    for (int i = 0; i < m_interfaces.size(); ++i)
    {
        //BoundaryInterface_d temp = topology.interface(i);
        //m_interfaces.at(i) = temp;
        m_interfaces.at(i) = topology.interface(i);
    }
}
