#include "Boundary.h"
#include <cassert> 

void BoxSide::getContainedCorners(int dim, std::vector<BoxCorner> &corners) const
{
    assert(dim > 0);
    corners.clear();
    corners.reserve( 1ULL<<(dim-1) );
    const int dir = direction();
    const bool par = parameter();
    for (BoxCorner c=BoxCorner::getFirst(dim); c.index()<BoxCorner::getEnd(dim).index();++c)
    {
        if (c.parameters(dim)[dir] == par)
            corners.push_back(c);
    }
}