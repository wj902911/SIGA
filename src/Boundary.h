#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Core>

struct boundary
{
    enum side { west  = 1,  east = 2, south = 3, north= 4, front=5, back=6,
                stime = 7, etime = 8,
                left  = 1, right = 2, down  = 3, up   = 4 , none = 0 };

    enum corner { southwestfront = 1, southeastfront = 2, northwestfront = 3, northeastfront = 4,
                  southwestback  = 5, southeastback  = 6, northwestback  = 7, northeastback  = 8,
                  southwest      = 1, southeast      = 2, northwest      = 3, northeast      = 4
    };
};

struct BoxCorner;
struct PatchCorner;

class BoxSide
{
public:
    BoxSide (): m_index(0) {}

    BoxSide (int dir, bool par) : m_index(index(dir,par))
    { if(dir < 0) std::cerr << "invalid side"; }

    BoxSide (int a): m_index(a) 
    { if(a < 0) std::cerr << "invalid side"; }
    
    BoxSide (boundary::side a): m_index(a) 
    { if(a < 0) std::cerr << "invalid side"; }

    operator boundary::side() const {return static_cast<boundary::side>(m_index);}

    int direction () const {return static_cast<int>((m_index-1) / 2);}

    bool parameter () const {return (m_index-1)%2 != 0;}

    int index () const {return m_index;}

    static int index (int dir, bool par) {return static_cast<int>(par?2*dir+2:2*dir+1);}

    static BoxSide getFirst (int)     { return BoxSide(1); }
    static BoxSide getLast  (int dim) {return BoxSide(static_cast<int>(2*dim));}
    static BoxSide getEnd   (int dim) {return BoxSide(static_cast<int>(2*dim+1));}

    BoxSide& operator++ () { ++m_index; return *this;}
    BoxSide& operator-- () { --m_index; return *this;}

    void getContainedCorners (int dim, std::vector<BoxCorner> &corners) const;

private:
    int m_index;

};

struct PatchSide : public BoxSide
{
public:
    PatchSide (): BoxSide(), m_patchIndex(0) {}

    PatchSide (int p, BoxSide s) : BoxSide(s), m_patchIndex(p) {}

    PatchSide (int p, boundary::side s): BoxSide(s), m_patchIndex(p) {}

    BoxSide& side() {return *this;}
    const BoxSide& side() const {return *this;}

    int patchIndex() const {return m_patchIndex;}

private:
    int m_patchIndex;
};

struct BoxCorner
{
public:
    BoxCorner (int a = 0): m_index(a)
    { if(a < 0) std::cerr << "invalid side"; }

    BoxCorner (boundary::corner a): m_index(a) 
    { if(a < 0) std::cerr << "invalid side"; }

    BoxCorner (std::vector<bool> v) : m_index(1)
    {
        for (int i = 0; i < v.size(); i++)
        {
            m_index += v[i] ? 1 << i : 0;
        }
    }

    void parameters_into (int dim, Eigen::Vector<bool, -1> &param) const
    {
        param.resize(dim);
        for (int i=0; i<dim; ++i)
            param[i]=((m_index-1)>>i)&1;
    }

    static BoxCorner getFirst (int)     { return BoxCorner(1); }
    static BoxCorner getLast  (int dim) { return BoxCorner(static_cast<int>(1<<dim)); }
    static BoxCorner getEnd   (int dim) { return BoxCorner(static_cast<int>(1<<dim)+1); }

    BoxCorner& operator++ () { ++m_index; return *this;}
    BoxCorner& operator-- () { --m_index; return *this;}

    int index () const {return m_index;}

    Eigen::Vector<bool, -1> parameters(int dim) const
    {
        Eigen::Vector<bool, -1> r;
        parameters_into(dim, r); 
        return r;
    }
private:
    int m_index;
};

struct PatchCorner : public BoxCorner
{
public:
    PatchCorner (int p, int c): BoxCorner(c), m_patchIndex(p) {}

    PatchCorner (int p, BoxCorner c) : BoxCorner(c), m_patchIndex(p) {}

    PatchCorner (int p, boundary::corner c): BoxCorner(c), m_patchIndex(p) {}

    BoxCorner& corner() {return *this;}
    const BoxCorner& corner() const {return *this;}

    int patchIndex() const {return m_patchIndex;}

private:
    int m_patchIndex;
};

struct BoundaryInterface
{
public:
    BoundaryInterface() { }

    BoundaryInterface(PatchSide const & _ps1,
                      PatchSide const & _ps2,
                      bool o1)
        : m_ps1(_ps1), m_ps2(_ps2) 
    {
        m_directionMap.resize(2);
        m_directionOrientation.resize(2);
        m_directionMap[m_ps1.direction()]=m_ps2.direction();
        m_directionMap[1-m_ps1.direction()]=1-m_ps2.direction();
        m_directionOrientation[m_ps1.direction()]= (m_ps1.parameter()==m_ps2.parameter());
        m_directionOrientation[1-m_ps1.direction()]=o1;
    }

    BoundaryInterface(PatchSide const & _ps1,
                      PatchSide const & _ps2,
                      int dim)
        : m_ps1(_ps1), m_ps2(_ps2)
    {
        m_directionMap.resize(dim);
        m_directionOrientation.resize(dim);

        for (int i = 1 ; i < dim; ++i)
        {
        const int o = (m_ps1.direction()+i)%dim;
        const int d = (m_ps2.direction()+i)%dim;

        m_directionMap[o]=d;
        m_directionOrientation[o]=true;
        }
        m_directionMap[m_ps1.direction()]=m_ps2.direction();
        m_directionOrientation[m_ps1.direction()]= (m_ps1.parameter()==m_ps2.parameter());
    }

    BoundaryInterface(PatchSide const & _ps1,
                      PatchSide const & _ps2,
                      Eigen::VectorXi const & map_info,
                      Eigen::Vector<bool, -1> const & orient_flags)
        : m_ps1(_ps1), m_ps2(_ps2),
          m_directionMap(map_info),
          m_directionOrientation(orient_flags)
    {
        m_directionMap(m_ps1.direction())=m_ps2.direction();
        m_directionOrientation(m_ps1.direction())=(m_ps1.parameter()==m_ps2.parameter());
    }

    PatchSide& first ()              {return m_ps1;}
    const PatchSide& first () const  {return m_ps1;}

    PatchSide& second ()              {return m_ps2;}
    const PatchSide& second () const  {return m_ps2;}

    const Eigen::VectorXi & dirMap() const
    { return m_directionMap; }

    const Eigen::Vector<bool, -1> & dirOrientation()  const
    { return m_directionOrientation; }

private:
    PatchSide m_ps1;
    PatchSide m_ps2;
    Eigen::VectorXi m_directionMap;
    Eigen::Vector<bool, -1> m_directionOrientation;
};
