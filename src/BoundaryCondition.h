#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <deque>
#include <map>
#include "Boundary.h"

struct condition_type
{
    enum type
    {
        dirichlet = 0,
        neumann = 1,
        robin = 2,
        double_stress = 3
    };
};

struct boundary_condition
{
public:
    boundary_condition(int p, BoxSide s, const std::vector<double>& v,
                       const std::string& lable, int unknown, int unkcomp)
        : m_ps(p, s), 
          m_values(v), 
          m_label(lable), 
          m_unknown(unknown), 
          m_unkcomp(unkcomp)
    {
    }

    boundary_condition(int p, BoxSide s, const std::vector<double>& v,
                       condition_type::type t, int unknown)
        : m_ps(p, s), 
          m_values(v), 
          m_unknown(unknown), 
          m_unkcomp(-1)
    {
        switch (t)
        {
        case condition_type::dirichlet:
            m_label = "Dirichlet";
            break;
        case condition_type::neumann:
            m_label = "Neumann";
            break;
        case condition_type::robin:
            m_label = "Robin";
            break;
        case condition_type::double_stress:
            m_label = "Double stress";
            break;
        }
    }

    boundary_condition(int p, BoxSide s, const std::vector<double>& v,
                       condition_type::type t, int unknown, int unkcomp)
        : m_ps(p, s), 
          m_values(v), 
          m_unknown(unknown), 
          m_unkcomp(unkcomp)
    {
        switch (t)
        {
        case condition_type::dirichlet:
            m_label = "Dirichlet";
            break;
        case condition_type::neumann:
            m_label = "Neumann";
            break;
        case condition_type::robin:
            m_label = "Robin";
            break;
        case condition_type::double_stress:
            m_label = "Double stress";
            break;
        }
    }

    const std::string& label() const { return m_label; }

    const PatchSide& patchSide() const { return m_ps; }

    const BoxSide& side() const { return m_ps.side(); }

    int patchIndex() const { return m_ps.patchIndex(); }

    int unknown()  const { return m_unknown; }

    int unkcomp()  const { return m_unkcomp; }

    double value(int i) const { return m_values[i]; }

    std::vector<double> values() const { return m_values; }

private:
    PatchSide m_ps;
    std::vector<double> m_values;
    std::string m_label;
    int m_unknown;
    int m_unkcomp;
};

struct corner_condition
{
public:
    corner_condition(int p, BoxCorner c, const std::vector<double>& v,
                 const std::string& lable, int unknown, int unkcomp)
        : m_pc(p, c), 
          m_values(v), 
          m_label(lable), 
          m_unknown(unknown), 
          m_unkcomp(unkcomp)
    {
    }

    corner_condition(int p, BoxCorner c, const std::vector<double>& v,
                 condition_type::type t, int unknown)
        : m_pc(p, c), 
          m_values(v), 
          m_unknown(unknown), 
          m_unkcomp(-1)
    {
        switch (t)
        {
        case condition_type::dirichlet:
            m_label = "Dirichlet";
            break;
        case condition_type::neumann:
            m_label = "Neumann";
            break;
        case condition_type::robin:
            m_label = "Robin";
            break;
        case condition_type::double_stress:
            m_label = "Double stress";
            break;
        }
    }

    corner_condition(int p, BoxCorner c, const std::vector<double>& v,
                 condition_type::type t, int unknown, int unkcomp)
        : m_pc(p, c), 
          m_values(v), 
          m_unknown(unknown), 
          m_unkcomp(unkcomp)
    {
        switch (t)
        {
        case condition_type::dirichlet:
            m_label = "Dirichlet";
            break;
        case condition_type::neumann:
            m_label = "Neumann";
            break;
        case condition_type::robin:
            m_label = "Robin";
            break;
        case condition_type::double_stress:
            m_label = "Double stress";
            break;
        }
    }

    const std::string& label() const { return m_label; }

    const PatchCorner& patchCorner() const { return m_pc; }

    const BoxCorner& corner() const { return m_pc.corner(); }

    int patchIndex() const { return m_pc.patchIndex(); }

    int unknown()  const { return m_unknown; }

    double value(int i) const { return m_values[i]; }
    
private:
    PatchCorner m_pc;
    std::vector<double> m_values;
    std::string m_label;
    int m_unknown;
    int m_unkcomp;
};

class BoundaryConditions
{
public:
    typedef typename std::deque<boundary_condition> bcContainer;
    typedef typename bcContainer::iterator iterator;
    typedef typename bcContainer::const_iterator const_iterator;

    typedef std::map<std::string, bcContainer> bcData;
    typedef typename bcData::iterator bciterator;
    typedef typename bcData::const_iterator const_bciterator;

    typedef typename std::deque<corner_condition> cornerContainer;
    typedef typename cornerContainer::iterator corner_iterator;
    typedef typename cornerContainer::const_iterator const_corner_iterator;

    typedef std::map<std::string, cornerContainer> cornerData;

    BoundaryConditions() = default;

    void clear()
    {
        m_bc.clear();
        m_cc.clear();
    }

    const bcContainer & container(const std::string & label) const 
    {return m_bc.at(label); }

    void addCondition(int p, BoxSide s, condition_type::type t, 
                      const std::vector<double>& v, 
                      int unknown = 0, int unkcomp = -1)
    {
        boundary_condition bc(p, s, v, t, unknown, unkcomp);
        m_bc[bc.label()].push_back(bc);
    }

    void addCondition(int p, BoxCorner c, condition_type::type t, 
                      const std::vector<double>& v, 
                      int unknown = 0, int unkcomp = -1)
    {
        corner_condition cc(p, c, v, t, unknown, unkcomp);
        m_cc[cc.label()].push_back(cc);
    }

    void addCondition(int p, boundary::side s, condition_type::type t, 
                      const std::vector<double>& v, 
                      int unknown = 0, int unkcomp = -1)
    {
        addCondition(p, BoxSide(s), t, v, unknown, unkcomp);
    }

    void addCondition(int p, boundary::corner c, condition_type::type t, 
                      const std::vector<double>& v, 
                      int unknown = 0, int unkcomp = -1)
    {
        addCondition(p, BoxCorner(c), t, v, unknown, unkcomp);
    }

    const_iterator dirichletBegin() const
    { return m_bc["Dirichlet"].begin(); }

    const_iterator dirichletEnd() const
    { return m_bc["Dirichlet"].end(); }

    const_iterator neumannBegin() const
    { return m_bc["Neumann"].begin(); }

    const_iterator neumannEnd() const
    { return m_bc["Neumann"].end(); }

    const_iterator robinBegin() const
    { return m_bc["Robin"].begin(); }

    const_iterator robinEnd() const
    { return m_bc["Robin"].end(); }

    const_iterator doubleStressBegin() const
    { return m_bc["Double stress"].begin(); }

    const_iterator doubleStressEnd() const
    { return m_bc["Double stress"].end(); }

    const_corner_iterator dirichletCornerBegin() const
    { return m_cc["Dirichlet"].begin(); }

    const_corner_iterator dirichletCornerEnd() const
    { return m_cc["Dirichlet"].end(); }

    const_corner_iterator neumannCornerBegin() const
    { return m_cc["Neumann"].begin(); }

    const_corner_iterator neumannCornerEnd() const
    { return m_cc["Neumann"].end(); }

    const_corner_iterator robinCornerBegin() const
    { return m_cc["Robin"].begin(); }

    const_corner_iterator robinCornerEnd() const
    { return m_cc["Robin"].end(); }

    const_corner_iterator doubleStressCornerBegin() const
    { return m_cc["Double stress"].begin(); }

    const_corner_iterator doubleStressCornerEnd() const
    { return m_cc["Double stress"].end(); }

    const bcContainer & neumannSides()   const { return m_bc["Neumann"]; }

private:
    mutable bcData m_bc;
    mutable cornerData m_cc;
};