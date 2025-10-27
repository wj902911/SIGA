#pragma once

#include <cuda_runtime.h>
#include <DeviceVector.h>
#include <Boundary.h>
#include <DeviceObjectArray.h>

class BoxSide_d
{
public:
    __host__ __device__
    BoxSide_d (): m_index(0) {}
    __host__ __device__
    BoxSide_d (int a): m_index(a) 
    { if(a < 0) printf("invalid side"); }
    __host__ __device__
    BoxSide_d (const BoxSide_d &s): m_index(s.m_index) {}
    __host__ __device__
    BoxSide_d (boundary::side a) : m_index(a) {}
    __device__
    operator boundary::side() const 
    { return static_cast<boundary::side>(m_index); }
    __host__ __device__
    int direction () const { return static_cast<int>((m_index-1) / 2);}
    __host__ __device__
    bool parameter () const { return (m_index-1)%2 != 0; }
    __host__ __device__
    int index () const {return m_index;}
    __host__ __device__
    static int index (int dir, bool par) {return static_cast<int>(par?2*dir+2:2*dir+1);}

private:
    int m_index;
};

struct PatchSide_d : public BoxSide_d
{
public:
    __host__ __device__
    PatchSide_d (): BoxSide_d(), m_patchIndex(0) {}

    __host__ __device__
    PatchSide_d (int p, BoxSide_d s) : BoxSide_d(s), m_patchIndex(p) {}

    __host__ __device__
    PatchSide_d (int p, boundary::side s): BoxSide_d(s), m_patchIndex(p) {}

    __host__ __device__
    PatchSide_d (const PatchSide &s)
        : BoxSide_d(s.side()), m_patchIndex(s.patchIndex()) {}
    __host__ __device__
    BoxSide_d& side() {return *this;}
    __host__ __device__
    const BoxSide_d& side() const {return *this;}

    __host__ __device__
    int patchIndex() const {return m_patchIndex;}

    __host__
    PatchSide_d clone() const
    {
        return PatchSide_d(*this);
    }

private:
    int m_patchIndex;
};

__device__
inline int sideOrientation(int s)
{
    return ( ( s + (s+1) / 2 ) % 2 ? 1 : -1 );
}

struct BoundaryInterface_d
{
private:
    PatchSide_d m_ps1;
    PatchSide_d m_ps2;
    DeviceVector<int> m_directionMap;
    DeviceVector<bool> m_directionOrientation;

public:
    __host__
    BoundaryInterface_d() = default;

    __host__
    BoundaryInterface_d(const BoundaryInterface &bi)
        : m_ps1(bi.first()), m_ps2(bi.second()),
          m_directionMap(bi.dirMap()), m_directionOrientation(bi.dirOrientation()) {}

    //copy constructor
    __host__ __device__
    BoundaryInterface_d(const BoundaryInterface_d &bi)
        : m_ps1(bi.m_ps1), m_ps2(bi.m_ps2),
          m_directionMap(bi.m_directionMap), 
          m_directionOrientation(bi.m_directionOrientation) {}

    //copy assignment operator
    __host__ __device__
    BoundaryInterface_d& operator=(const BoundaryInterface_d &bi)
    {
        if (this != &bi) // self-assignment check
        {
            m_ps1 = bi.m_ps1;
            m_ps2 = bi.m_ps2;
            m_directionMap = bi.m_directionMap;
            m_directionOrientation = bi.m_directionOrientation;
        }
        return *this;
    }

    //move constructor
    __host__ __device__
    BoundaryInterface_d(BoundaryInterface_d &&bi) noexcept
        : m_ps1(std::move(bi.m_ps1)), m_ps2(std::move(bi.m_ps2)),
          m_directionMap(std::move(bi.m_directionMap)), 
          m_directionOrientation(std::move(bi.m_directionOrientation)) {}
    
    //move assignment operator
    __host__ __device__
    BoundaryInterface_d& operator=(BoundaryInterface_d &&bi) noexcept
    {
        if (this != &bi) // self-assignment check
        {
            m_ps1 = std::move(bi.m_ps1);
            m_ps2 = std::move(bi.m_ps2);
            m_directionMap = std::move(bi.m_directionMap);
            m_directionOrientation = std::move(bi.m_directionOrientation);
        }
        return *this;
    }

    __host__
    BoundaryInterface_d clone() const
    {
        return BoundaryInterface_d(*this);
    }

    __host__ __device__
    PatchSide_d& first ()              {return m_ps1;}
    __host__ __device__
    const PatchSide_d& first () const  {return m_ps1;}
    __host__ __device__
    PatchSide_d& second ()              {return m_ps2;}
    __host__ __device__
    const PatchSide_d& second () const  {return m_ps2;}

    __host__ __device__
    const DeviceVector<int> & dirMap() const { return m_directionMap; }
    __host__ __device__
    const DeviceVector<bool> & dirOrientation()  const { return m_directionOrientation; }
};

