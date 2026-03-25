#pragma once

#include <cuda_runtime.h>
#include <iterator>
#include <Utility_d.h>


class KnotGPUIterator;
class KnotVectorDeviceView;

class UKnotGPUIterator
{
private:
    const int* m_mlt;
    const double* m_raw;
    int m_upos;
    int m_sh;
    int m_dbg;

public:
    friend class KnotVectorDeviceView;
    friend class KnotGPUIterator;

    typedef std::random_access_iterator_tag iterator_category;
    typedef double value_type;
    typedef std::ptrdiff_t difference_type;
    typedef const double& reference;
    typedef const double* pointer;

    __device__
    UKnotGPUIterator()
    : m_mlt(nullptr), m_raw(nullptr), m_upos(0), m_sh(0), m_dbg(0)
    { }

    __device__ __forceinline__
    UKnotGPUIterator(const KnotVectorDeviceView & KV, const int upos = 0, const int s = 0);

    __device__ __forceinline__
    static UKnotGPUIterator End(const KnotVectorDeviceView & KV);

    __device__
    int firstAppearance() const
    { return 0 == m_upos ? 0 : m_mlt[m_upos-1]; }

    __device__
    int lastAppearance() const
    { return m_mlt[m_upos] - 1; }

    __device__
    reference operator*  () const
    { return m_raw[m_mlt[m_upos]-1]; }

    __device__
    pointer operator-> () const {return m_raw+m_mlt[m_upos]-1 ;}

    __device__
    UKnotGPUIterator& operator++()
    {
        ++m_upos;
        return *this;
    }

    __device__
    UKnotGPUIterator& operator--()
    {
        --m_upos;
        return *this;
    }

    __device__
    UKnotGPUIterator operator++(int) { UKnotGPUIterator tmp(*this); ++(*this); return tmp; }
    __device__
    UKnotGPUIterator operator--(int) { UKnotGPUIterator tmp(*this); --(*this); return tmp; }

    __device__
    bool operator == (const UKnotGPUIterator& other) const
    { return m_upos == other.m_upos;}

    __device__
    bool operator != (const UKnotGPUIterator& other) const {return m_upos != other.m_upos;}
    __device__
    bool operator <  (const UKnotGPUIterator& other) const {return m_upos <  other.m_upos;}
    __device__
    bool operator >  (const UKnotGPUIterator& other) const {return m_upos >  other.m_upos;}
    __device__
    bool operator <= (const UKnotGPUIterator& other) const {return m_upos <= other.m_upos;}
    __device__
    bool operator >= (const UKnotGPUIterator& other) const {return m_upos >= other.m_upos;}

    __device__
    reference operator[] (difference_type a) const
    { return m_raw[m_mlt[m_upos+a]-1]; }

    __device__
    UKnotGPUIterator& operator+=(const difference_type & a)
    {
        m_upos += a;
        return *this;
    }

    __device__
    UKnotGPUIterator operator+(const difference_type & a) const
    {
        UKnotGPUIterator tmp(*this);
        return tmp+=a;
    }

    __device__
    UKnotGPUIterator& operator-=(const difference_type & a)
    { return operator+=(-a); }

    __device__
    UKnotGPUIterator operator-(const difference_type & a) const
    {
        UKnotGPUIterator tmp(*this);
        return tmp-=a;
    }

    __device__
    friend difference_type operator-(const UKnotGPUIterator & l, 
                                     const UKnotGPUIterator & r)
    {return l.m_upos - r.m_upos; }

    __device__
    int multSum() const
    { return m_mlt[m_upos];}

    __device__
    int multiplicity() const
    {
        if ( 0 == m_upos )//is it the first unique knot?
            return *m_mlt;
        else
        {
            const int* mp = m_mlt + m_upos;
            return *mp - *(mp-1);
        }
    }
};

class KnotGPUIterator
{
private:
    UKnotGPUIterator m_uit;
    int m_pos;
public:
    friend class KnotVectorDeviceView;

    typedef std::random_access_iterator_tag iterator_category;
    typedef const int* mltpointer;
    typedef double value_type;
    typedef std::ptrdiff_t difference_type;
    typedef const double& reference;
    typedef const double* pointer;

    __device__
    KnotGPUIterator() : m_uit(), m_pos(0) {}

    __device__
    explicit KnotGPUIterator(const KnotVectorDeviceView & KV, 
        const int upos = 0, const int s = 0)
    : m_uit(KV,upos,s), m_pos(firstAppearance())
    { }

    __device__ __forceinline__
    static KnotGPUIterator End(const KnotVectorDeviceView & KV);

    __device__
    reference operator* () const
    { return m_uit.m_raw[m_pos]; }

    __device__
    pointer get() const {return  m_uit.m_raw+m_pos;}
    __device__
    pointer operator-> () const {return  get();}

    __device__
    KnotGPUIterator& operator++()
    {
        if (++m_pos == m_uit.multSum())
            ++m_uit;
        return *this;
    }

    __device__
    KnotGPUIterator& operator--()
    {
        if ( m_pos-- == firstAppearance() )
            --m_uit;
        return *this;
    }

    __device__
    KnotGPUIterator operator++(int) 
    { KnotGPUIterator tmp(*this); ++(*this); return tmp; }
    __device__
    KnotGPUIterator operator--(int) 
    { KnotGPUIterator tmp(*this); --(*this); return tmp; }

    __device__
    bool operator == (const KnotGPUIterator& other) const
    { return m_pos == other.m_pos;}

    __device__
    bool operator != (const KnotGPUIterator& other) const 
    {return m_pos != other.m_pos;}
    __device__
    bool operator <  (const KnotGPUIterator& other) const 
    {return m_pos <  other.m_pos;}
    __device__
    bool operator >  (const KnotGPUIterator& other) const 
    {return m_pos >  other.m_pos;}
    __device__
    bool operator <= (const KnotGPUIterator& other) const 
    {return m_pos <= other.m_pos;}
    __device__
    bool operator >= (const KnotGPUIterator& other) const 
    {return m_pos >= other.m_pos;}

    __device__
    reference operator[] (difference_type a) const
    { return m_uit.m_raw[m_pos+a]; }

    __device__
    reference operator() (difference_type a) const
    { return m_uit[a]; }

    __device__
    int firstAppearance() const
    { return m_uit.firstAppearance();}

    __device__
    int lastAppearance() const
    { return m_uit.lastAppearance(); }

    __device__
    KnotGPUIterator& operator+=(const difference_type & a)
    {
        m_pos += a;
        if (a<0)
        {
            mltpointer end = m_uit.m_mlt + m_uit.m_upos;
            mltpointer beg = end + a;
            if (beg < m_uit.m_mlt) beg = m_uit.m_mlt;
            m_uit.m_upos = upper_bound_ptr(beg, end, m_pos) - m_uit.m_mlt;
        }
        else 
        {
            mltpointer beg = m_uit.m_mlt + m_uit.m_upos;
            mltpointer end = dmin(m_uit.m_mlt + m_uit.m_dbg-1, beg + a);
            m_uit.m_upos = upper_bound_ptr(beg, end, m_pos) - m_uit.m_mlt;
        }
        return *this;
    }

    __device__
    KnotGPUIterator operator+(const difference_type & a) const
    {
        KnotGPUIterator tmp(*this);
        return tmp+=a;
    }

    __device__
    KnotGPUIterator operator-(const difference_type & a) const
    {
        KnotGPUIterator tmp(*this);
        return tmp-=a;
    }
    
    __device__
    KnotGPUIterator& operator-=(const difference_type & a)
    { return operator+=(-a); }

    __device__
    const UKnotGPUIterator & uIterator() const
    { return m_uit; }
};