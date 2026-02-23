#pragma once

#include <iterator>

class KnotIterator;

class UKnotIterator
{
private:
    const int* m_mlt;
    const double* m_raw;
    int m_upos;
    int m_sh;
    int m_dbg;
public:
    friend class KnotVector;
    friend class KnotIterator;

    typedef std::random_access_iterator_tag iterator_category;
    typedef double value_type;
    typedef std::ptrdiff_t difference_type;
    typedef const double&       reference;
    typedef const double*       pointer;


    UKnotIterator()
    : m_mlt(NULL), m_raw(NULL), m_upos(0), m_sh(0), m_dbg(0)
    { }

    UKnotIterator(const KnotVector & KV, const int upos = 0, const int s = 0);
    static UKnotIterator End(const KnotVector & KV);


    int firstAppearance() const
    { return 0 == m_upos ? 0 : m_mlt[m_upos-1]; }

    int lastAppearance() const
    { return m_mlt[m_upos] - 1; }

    reference operator*  () const
    { return m_raw[m_mlt[m_upos]-1]; }

    pointer operator-> () const {return m_raw+m_mlt[m_upos]-1 ;}

    UKnotIterator& operator++()
    {
        ++m_upos;
        return *this;
    }

    UKnotIterator& operator--()
    {
        --m_upos;
        return *this;
    }

    UKnotIterator operator++(int) { UKnotIterator tmp(*this); ++(*this); return tmp; }
    UKnotIterator operator--(int) { UKnotIterator tmp(*this); --(*this); return tmp; }

    bool operator == (const UKnotIterator& other) const
    { return m_upos == other.m_upos;}

    bool operator != (const UKnotIterator& other) const {return m_upos != other.m_upos;}
    bool operator <  (const UKnotIterator& other) const {return m_upos <  other.m_upos;}
    bool operator >  (const UKnotIterator& other) const {return m_upos >  other.m_upos;}
    bool operator <= (const UKnotIterator& other) const {return m_upos <= other.m_upos;}
    bool operator >= (const UKnotIterator& other) const {return m_upos >= other.m_upos;}

    reference operator[] (ptrdiff_t a) const
    { return m_raw[m_mlt[m_upos+a]-1]; }

    UKnotIterator& operator+=(const std::ptrdiff_t & a)
    {
        m_upos += a;
        return *this;
    }

    UKnotIterator operator+(const difference_type & a) const
    {
        UKnotIterator tmp(*this);
        return tmp+=a;
    }

    UKnotIterator& operator-=(const std::ptrdiff_t & a)
    { return operator+=(-a); }

    UKnotIterator operator-(const difference_type & a) const
    {
        UKnotIterator tmp(*this);
        return tmp-=a;
    }

    friend difference_type operator-(const UKnotIterator & l, 
                                     const UKnotIterator & r)
    {return l.m_upos - r.m_upos; }

    int multSum() const
    { return m_mlt[m_upos];}

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

class KnotIterator
{
private:
    UKnotIterator m_uit;
    int m_pos;
public:
    friend class KnotVector;

    typedef std::random_access_iterator_tag iterator_category;
    typedef const int* mltpointer;
    typedef double value_type;
    typedef std::ptrdiff_t difference_type;
    typedef const double& reference;
    typedef const double* pointer;

    KnotIterator()
    : m_uit(), m_pos(0)
    { }

    explicit KnotIterator(const KnotVector & KV, const int upos = 0, const int s = 0)
    : m_uit(KV,upos,s), m_pos(firstAppearance())
    { }

    static KnotIterator End(const KnotVector & KV);

    reference operator* () const
    { return m_uit.m_raw[m_pos]; }

    pointer get() const {return  m_uit.m_raw+m_pos;}
    pointer operator-> () const {return  get();}

    KnotIterator& operator++()
    {
        if (++m_pos == m_uit.multSum())
            ++m_uit;
        return *this;
    }

    KnotIterator& operator--()
    {
        if ( m_pos-- == firstAppearance() )
            --m_uit;
        return *this;
    }

    KnotIterator operator++(int) { KnotIterator tmp(*this); ++(*this); return tmp; }
    KnotIterator operator--(int) { KnotIterator tmp(*this); --(*this); return tmp; }

    bool operator == (const KnotIterator& other) const
    { return m_pos == other.m_pos;}

    bool operator != (const KnotIterator& other) const {return m_pos != other.m_pos;}
    bool operator <  (const KnotIterator& other) const {return m_pos <  other.m_pos;}
    bool operator >  (const KnotIterator& other) const {return m_pos >  other.m_pos;}
    bool operator <= (const KnotIterator& other) const {return m_pos <= other.m_pos;}
    bool operator >= (const KnotIterator& other) const {return m_pos >= other.m_pos;}

    reference operator[] (ptrdiff_t a) const
    { return m_uit.m_raw[m_pos+a]; }

    reference operator() (ptrdiff_t a) const
    { return m_uit[a]; }

    int firstAppearance() const
    { return m_uit.firstAppearance();}

    int lastAppearance() const
    { return m_uit.lastAppearance(); }

    KnotIterator& operator+=(const std::ptrdiff_t & a)
    {
        m_pos += a;

        if (a<0)
        {
            mltpointer end = m_uit.m_mlt + m_uit.m_upos;
            mltpointer beg = end + a;
            if (beg < m_uit.m_mlt) beg = m_uit.m_mlt;
            m_uit.m_upos = std::upper_bound(beg, end, m_pos) - m_uit.m_mlt;
        }
        else 
        {
            mltpointer beg = m_uit.m_mlt + m_uit.m_upos;
            mltpointer end = std::min(m_uit.m_mlt + m_uit.m_dbg-1, beg + a);
            m_uit.m_upos = std::upper_bound(beg, end, m_pos) - m_uit.m_mlt;
        }

        return *this;
    }

    KnotIterator operator+(const difference_type & a) const
    {
        KnotIterator tmp(*this);
        return tmp+=a;
    }

    KnotIterator operator-(const difference_type & a) const
    {
        KnotIterator tmp(*this);
        return tmp-=a;
    }

    KnotIterator& operator-=(const std::ptrdiff_t & a)
    { return operator+=(-a); }

    const UKnotIterator & uIterator() const
    { return m_uit; }
};