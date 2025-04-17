#pragma once

//#include <cuda_runtime.h>
//#include <thrust/device_vector.h>

template<class Vec>
inline bool nextLexicographic(Vec& cur, const Vec& size)
{
    const int d = cur.size();
    assert(d == size.size());
    for (int i = 0; i < d; i++)
    {
        if (++cur[i] == size[i])
        {
            if (i == d - 1)
                return false;
            else
                cur[i] = 0;
        }
        else
            return true;
    }
    assert("Something went wrong in nextLexicographic, wrong input?");
    return false;
}

template<class Vec>
bool nextLexicographic(Vec& cur, const Vec& start, const Vec& end)
{
    const int d = cur.size();
    assert( d == start.size() && d == end.size());

    for (int i = 0; i < d; ++i)
    {
        if (++cur[i] == end[i])
        {
            if (i == d - 1)
                return false;
            else
                cur[i] = start[i];
        }
        else
            return true;
    }
    assert("Something went wrong in nextLexicographic, wrong input?");
    return false;
}

