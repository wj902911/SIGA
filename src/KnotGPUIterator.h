#pragma once

#include <cuda_runtime.h>


class KnotGPUIterator;

class UKnotGPUIterator
{
private:
    const int* m_mlt;
    const double* m_raw;
    int m_upos;
    int m_sh;
    int m_dbg;

public:
    __device__
    UKnotGPUIterator()
    : m_mlt(nullptr), m_raw(nullptr), m_upos(0), m_sh(0), m_dbg(0)
    { }

    __device__
    UKnotGPUIterator(const KnotGPUIterator & KV, const int upos = 0, const int s = 0);

};