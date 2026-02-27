#pragma once
// KnotVectorDeviceView must be fully defined before including this

__device__ __forceinline__
UKnotGPUIterator::UKnotGPUIterator(const KnotVectorDeviceView &KV, int upos, int s)
: m_mlt ( KV.multSumData() ), m_raw ( KV.data() ),
  m_upos( upos             ), m_sh  ( s         )
{
    m_dbg = KV.uSize() + 1;
}

__device__ __forceinline__ UKnotGPUIterator 
UKnotGPUIterator::End(const KnotVectorDeviceView &KV)
{
    return UKnotGPUIterator(KV, KV.uSize(),KV.numLeftGhosts());
}

__device__ __forceinline__ KnotGPUIterator 
KnotGPUIterator::End(const KnotVectorDeviceView &KV)
{
    return KnotGPUIterator(KV, KV.uSize(), KV.numLeftGhosts());
}