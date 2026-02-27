#include <KnotGPUIterator.h>
#include <KnotVectorDeviceView.h>

#if 0
__device__
UKnotGPUIterator::UKnotGPUIterator(const KnotVectorDeviceView &KV, const int upos, const int s)
: m_mlt ( KV.multSumData() ), m_raw ( KV.data() ),
  m_upos( upos             ), m_sh  (s          )
{ m_dbg = KV.uSize()+1; }
#endif


