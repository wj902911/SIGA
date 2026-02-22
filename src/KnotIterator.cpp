#include "KnotIterator.h"
#include <KnotVector.h>

UKnotIterator::UKnotIterator(const KnotVector &KV, const int upos, const int s)
: m_mlt ( KV.multSumData() ), m_raw ( KV.data() ),
  m_upos( upos             ), m_sh  (s          )
{ m_dbg = KV.uSize()+1; }

UKnotIterator UKnotIterator::End(const KnotVector &KV)
{ return UKnotIterator(KV, KV.uSize(),KV.numLeftGhosts()); }

KnotIterator KnotIterator::End(const KnotVector &KV)
{ return KnotIterator(KV, KV.uSize(),KV.numLeftGhosts()); }
