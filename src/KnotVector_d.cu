#include <KnotVector_d.h>

#if 0
int KnotVector_d::upperBound(double value) const
{
    int low = m_order - 1;
        int high = m_knots.size();;
        while (low < high)
        {
            int mid = low + (high - low) / 2;
            if (m_knots[mid] <= value)
            {
                low = mid + 1;
            }
            else
            {
                high = mid;
            }
        }
        return low;
}
#endif

__global__ 
void destructKernel(KnotVector_d* ptr, size_t count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) 
        ptr[idx].~KnotVector_d();
}

__global__ 
void deviceDeepCopyKernel(KnotVector_d* device_dst, KnotVector_d* host_src)
{
    new (device_dst) KnotVector_d(*host_src);
}
