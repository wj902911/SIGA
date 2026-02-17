#include <MultiPatchDeviceView.h>

__global__
void edgeLengthesKernel(const MultiPatchDeviceView& patches, 
                        DeviceVectorView<double> lengths)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < patches.totalNumBdGPs(); idx += blockDim.x * gridDim.x)
    {
        int patch(0);
        int point_idx = patches.threadPatch_edge(idx, patch);
        int dir(0);
        point_idx = patches.threadEdgeDir(point_idx, patch, dir);
        int edgeIdx(0);
        point_idx = patches.threadEdge(point_idx, patch, dir, edgeIdx);
        int basisDim = patches.domainDim();
    }
}   