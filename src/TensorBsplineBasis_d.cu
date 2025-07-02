#include <TensorBsplineBasis_d.h>

__global__
void retrieveKnotData(int dir, DeviceObjectArray<KnotVector_d>* d_knots_d, double* d_result)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < (*d_knots_d)[dir].getNumKnots(); idx += blockDim.x * gridDim.x)
    {
        d_result[idx] = (*d_knots_d)[dir].getKnots()[idx];
        //printf("retrieveKnotData: %d, %f\n", idx, d_result[idx]);
    }   
        //(*d_knots_d)[dir].getKnots().print();
}

__global__
void retrieveKnotSize(DeviceObjectArray<KnotVector_d>* d_knots_d, int* d_result)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < d_knots_d->size(); idx += blockDim.x * gridDim.x)
        d_result[idx] = (*d_knots_d)[idx].getNumKnots();
}

__global__
void retrieveKnotSizeAndOrder(DeviceObjectArray<KnotVector_d>* d_knots_d, 
                              int* d_order, int* d_size)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < d_knots_d->size(); idx += blockDim.x * gridDim.x)
    {
        d_order[idx] = (*d_knots_d)[idx].getOrder();
        d_size[idx] = (*d_knots_d)[idx].getNumKnots();
    }
}

__global__
void retrieveKnotOrder(DeviceObjectArray<KnotVector_d>* d_knots_d, int* d_result)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < d_knots_d->size(); idx += blockDim.x * gridDim.x)
        d_result[idx] = (*d_knots_d)[idx].getOrder();
}

__global__
void deviceConstructKnotVector(KnotVector_d* knotVector, KnotVector_d* input)
{
    new (knotVector) KnotVector_d(*input);
}