#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>

#include <iostream>

#include <Assembler.h>
#include <MultiPatch_d.h>
#include <MultiBasis_d.h>
#include <MultiBasis.h>
#include <GaussPoints_d.h>

#if 0
__device__
int lower_bound(double* arr, int size, double val);
__device__
int upper_bound(double* arr, int size, double val);

__global__ void tensorGridKernel(double* u, double* v, int v_size, double* grid);
__global__ void tensorGridKernel(double* u, double* v, double* w, int v_size, int w_size, double* grid);
//__global__ void tensorGridKernel(double* vecs, int* sizes, int dim, int num_patch, double* grid, int total_pts);
__global__ void assembleKernel(
	double* knots_ref,
	int* knot_sizes_ref,
	int* knot_degrees_ref,
	double* cps_ref,
	double* knots,
	int* knot_sizes, 
	int* knot_degrees, 
	double* cps, 
	int* dof_indices, 
	int dim, 
	int num_patches, 
	int der_order,
	double* global_mat,
	double* global_rhs);

void getGuessPts(int num_gausspts, thrust::device_vector<double>& gausspts, thrust::device_vector<double>& weight);
thrust::device_vector<double> tensorGrid(thrust::device_vector<double>& u, thrust::device_vector<double>& v);
thrust::device_vector<double> tensorGrid(thrust::device_vector<double>& u, thrust::device_vector<double>& v, thrust::device_vector<double>& w);
thrust::device_vector<double> tensorGrid(thrust::device_vector<double>& vecs, thrust::device_vector<int>& sizes, int dim, int num_patch);
void assemble(
	thrust::device_vector<double>& knots_ref, 
	thrust::device_vector<int>& knot_sizes_ref, 
	thrust::device_vector<int>& knot_degrees_ref, 
	thrust::device_vector<double>& cps_ref, 
	thrust::device_vector<double>& knots, 
	thrust::device_vector<int>& knot_sizes, 
	thrust::device_vector<int>& knot_degrees, 
	thrust::device_vector<double>& cps, 
	thrust::device_vector<int>& dof_indices, 
	int dim, int num_patches);
#endif
#if 1
template __global__ void destructKernel<DeviceObjectArray<double>>(DeviceObjectArray<double>* a, size_t n);
//template __global__ void destructKernel<DeviceObjectArray<DeviceMatrix<int>>>(DeviceObjectArray<DeviceMatrix<int>>* a, size_t n);
//template __global__ void destructKernel<DeviceObjectArray<DeviceMatrixBase<DeviceMatrix<int>, int>>>(DeviceObjectArray<DeviceMatrixBase<DeviceMatrix<int>, int>>* a, size_t n);
template __global__ void destructKernel<DeviceVector<int>>(DeviceVector<int>* a, size_t n);
template __global__ void destructKernel<DeviceVector<double>>(DeviceVector<double>* a, size_t n);
//template __global__ void destructKernel<DeviceMatrix<int>>(DeviceMatrix<int>* a, size_t n);
#endif
template __global__ void destructKernel<KnotVector_d>(KnotVector_d* a, size_t n);
template __global__ void destructKernel<TensorBsplineBasis_d>(TensorBsplineBasis_d* a, size_t n);
template __global__ void destructKernel<Patch_d>(Patch_d* a, size_t n);
template __global__ void destructKernel<DofMapper_d>(DofMapper_d* a, size_t n);
template __global__ void destructKernel<GaussPoints_d>(GaussPoints_d* a, size_t n);
#if 1
template __global__ void deviceDeepCopyKernel<DeviceObjectArray<double>>(DeviceObjectArray<double>* a, 
	                                                                     DeviceObjectArray<double>* b);
#endif

__global__ void testKernel(MultiPatch_d* d_multiPatch_d)
{
	d_multiPatch_d->patch(0).basis().getKnotVector(0).getKnots().print();
	d_multiPatch_d->patch(0).basis().getKnotVector(1).getKnots().print();
	d_multiPatch_d->patch(0).controlPoints().print();
	d_multiPatch_d->patch(1).controlPoints().print();
	printf("testkernel done\n");
}

__global__
void MultiBasis_dTestKenel(MultiBasis_d* d_bases_d)
{
    d_bases_d->basis(0).getKnotVector(0).getKnots().print();
	d_bases_d->basis(0).getKnotVector(1).getKnots().print();
	d_bases_d->basis(1).getKnotVector(0).getKnots().print();
	d_bases_d->basis(1).getKnotVector(1).getKnots().print();
    printf("MultiBasis_dTestKenel done\n");
}

__global__ void checkData(DeviceObjectArray<DeviceObjectArray<double>>* d_knot_DOA)
{
	d_knot_DOA->operator[](0).print(); // Access the first patch's knot vector (u)
	d_knot_DOA->operator[](1).print();
}

__global__ 
void constructBasisArray(TensorBsplineBasis_d* d_basis_array, 
									int num_patches)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_patches; idx += blockDim.x * gridDim.x)
	{
		// Construct the basis array here
	}
}

__global__ void constructDOADouble(DeviceObjectArray<double>* d_knot_DOA, 
	                               int size, double* d_data)
{
	new (d_knot_DOA) DeviceObjectArray<double>(size, d_data);
}

int main()
{
	int knot_u_order = 1;
	int knot_v_order = 1;
	std::vector<double> knot_u{ 0., 0., 1., 1. };
	std::vector<double> knot_v{ 0., 0., 1., 1. };
	Eigen::MatrixXd control_points(4, 2);
	control_points <<
		0.000, 0.000, 
		1.000, 0.000, 
		0.000, 1.000,
		1.000, 1.000;

#if 0
	//DeviceObjectArray<double>* knot_DOA = new DeviceObjectArray<double>[2];
	//knot_DOA[0]=DeviceObjectArray<double>(knot_u.size(), knot_u.data());
	//knot_DOA[1]=DeviceObjectArray<double>(knot_v.size(), knot_v.data());
	//DeviceObjectArray<double> knot_u_DOA_built(knot_u.size(), knot_u.data());
	//DeviceObjectArray<double> knot_v_DOA_built(knot_v.size(), knot_v.data());
	DeviceObjectArray<DeviceObjectArray<double>> h_knot_DOA(2);
	h_knot_DOA.at(0) = DeviceObjectArray<double>(knot_u.size(), knot_u.data());
	h_knot_DOA.at(1) = DeviceObjectArray<double>(knot_v.size(), knot_v.data());
	//h_knot_DOA.at(0) = knot_u_DOA_built;
	//h_knot_DOA.at(1) = knot_v_DOA_built;
	//h_knot_DOA.parallelDataSetting(knot_DOA, 2); 
	//delete[] knot_DOA;

	//DeviceObjectArray<double> knot_u_DOA1 = h_knot_DOA.at(0);
	//knot_u_DOA1.print(); // Print the contents of the first patch's knot vector (u)

	DeviceObjectArray<DeviceObjectArray<double>>* d_knot_DOA = nullptr;
	cudaMalloc((void**)&d_knot_DOA, sizeof(DeviceObjectArray<DeviceObjectArray<double>>));
	cudaMemcpy(d_knot_DOA, &h_knot_DOA, sizeof(DeviceObjectArray<DeviceObjectArray<double>>), 
			   cudaMemcpyHostToDevice);
	#if 0
	testKernel<<<1, 1>>>(d_knot_DOA);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		std::cerr << "Error after kernel testKernel launch: " << cudaGetErrorString(err) << std::endl;
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		std::cerr << "CUDA error during device synchronization (testKernel): " << cudaGetErrorString(err) << std::endl;	
	#endif
	//std::vector<double> knot_u_vec = DeviceObjectArray<double>(h_knot_DOA.at(0)).retrieveDataToHost();
	//std::vector<double> knot_v_vec = DeviceObjectArray<double>(h_knot_DOA.at(1)).retrieveDataToHost();
	DeviceObjectArray<double> knot_u_DOA = h_knot_DOA.at(0);
	//DeviceObjectArray<double> knot_u_DOA = DeviceObjectArray<double>(knot_u_vec.size(), knot_u_vec.data()); // Access the first patch's knot vector (u)
	knot_u_DOA.print(); 

	DeviceObjectArray<double> knot_v_DOA = h_knot_DOA.at(1); // Access the second patch's knot vector (v)
	//DeviceObjectArray<double> knot_v_DOA = DeviceObjectArray<double>(knot_v_vec.size(), knot_v_vec.data()); // Access the second patch's knot vector (v)
	knot_v_DOA.print();

	#if 0
	cudaFree(d_knot_DOA); // Free the device memory for knot_DOA

	knot_u_DOA.print(); 

	cudaMalloc((void**)&d_knot_DOA, sizeof(DeviceObjectArray<DeviceObjectArray<double>>));
	cudaMemcpy(d_knot_DOA, &h_knot_DOA, sizeof(DeviceObjectArray<DeviceObjectArray<double>>), 
			   cudaMemcpyHostToDevice);

	DeviceObjectArray<double> knot_u_DOA1 = h_knot_DOA.at(0); // Access the first patch's knot vector (u)
	knot_u_DOA1.print();

	DeviceObjectArray<double>* d_knot_u_DOA = nullptr;
	cudaMalloc((void**)&d_knot_u_DOA, sizeof(DeviceObjectArray<double>));
	cudaMemcpy(d_knot_u_DOA, &knot_u_DOA, sizeof(DeviceObjectArray<double>), 
			   cudaMemcpyHostToDevice);
	// Launch a kernel to check the contents of knot_u_DOA
	checkData<<<1, 1>>>(d_knot_u_DOA);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		std::cerr << "Error after kernel checkData launch: " << cudaGetErrorString(err) << std::endl;
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		std::cerr << "CUDA error during device synchronization (checkData): " << cudaGetErrorString(err) << std::endl;
	//cudaFree(d_knot_u_DOA); // Free the device memory for knot_u_DOA
	//double* d_knot_u_DOA_data = knot_u_DOA.data();
	//double* h_knot_u_DOA_data = new double[knot_u_DOA.size()];
	//cudaMemcpy(h_knot_u_DOA_data, d_knot_u_DOA_data, knot_u_DOA.size() * sizeof(double), cudaMemcpyDeviceToHost);
	knot_u_DOA.print();
	#endif

	DeviceObjectArray<DeviceObjectArray<double>> h_knot_DOA_copy(2);
	//h_knot_DOA_copy.at(0) = DeviceObjectArray<double>(h_knot_DOA.at(0)); // Copy u knot vector
	//h_knot_DOA_copy.at(1) = DeviceObjectArray<double>(h_knot_DOA.at(1)); // Copy v knot vector
	//h_knot_DOA_copy.at(0) = knot_u_DOA.clone(); // Copy u knot vector
	//h_knot_DOA_copy.at(1) = knot_v_DOA.clone(); // Copy v knot vector
	h_knot_DOA_copy.at(0) = knot_u_DOA; // Copy u knot vector
	h_knot_DOA_copy.at(1) = knot_v_DOA; // Copy v knot vector
	//h_knot_DOA_copy.at(0) = DeviceObjectArray<double>(knot_u_vec.size(), knot_u_vec.data());
	//h_knot_DOA_copy.at(1) = DeviceObjectArray<double>(knot_v_vec.size(), knot_v_vec.data());

	DeviceObjectArray<DeviceObjectArray<double>>* d_knot_DOA_copy = nullptr;
	cudaMalloc((void**)&d_knot_DOA_copy, sizeof(DeviceObjectArray<DeviceObjectArray<double>>));
	cudaMemcpy(d_knot_DOA_copy, &h_knot_DOA_copy, sizeof(DeviceObjectArray<DeviceObjectArray<double>>), 
			   cudaMemcpyHostToDevice);

	//cudaFree(d_knot_DOA); // Free the device memory for knot_DOA

	DeviceObjectArray<double> knot_u_DOA_copy = h_knot_DOA_copy.at(0); // Access the first patch's knot vector (u)
	knot_u_DOA_copy.print();
#endif
	
	std::vector<double> knot_u2{ 0., 0., 1., 1. };
	std::vector<double> knot_v2{ 0., 0., 1., 1. };
	Eigen::MatrixXd control_points2(4, 2);
	control_points2 <<
		1.000, 0.000, 
		2.000, 0.000,
		1.000, 1.000,
		2.000, 1.000;

	KnotVector u1(knot_u_order,knot_u);
	KnotVector v1(knot_v_order,knot_v);
	KnotVector u2(knot_u_order,knot_u2);
	KnotVector v2(knot_v_order,knot_v2);
	Patch patch(u1, v1, control_points);
	Patch patch2(u2, v2, control_points2);

	MultiPatch multiPatch;
	multiPatch.addPatch(patch);
	multiPatch.addPatch(patch2);
	multiPatch.computeTopology();

	MultiBasis bases(multiPatch);

#if 0
	DeviceObjectArray<DeviceObjectArray<double>> h_knot_DOA(2);
	h_knot_DOA.at(0) = DeviceObjectArray<double>(knot_u.size(), knot_u.data());
	h_knot_DOA.at(1) = DeviceObjectArray<double>(knot_v.size(), knot_v.data());
	DeviceObjectArray<DeviceObjectArray<double>>* d_knot_DOA = nullptr;
	cudaMalloc((void**)&d_knot_DOA, sizeof(DeviceObjectArray<DeviceObjectArray<double>>));
	cudaMemcpy(d_knot_DOA, &h_knot_DOA, sizeof(DeviceObjectArray<DeviceObjectArray<double>>), 
			   cudaMemcpyHostToDevice);
	checkData<<<1, 1>>>(d_knot_DOA);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		std::cerr << "Error after kernel checkData launch: " << cudaGetErrorString(err) << std::endl;
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		std::cerr << "CUDA error during device synchronization (checkData): " << cudaGetErrorString(err) << std::endl;
	cudaFree(d_knot_DOA); // Free the device memory for knot_DOA
	DeviceObjectArray<double> h_knot_u_DOA_copy = h_knot_DOA.at(0);
	DeviceObjectArray<double> h_knot_v_DOA_copy = h_knot_DOA.at(1);
	h_knot_u_DOA_copy.print();
	h_knot_v_DOA_copy.print();
#endif
#if 0
	double* d_knot_u_data = nullptr;
	cudaMalloc((void**)&d_knot_u_data, knot_u.size() * sizeof(double));
	cudaMemcpy(d_knot_u_data, knot_u.data(), knot_u.size() * sizeof(double), 
	           cudaMemcpyHostToDevice);
	constructDOADouble<<<1, 1>>>(h_knot_DOA.data(), knot_u.size(), d_knot_u_data);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		std::cerr << "Error after kernel constructDOADouble launch: " 
		          << cudaGetErrorString(err) << std::endl;
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		std::cerr << "CUDA error during device synchronization (constructDOADouble): " 
		          << cudaGetErrorString(err) << std::endl;
#endif
	//MultiPatch_d multiPatch_d(multiPatch);
	//MultiBasis bases(multiPatch);
	//MultiBasis_d bases_d(bases);
#if 0
	MultiBasis_d* d_bases_d = nullptr;
	cudaMalloc((void**)&d_bases_d, sizeof(MultiBasis_d));
	cudaMemcpy(d_bases_d, &bases_d, sizeof(MultiBasis_d), cudaMemcpyHostToDevice);
	MultiBasis_dTestKenel<<<1, 1>>>(d_bases_d);
	if (cudaGetLastError() != cudaSuccess)
		std::cerr << "Error after kernel MultiBasis_dTestKenel launch: " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	if (cudaDeviceSynchronize() != cudaSuccess)
		std::cerr << "CUDA error during device synchronization (MultiBasis_dTestKenel): " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
#endif
#if 0	
	MultiBasis_d* d_bases_d = nullptr;
	cudaMalloc((void**)&d_bases_d, sizeof(MultiBasis_d));
	cudaMemcpy(d_bases_d, &bases_d, sizeof(MultiBasis_d), cudaMemcpyHostToDevice);
	testKernel<<<1, 1>>>(d_bases_d);
	if (cudaGetLastError() != cudaSuccess)
		std::cerr << "Error after kernel testKernel launch: " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	if (cudaDeviceSynchronize() != cudaSuccess)
		std::cerr << "CUDA error during device synchronization (testKernel): " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	cudaFree(d_bases_d);
#endif

#if 0
	TensorBsplineBasis_d h_basis_d(u1,v1);
	TensorBsplineBasis_d* d_basis_d = nullptr;
	cudaMalloc((void**)&d_basis_d, sizeof(TensorBsplineBasis_d));
	cudaMemcpy(d_basis_d, &h_basis_d, sizeof(TensorBsplineBasis_d), cudaMemcpyHostToDevice);
	testKernel<<<1, 1>>>(d_basis_d);
	if (cudaGetLastError() != cudaSuccess)
		std::cerr << "Error after kernel testKernel launch: " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	if (cudaDeviceSynchronize() != cudaSuccess)
		std::cerr << "CUDA error during device synchronization (testKernel): " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	cudaFree(d_basis_d);

	KnotVector_d h_knot_u_d_retrieved = h_basis_d.getKnotVector_h(0);
	h_knot_u_d_retrieved.getKnots().print();

#endif
#if 0
	KnotVector_d* h_knots_d = new KnotVector_d[2];
	h_knots_d[0] = KnotVector_d(u1);
	h_knots_d[1] = KnotVector_d(v1);
	DeviceObjectArray<KnotVector_d> knots_d(2);
	knots_d.parallelDataSetting(h_knots_d, 2);
	delete[] h_knots_d;
	//KnotVector_d h_knot_u_d(static_cast<const DeviceObjectArray<KnotVector_d>>(knots_d)[0]);
	//h_knot_u_d.getKnots().print();
	//h_knots_d = new KnotVector_d[2];
	//knots_d.parallelDataReading(h_knots_d, 2);
	//h_knots_d[0].getKnots().print();
	//delete[] h_knots_d;
	//KnotVector_d u1_d(u1);
	//knots_d.at(0) = KnotVector_d(u1);
	//knots_d.at(1) = KnotVector_d(v1);
	DeviceObjectArray<KnotVector_d>* d_knots_d = nullptr;
	cudaMalloc((void**)&d_knots_d, sizeof(DeviceObjectArray<KnotVector_d>));
	cudaMemcpy(d_knots_d, &knots_d, sizeof(DeviceObjectArray<KnotVector_d>), cudaMemcpyHostToDevice);
	testKernel<<<1, 1>>>(d_knots_d);
	if (cudaGetLastError() != cudaSuccess)
		std::cerr << "Error after kernel testKernel launch: " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	if (cudaDeviceSynchronize() != cudaSuccess)
		std::cerr << "CUDA error during device synchronization (testKernel): " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	cudaFree(d_knots_d);
#endif
#if 0
	DeviceObjectArray<Patch_d> patches(2);
	Patch_d patch_d(patch);
	patches.at(0) = patch_d;
	Patch_d patch_d2(patch2);
	patches.at(1) = patch_d2;
	//Patch_d patch_d1(patches.at(0));
	//patch_d1.getControlPoints().print();
	DeviceObjectArray<Patch_d>* d_patches = nullptr;
	cudaMalloc((void**)&d_patches, sizeof(DeviceObjectArray<Patch_d>));
	cudaMemcpy(d_patches, &patches, sizeof(DeviceObjectArray<Patch_d>), cudaMemcpyHostToDevice);
	testKernel<<<1, 1>>>(d_patches);
	if (cudaGetLastError() != cudaSuccess)
		std::cerr << "Error after kernel testKernel launch: " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	if (cudaDeviceSynchronize() != cudaSuccess)
		std::cerr << "CUDA error during device synchronization (testKernel): " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	cudaFree(d_patches);
#endif
#if 0
	Patch_d patch_d(patch);
	Patch_d* d_patch = nullptr;
	cudaMalloc((void**)&d_patch, sizeof(Patch_d));
	cudaMemcpy(d_patch, &patch_d, sizeof(Patch_d), cudaMemcpyHostToDevice);
	testKernel<<<1, 1>>>(d_patch);
	if (cudaGetLastError() != cudaSuccess)
		std::cerr << "Error after kernel testKernel launch: " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	if (cudaDeviceSynchronize() != cudaSuccess)
		std::cerr << "CUDA error during device synchronization (testKernel): " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	cudaFree(d_patch);
#endif
#if 0
	MultiPatch_d multiPatch_d(multiPatch);
	MultiPatch_d* d_multiPatch = nullptr;
	cudaMalloc((void**)&d_multiPatch, sizeof(MultiPatch_d));
	cudaMemcpy(d_multiPatch, &multiPatch_d, sizeof(MultiPatch_d), cudaMemcpyHostToDevice);
	testKernel<<<1, 1>>>(d_multiPatch);
	if (cudaGetLastError() != cudaSuccess)
		std::cerr << "Error after kernel testKernel launch: " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	if (cudaDeviceSynchronize() != cudaSuccess)
		std::cerr << "CUDA error during device synchronization (testKernel): " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	cudaFree(d_multiPatch);
#endif
#if 0
	BoundaryInterface_d interface(multiPatch.topology().interface(0));
	BoundaryInterface_d* d_interface = nullptr;
	cudaMalloc((void**)&d_interface, sizeof(BoundaryInterface_d));
	cudaMemcpy(d_interface, &interface, sizeof(BoundaryInterface_d), cudaMemcpyHostToDevice);
	testKernel<<<1, 1>>>(d_interface);
	if (cudaGetLastError() != cudaSuccess)
		std::cerr << "Error after kernel testKernel launch: " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	if (cudaDeviceSynchronize() != cudaSuccess)
		std::cerr << "CUDA error during device synchronization (testKernel): " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	cudaFree(d_interface);
	DeviceObjectArray<BoundaryInterface_d> interfaces(1);
	interfaces.at(0) = multiPatch.topology().interface(0);
	DeviceObjectArray<BoundaryInterface_d>* d_interfaces = nullptr;
	cudaMalloc((void**)&d_interfaces, sizeof(DeviceObjectArray<BoundaryInterface_d>));
	cudaMemcpy(d_interfaces, &interfaces, sizeof(DeviceObjectArray<BoundaryInterface_d>), cudaMemcpyHostToDevice);
	testKernel<<<1, 1>>>(d_interfaces);
	if (cudaGetLastError() != cudaSuccess)
		std::cerr << "Error after kernel testKernel launch: " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	if (cudaDeviceSynchronize() != cudaSuccess)
		std::cerr << "CUDA error during device synchronization (testKernel): " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	cudaFree(d_interfaces);
#endif
#if 0
	BoxTopology_d topology(multiPatch.topology());
	std::cout << "Number of patches: " << topology.nboxes() << std::endl;
	//topology.interface(0).dirMap().print();
	BoxTopology_d* d_topology = nullptr;
	cudaMalloc((void**)&d_topology, sizeof(BoxTopology_d));
	cudaMemcpy(d_topology, &topology, sizeof(BoxTopology_d), cudaMemcpyHostToDevice);
	testKernel<<<1, 1>>>(d_topology);
	if (cudaGetLastError() != cudaSuccess)
		std::cerr << "Error after kernel testKernel launch: " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	if (cudaDeviceSynchronize() != cudaSuccess)
		std::cerr << "CUDA error during device synchronization (testKernel): " 
		          << cudaGetErrorString(cudaGetLastError()) 
				  << std::endl;
	cudaFree(d_topology);
#endif
#if 1
#if 0
	std::vector<int> numKnots = multiPatch.getBasisNumKnots();
	std::vector<double> knots = multiPatch.getBasisKnots();
	std::vector<int> numGpAndEle = multiPatch.getNumGpAndEle();
	std::cout << "Before refine:" << std::endl;
	std::cout << "NumKnots: ";
	for (int i = 0; i < numKnots.size(); i++)
	{
		std::cout << numKnots[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "Knots: ";
	for (int i = 0; i < knots.size(); i++)
	{
		std::cout << knots[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "NumGpAndEle: ";
	for (int i = 0; i < numGpAndEle.size(); i++)
	{
		std::cout << numGpAndEle[i] << " ";
	}
	std::cout << std::endl;
#endif

	bases.uniformRefine();

	BoundaryConditions bcInfo;
	for (int d = 0; d < 2; ++d)
		bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, std::vector<double>{0.0, 0.0}, d);

	//std::vector<double> neumannValue{ 100e4, 0.0 };
	std::vector<double> disp{ 0.1, 0.0 };
    //bcInfo.addCondition(1, boundary::east, condition_type::neumann, neumannValue);
    bcInfo.addCondition(1, boundary::east, condition_type::dirichlet, disp, 0);

	Eigen::VectorXd bodyForce(2);
	bodyForce << 0.0, 0.0;

#if 0
	numKnots = multiPatch.getBasisNumKnots();
	knots = multiPatch.getBasisKnots();
	numGpAndEle = multiPatch.getNumGpAndEle();
	std::cout << "After refine:" << std::endl;
	std::cout << "NumKnots: ";
	for (int i = 0; i < numKnots.size(); i++)
	{
		std::cout << numKnots[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "Knots: ";
	for (int i = 0; i < knots.size(); i++)
	{
		std::cout << knots[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "NumGpAndEle: ";
	for (int i = 0; i < numGpAndEle.size(); i++)
	{
		std::cout << numGpAndEle[i] << " ";
	}
	std::cout << std::endl;

	thrust::device_vector<double> knot_refined;
	thrust::device_vector<double> knot_geo;
	for (int i = 0; i < multiPatch.getNumPatches(); i++)
	{
		for (int d = 0; d < multiPatch.getDim(); d++)
		{
			knot_refined = multiPatch.getBasisKnots(i, d);
			std::cout << "Basis " << i << " dim " << d << ": ";
			for (int j = 0; j < knot_refined.size(); j++)
			{
				std::cout << knot_refined[j] << " ";
			}
			std::cout << std::endl;
			knot_geo = multiPatch.getGeoKnots(i, d);
			std::cout << "Geo " << i << " dim " << d << ": ";
			for (int j = 0; j < knot_geo.size(); j++)
			{
				std::cout << knot_geo[j] << " ";
			}
			std::cout << std::endl;
		}
	}
	Eigen::VectorXi bnd = multiPatch.coefSlice(0, 0, 1);
	std::cout << "CoefSlice: ";
	for (int i = 0; i < bnd.size(); i++)
	{
		std::cout << bnd[i] << " ";
	}
	std::cout << std::endl;
#endif
	Assembler assembler(multiPatch, bases, bcInfo, bodyForce);
	DeviceVector<double> solution(assembler.numDofs());
	solution.setZero();
	assembler.assemble(solution);
#endif
#if 0
	thrust::device_vector<int> dofs_u{ 2,0,3,1 };
	thrust::device_vector<int> dofs_v{ 2,0,3,1 };

	//int numEles = (knot_u.size() - knot_u_order * 2 - 1) * (knot_v.size() - knot_v_order * 2 - 1);
	thrust::device_vector<double> knots_vecs_ref(knot_u_ref.size() + knot_v_ref.size());
	thrust::copy(knot_u_ref.begin(), knot_u_ref.end(), knots_vecs_ref.begin());
	thrust::copy(knot_v_ref.begin(), knot_v_ref.end(), knots_vecs_ref.begin() + knot_u_ref.size());
	thrust::device_vector<int> knots_sizes_ref{ static_cast<int>(knot_u_ref.size()), static_cast<int>(knot_v_ref.size()) };


	thrust::device_vector<double> knots_vecs(knot_u.size() + knot_v.size());
	thrust::copy(knot_u.begin(), knot_u.end(), knots_vecs.begin());
	thrust::copy(knot_v.begin(), knot_v.end(), knots_vecs.begin() + knot_u.size());
	thrust::device_vector<int> knots_sizes{ static_cast<int>(knot_u.size()), static_cast<int>(knot_v.size()) };
	thrust::device_vector<int> knot_degrees{ knot_u_order, knot_v_order };
	thrust::device_vector<int> dof_indices(dofs_u.size() + dofs_v.size());
	thrust::copy(dofs_u.begin(), dofs_u.end(), dof_indices.begin());
	thrust::copy(dofs_v.begin(), dofs_v.end(), dof_indices.begin() + dofs_u.size());

	assemble(
		knots_vecs_ref, 
		knots_sizes_ref, 
		knot_degrees,
		control_points, 
		knots_vecs, 
		knots_sizes, 
		knot_degrees, 
		control_points, 
		dof_indices, 
		dim, 
		1);

	int num_gausspts_u = knot_u_order + 1;
	int num_gausspts_v = knot_v_order + 1;

	thrust::device_vector<double> gausspts_u;
	thrust::device_vector<double> weight_u;
	getGuessPts(num_gausspts_u, gausspts_u, weight_u);
	thrust::device_vector<double> gausspts_v;
	thrust::device_vector<double> weight_v;
	getGuessPts(num_gausspts_v, gausspts_v, weight_v);

	
	thrust::device_vector<double> gausspts_vecs(gausspts_u.size() + gausspts_v.size());
	thrust::copy(gausspts_u.begin(), gausspts_u.end(), gausspts_vecs.begin());
	thrust::copy(gausspts_v.begin(), gausspts_v.end(), gausspts_vecs.begin() + gausspts_u.size());
	thrust::device_vector<int> gausspts_sizes{ num_gausspts_u, num_gausspts_v };

	thrust::device_vector<double> gauss_grid = tensorGrid(gausspts_vecs, gausspts_sizes, dim, 1);

	for(int i = 0; i < gauss_grid.size(); i++)
	{
		std::cout << gauss_grid[i] << std::endl;
	}
#endif

	return 0;
}

#if 0
__device__
int lower_bound(double* arr, int size, double val)
{
	int low = 0;
	int high = size;
	while (low < high) {
		int mid = low + (high - low) / 2;
		if (arr[mid] < val) {
			low = mid + 1;
		}
		else {
			high = mid;
		}
	}
	return low;
}

__device__
int upper_bound(double* arr, int size, double val)
{
	int low = 0;
	int high = size;
	while (low < high) {
		int mid = low + (high - low) / 2;
		if (arr[mid] <= val) {
			low = mid + 1;
		}
		else {
			high = mid;
		}
	}
	return low;
}

__global__ void tensorGridKernel(double* u, double* v, int v_size, double* grid)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int u_idx = idx / v_size;
	int v_idx = idx % v_size;

	grid[2 * idx] = u[u_idx];
	grid[2 * idx + 1] = v[v_idx];
}

__global__ void tensorGridKernel(double* u, double* v, double* w, int v_size, int w_size, double* grid)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int u_idx = idx / (v_size * w_size);
	int v_idx = (idx / w_size) % v_size;
	int w_idx = idx % w_size;

	grid[3 * idx] = u[u_idx];
	grid[3 * idx + 1] = v[v_idx];
	grid[3 * idx + 2] = w[w_idx];
}

__global__ void tensorGridKernel(double* vecs, int* sizes, int dim, int num_patch, double* grid, int total_points)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_points; idx += blockDim.x * gridDim.x)
	{
		int patch_idx = 0;
		int patch_grid_offset = 0;
		int coords[3] = { 0 }; // Supports up to 3D
		tensorGridDevice(idx, sizes, dim, num_patch, patch_idx, patch_grid_offset, coords);
		

		int patch_offset = patch_idx * dim;
		int vec_offset = 0;
		for (int i = 0; i < patch_offset * dim; ++i) {
			vec_offset += sizes[i];
		}

		double* patch_vecs = &vecs[vec_offset];
		int vector_start = 0;
		for (int d = 0; d < dim; ++d) {
			grid[patch_grid_offset + idx * dim + d] = patch_vecs[vector_start + coords[d]];
			vector_start += sizes[patch_offset + d];
		}
	}
}

__global__ void assembleKernel(
	double* knots_ref,
	int* knot_sizes_ref,
	int* knot_degrees_ref,
	double* cps_ref,
	double* knots,
	int* knot_sizes, 
	int* knot_orders, 
	double* cps, 
	int* dof_indices, 
	int dim, 
	int num_patches, 
	int der_order,
	double* global_mat, 
	double* global_rhs)
{
	int patch_idx = 0;
	//int patch_grid_offset = 0;
	int coords[6] = { 0 }; // Supports up to 3D

	int* sizes = new int[num_patches * dim * 2];
	for (int i = 0; i < num_patches * dim; i++) {
		sizes[i] = knot_orders[i] + 1;
		sizes[i + dim] = knot_sizes[i] - knot_orders[i] * 2 - 1;
	}

	int num_guspts = 0;
	for(int i = 0; i < num_patches; i++) {
		int patch_offset = i * dim * 2;
		int patch_points = 1;
		for (int j = 0; j < dim * 2; ++j) {
			patch_points *= sizes[patch_offset + j];
		}
		num_guspts += patch_points;
	}

	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_guspts; idx += blockDim.x * gridDim.x) {

		printf("GaussPt %d\n", idx);

		//tensorGridDevice(idx, sizes, dim * 2, num_patches, patch_idx, /*patch_grid_offset,*/ coords);

		int* patchGusptVecSizes = &sizes[patch_idx * dim * 2];
		//int gusptSize_patchOffset = patch_idx * dim * 2;
		int total_gusptVecSize = 0;
		for (int i = 0; i < dim; ++i) {
			total_gusptVecSize += patchGusptVecSizes[i];
		}
		double* gusptVecs = new double[total_gusptVecSize];
		double* gusptWeights = new double[total_gusptVecSize];
		for(int i = 0; i < dim; i++) {
			//getGaussPtsDevice(patchGusptVecSizes[i], &gusptVecs[i * dim], &gusptWeights[i * dim]);
		}

		double gausspt[3] = { 0 };
		double weight = 1.0;
		int gusptVec_start = 0;
		for (int d = 0; d < dim; ++d) {
			gausspt[d] = gusptVecs[gusptVec_start + coords[d]];
			weight *= gusptWeights[gusptVec_start + coords[d]];
			gusptVec_start += patchGusptVecSizes[d];
			//printf("%d\n", d);
			//printf("%f\n", gausspt[d]);
			//printf("%f\n", weight);
			//printf("\n");
		}

		int patchSize_offset = patch_idx * dim;
		int knots_offset = 0;
		for (int i = 0; i < patchSize_offset * dim; ++i) {
			knots_offset += knot_sizes[i];
		}

		double* patch_knots = &knots[knots_offset];
		int* patchKnotDegrees = &knot_orders[patchSize_offset];
		int* patchKnotSizes = &knot_sizes[patchSize_offset];

		double lower[3] = { 0 };
		double upper[3] = { 0 };

		int knot_start = 0;
		for (int d = 0; d < dim; ++d) {
			lower[d] = patch_knots[knot_start + patchKnotDegrees[d] + coords[d + dim]];
			upper[d] = patch_knots[knot_start + patchKnotDegrees[d] + 1 + coords[d + dim]];
			knot_start += knot_sizes[patchSize_offset + d];
			//printf("dim %d:lower %f upper %f\n", d, lower[d], upper[d]);
		}

		double hprod = 1.0;
		for (int d = 0; d < dim; ++d) {
			double h = (upper[d] - lower[d]) / 2.0;
			//printf("%f\n", h);
			gausspt[d] = h * (gausspt[d] + 1.0) + lower[d];
			printf("gausspt %f\n", gausspt[d]);
			hprod *= h;
		}
		weight *= hprod;

		double* values = new double[patchGusptVecSizes[dim] * dim];
		for (int d = 0; d < dim; ++d) {
			int p1 = patchKnotDegrees[d] + 1;

			double* ndu = new double[p1 * p1];
			double* left = new double[p1];
			double* right = new double[p1];
			double* a = new double[2 * p1];

			int span = upper_bound(patch_knots, patchKnotSizes[d], gausspt[d]) - 1;
			//printf("span %d\n", span);
			ndu[0] = 1.0;
			for (int j = 1; j <= patchKnotDegrees[d]; ++j) {
				left[j] = gausspt[d] - patch_knots[span + 1 - j];
				right[j] = patch_knots[span + j] - gausspt[d];
				double saved = 0.0;
				for (int r = 0; r < j; ++r) {
					ndu[j * p1 + r] = right[r + 1] + left[j - r];
					double temp = ndu[r * p1 + j - 1] / ndu[j * p1 + r];
					ndu[r * p1 + j] = saved + right[r + 1] * temp;
					saved = left[j - r] * temp;
				}
				ndu[j * p1 + j] = saved;
			}

			for(int j = 0; j <= patchKnotDegrees[d]; j++) {
				if(span - patchKnotDegrees[d] + j >= 0 && span - patchKnotDegrees[d] + j < patchKnotSizes[d]) {
					values[d * dim + j] = ndu[j * p1 + patchKnotDegrees[d]];
				}
				else {
					values[d * dim + j] = 0.0;
				}
				printf("%f\n", values[d * dim + j]);
			}
			delete[] ndu, left, right, a;
		}
		printf("\n");

		delete[] values;
		delete[] gusptVecs;
		delete[] gusptWeights;
	}

	delete sizes;
}

thrust::device_vector<double> tensorGrid(thrust::device_vector<double>& u, thrust::device_vector<double>& v)
{
	thrust::device_vector<double> grid(2 * u.size() * v.size());

	int block_size = 256;
	int num_blocks = (u.size() * v.size() + block_size - 1) / block_size;

	tensorGridKernel << <num_blocks, block_size >> > (
		thrust::raw_pointer_cast(u.data()),
		thrust::raw_pointer_cast(v.data()),
		v.size(),
		thrust::raw_pointer_cast(grid.data())
		);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Error after kernel tensorGridKernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization (tensorGridKernel): " << cudaGetErrorString(err) << std::endl;
	}
	return grid;
}

thrust::device_vector<double> tensorGrid(thrust::device_vector<double>& u, thrust::device_vector<double>& v, thrust::device_vector<double>& w)
{
	thrust::device_vector<double> grid(3 * u.size() * v.size() * w.size());

	int block_size = 256;
	int num_blocks = (u.size() * v.size() * w.size() + block_size - 1) / block_size;

	tensorGridKernel << <num_blocks, block_size >> > (
		thrust::raw_pointer_cast(u.data()),
		thrust::raw_pointer_cast(v.data()),
		thrust::raw_pointer_cast(w.data()),
		v.size(),
		w.size(),
		thrust::raw_pointer_cast(grid.data())
		);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Error after kernel tensorGridKernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization (tensorGridKernel): " << cudaGetErrorString(err) << std::endl;
	}
	return grid;
}

#if 0
thrust::device_vector<double> tensorGrid(thrust::device_vector<double>& vecs, thrust::device_vector<int>& sizes, int dim, int num_patch)
{
	int total_points = 0;
	for (int i = 0; i < num_patch; ++i) {
		int patch_offset = i * dim;
		int patch_points = 1;
		for (int j = 0; j < dim; ++j) {
			patch_points *= sizes[patch_offset + j];
		}
		total_points += patch_points;
	}

	thrust::device_vector<double> grid(total_points * dim);

	int block_size = 256;
	int num_blocks = (total_points + block_size - 1) / block_size;

	tensorGridKernel << <num_blocks, block_size >> > (
		thrust::raw_pointer_cast(vecs.data()),
		thrust::raw_pointer_cast(sizes.data()),
		dim,
		num_patch,
		thrust::raw_pointer_cast(grid.data()),
		total_points
		);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Error after kernel tensorGridKernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization (tensorGridKernel): " << cudaGetErrorString(err) << std::endl;
	}
	return grid;
}
#endif

void assemble(
	thrust::device_vector<double>& knots_ref, 
	thrust::device_vector<int>& knot_sizes_ref, 
	thrust::device_vector<int>& knot_degrees_ref, 
	thrust::device_vector<double>& cps_ref, 
	thrust::device_vector<double>& knots, 
	thrust::device_vector<int>& knot_sizes, 
	thrust::device_vector<int>& knot_degrees, 
	thrust::device_vector<double>& cps, 
	thrust::device_vector<int>& dof_indices, 
	int dim, int num_patches)
{
	//int block_size = 256;
	//int num_blocks = (num_eles + block_size - 1) / block_size;

	//assembleKernel << <num_blocks, block_size >> > (
	assembleKernel << <1, 1 >> > (
		thrust::raw_pointer_cast(knots_ref.data()),
		thrust::raw_pointer_cast(knot_sizes_ref.data()),
		thrust::raw_pointer_cast(knot_degrees_ref.data()),
		thrust::raw_pointer_cast(cps.data()),
		thrust::raw_pointer_cast(knots.data()),
		thrust::raw_pointer_cast(knot_sizes.data()),
		thrust::raw_pointer_cast(knot_degrees.data()),
		thrust::raw_pointer_cast(cps.data()),
		thrust::raw_pointer_cast(dof_indices.data()),
		dim,
		num_patches,
		1,
		nullptr,
		nullptr
		);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Error after kernel assembleKernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization (assembleKernel): " << cudaGetErrorString(err) << std::endl;
	}
}
#endif