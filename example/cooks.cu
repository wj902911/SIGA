#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>

#include <iostream>

#include <Solver.h>
//#include <Assembler.h>
#include <MultiPatch_d.h>
#include <MultiBasis_d.h>
#include <MultiBasis.h>
#include <GaussPoints_d.h>
#include <Postprocessor.h>

#include <filesystem>

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
//template __global__ void destructKernel<KnotVector_d>(KnotVector_d* a, size_t n);
//template __global__ void destructKernel<TensorBsplineBasis_d>(TensorBsplineBasis_d* a, size_t n);
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
//template __global__ void parallPlus<double>(double* a, double* b, double* c, int n);
template __global__ void parallPlus<double>(double* a, double b, double* c, int n);
template __global__ void squareNormKernel<double>(double* a, double* result, int n);

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
	int numRefinements = 1;
	double deltaDisplacement = 0.1;
	double maxDisplacement = 0.2;
	Eigen::VectorXi numPoints(2);
	numPoints << 10, 10;

	if (!std::filesystem::exists("./TwoPatchesTest"))
		std::filesystem::create_directory("./TwoPatchesTest");
	std::string filenameParaview = "TwoPatchesTest_";
	std::string outputFolder = "./TwoPatchesTest/" + filenameParaview + "output";
	if (!std::filesystem::exists(outputFolder))
		std::filesystem::create_directory(outputFolder);
	std::string fileNameWithPath = outputFolder + "/" + filenameParaview;
	ParaviewCollection collection(fileNameWithPath);

	int knot_u_order = 1;
	int knot_v_order = 1;
	std::vector<double> knot_u{ 0., 0., 1., 1. };
	std::vector<double> knot_v{ 0., 0., 1., 1. };
	Eigen::MatrixXd control_points(4, 2);
	control_points <<
		0.000, 0.000, 
		2.000, 0.000, 
		0.000, 1.000,
		2.000, 1.000;

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
		2.000, 0.000, 
		4.000, 0.000,
		2.000, 1.000,
		4.000, 1.000;
#if 0
	std::vector<double> knot_u3{ 0., 0., 1., 1. };
	std::vector<double> knot_v3{ 0., 0., 1., 1. };
	Eigen::MatrixXd control_points3(4, 2);
	control_points3 <<
		4.000, 0.000, 
		6.000, 0.000,
		4.000, 1.000,
		6.000, 1.000;
#endif

	KnotVector u1(knot_u_order,knot_u);
	KnotVector v1(knot_v_order,knot_v);
	KnotVector u2(knot_u_order,knot_u2);
	KnotVector v2(knot_v_order,knot_v2);
	//KnotVector u3(knot_u_order,knot_u3);
	//KnotVector v3(knot_v_order,knot_v3);
	Patch patch(u1, v1, control_points);
	Patch patch2(u2, v2, control_points2);
	//Patch patch3(u3, v3, control_points3);

	MultiPatch multiPatch;
	multiPatch.addPatch(patch);
	multiPatch.addPatch(patch2);
	//multiPatch.addPatch(patch3);
	multiPatch.computeTopology();

	MultiBasis bases(multiPatch);

	for (int r = 0; r < numRefinements; ++r)
		bases.uniformRefine();

	BoundaryConditions bcInfo;
	for (int d = 0; d < 2; ++d)
		bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, std::vector<double>{0.0, 0.0}, d);

	//std::vector<double> neumannValue{ 100e4, 0.0 };
	std::vector<double> disp{ deltaDisplacement, 0.0 };
    //bcInfo.addCondition(1, boundary::east, condition_type::neumann, neumannValue);
    bcInfo.addCondition(1, boundary::east, condition_type::dirichlet, disp, 0);

	Eigen::VectorXd bodyForce(2);
	bodyForce << 0.0, 0.0;


	int step = 0;

	Assembler assembler(multiPatch, bases, bcInfo, bodyForce);
	//DeviceVector<double> solution(assembler.numDofs());
	//solution.setZero();
	//assembler.assemble(solution);
	Solver solver(assembler);

#if 1
	MultiPatch displacement;
	solver.constructSolution(displacement);
	//std::cout << displacement.patch(0).getControlPoints() << std::endl << std::endl;
	//std::cout << displacement.patch(1).getControlPoints() << std::endl << std::endl;

	DisplacementFunction displacementFunction(displacement);

	PostProcessor postProcessor(multiPatch);
	postProcessor.addFunction("Displacement", &displacementFunction);
	

	collection.initalize();
    cudaDeviceSetLimit(cudaLimitStackSize, 4*1024);
	postProcessor.outputToParaview(fileNameWithPath, numPoints, step, collection);

	double totalDisplacement = 0.0;
	while (totalDisplacement <= maxDisplacement)
	{
		std::cout << "Step " << step << ":" << std::endl;
		assembler.refresh();
		solver.solve();
		solver.constructSolution(displacement);

		postProcessor.outputToParaview(fileNameWithPath, numPoints, step + 1, collection);

		totalDisplacement += deltaDisplacement;
		step++;
	}

#else
	solver.solve();

	MultiPatch displacement;
	solver.constructSolution(displacement);
	DisplacementFunction displacementFunction(displacement);

	collection.initalize();
	PostProcessor postProcessor(multiPatch);
	postProcessor.addFunction("Displacement", &displacementFunction);
	postProcessor.outputToParaview(fileNameWithPath, numPoints, step, collection);
#endif
	collection.save();



	return 0;
}