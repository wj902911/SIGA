#define DEBUG

#include "device_launch_parameters.h"

#include <KnotVectorDeviceView.h>
#include <Eigen/Core>
#include <GPUSolver.h>
#include <GPUPostProcessor.h>
#include <filesystem>

int main(int argc, char* argv[])
{
	if (argc < 3)
    {
        std::cerr << "Usage: ./test_3D <numRefinements> <numDegElev>\n";
        return 1;
    }
	double YM = 1.0;
	double PR = 0.3;

	int numRefinements = std::stoi(argv[1]);
	int numDegElev = std::stoi(argv[2]);
	double deltaDisplacement = 0.1;
	double maxDisplacement = 1.0;
	std::vector<int> numPointsPerPatch{ 10000, 5000 };

	if (!std::filesystem::exists("./TwoPatchesTest_3D"))
		std::filesystem::create_directory("./TwoPatchesTest_3D");
	std::string filenameParaview = "TwoPatchesTest_3D_";
	std::string outputFolder = "./TwoPatchesTest_3D/" + filenameParaview + "output";
	if (!std::filesystem::exists(outputFolder))
		std::filesystem::create_directory(outputFolder);
	std::string fileNameWithPath = outputFolder + "/" + filenameParaview;
	ParaviewCollection collection(fileNameWithPath);


	int knot_u_order = 1;
	int knot_v_order = 1;
	int knot_w_order = 1;
	std::vector<double> knot_u{ 0., 0., 0.5, 1., 1. };
	std::vector<double> knot_v{ 0., 0., 1., 1. };
	std::vector<double> knot_w{ 0., 0., 1., 1. };
	Eigen::MatrixXd control_points(12, 3);
	control_points <<
		0.000, 0.000, 0.000,
		1.000, 0.000, 0.000,
		2.000, 0.000, 0.000,
		0.000, 1.000, 0.000,
		1.000, 1.000, 0.000,
		2.000, 1.000, 0.000,
		0.000, 0.000, 1.000,
		1.000, 0.000, 1.000,
		2.000, 0.000, 1.000,
		0.000, 1.000, 1.000,
		1.000, 1.000, 1.000,
		2.000, 1.000, 1.000;
	
	std::vector<double> knot_u2{ 0., 0., 1., 1. };
	std::vector<double> knot_v2{ 0., 0., 1., 1. };
	std::vector<double> knot_w2{ 0., 0., 1., 1. };
	Eigen::MatrixXd control_points2(8, 3);
	control_points2 <<
		2.000, 0.000, 0.000,
		3.000, 0.000, 0.000,
		2.000, 1.000, 0.000,
		3.000, 1.000, 0.000,
		2.000, 0.000, 1.000,
		3.000, 0.000, 1.000,
		2.000, 1.000, 1.000,
		3.000, 1.000, 1.000;
#if 0
    std::vector<double> knot_u3{ 0., 0., 1., 1. };
	std::vector<double> knot_v3{ 0., 0., 1., 1. };
	Eigen::MatrixXd control_points3(4, 2);
	control_points3 <<
		4.000, 0.000, 
		//5.000, 0.000, 
		6.000, 0.000,
		4.000, 1.000,
		//5.000, 1.000,
		6.000, 1.000;
#endif

    KnotVector u1(knot_u_order,knot_u);
	KnotVector v1(knot_v_order,knot_v);
	KnotVector w1(knot_w_order,knot_w);
	KnotVector u2(knot_u_order,knot_u2);
	KnotVector v2(knot_v_order,knot_v2);
	KnotVector w2(knot_w_order,knot_w2);
    //KnotVector u3(knot_u_order,knot_u3);
	//KnotVector v3(knot_v_order,knot_v3);

	//std::vector<double> breaks = u1.breaks();

    Patch patch(u1, v1, w1, control_points);
	Patch patch2(u2, v2, w2, control_points2);
	//Patch patch3(u3, v3, control_points3);

    MultiPatch multiPatch;
	multiPatch.addPatch(patch);
	multiPatch.addPatch(patch2);
    //multiPatch.addPatch(patch3);
	multiPatch.computeTopology();

	MultiBasis bases(multiPatch);

	for (int i = 0; i < numDegElev; ++i)
		bases.degreeElevate(false);

    for (int r = 0; r < numRefinements; ++r)
		bases.uniformRefine();

    BoundaryConditions bcInfo;
	for (int d = 0; d < 3; ++d)
		bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, 
                            std::vector<double>{0.0, 0.0, 0.0}, d);
	std::vector<double> disp{ deltaDisplacement, 0.0, 0.0 };
    bcInfo.addCondition(1, boundary::east, condition_type::dirichlet, disp, 0);

	Eigen::VectorXd bodyForce(3);
	bodyForce << 0.0, 0.0, 0.0;

	std::cout << "Initializing assembler..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	GPUAssembler assembler(multiPatch, bases, bcInfo, bodyForce);
	//std::cout << "Setting material properties..." << std::endl;
	assembler.options().setReal("youngs_modulus", YM);
	assembler.options().setReal("poissons_ratio", PR);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Initialized assembler in " << elapsed.count() << " s." << std::endl;
	//assembler.print();
    GPUSolver solver(assembler);

	std::cout << "Initializing post-processor..." << std::endl;
	start = std::chrono::high_resolution_clock::now();
	GPUPostProcessor postProcessor(assembler, numPointsPerPatch, true, 2);
	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start;
	std::cout << "Initialized post-processor in " << elapsed.count() << " s." << std::endl;

	MultiPatch displacementHost;
	bases.giveBasis(displacementHost, 3);
	//MultiPatchDeviceData displacementDeviceData(displacementHost);
	//assembler.constructSolution(solver.solutionView(), 
	//                            solver.allFixedDofsView(), 
	//							displacementDeviceData.deviceView());

	

	GPUDisplacementFunction displacementFunction(displacementHost);

	//printKernel<<<1, 1>>>(displacementFunction.displacementDeviceView());
	//cudaDeviceSynchronize();

	postProcessor.addFunction("displacement", &displacementFunction);
	collection.initalize();
	start = std::chrono::high_resolution_clock::now();
	postProcessor.outputToParaview(fileNameWithPath, 0, collection);
	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start;
	std::cout << "Output initial geometry in " << elapsed.count() << " s." << std::endl;

	std::cout << "Initialized system with " << assembler.numElements() << " elements and " << assembler.numDofs() << " dofs." << std::endl;

	int step = 1;
	double totalDisplacement = deltaDisplacement;
	auto solvestart = std::chrono::high_resolution_clock::now();
	while (totalDisplacement <= maxDisplacement)
	{
		std::cout << "Step " << step << " with displacement: " << totalDisplacement
		<< " and step length: " << deltaDisplacement << std::endl;
		solver.solve();

		assembler.constructSolution(solver.solutionView(),
			solver.allFixedDofsView(),
			displacementFunction);
		start = std::chrono::high_resolution_clock::now();
		postProcessor.outputToParaview(fileNameWithPath, step, collection);
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - start;
		std::cout << "Output deformed geometry in " << elapsed.count() << " s." << std::endl;

		totalDisplacement += deltaDisplacement;
		step++;
	}
	collection.save();

	auto solveend = std::chrono::high_resolution_clock::now();
	elapsed = solveend - solvestart;
	std::cout << "Solved the system in " << elapsed.count() << " s." << std::endl;
    return 0;
}