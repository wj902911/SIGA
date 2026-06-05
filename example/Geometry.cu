#include "device_launch_parameters.h"
#include <GPUPostProcessor.h>
#include <filesystem>

int main(int argc, char* argv[])
{
    MultiPatch multiPatch;
#if 1 // 1 1st order; 0 2nd order
    int knot_u_order = 1;
	int knot_v_order = 1;
	int knot_w_order = 1; // 3D case
	std::vector<double> knot_u{ 0., 0., 1., 1. };
	std::vector<double> knot_v{ 0., 0., 1., 1. };
	std::vector<double> knot_w{ 0., 0., 1., 1. }; // 3D case
#if 0 // 1 2D case; 0 3D case
#if 1
	Eigen::MatrixXd control_points(4, 2);
	control_points <<
		0.000, 0.000, 
		//1.000, 0.000, 
		1.000, 0.000, 
		0.000, 1.000,
		//1.000, 1.000,
		1.500, 1.500;
#else
	Eigen::MatrixXd control_points(9, 2);
	control_points <<
		0.000, 0.000, 
		0.500, 0.000, 
		1.000, 0.000, 
		0.000, 0.500,
		0.500, 0.500,
		1.000, 0.500,
		0.000, 1.000,
		0.500, 1.000,
		1.500, 1.500;
#endif
#else
	Eigen::MatrixXd control_points(8, 3);
	control_points <<
		0.000, 0.000, 0.000,
		1.000, 0.000, 0.000,
		//2.000, 0.000, 0.000,
		0.000, 1.000, 0.000,
		1.000, 1.000, 0.000,
		//2.000, 1.000, 0.000,
		0.000, 0.000, 1.000,
		1.000, 0.000, 1.000,
		//2.000, 0.000, 1.000,
		0.000, 1.000, 1.000,
		1.000, 1.000, 1.000;
		//2.000, 1.000, 1.000;
#endif
#else
	int knot_u_order = 2;
	int knot_v_order = 2;
	std::vector<double> knot_u{ 0., 0., 0., 1., 1., 1. };
	std::vector<double> knot_v{ 0., 0., 0., 1., 1., 1. };
	Eigen::MatrixXd control_points(9, 2);
	control_points <<
		0.000, 0.000, 
		0.500, 0.000, 
		1.000, 0.000, 
		0.000, 0.500,
		0.500, 0.500,
		1.000, 0.500,
		0.000, 1.000,
		0.500, 1.000,
		1.500, 1.500;
#endif
    KnotVector u1(knot_u_order,knot_u);
	KnotVector v1(knot_v_order,knot_v);
	KnotVector w1(knot_w_order,knot_w); // 3D case
    //Patch patch(u1, v1, control_points);
    Patch patch(u1, v1, w1, control_points); // 3D case
	multiPatch.addPatch(patch);

#if 1 // 1 3D 2nd patch; 0 only 1 patch
    std::vector<double> knot_u2{ 0., 0., 1., 1. };
	std::vector<double> knot_v2{ 0., 0., 1., 1. };
	std::vector<double> knot_w2{ 0., 0., 1., 1. };
	Eigen::MatrixXd control_points2(8, 3);
	control_points2 <<
		1.000, 0.000, 0.000,
		2.000, 0.000, 0.000,
		1.000, 1.000, 0.000,
		2.000, 1.000, 0.000,
		1.000, 0.000, 1.000,
		2.000, 0.000, 1.000,
		1.000, 1.000, 1.000,
		2.000, 1.000, 1.000;
    KnotVector u2(knot_u_order,knot_u2);
	KnotVector v2(knot_v_order,knot_v2);
	KnotVector w2(knot_w_order,knot_w2);
	Patch patch2(u2, v2, w2, control_points2);
	multiPatch.addPatch(patch2);
#endif

	multiPatch.computeTopology();
	MultiBasis bases(multiPatch);

    int numPoints = 1000;
    std::vector<int> numPointsPerPatch;
    for (int i = 0; i < multiPatch.getNumPatches(); ++i)
        numPointsPerPatch.push_back(numPoints);

     if (!std::filesystem::exists("./Geometry"))
		std::filesystem::create_directory("./Geometry");
	std::string filenameParaview = "Geometry_";
	std::string outputFolder = "./Geometry/" + filenameParaview + "output";
	if (!std::filesystem::exists(outputFolder))
		std::filesystem::create_directory(outputFolder);
	std::string fileNameWithPath = outputFolder + "/" + filenameParaview;
	ParaviewCollection collection(fileNameWithPath);
	//collection.initalize();

    BoundaryConditions bcInfo;
    //std::vector<double> zeros{0.0, 0.0};
    //for (int d = 0; d < 2; ++d)
	//	bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, zeros, d);
	//std::vector<double> disp { 0.1, 0.0 };
    //bcInfo.addCondition(1, boundary::east, condition_type::dirichlet, disp, 0);

	Eigen::VectorXd bodyForce(2);
	//bodyForce << 0.0, 0.0;
    
	GPUAssembler assembler(multiPatch, bases, bcInfo, bodyForce);
	GPUPostProcessor postProcessor(assembler, numPointsPerPatch, true, 20);

    //MultiPatch displacementHost;
	//bases.giveBasis(displacementHost, 2);
	//GPUFunction displacementFunction(displacementHost);
	//postProcessor.addFunction("displacement", &displacementFunction);

	postProcessor.outputToParaview(fileNameWithPath, 0, collection);
	collection.save();

    std::cout << "Geometry is generated." << std::endl;
    return 0;
}
