#define DEBUG

#include "device_launch_parameters.h"

#include <KnotVectorDeviceView.h>
#include <Eigen/Core>
#include <GPUSolver.h>

int main()
{
	int numRefinements = 0;
	double deltaDisplacement = 0.1;
	double maxDisplacement = 0.2;


	int knot_u_order = 1;
	int knot_v_order = 1;
	std::vector<double> knot_u{ 0., 0., 1., 1. };
	std::vector<double> knot_v{ 0., 0., 1., 1. };
	Eigen::MatrixXd control_points(4, 2);
	control_points <<
		0.000, 0.000, 
		//1.000, 0.000, 
		2.000, 0.000, 
		0.000, 1.000,
		//1.000, 1.000,
		2.000, 1.000;
	
	std::vector<double> knot_u2{ 0., 0., 1., 1. };
	std::vector<double> knot_v2{ 0., 0., 1., 1. };
	Eigen::MatrixXd control_points2(4, 2);
	control_points2 <<
		2.000, 0.000, 
		//3.000, 0.000, 
		4.000, 0.000,
		2.000, 1.000,
		//3.000, 1.000,
		4.000, 1.000;
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
		bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, 
                            std::vector<double>{0.0, 0.0}, d);
	std::vector<double> disp{ deltaDisplacement, 0.0 };
    bcInfo.addCondition(1, boundary::east, condition_type::dirichlet, disp, 0);

	Eigen::VectorXd bodyForce(2);
	bodyForce << 0.0, 0.0;

	GPUAssembler assembler(multiPatch, bases, bcInfo, bodyForce);
	//assembler.print();
    GPUSolver solver(assembler);

	assembler.assemble(solver.solutionView(), 0, solver.allFixedDofsView());

    return 0;
}