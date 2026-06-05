#include <GPUSolver.h>
#include <GPUPostProcessor.h>
#include <filesystem>
#include <GPUStrainGradientElasticityAssembler.h>

int main(int argc, char* argv[])
{
    double YM = 1000.0;
	double PR = 0.495;
    double lengthScale = 0.5;

    int numRefinements = std::stoi(argv[1]);
    int numDegElev = std::stoi(argv[2]);

    double initialDeltaDisp = 0.02;
    double maxDisp = 0.2;
    std::vector<int> numPointsPerPatch{ 1000 };

    if (!std::filesystem::exists("./strainGradient_2DTrapezoid"))
		std::filesystem::create_directory("./strainGradient_2DTrapezoid");
	std::string filenameParaview = "strainGradient_2DTrapezoid_";
	std::string outputFolder = "./strainGradient_2DTrapezoid/" + filenameParaview + "output";
    if (!std::filesystem::exists(outputFolder))
		std::filesystem::create_directory(outputFolder);
	std::string fileNameWithPath = outputFolder + "/" + filenameParaview;
	ParaviewCollection collection(fileNameWithPath);

    int knot_u_order = 1;
    int knot_v_order = 1;
	std::vector<double> knot_u{ 0., 0., 1., 1. };
    std::vector<double> knot_v{ 0., 0., 1., 1. };
	Eigen::MatrixXd control_points(4, 2);
    control_points << 0., 0.,
                      1., 0.,
                      0., 1.,
                      1., .5;
    KnotVector u1(knot_u_order,knot_u);
	KnotVector v1(knot_v_order,knot_v);
    Patch patch(u1, v1, control_points);
    MultiPatch geometry;
	geometry.addPatch(patch);
	geometry.computeTopology();
	MultiBasis basisDisplacement(geometry);

    for (int i = 0; i < numDegElev; ++i)
        basisDisplacement.degreeElevate();
    for (int i = 0; i < numRefinements; ++i)
        basisDisplacement.uniformRefine();

    BoundaryConditions bcInfo;
    std::vector<double> zeros{0.0, 0.0};
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, zeros, 0);
    bcInfo.addCondition(0, boundary::south, condition_type::dirichlet, zeros, 1);
    std::vector<double> disp{initialDeltaDisp, 0.0};
    bcInfo.addCondition(0, boundary::east, condition_type::dirichlet, disp, 0);

    Eigen::VectorXd bodyForce(2);
	bodyForce << 0.0, 0.0;

    GPUStrainGradientElasticityAssembler assembler(geometry, basisDisplacement, bcInfo, bodyForce);
    //GPUAssembler assembler(geometry, basisDisplacement, bcInfo, bodyForce);
    assembler.options().setReal("youngs_modulus", YM);
	assembler.options().setReal("poissons_ratio", PR);
    assembler.options().setReal("length_scale", lengthScale);
    std::cout << "Initialized system with " << assembler.numDofs() << " dofs.\n";
    
    GPUSolver solver(assembler);

	GPUPostProcessor postProcessor(assembler, numPointsPerPatch, true, 2);

	MultiPatch displacementHost;
    basisDisplacement.giveBasis(displacementHost, 2);
	GPUFunction displacementFunction(displacementHost);
	postProcessor.addFunction("displacement", &displacementFunction);

	postProcessor.outputToParaview(fileNameWithPath, 0, collection);

    int step = 1;
    double appliedDisp = 0.0;
    double deltaDisp = initialDeltaDisp;
	auto solvestart = std::chrono::high_resolution_clock::now();
    while (abs(appliedDisp - maxDisp) > 1e-6)
    {
        appliedDisp += deltaDisp;
        
	    solver.solve();

        std::cout << "Step " << step << ": applying displacement " << appliedDisp << "\n";
	    auto start = std::chrono::high_resolution_clock::now();
        double smallestEigenvalue = solver.smallestEigenValue();
	    auto end = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double> elapsed = end - start;
        std::cout << "Smallest eigenvalue: " << smallestEigenvalue << "\n";
        std::cout << "Time taken: " << elapsed.count() << " seconds\n";

        assembler.constructSolution(solver.solutionView(), solver.allFixedDofsView(), displacementFunction);
		postProcessor.outputToParaview(fileNameWithPath, step, collection);
        step++;
    }

	collection.save();
	auto solveend = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = solveend - solvestart;
    std::cout << "Total solve time: " << elapsed.count() << " seconds\n";

    return 0;
}
