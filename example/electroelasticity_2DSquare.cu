#include <GPUSolver.h>
#include <GPUPostProcessor.h>
#include <filesystem>
#include <GPUElectroelasticityAssembler.h>

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: ./electroelasticity_2DSquare <numRefinements> <numDegElev>\n";
        return 1;
    }

	double YM = 1000.0;
	double PR = 0.495;
    double rp = 4.0;
    double fsp = 8.8542;
    
	int numRefinements = std::stoi(argv[1]);
    int numDegElev = std::stoi(argv[2]);
	double initialDeltaElePotential = 0.1;
    double maxElePotential = 1.5;
	std::vector<int> numPointsPerPatch{ 1000 };

    int critNumIter = 7;

    if (!std::filesystem::exists("./electroelasticity_2DSquare"))
		std::filesystem::create_directory("./electroelasticity_2DSquare");
	std::string filenameParaview = "electroelasticity_2DSquare_";
	std::string outputFolder = "./electroelasticity_2DSquare/" + filenameParaview + "output";
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
                      1., 1.;
    KnotVector u1(knot_u_order,knot_u);
	KnotVector v1(knot_v_order,knot_v);
    Patch patch(u1, v1, control_points);
    MultiPatch geometry;
	geometry.addPatch(patch);
	geometry.computeTopology();
	MultiBasis basisDisplacement(geometry);
    MultiBasis basisElePotential(geometry);

	for (int i = 0; i < numDegElev; ++i) {
        basisDisplacement.degreeElevate(true, 1);
        basisElePotential.degreeElevate(true, 1);
    }
    for (int i = 0; i < numRefinements; ++i) {
        basisDisplacement.uniformRefine(1);
        basisElePotential.uniformRefine(1);
    }
    BoundaryConditions bcInfo;
    std::vector<double> zeros{0.0, 0.0};
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, zeros, 0);
    bcInfo.addCondition(0, boundary::south, condition_type::dirichlet, zeros, 1);
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, zeros, 2);
    std::vector<double> voltage{initialDeltaElePotential, 0.0};
    bcInfo.addCondition(0, boundary::east, condition_type::dirichlet, voltage, 2);

	Eigen::VectorXd bodyForce(2);
	bodyForce << 0.0, 0.0;

    GPUElectroelasticityAssembler assembler(geometry, basisDisplacement, basisElePotential, bcInfo, bodyForce);
    assembler.options().setReal("youngs_modulus", YM);
	assembler.options().setReal("poissons_ratio", PR);
    assembler.options().setReal("dielectric_permittivity", fsp * rp);

    GPUSolver solver(assembler);
    //voltage[0] = 0.05;
    //assembler.refreshFixedDofs();
	GPUPostProcessor postProcessor(assembler, numPointsPerPatch, true, 2);

	MultiPatch displacementHost;
    basisDisplacement.giveBasis(displacementHost, 2);
	GPUDisplacementFunction displacementFunction(displacementHost);
	postProcessor.addFunction("displacement", &displacementFunction);

	postProcessor.outputToParaview(fileNameWithPath, 0, collection);


	int step = 1;
    double appliedVoltage = 0;
	double voltagePerStep = initialDeltaElePotential;
    bool endingSimulation = false;
	auto solvestart = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd hostOldSol;
    solver.solutionToHost(hostOldSol);
    Eigen::VectorXd hostOldFixedDofs;
    solver.fixedDofsToHost(hostOldFixedDofs);
    while (abs(appliedVoltage) < abs(maxElePotential) && !endingSimulation)
    {
        if (abs(appliedVoltage + voltagePerStep) > abs(maxElePotential))
        {
            voltagePerStep = maxElePotential - appliedVoltage;
            voltage[0] = voltagePerStep;
            assembler.refreshFixedDofs();
            appliedVoltage = maxElePotential;
            endingSimulation = true;
        }
        else
        {
            appliedVoltage += voltagePerStep;
        }

        std::cout << "Step " << step << ": Applying voltage " << appliedVoltage << "\n";
	    solver.solve();
        if(!solver.isConverged())
        {
            if (endingSimulation)
                endingSimulation = false;
            appliedVoltage -= voltagePerStep;
            voltagePerStep *= 0.5;
            voltage[0] = voltagePerStep;
            assembler.refreshFixedDofs();
            solver.setSolutionFromHost(hostOldSol);
            solver.setFixedDofsFromHost(hostOldFixedDofs);
            std::cout << "Solver did not converge at step " << step << ". Reverting to previous solution and reducing voltage per step to " << voltagePerStep << ".\n";
            continue;
        }
        solver.solutionToHost(hostOldSol);
        //std::cout << "Old solution:\n" << hostOldSol << "\n";
        solver.fixedDofsToHost(hostOldFixedDofs);
        //std::cout << "Old fixed DoFs:\n" << hostOldFixedDofs.transpose() << "\n";
        
        assembler.constructSolution(solver.solutionView(), solver.allFixedDofsView(), displacementFunction);
		postProcessor.outputToParaview(fileNameWithPath, step, collection);

        int numIter = solver.numIterations();
        if (numIter > critNumIter && !endingSimulation)
        {
            voltagePerStep *= sqrt(static_cast<double>(critNumIter) / numIter);
            //voltagePerStep *= static_cast<double>(critNumIter) / static_cast<double>(numIter);
            voltage[0] = voltagePerStep;
            assembler.refreshFixedDofs();
            std::cout << "Adjusting voltage step to " << voltagePerStep << " for next step due to convergence issues.\n"; 
        }

        step++;
    }

	collection.save();
	auto solveend = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = solveend - solvestart;
    std::cout << "Solver time: " << elapsed.count() << " seconds\n";
    return 0;
}