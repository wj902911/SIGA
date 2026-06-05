#include "device_launch_parameters.h"

#include <Eigen/Core>
#include <GPUPostProcessor.h>
#include <GPUSolver.h>
#include <filesystem>
#include <iostream>
#include <TeeLogger.h>

int main(int argc, char* argv[])
{
    const double length = 10.0;
    const double height = 1.0;
    const double YM = 1.0e5;
    const double PR = 0.3;

    const int numRefinements = argc > 1 ? std::stoi(argv[1]) : 4;
    const int numDegreeElevations = argc > 2 ? std::stoi(argv[2]) : 1;
    const double tractionY = argc > 3 ? std::stod(argv[3]) : -100.0;

    const std::string rootFolder = "./2DBeamBending_traction";
    const std::string outputFolder = rootFolder + "/2DBeamBending_traction_output";
    std::filesystem::create_directories(outputFolder);
    TeeLogger log(outputFolder + "/log.txt");

    std::cout << "2D cantilever beam bending with end traction\n";
    std::cout << "Refinements: " << numRefinements
              << ", degree elevations: " << numDegreeElevations
              << ", east-edge traction: (0, " << tractionY << ")\n";

    const int knot_u_order = 1;
    const int knot_v_order = 1;
    std::vector<double> knot_u{0.0, 0.0, 1.0, 1.0};
    std::vector<double> knot_v{0.0, 0.0, 1.0, 1.0};

    Eigen::MatrixXd controlPoints(4, 2);
    controlPoints <<
        0.0,    0.0,
        length, 0.0,
        0.0,    height,
        length, height;

    KnotVector u(knot_u_order, knot_u);
    KnotVector v(knot_v_order, knot_v);
    Patch patch(u, v, controlPoints);

    MultiPatch geometry;
    geometry.addPatch(patch);
    geometry.computeTopology();

    MultiBasis basis(geometry);
    for (int i = 0; i < numDegreeElevations; ++i)
        basis.degreeElevate(false);
    for (int r = 0; r < numRefinements; ++r)
        basis.uniformRefine();

    BoundaryConditions bcInfo;
    const std::vector<double> zeros{0.0, 0.0};
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, zeros, 0);
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, zeros, 1);

    const std::vector<double> endTraction{0.0, tractionY};
    bcInfo.addCondition(0, boundary::east, condition_type::neumann, endTraction, 0);

    Eigen::VectorXd bodyForce(2);
    bodyForce << 0.0, 0.0;

    GPUAssembler assembler(geometry, basis, bcInfo, bodyForce);
    assembler.options().setReal("youngs_modulus", YM);
    assembler.options().setReal("poissons_ratio", PR);
    std::cout << "Initialized system with " << assembler.numDofs() << " dofs.\n";

    GPUSolver solver(assembler);
    solver.solve();

    MultiPatch displacementHost;
    basis.giveBasis(displacementHost, 2);
    GPUFunction displacementFunction(displacementHost);
    assembler.constructSolution(solver.solutionView(),
                                solver.allFixedDofsView(),
                                displacementFunction);

    MultiPatch cauchyStressHost;
    basis.giveBasis(cauchyStressHost, assembler.dimTensor());
    GPUFunction cauchyStressFunction(cauchyStressHost);
    assembler.constructCauchyStressFunction(displacementFunction, cauchyStressFunction);

    const std::string filePrefix = outputFolder + "/2DBeamBending_traction_";
    ParaviewCollection collection(filePrefix);

    std::vector<int> numPointsPerPatch{2000};
    GPUPostProcessor postProcessor(assembler, numPointsPerPatch, true, 2);
    postProcessor.addFunction("displacement", &displacementFunction);
    postProcessor.addFunction("stress_cauchy", &cauchyStressFunction);
    postProcessor.outputToParaview(filePrefix, 0, collection);
    collection.save();

    Eigen::VectorXd solution;
    solver.solutionToHost(solution);
    std::cout << "Solved. Solution norm: " << solution.norm() << "\n";
    std::cout << "Paraview output: " << outputFolder << "\n";
    return 0;
}
