#include "device_launch_parameters.h"

#include <Eigen/Core>
#include <GPUPostProcessor.h>
#include <GPUSolver.h>
#include <GPUStrainGradientElasticityAssembler.h>
#include <TeeLogger.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char* argv[])
{
    const double length = 2.0;
    const double height = 0.1;
    const double YM = 1.725;
    const double PR = 0.3;

    const int numRefinements = argc > 1 ? std::stoi(argv[1]) : 4;
    const int numDegreeElevations = argc > 2 ? std::stoi(argv[2]) : 1;
    const double pointLoadY = argc > 3 ? std::stod(argv[3]) : -1.0e-8;
    const double lengthScale = argc > 4 ? std::stod(argv[4]) : 0.1;
    const double initialLoadStep = argc > 5 ? std::stod(argv[5]) : 0.1;
    const int targetNumIterations = argc > 6 ? std::stoi(argv[6]) : 5;
    if (initialLoadStep <= 0.0)
        throw std::invalid_argument("initialLoadStep must be positive.");

    const std::string rootFolder = "./strainGradient_2DBeamBending_cornerPointLoad_output";
    std::string outputFolder = rootFolder + "/strainGradient_2DBeamBending_cornerPointLoad_output";
    for (int i = 1; i < argc; ++i)
        outputFolder += "_" + std::string(argv[i]);
    std::filesystem::create_directories(outputFolder);
    TeeLogger log(outputFolder + "/log.txt");

    std::cout << "2D strain-gradient cantilever beam bending with corner point load\n";
    std::cout << "Refinements: " << numRefinements
              << ", degree elevations: " << numDegreeElevations
              << ", southeast corner point load: (0, " << pointLoadY << ")\n";
    std::cout << "Material: Y = " << YM << ", nu = " << PR
              << ", length scale = " << lengthScale << "\n";
    std::cout << "Initial load step factor: " << initialLoadStep << "\n";
    std::cout << "Target solver iterations per load step: " << targetNumIterations << "\n";

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

    const std::vector<double> cornerPointLoad{0.0, pointLoadY};
    bcInfo.addCondition(0, boundary::southeast, condition_type::neumann, cornerPointLoad);

    Eigen::VectorXd bodyForce(2);
    bodyForce << 0.0, 0.0;

    GPUStrainGradientElasticityAssembler assembler(geometry, basis, bcInfo, bodyForce);
    assembler.options().setReal("youngs_modulus", YM);
    assembler.options().setReal("poissons_ratio", PR);
    assembler.options().setReal("length_scale", lengthScale);
    assembler.options().setReal("neumann_load_scaling", 0.0);
    assembler.options().setInt("material_law", 1);
    std::cout << "Initialized system with " << assembler.numDofs() << " dofs.\n";

    GPUSolver solver(assembler);

    MultiPatch displacementHost;
    basis.giveBasis(displacementHost, 2);
    GPUFunction displacementFunction(displacementHost);
    assembler.constructSolution(solver.solutionView(),
                                solver.allFixedDofsView(),
                                displacementFunction);

    MultiPatch firstPiolaHost;
    basis.giveBasis(firstPiolaHost, 4);
    GPUFunction firstPiolaFunction(firstPiolaHost);

    MultiPatch cauchyStressHost;
    basis.giveBasis(cauchyStressHost, assembler.dimTensor());
    GPUFunction cauchyStressFunction(cauchyStressHost);
    assembler.constructStrainGradientStressFunctions(displacementFunction,
                                                     firstPiolaFunction,
                                                     cauchyStressFunction);

    const std::string filePrefix =
        outputFolder + "/strainGradient_2DBeamBending_cornerPointLoad_";
    ParaviewCollection collection(filePrefix);

    std::vector<int> numPointsPerPatch{2000};
    GPUPostProcessor postProcessor(assembler, numPointsPerPatch, true, 2);
    postProcessor.addFunction("displacement", &displacementFunction);
    postProcessor.addFunction("first_piola", &firstPiolaFunction);
    postProcessor.addFunction("stress_cauchy", &cauchyStressFunction);

    auto writeParaviewOutput = [&](int outputStep, double loadFactor)
    {
        assembler.constructSolution(solver.solutionView(),
                                    solver.allFixedDofsView(),
                                    displacementFunction);
        assembler.constructStrainGradientStressFunctions(displacementFunction,
                                                         firstPiolaFunction,
                                                         cauchyStressFunction);
        postProcessor.outputToParaview(filePrefix, outputStep, collection);
        std::cout << "Wrote result output " << outputStep
                  << " at load factor " << loadFactor
                  << " and point load (0, " << loadFactor * pointLoadY << ").\n";
    };

    int outputStep = 0;
    writeParaviewOutput(outputStep++, 0.0);

    const double loadTolerance = 1e-12;
    const double minLoadStep = 1e-8;
    double loadFactor = 0.0;
    double loadStep = std::min(initialLoadStep, 1.0);
    int step = 1;

    while (std::abs(loadFactor - 1.0) > loadTolerance)
    {
        if (loadFactor + loadStep > 1.0)
            loadStep = 1.0 - loadFactor;

        Eigen::VectorXd previousSolution;
        solver.solutionToHost(previousSolution);
        const double previousLoadFactor = loadFactor;

        loadFactor += loadStep;
        assembler.options().setReal("neumann_load_scaling", loadFactor);

        std::cout << "Step " << step << ": load factor " << loadFactor
                  << ", point load (0, " << loadFactor * pointLoadY
                  << "), step length " << loadStep << "\n";

        solver.solve();
        const int stepNumIterations = solver.numIterations();

        if (!solver.isConverged())
        {
            std::cout << "Step " << step << " did not converge after "
                      << stepNumIterations << " iterations. Reducing load step.\n";
            solver.setSolutionFromHost(previousSolution);
            loadFactor = previousLoadFactor;
            assembler.options().setReal("neumann_load_scaling", loadFactor);
            loadStep *= 0.5;
            if (loadStep < minLoadStep)
                throw std::runtime_error("Load step became too small before convergence.");
            continue;
        }

        writeParaviewOutput(outputStep++, loadFactor);

        if (targetNumIterations > 0 && std::abs(loadFactor - 1.0) > loadTolerance)
        {
            const double oldLoadStep = loadStep;
            if (stepNumIterations > targetNumIterations)
            {
                const double factor = std::max(
                    0.5,
                    std::sqrt(static_cast<double>(targetNumIterations) /
                              static_cast<double>(stepNumIterations)));
                loadStep *= factor;
                std::cout << "Reducing next load step to " << loadStep
                          << " because " << stepNumIterations
                          << " iterations exceeded target "
                          << targetNumIterations << ".\n";
            }
            else if (stepNumIterations < targetNumIterations)
            {
                const int denominator = std::max(1, stepNumIterations);
                const double factor = std::min(
                    2.0,
                    std::sqrt(static_cast<double>(targetNumIterations) /
                              static_cast<double>(denominator)));
                loadStep *= factor;
                std::cout << "Increasing next load step to " << loadStep
                          << " because " << stepNumIterations
                          << " iterations were below target "
                          << targetNumIterations << ".\n";
            }
            else
            {
                std::cout << "Maintaining next load step " << loadStep
                          << " because solver iterations matched target "
                          << targetNumIterations << ".\n";
            }

            const double remainingLoad = 1.0 - loadFactor;
            if (loadStep > remainingLoad)
            {
                loadStep = remainingLoad;
                if (loadStep != oldLoadStep)
                    std::cout << "Capping next load step to remaining load factor "
                              << loadStep << ".\n";
            }
        }

        ++step;
    }

    collection.save();

    Eigen::VectorXd solution;
    solver.solutionToHost(solution);
    std::cout << "Solved at full load. Solution norm: " << solution.norm() << "\n";
    std::cout << "Paraview output: " << outputFolder << "\n";
    return solver.isConverged() ? 0 : 2;
}
