#include <GPUSolver.h>
#include <GPUPostProcessor.h>
#include <GPUFlexoelectriciyAssembler.h>
#include <TeeLogger.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

//#define FLEXO_NUMERIC_JACOBIAN_TEST

int main(int argc, char* argv[])
{
    try
    {
    const double length = argc > 1 ? std::stod(argv[1]) : 2.0;
    const double height = argc > 2 ? std::stod(argv[2]) : 0.1;
    const int numEle_L = argc > 3 ? std::stoi(argv[3]) : 200;
    const int numEle_H = argc > 4 ? std::stoi(argv[4]) : 10;
    const int numDegreeElevations = argc > 5 ? std::stoi(argv[5]) : 2;
    const double pointLoadY = argc > 6 ? std::stod(argv[6]) : -2.0e-5;
    const double muL = argc > 7 ? std::stod(argv[7]) : 0.0;
    const double muT = argc > 8 ? std::stod(argv[8]) : 10.0;
    const double muS = argc > 9 ? std::stod(argv[9]) : 0.0;
    const double lengthScale = argc > 10 ? std::stod(argv[10]) : 0.1;
    const double YM = argc > 11 ? std::stod(argv[11]) : 1.725;
    const double PR = argc > 12 ? std::stod(argv[12]) : 0.3;
    const double dielectricPermittivity = argc > 13 ? std::stod(argv[13]) : 0.092;
    const double initialLoadStep = argc > 14 ? std::stod(argv[14]) : 0.1;
    const int targetNumIterations = argc > 15 ? std::stoi(argv[15]) : 5;
    const int materialLaw = argc > 16 ? std::stoi(argv[16]) : 1;
    const bool adaptiveLoadStep = argc > 17 ? std::stoi(argv[17]) != 0 : true;
    const int includeHbarFlexoCorrection = argc > 18 ? std::stoi(argv[18]) : 0;
    const double meterToMicrometer = 1.0e-6;
    const double dielectricPermittivityModel = dielectricPermittivity * meterToMicrometer;
    const double muLModel = muL * meterToMicrometer;
    const double muTModel = muT * meterToMicrometer;
    const double muSModel = muS * meterToMicrometer;
    if (initialLoadStep <= 0.0)
        throw std::invalid_argument("initialLoadStep must be positive.");
    if (materialLaw != 0 && materialLaw != 1)
        throw std::invalid_argument("materialLaw must be 0 (StVK) or 1 (neo-Hookean).");

    const std::string rootFolder = "./flexoelectricity_CodonyOpenCircuitCantilever";
    std::string outputFolder = rootFolder + "/output";
    for (int i = 1; i < argc; ++i)
        outputFolder += "_" + std::string(argv[i]);
    std::filesystem::create_directories(outputFolder);
    TeeLogger log(outputFolder + "/log.txt");

    std::cout << "Codony et al. Section 5.1 open-circuit flexoelectric cantilever\n";
    std::cout << "Beam: L = " << length << ", H = " << height << "\n";
    std::cout << "Elements: " << numEle_L << " x " << numEle_H
              << ", degree = " << numDegreeElevations + 1 << "\n";
    std::cout << "Material input: Y = " << YM << " GPa, nu = " << PR
              << ", epsilon = " << dielectricPermittivity << " nJ/(V^2 m)"
              << ", length scale = " << lengthScale << "\n";
    std::cout << "Material model units: epsilon = " << dielectricPermittivityModel
              << " nJ/(V^2 um)\n";
    std::cout << "Material law: "
              << (materialLaw == 0 ? "St. Venant-Kirchhoff" : "neo-Hookean")
              << " (" << materialLaw << ")\n";
    std::cout << "Flexoelectric tensor input: mu_L = " << muL
              << ", mu_T = " << muT
              << ", mu_S = " << muS << " nJ/(V m)\n";
    std::cout << "Flexoelectric tensor model units: mu_L = " << muLModel
              << ", mu_T = " << muTModel
              << ", mu_S = " << muSModel << " nJ/(V um)\n";
    std::cout << "Southeast corner point load: (0, " << pointLoadY << ")\n";
    std::cout << "Initial load step factor: " << initialLoadStep << "\n";
    std::cout << "Target solver iterations per load step: " << targetNumIterations << "\n";
    std::cout << "Adaptive load step: " << (adaptiveLoadStep ? "on" : "off") << "\n";
    std::cout << "hbar flexoelectric correction: "
              << (includeHbarFlexoCorrection ? "on" : "off") << "\n";
    std::cout << "Electrical BC: free east end grounded, remaining boundaries open circuit\n";

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

    MultiBasis basisDisplacement(geometry);
    MultiBasis basisElectricPotential(geometry);
    for (int i = 0; i < numDegreeElevations; ++i)
    {
        basisDisplacement.degreeElevate();
        basisElectricPotential.degreeElevate();
    }
    basisDisplacement.uniformRefine(0, numEle_L - 1);
    basisDisplacement.uniformRefine(1, numEle_H - 1);
    basisElectricPotential.uniformRefine(0, numEle_L - 1);
    basisElectricPotential.uniformRefine(1, numEle_H - 1);

    BoundaryConditions bcInfo;
    const std::vector<double> zeros{0.0, 0.0};
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, zeros, 0);
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, zeros, 1);
    bcInfo.addCondition(0, boundary::east, condition_type::dirichlet, zeros, 2);

    const std::vector<double> cornerPointLoad{0.0, pointLoadY};
    bcInfo.addCondition(0, boundary::southeast, condition_type::neumann,
                        cornerPointLoad);

    Eigen::VectorXd bodyForce(2);
    bodyForce << 0.0, 0.0;

    GPUFlexoelectriciyAssembler assembler(geometry, basisDisplacement,
                                          basisElectricPotential, bcInfo,
                                          bodyForce);
    assembler.options().setReal("youngs_modulus", YM);
    assembler.options().setReal("poissons_ratio", PR);
    assembler.options().setReal("dielectric_permittivity", dielectricPermittivityModel);
    assembler.options().setReal("length_scale", lengthScale);
    assembler.options().setReal("flexoelectric_mu_L", muLModel);
    assembler.options().setReal("flexoelectric_mu_T", muTModel);
    assembler.options().setReal("flexoelectric_mu_S", muSModel);
    assembler.options().setReal("neumann_load_scaling", 0.0);
    assembler.options().setInt("material_law", materialLaw);
    assembler.options().setInt("include_hbar_flexo_correction",
                               includeHbarFlexoCorrection);
    std::cout << "Initialized system with " << assembler.numDofs() << " dofs.\n";

    GPUSolver solver(assembler);

    MultiPatch displacementHost;
    basisDisplacement.giveBasis(displacementHost, 2);
    GPUFunction displacementFunction(displacementHost);
    //assembler.constructSolution(solver.solutionView(), solver.allFixedDofsView(),
    //                            displacementFunction);

    MultiPatch electricPotentialHost;
    basisElectricPotential.giveBasis(electricPotentialHost, 1);
    GPUFunction electricPotentialFunction(electricPotentialHost);
    //assembler.constructElecSolution(solver.solutionView(), solver.allFixedDofsView(),
    //                                electricPotentialFunction);

    MultiPatch electricFieldHost;
    basisElectricPotential.giveBasis(electricFieldHost, 2);
    GPUFunction electricFieldFunction(electricFieldHost);

    const std::string filePrefix =
        outputFolder + "/flexoelectricity_CodonyOpenCircuitCantilever_";
    ParaviewCollection collection(filePrefix);

    std::vector<int> numPointsPerPatch{10000};
    GPUPostProcessor postProcessor(assembler, numPointsPerPatch, true, 2);
    postProcessor.addFunction("displacement", &displacementFunction);
    postProcessor.addFunction("electric_potential", &electricPotentialFunction);
    postProcessor.addFunction("electric_field", &electricFieldFunction);

    std::ofstream displacementHistory(outputFolder + "/displacement.txt");
    std::ofstream loadHistory(outputFolder + "/load.txt");
    std::ofstream electricFieldHistory(outputFolder + "/electricField.txt");
    displacementHistory << std::scientific << std::setprecision(12);
    loadHistory << std::scientific << std::setprecision(12);
    electricFieldHistory << std::scientific << std::setprecision(12);
    if (!displacementHistory || !loadHistory || !electricFieldHistory)
        throw std::runtime_error("Failed to open displacement/load/electric-field history files.");

    auto writeParaviewOutput = [&](int outputStep, double loadFactor)
    {
        assembler.constructSolution(solver.solutionView(), solver.allFixedDofsView(),
                                    displacementFunction);
        assembler.constructElecSolution(solver.solutionView(), solver.allFixedDofsView(),
                                        electricPotentialFunction);
        assembler.constructElectricFieldFunction(electricPotentialFunction,
                                                 electricFieldFunction);

        Eigen::MatrixXd northeastPoint(2, 1);
        northeastPoint << 1.0, 1.0;
        const Eigen::MatrixXd northeastDisplacement =
            displacementFunction.eval(0, northeastPoint);
        Eigen::MatrixXd fixedEndCenterPoint(2, 1);
        fixedEndCenterPoint << 0.0, 0.5;
        const Eigen::MatrixXd fixedEndCenterElectricField =
            electricFieldFunction.eval(0, fixedEndCenterPoint);
        displacementHistory << northeastDisplacement(1, 0) << "\n";
        loadHistory << loadFactor * pointLoadY << "\n";
        electricFieldHistory << fixedEndCenterElectricField(1, 0) << "\n";
        displacementHistory.flush();
        loadHistory.flush();
        electricFieldHistory.flush();

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

#ifdef FLEXO_NUMERIC_JACOBIAN_TEST
        {
            Eigen::VectorXd jacobianCheckSolution;
            solver.solutionToHost(jacobianCheckSolution);
            std::cout << "Checking numerical Jacobian at step " << step
                      << ", load factor " << loadFactor << ".\n";
            assembler.checkNumericalJacobian(jacobianCheckSolution,
                                             solver.allFixedDofsView());
        }
#endif

        writeParaviewOutput(outputStep++, loadFactor);

        if (adaptiveLoadStep && targetNumIterations > 0 &&
            std::abs(loadFactor - 1.0) > loadTolerance)
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
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
