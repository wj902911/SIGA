#include "device_launch_parameters.h"

#include <Eigen/Core>
#include <GPUPostProcessor.h>
#include <GPUSolver.h>
#include <GPUStrainGradientElasticityAssembler.h>
#include <TeeLogger.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
std::string trim(const std::string& value)
{
    const std::size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return "";
    const std::size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::map<std::string, std::string> readParameterFile(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open parameter file: " + path);

    std::map<std::string, std::string> parameters;
    std::string line;
    int lineNumber = 0;
    while (std::getline(in, line))
    {
        ++lineNumber;
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos)
            line = line.substr(0, comment);

        line = trim(line);
        if (line.empty())
            continue;

        const std::size_t separator = line.find(':');
        if (separator == std::string::npos)
            throw std::runtime_error("Expected key: value in " + path +
                                     " at line " + std::to_string(lineNumber));

        const std::string key = trim(line.substr(0, separator));
        const std::string value = trim(line.substr(separator + 1));
        if (key.empty() || value.empty())
            throw std::runtime_error("Empty key or value in " + path +
                                     " at line " + std::to_string(lineNumber));
        parameters[key] = value;
    }

    return parameters;
}

double parameterDouble(const std::map<std::string, std::string>& parameters,
                       const std::string& key,
                       double defaultValue)
{
    const auto it = parameters.find(key);
    if (it == parameters.end())
        return defaultValue;
    return std::stod(it->second);
}

int parameterInt(const std::map<std::string, std::string>& parameters,
                 const std::string& key,
                 int defaultValue)
{
    const auto it = parameters.find(key);
    if (it == parameters.end())
        return defaultValue;
    return std::stoi(it->second);
}

std::string parameterString(const std::map<std::string, std::string>& parameters,
                            const std::string& key,
                            const std::string& defaultValue)
{
    const auto it = parameters.find(key);
    if (it == parameters.end())
        return defaultValue;
    return it->second;
}
} // namespace

int main(int argc, char* argv[])
{
    double L = 150.0;
    double H = 15.0;
    double YM = 1.0;
    double PR = 0.495;
    double lengthScale = 5.0;
    double doubleStressX = 0.0;
    double doubleStressY = -0.25;
    int numEle_L = 10;
    int numEle_H = 1;
    int numRefinements = 4;
    int numDegElev = 2;
    int numPointsPerPatchValue = 1000;
    int materialLaw = 1;
    int numLoadSteps = 10;
    double initialLoadStep = 0.005;
    int maxNewtonIterations = 50;
    int targetNumIterations = 5;
    std::string outputPostfix = "default";

    const bool useParameterFile = argc == 2 &&
        std::filesystem::path(argv[1]).extension() == ".txt";
    if (useParameterFile)
    {
        const std::map<std::string, std::string> parameters =
            readParameterFile(argv[1]);
        L = parameterDouble(parameters, "L", L);
        H = parameterDouble(parameters, "H", H);
        YM = parameterDouble(parameters, "YM", YM);
        PR = parameterDouble(parameters, "PR", PR);
        lengthScale = parameterDouble(parameters, "lengthScale", lengthScale);
        doubleStressX = parameterDouble(parameters, "doubleStressX", doubleStressX);
        doubleStressY = parameterDouble(parameters, "doubleStressY", doubleStressY);
        numEle_L = parameterInt(parameters, "numEle_L", numEle_L);
        numEle_H = parameterInt(parameters, "numEle_H", numEle_H);
        numRefinements = parameterInt(parameters, "numRefinements", numRefinements);
        numDegElev = parameterInt(parameters, "numDegElev", numDegElev);
        numPointsPerPatchValue = parameterInt(parameters, "numPointsPerPatch",
                                             numPointsPerPatchValue);
        materialLaw = parameterInt(parameters, "materialLaw", materialLaw);
        numLoadSteps = parameterInt(parameters, "numLoadSteps", numLoadSteps);
        initialLoadStep = parameterDouble(parameters, "initialLoadStep",
                                          initialLoadStep);
        maxNewtonIterations = parameterInt(parameters, "maxNewtonIterations",
                                           maxNewtonIterations);
        targetNumIterations = parameterInt(parameters, "targetNumIterations",
                                           targetNumIterations);
        outputPostfix = parameterString(parameters, "outputPostfix", outputPostfix);
    }
    else
    {
        if (argc > 1) numEle_L = std::stoi(argv[1]);
        if (argc > 2) numEle_H = std::stoi(argv[2]);
        if (argc > 3) numRefinements = std::stoi(argv[3]);
        if (argc > 4) numDegElev = std::stoi(argv[4]);
        if (argc > 5) doubleStressY = std::stod(argv[5]);
        if (argc > 6) lengthScale = std::stod(argv[6]);
        if (argc > 7) L = std::stod(argv[7]);
        if (argc > 8) H = std::stod(argv[8]);
        if (argc > 9) YM = std::stod(argv[9]);
        if (argc > 10) PR = std::stod(argv[10]);
        if (argc > 11) materialLaw = std::stoi(argv[11]);
        if (argc > 12) numLoadSteps = std::stoi(argv[12]);
        if (argc > 13) initialLoadStep = std::stod(argv[13]);
        if (argc > 14) maxNewtonIterations = std::stoi(argv[14]);
        if (argc > 15) targetNumIterations = std::stoi(argv[15]);
        outputPostfix = "manual";
        for (int i = 1; i < argc; ++i)
            outputPostfix += "_" + std::string(argv[i]);
    }

    if (L <= 0.0 || H <= 0.0)
        throw std::invalid_argument("L and H must be positive.");
    if (numEle_L < 1 || numEle_H < 1)
        throw std::invalid_argument("numEle_L and numEle_H must be positive.");
    if (numRefinements < 0 || numDegElev < 0)
        throw std::invalid_argument("numRefinements and numDegElev must be nonnegative.");
    if (materialLaw != 0 && materialLaw != 1)
        throw std::invalid_argument("materialLaw must be 0 (StVK) or 1 (neo-Hookean).");
    if (numLoadSteps < 1)
        throw std::invalid_argument("numLoadSteps must be positive.");
    if (initialLoadStep <= 0.0 || initialLoadStep > 1.0)
        throw std::invalid_argument("initialLoadStep must be in (0, 1].");
    if (maxNewtonIterations < 1)
        throw std::invalid_argument("maxNewtonIterations must be positive.");

    const std::string rootFolder = "./strainGradient_2DBeamBending_doubleStress";
    const std::string filenameParaview =
        "strainGradient_2DBeamBending_doubleStress_";
    const std::string outputFolder =
        rootFolder + "/" + filenameParaview + "output_" + outputPostfix;
    std::filesystem::create_directories(outputFolder);
    TeeLogger log(outputFolder + "/log.txt");

    std::cout << "2D strain-gradient cantilever beam bending with double stress\n";
    std::cout << "Beam length: " << L << "\n";
    std::cout << "Beam height: " << H << "\n";
    std::cout << "Number of elements: "
              << numEle_L * std::pow(2, numRefinements) << " x "
              << numEle_H * std::pow(2, numRefinements) << "\n";
    std::cout << "Refinements: " << numRefinements << "\n";
    std::cout << "Degree elevations: " << numDegElev << "\n";
    std::cout << "Material: Y = " << YM << ", nu = " << PR
              << ", length scale = " << lengthScale << "\n";
    std::cout << "Right-end double stress: (" << doubleStressX
              << ", " << doubleStressY << ")\n";
    std::cout << "Load steps: " << numLoadSteps << "\n";
    std::cout << "Initial load step factor: " << initialLoadStep << "\n";
    std::cout << "Max Newton iterations per load step: "
              << maxNewtonIterations << "\n";
    std::cout << "Target solver iterations per load step: "
              << targetNumIterations << "\n";

    const int knotUOrder = 1;
    const int knotVOrder = 1;
    std::vector<double> knotU{0.0, 0.0, 1.0, 1.0};
    std::vector<double> knotV{0.0, 0.0, 1.0, 1.0};

    Eigen::MatrixXd controlPoints(4, 2);
    controlPoints <<
        0.0, 0.0,
        L,   0.0,
        0.0, H,
        L,   H;

    KnotVector u(knotUOrder, knotU);
    KnotVector v(knotVOrder, knotV);
    Patch patch(u, v, controlPoints);

    MultiPatch geometry;
    geometry.addPatch(patch);
    geometry.computeTopology();

    MultiBasis basis(geometry);
    for (int i = 0; i < numDegElev; ++i)
        basis.degreeElevate(false);
    basis.uniformRefine(0, numEle_L - 1);
    basis.uniformRefine(1, numEle_H - 1);
    for (int r = 0; r < numRefinements; ++r)
        basis.uniformRefine();

    BoundaryConditions bcInfo;
    const std::vector<double> zeros{0.0, 0.0};
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, zeros, 0);
    bcInfo.addCondition(0, boundary::southwest, condition_type::dirichlet, zeros, 1);

    const std::vector<double> doubleStress{doubleStressX, doubleStressY};
    bcInfo.addCondition(0, boundary::east, condition_type::double_stress,
                        doubleStress, 0);

    Eigen::VectorXd bodyForce(2);
    bodyForce << 0.0, 0.0;

    GPUStrainGradientElasticityAssembler assembler(geometry, basis, bcInfo, bodyForce);
    assembler.options().setReal("youngs_modulus", YM);
    assembler.options().setReal("poissons_ratio", PR);
    assembler.options().setReal("length_scale", lengthScale);
    assembler.options().setReal("neumann_load_scaling", 0.0);
    assembler.options().setInt("material_law", materialLaw);
    std::cout << "Initialized system with " << assembler.numDofs() << " dofs.\n";

    GPUSolver solver(assembler);
    solver.setTolerance(1e-8, 1e-3);
    solver.setMaxIterations(maxNewtonIterations);

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

    const std::string filePrefix = outputFolder + "/" + filenameParaview;
    ParaviewCollection collection(filePrefix);

    std::vector<int> numPointsPerPatch{numPointsPerPatchValue};
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
                  << " and double stress (" << loadFactor * doubleStressX
                  << ", " << loadFactor * doubleStressY << ").\n";
    };

    int outputStep = 0;
    writeParaviewOutput(outputStep++, 0.0);

    const double loadTolerance = 1e-12;
    const double minLoadStep = 1e-8;
    double loadFactor = 0.0;
    double loadStep = initialLoadStep;
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
                  << ", double stress (" << loadFactor * doubleStressX
                  << ", " << loadFactor * doubleStressY
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
