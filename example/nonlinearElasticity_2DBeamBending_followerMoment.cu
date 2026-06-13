#include "device_launch_parameters.h"

#include <Eigen/Core>
#include <GPUPostProcessor.h>
#include <GPUSolver.h>
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
    double L = 10.0;
    double H = 1.0;
    double YM = 1.0e5;
    double PR = 0.3;
    double followerMoment = -100.0;
    int numEle_L = 16;
    int numEle_H = 2;
    int numRefinements = 0;
    int numDegElev = 2;
    int numPointsPerPatchValue = 2000;
    int materialLaw = 1;
    int numLoadSteps = 1;
    int maxNewtonIterations = 100;
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
        followerMoment = parameterDouble(parameters, "followerMoment", followerMoment);
        numEle_L = parameterInt(parameters, "numEle_L", numEle_L);
        numEle_H = parameterInt(parameters, "numEle_H", numEle_H);
        numRefinements = parameterInt(parameters, "numRefinements", numRefinements);
        numDegElev = parameterInt(parameters, "numDegElev", numDegElev);
        numPointsPerPatchValue = parameterInt(parameters, "numPointsPerPatch",
                                             numPointsPerPatchValue);
        materialLaw = parameterInt(parameters, "materialLaw", materialLaw);
        numLoadSteps = parameterInt(parameters, "numLoadSteps", numLoadSteps);
        maxNewtonIterations = parameterInt(parameters, "maxNewtonIterations",
                                           maxNewtonIterations);
        outputPostfix = parameterString(parameters, "outputPostfix", outputPostfix);
    }
    else
    {
        if (argc > 1) L = std::stod(argv[1]);
        if (argc > 2) H = std::stod(argv[2]);
        if (argc > 3) followerMoment = std::stod(argv[3]);
        if (argc > 4) numEle_L = std::stoi(argv[4]);
        if (argc > 5) numEle_H = std::stoi(argv[5]);
        if (argc > 6) numRefinements = std::stoi(argv[6]);
        if (argc > 7) numDegElev = std::stoi(argv[7]);
        if (argc > 8) YM = std::stod(argv[8]);
        if (argc > 9) PR = std::stod(argv[9]);
        if (argc > 10) materialLaw = std::stoi(argv[10]);
        if (argc > 11) numPointsPerPatchValue = std::stoi(argv[11]);
        if (argc > 12) numLoadSteps = std::stoi(argv[12]);
        if (argc > 13) maxNewtonIterations = std::stoi(argv[13]);
        outputPostfix = "manual";
        for (int i = 1; i < argc; ++i)
            outputPostfix += "_" + std::string(argv[i]);
    }

    if (materialLaw != 0 && materialLaw != 1)
        throw std::invalid_argument("materialLaw must be 0 (StVK) or 1 (neo-Hookean).");
    if (numEle_L < 1 || numEle_H < 1)
        throw std::invalid_argument("numEle_L and numEle_H must be positive.");
    if (numLoadSteps < 1)
        throw std::invalid_argument("numLoadSteps must be positive.");
    if (maxNewtonIterations < 1)
        throw std::invalid_argument("maxNewtonIterations must be positive.");

    const std::string rootFolder =
        "./nonlinearElasticity_2DBeamBending_followerMoment";
    const std::string filenameParaview =
        "nonlinearElasticity_2DBeamBending_followerMoment_";
    const std::string outputFolder =
        rootFolder + "/" + filenameParaview + "output_" + outputPostfix;
    std::filesystem::create_directories(outputFolder);
    TeeLogger log(outputFolder + "/log.txt");

    std::cout << "Nonlinear elasticity 2D cantilever beam bending with follower moment\n";
    std::cout << "Young's Modulus: " << YM << "\n";
    std::cout << "Poisson's Ratio: " << PR << "\n";
    std::cout << "Material law: "
              << (materialLaw == 0 ? "StVK" : "neo-Hookean")
              << " (" << materialLaw << ")\n";
    std::cout << "Beam length: " << L << "\n";
    std::cout << "Beam height: " << H << "\n";
    std::cout << "Number of elements: "
              << numEle_L * std::pow(2, numRefinements) << " x "
              << numEle_H * std::pow(2, numRefinements) << "\n";
    std::cout << "Basis function degree: " << numDegElev + 1 << "\n";
    std::cout << "Follower moment: " << followerMoment << "\n";
    std::cout << "Load steps: " << numLoadSteps << "\n";
    std::cout << "Max Newton iterations per load step: "
              << maxNewtonIterations << "\n";

    const int knot_u_order = 1;
    const int knot_v_order = 1;
    std::vector<double> knot_u{0.0, 0.0, 1.0, 1.0};
    std::vector<double> knot_v{0.0, 0.0, 1.0, 1.0};
    Eigen::MatrixXd controlPoints(4, 2);
    controlPoints <<
        0.0, 0.0,
        L,   0.0,
        0.0, H,
        L,   H;

    KnotVector u(knot_u_order, knot_u);
    KnotVector v(knot_v_order, knot_v);
    Patch patch(u, v, controlPoints);
    MultiPatch geometry;
    geometry.addPatch(patch);
    geometry.computeTopology();

    MultiBasis basis(geometry);
    for (int i = 0; i < numDegElev; ++i)
        basis.degreeElevate();
    basis.uniformRefine(0, numEle_L - 1);
    basis.uniformRefine(1, numEle_H - 1);
    for (int i = 0; i < numRefinements; ++i)
        basis.uniformRefine();

    BoundaryConditions bcInfo;
    const std::vector<double> zeros{0.0, 0.0};
    const std::vector<double> moment{followerMoment};
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, zeros, 0);
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, zeros, 1);
    bcInfo.addCondition(0, boundary::east, condition_type::follower_moment,
                        moment, 0);

    Eigen::VectorXd bodyForce(2);
    bodyForce << 0.0, 0.0;

    GPUAssembler assembler(geometry, basis, bcInfo, bodyForce);
    assembler.options().setReal("youngs_modulus", YM);
    assembler.options().setReal("poissons_ratio", PR);
    assembler.options().setInt("material_law", materialLaw);
    assembler.options().setSwitch("use_nonsymmetric_newton_solver", true);
    std::cout << "Initialized system with " << assembler.numDofs() << " dofs.\n";

    GPUSolver solver(assembler);
    solver.setTolerance(1e-8, 1e-3);
    solver.setMaxIterations(maxNewtonIterations);
    for (int step = 1; step <= numLoadSteps; ++step)
    {
        const double loadScale =
            static_cast<double>(step) / static_cast<double>(numLoadSteps);
        assembler.options().setReal("neumann_load_scaling", loadScale);
        std::cout << "Load step " << step << "/" << numLoadSteps
                  << ", load scale: " << loadScale << "\n";
        solver.solve();
        if (!solver.isConverged())
            break;
    }

    MultiPatch displacementHost;
    basis.giveBasis(displacementHost, 2);
    GPUFunction displacementFunction(displacementHost);
    assembler.constructSolution(solver.solutionView(),
                                solver.allFixedDofsView(),
                                displacementFunction);

    MultiPatch cauchyStressHost;
    basis.giveBasis(cauchyStressHost, assembler.dimTensor());
    GPUFunction cauchyStressFunction(cauchyStressHost);
    assembler.constructCauchyStressFunction(displacementFunction,
                                            cauchyStressFunction);

    const std::string filePrefix = outputFolder + "/" + filenameParaview;
    ParaviewCollection collection(filePrefix);

    std::vector<int> numPointsPerPatch{numPointsPerPatchValue};
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
