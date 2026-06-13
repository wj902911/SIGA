#include <GPUSolver.h>
#include <GPUPostProcessor.h>
#include <TeeLogger.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <GPUStrainGradientElasticityAssembler.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <stdexcept>

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

bool hasParameter(const std::map<std::string, std::string>& parameters,
                  const std::string& key)
{
    return parameters.find(key) != parameters.end();
}

double parameterDouble(const std::map<std::string, std::string>& parameters,
                       const std::string& key)
{
    const auto it = parameters.find(key);
    if (it == parameters.end())
        throw std::runtime_error("Missing required parameter: " + key);
    return std::stod(it->second);
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
                 const std::string& key)
{
    const auto it = parameters.find(key);
    if (it == parameters.end())
        throw std::runtime_error("Missing required parameter: " + key);
    return std::stoi(it->second);
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

std::string parameterString(
    const std::map<std::string, std::string>& parameters,
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
    const bool useParameterFile = argc == 2 &&
        std::filesystem::path(argv[1]).extension() == ".txt";
    if (!useParameterFile && argc < 10)
    {
        std::cerr
            << "Usage: " << argv[0]
            << " <parameterFile.txt>\n"
            << "   or: " << argv[0]
            << " <lengthScale> <A> <R> <numEle_L> <numEle_H>"
            << " <numRefinements> <numDegElev> <initialDeltaDisp> <maxStrain>"
            << " [numBisectionIterations] [targetNumIterations]"
            << " [bucklingPerturbationAmplitude] [postBisectionDeltaDisp]"
            << " [resultOutputStrainInterval]\n"
            << "Example: " << argv[0]
            << " 5 10000 100 10 1 0 1 -0.1 -0.001 20\n";
        return 1;
    }

    double YM = 1.0;
	double PR = 0.495;
    double lengthScale = 0.0;
    double L = 0.0;
    double H = 0.0;
    int numEle_L = 0;
    int numEle_H = 0;
    int numRefinements = 0;
    int numDegElev = 0;
    double initialDeltaDisp = 0.0;
    double maxStrain = 0.0;
    int numBisectionIterations = 20;
    int targetNumIterations = 5;
    double bucklingPerturbationAmplitude = 0.0;
    double postBisectionDeltaDisp = 0.0;
    double resultOutputStrainInterval = 0.0;
    int numPointsPerPatchValue = 1000;
    std::string outputSuffix;

    if (useParameterFile)
    {
        const std::map<std::string, std::string> parameters =
            readParameterFile(argv[1]);
        lengthScale = parameterDouble(parameters, "lengthScale");
        if (hasParameter(parameters, "L") && hasParameter(parameters, "H"))
        {
            L = parameterDouble(parameters, "L");
            H = parameterDouble(parameters, "H");
        }
        else
        {
            const double A = parameterDouble(parameters, "A");
            const double R = parameterDouble(parameters, "R");
            H = sqrt(A / R);
            L = sqrt(A * R);
        }
        numEle_L = parameterInt(parameters, "numEle_L");
        numEle_H = parameterInt(parameters, "numEle_H");
        numRefinements = parameterInt(parameters, "numRefinements", 0);
        numDegElev = parameterInt(parameters, "numDegElev");
        initialDeltaDisp = parameterDouble(parameters, "initialDeltaDisp");
        maxStrain = parameterDouble(parameters, "maxStrain");
        numBisectionIterations = parameterInt(parameters, "numBisectionIterations",
                                             numBisectionIterations);
        targetNumIterations = parameterInt(parameters, "targetNumIterations",
                                          targetNumIterations);
        bucklingPerturbationAmplitude = parameterDouble(
            parameters, "bucklingPerturbationAmplitude",
            1e-3 * abs(initialDeltaDisp));
        postBisectionDeltaDisp = parameterDouble(
            parameters, "postBisectionDeltaDisp", 0.1 * initialDeltaDisp);
        resultOutputStrainInterval = parameterDouble(
            parameters, "resultOutputStrainInterval", resultOutputStrainInterval);
        numPointsPerPatchValue = parameterInt(parameters, "numPointsPerPatch",
                                             numPointsPerPatchValue);
        YM = parameterDouble(parameters, "YM", YM);
        PR = parameterDouble(parameters, "PR", PR);
        const std::string outputPostfix = parameterString(
            parameters, "outputPostfix",
            std::filesystem::path(argv[1]).stem().string());
        if (!outputPostfix.empty())
            outputSuffix = "_" + outputPostfix;
    }
    else
    {
        lengthScale = std::stod(argv[1]);
        const double A = std::stod(argv[2]);
        const double R = std::stod(argv[3]);
        H = sqrt(A / R);
        L = sqrt(A * R);
        numEle_L = std::stoi(argv[4]);
        numEle_H = std::stoi(argv[5]);
        numRefinements = std::stoi(argv[6]);
        numDegElev = std::stoi(argv[7]);
        initialDeltaDisp = std::stod(argv[8]);
        maxStrain = std::stod(argv[9]);
        if (argc > 10)
            numBisectionIterations = std::stoi(argv[10]);
        if (argc > 11)
            targetNumIterations = std::stoi(argv[11]);
        bucklingPerturbationAmplitude = 1e-3 * abs(initialDeltaDisp);
        if (argc > 12)
            bucklingPerturbationAmplitude = std::stod(argv[12]);
        postBisectionDeltaDisp = 0.1 * initialDeltaDisp;
        if (argc > 13)
            postBisectionDeltaDisp = std::stod(argv[13]);
        if (argc > 14)
            resultOutputStrainInterval = std::stod(argv[14]);
        if (argc > 15)
            numPointsPerPatchValue = std::stoi(argv[15]);
        for (int i = 1; i < argc; ++i)
        {
            if (i != 12 && i != 13)
                outputSuffix += "_" + std::string(argv[i]);
        }
    }
    std::vector<int> numPointsPerPatch{ numPointsPerPatchValue };

    if (!std::filesystem::exists("./strainGradient_2DBeamBuckling"))
		std::filesystem::create_directory("./strainGradient_2DBeamBuckling");
	std::string filenameParaview = "strainGradient_2DBeamBuckling_";
	std::string outputFolder = "./strainGradient_2DBeamBuckling/" + filenameParaview + "output" + outputSuffix;
    if (!std::filesystem::exists(outputFolder))
		std::filesystem::create_directory(outputFolder);
    TeeLogger log(outputFolder + "/log.txt");
    std::string fileNameWithPath = outputFolder + "/" + filenameParaview;
	ParaviewCollection collection(fileNameWithPath);
    std::string sectionDataFolder = outputFolder + "/SectionData";
    if (!std::filesystem::exists(sectionDataFolder))
        std::filesystem::create_directory(sectionDataFolder);
    const std::string onsetFile = outputFolder + "/instability_onset.txt";
    const std::string eigenvalueFile = outputFolder + "/instability_smallest_eigenvalue.txt";
    const std::string eigenvectorFile = outputFolder + "/instability_smallest_eigenvector.txt";

    double maxDisp = maxStrain * L;

    int knot_u_order = 1;
    int knot_v_order = 1;
	std::vector<double> knot_u{ 0., 0., 1., 1. };
    std::vector<double> knot_v{ 0., 0., 1., 1. };
	Eigen::MatrixXd control_points(4, 2);
    control_points << 0., 0.,
                      L, 0.,
                      0., H,
                      L, H;
    KnotVector u1(knot_u_order,knot_u);
	KnotVector v1(knot_v_order,knot_v);
    Patch patch(u1, v1, control_points);
    MultiPatch geometry;
	geometry.addPatch(patch);
	geometry.computeTopology();
	MultiBasis basisDisplacement(geometry);

    for (int i = 0; i < numDegElev; ++i)
        basisDisplacement.degreeElevate();

    basisDisplacement.uniformRefine(0, numEle_L - 1);
    basisDisplacement.uniformRefine(1, numEle_H - 1);

    for (int i = 0; i < numRefinements; ++i)
        basisDisplacement.uniformRefine();

    std::cout << "Young's Modulus: " << YM << "\n";
    std::cout << "Poisson's Ratio: " << PR << "\n";
    std::cout << "Length scale: " << lengthScale << "\n";
    std::cout << "Beam length: " << L << "\n";
    std::cout << "Beam height: " << H << "\n";
    std::cout << "Number of elements:" << numEle_L * pow(2, numRefinements) << " x " << numEle_H * pow(2, numRefinements) << "\n";
    std::cout << "Basis function degree: " << numDegElev + 1 << "\n";
    std::cout << "Max displacement: " << maxDisp << "\n";
    std::cout << "Bisection iterations: " << numBisectionIterations << "\n";
    std::cout << "Target solver iterations per step: " << targetNumIterations << "\n";
    std::cout << "Buckling perturbation amplitude: " << bucklingPerturbationAmplitude << "\n";
    std::cout << "Post-bisection step length: " << postBisectionDeltaDisp << "\n";
    if (resultOutputStrainInterval > 0.0)
        std::cout << "Result output strain interval: " << resultOutputStrainInterval << "\n";
    else
        std::cout << "Result output strain interval: every converged step\n";

    BoundaryConditions bcInfo;
    std::vector<double> zeros{0.0, 0.0};
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, zeros, 0);
    bcInfo.addCondition(0, boundary::southwest, condition_type::dirichlet, zeros, 1);
    bcInfo.addCondition(0, boundary::southeast, condition_type::dirichlet, zeros, 1);
    std::vector<double> disp{initialDeltaDisp, 0.0};
    bcInfo.addCondition(0, boundary::east, condition_type::dirichlet, disp, 0);
    bcInfo.addBoundaryCoupling(0, boundary::west, 1, 1);
    bcInfo.addBoundaryCoupling(0, boundary::west, 2, 1);
    bcInfo.addBoundaryCoupling(0, boundary::west, 3, 1);
    bcInfo.addBoundaryCoupling(0, boundary::east, 1, 1);
    bcInfo.addBoundaryCoupling(0, boundary::east, 2, 1);
    bcInfo.addBoundaryCoupling(0, boundary::east, 3, 1);

    Eigen::VectorXd bodyForce(2);
	bodyForce << 0.0, 0.0;

    GPUStrainGradientElasticityAssembler assembler(geometry, basisDisplacement, bcInfo, bodyForce);
    //GPUAssembler assembler(geometry, basisDisplacement, bcInfo, bodyForce);
    assembler.options().setReal("youngs_modulus", YM);
	assembler.options().setReal("poissons_ratio", PR);
    assembler.options().setReal("length_scale", lengthScale);
    std::cout << "Initialized system with " << assembler.numDofs() << " dofs.\n";

    GPUSolver solver(assembler);

    double storedCriticalDisp = 0.0;
    double storedCriticalEigenvalue = 0.0;
    Eigen::VectorXd storedCriticalEigenvector;
    bool hasStoredCriticalData = false;

    auto readCriticalDisplacement = [](const std::string& path, double& criticalDisp)
    {
        std::ifstream in(path);
        std::string key;
        double value = 0.0;
        while (in >> key >> value)
        {
            if (key == "critical_applied_displacement")
            {
                criticalDisp = value;
                return true;
            }
        }
        return false;
    };

    auto readScalar = [](const std::string& path, double& value)
    {
        std::ifstream in(path);
        return static_cast<bool>(in >> value);
    };

    auto readVector = [](const std::string& path, Eigen::VectorXd& vector)
    {
        std::ifstream in(path);
        std::vector<double> values;
        double value = 0.0;
        while (in >> value)
            values.push_back(value);

        vector.resize(static_cast<Eigen::Index>(values.size()));
        for (Eigen::Index i = 0; i < vector.size(); ++i)
            vector[i] = values[static_cast<std::size_t>(i)];

        return vector.size() > 0;
    };

    if (std::filesystem::exists(onsetFile) &&
        std::filesystem::exists(eigenvalueFile) &&
        std::filesystem::exists(eigenvectorFile) &&
        readCriticalDisplacement(onsetFile, storedCriticalDisp) &&
        readScalar(eigenvalueFile, storedCriticalEigenvalue) &&
        readVector(eigenvectorFile, storedCriticalEigenvector))
    {
        if (storedCriticalEigenvector.size() == assembler.numDofs())
        {
            hasStoredCriticalData = true;
            std::cout << "Loaded stored instability onset at displacement "
                      << storedCriticalDisp << " with smallest eigenvalue "
                      << storedCriticalEigenvalue << ".\n";
        }
        else
        {
            std::cout << "Ignoring stored instability eigenvector because it has "
                      << storedCriticalEigenvector.size() << " entries but the model has "
                      << assembler.numDofs() << " dofs.\n";
        }
    }

	GPUPostProcessor postProcessor(assembler, numPointsPerPatch, true, 2);

    MultiPatch displacementHost;
    basisDisplacement.giveBasis(displacementHost, 2);
	GPUFunction displacementFunction(displacementHost);
	postProcessor.addFunction("displacement", &displacementFunction);

    MultiPatch firstPiolaHost;
    basisDisplacement.giveBasis(firstPiolaHost, assembler.domainDim() * assembler.domainDim());
    GPUFunction firstPiolaFunction(firstPiolaHost);
    postProcessor.addFunction("first_piola_stress", &firstPiolaFunction);

    MultiPatch cauchyStressHost;
    const int cauchyStressDim = assembler.domainDim() * (assembler.domainDim() + 1) / 2;
    basisDisplacement.giveBasis(cauchyStressHost, cauchyStressDim);
    GPUFunction cauchyStressFunction(cauchyStressHost);
    postProcessor.addFunction("stress_cauchy", &cauchyStressFunction);


    int outputStep = 0;
    auto outputSectionStress11 = [&](int outputStepIndex)
    {
        constexpr int numSectionPoints = 101;
        Eigen::MatrixXd sectionPoints(2, numSectionPoints);
        for (int i = 0; i < numSectionPoints; ++i)
        {
            sectionPoints(0, i) = 0.5;
            sectionPoints(1, i) = static_cast<double>(i) / static_cast<double>(numSectionPoints - 1);
        }

        const Eigen::MatrixXd sectionStress = firstPiolaFunction.eval(0, sectionPoints);
        const Eigen::MatrixXd sectionCauchyStress = cauchyStressFunction.eval(0, sectionPoints);
        const std::string stepFolder = sectionDataFolder + "/step_" + std::to_string(outputStepIndex);
        if (!std::filesystem::exists(stepFolder))
            std::filesystem::create_directory(stepFolder);

        std::ofstream stressOut(stepFolder + "/SecStress11.txt");
        stressOut << std::setprecision(16);
        for (int i = 0; i < numSectionPoints; ++i)
            stressOut << sectionStress(0, i) << "\n";

        std::ofstream cauchyStressOut(stepFolder + "/SecCauStress11.txt");
        cauchyStressOut << std::setprecision(16);
        for (int i = 0; i < numSectionPoints; ++i)
            cauchyStressOut << sectionCauchyStress(0, i) << "\n";
    };

	postProcessor.outputToParaview(fileNameWithPath, outputStep++, collection);
    outputSectionStress11(outputStep - 1);
    double nextOutputStrain = resultOutputStrainInterval;
    const double outputStrainTolerance = std::max(1e-12, 1e-8 * std::abs(maxStrain));

    int step = 1;
    double appliedDisp = 0.0;
    double deltaDisp = initialDeltaDisp;
    bool instabilityOnsetStored = false;
    double previousAppliedDisp = appliedDisp;
    double previousStability = std::numeric_limits<double>::infinity();
    Eigen::VectorXd previousSolution;
    Eigen::VectorXd previousFixedDofs;
    solver.solutionToHost(previousSolution);
    solver.fixedDofsToHost(previousFixedDofs);
	auto solvestart = std::chrono::high_resolution_clock::now();
    while (abs(appliedDisp - maxDisp) > 1e-6)
    {
        if ((deltaDisp > 0.0 && appliedDisp + deltaDisp > maxDisp) ||
            (deltaDisp < 0.0 && appliedDisp + deltaDisp < maxDisp))
        {
            deltaDisp = maxDisp - appliedDisp;
            disp[0] = deltaDisp;
            assembler.refreshFixedDofs();
        }

        if (resultOutputStrainInterval > 0.0)
        {
            const double nextOutputDisp =
                (maxDisp >= 0.0 ? 1.0 : -1.0) * nextOutputStrain * L;
            const double deltaToNextOutput = nextOutputDisp - appliedDisp;
            if (abs(deltaToNextOutput) > outputStrainTolerance * L &&
                abs(deltaDisp) > abs(deltaToNextOutput))
            {
                deltaDisp = deltaToNextOutput;
                disp[0] = deltaDisp;
                assembler.refreshFixedDofs();
                std::cout << "Adjusted step length to " << deltaDisp
                          << " to hit next output displacement "
                          << nextOutputDisp << ".\n";
            }
        }

        if (hasStoredCriticalData && !instabilityOnsetStored)
        {
            const double nextAppliedDisp = appliedDisp + deltaDisp;
            const bool stepCrossesStoredCriticalDisp =
                (deltaDisp > 0.0 && appliedDisp < storedCriticalDisp && nextAppliedDisp > storedCriticalDisp) ||
                (deltaDisp < 0.0 && appliedDisp > storedCriticalDisp && nextAppliedDisp < storedCriticalDisp);
            const bool stepHitsStoredCriticalDisp =
                std::abs(nextAppliedDisp - storedCriticalDisp) <= 1e-12;

            if (stepCrossesStoredCriticalDisp || stepHitsStoredCriticalDisp)
            {
                deltaDisp = storedCriticalDisp - appliedDisp;
                disp[0] = deltaDisp;
                assembler.refreshFixedDofs();
                std::cout << "Adjusted step length to " << deltaDisp
                          << " to hit stored critical displacement "
                          << storedCriticalDisp << ".\n";
            }
        }

        previousAppliedDisp = appliedDisp;
        appliedDisp += deltaDisp;
        bool skipStepLengthAdjustment = false;

        std::cout << "Step " << step << ": applying displacement " << appliedDisp
                  << ", applied strain " << appliedDisp / L
                  << ", step length " << deltaDisp << "\n";

	    solver.solve();
        int stepNumIterations = solver.numIterations();

        double currentStability = solver.stability();
        std::cout << "Stability: " << currentStability << "\n";

        const bool reachedStoredCriticalDisp =
            hasStoredCriticalData && !instabilityOnsetStored &&
            std::abs(appliedDisp - storedCriticalDisp) <= 1e-10;

        if (!instabilityOnsetStored && (currentStability < 0.0 || reachedStoredCriticalDisp))
        {
            if (reachedStoredCriticalDisp)
            {
                std::cout << "Reached stored instability onset. Skipping bisection and "
                          << "using stored smallest eigenvalue "
                          << storedCriticalEigenvalue << ".\n";

                Eigen::VectorXd criticalSolution;
                Eigen::VectorXd criticalFixedDofs;
                solver.solutionToHost(criticalSolution);
                solver.fixedDofsToHost(criticalFixedDofs);

                if (criticalSolution.size() == storedCriticalEigenvector.size())
                    criticalSolution += bucklingPerturbationAmplitude * storedCriticalEigenvector;
                else
                    std::cout << "Skipping buckling perturbation because solution and eigenvector sizes differ.\n";

                solver.setSolutionFromHost(criticalSolution);
                solver.setFixedDofsFromHost(criticalFixedDofs);
                appliedDisp = storedCriticalDisp;
                deltaDisp = postBisectionDeltaDisp;
                disp[0] = deltaDisp;
                assembler.refreshFixedDofs();
                instabilityOnsetStored = true;
                skipStepLengthAdjustment = true;
                std::cout << "Restarting from stored instability onset with eigenvector perturbation.\n";
                std::cout << "Using post-bisection step length " << deltaDisp
                          << " for the next step.\n";
            }
            else
            {
                std::cout << "Instability detected. Starting bisection between "
                          << previousAppliedDisp << " and " << appliedDisp << ".\n";

                double stableDisp = previousAppliedDisp;
                double unstableDisp = appliedDisp;
                double stableStability = previousStability;
                double unstableStability = currentStability;

                for (int bisectionIter = 0; bisectionIter < numBisectionIterations; ++bisectionIter)
                {
                    const double trialDisp = 0.5 * (stableDisp + unstableDisp);
                    disp[0] = trialDisp - previousAppliedDisp;
                    assembler.refreshFixedDofs();
                    solver.setSolutionFromHost(previousSolution);
                    solver.setFixedDofsFromHost(previousFixedDofs);

                    std::cout << "Bisection " << bisectionIter + 1
                              << "/" << numBisectionIterations
                              << ": trial displacement " << trialDisp << "\n";

                    solver.solve();
                    const double trialStability = solver.stability();
                    std::cout << "Trial stability: " << trialStability << "\n";

                    if (trialStability < 0.0)
                    {
                        unstableDisp = trialDisp;
                        unstableStability = trialStability;
                    }
                    else
                    {
                        stableDisp = trialDisp;
                        stableStability = trialStability;
                    }
                }

                disp[0] = unstableDisp - previousAppliedDisp;
                assembler.refreshFixedDofs();
                solver.setSolutionFromHost(previousSolution);
                solver.setFixedDofsFromHost(previousFixedDofs);
                solver.solve();
                stepNumIterations = solver.numIterations();

                Eigen::VectorXd criticalEigenvector;
                const double criticalEigenvalue = solver.smallestEigenValue(criticalEigenvector);
                const double criticalStability = solver.stability();
                Eigen::VectorXd criticalSolution;
                Eigen::VectorXd criticalFixedDofs;
                solver.solutionToHost(criticalSolution);
                solver.fixedDofsToHost(criticalFixedDofs);

                std::ofstream onsetOut(onsetFile);
                onsetOut << std::setprecision(16)
                         << "critical_applied_displacement " << unstableDisp << "\n"
                         << "lower_stable_applied_displacement " << stableDisp << "\n"
                         << "upper_unstable_applied_displacement " << unstableDisp << "\n"
                         << "lower_stability " << stableStability << "\n"
                         << "upper_stability " << unstableStability << "\n"
                         << "critical_stability " << criticalStability << "\n"
                         << "buckling_perturbation_amplitude " << bucklingPerturbationAmplitude << "\n"
                         << "bisection_iterations " << numBisectionIterations << "\n";

                std::ofstream eigenvalueOut(eigenvalueFile);
                eigenvalueOut << std::setprecision(16) << criticalEigenvalue << "\n";

                std::ofstream eigenvectorOut(eigenvectorFile);
                eigenvectorOut << std::setprecision(16);
                for (Eigen::Index i = 0; i < criticalEigenvector.size(); ++i)
                    eigenvectorOut << criticalEigenvector[i] << "\n";

                std::cout << "Stored instability onset at displacement "
                          << unstableDisp << " with smallest eigenvalue "
                          << criticalEigenvalue << ".\n";

                if (criticalSolution.size() == criticalEigenvector.size())
                    criticalSolution += bucklingPerturbationAmplitude * criticalEigenvector;
                else
                    std::cout << "Skipping buckling perturbation because solution and eigenvector sizes differ.\n";

                solver.setSolutionFromHost(criticalSolution);
                solver.setFixedDofsFromHost(criticalFixedDofs);
                appliedDisp = unstableDisp;
                currentStability = criticalStability;
                deltaDisp = postBisectionDeltaDisp;
                disp[0] = deltaDisp;
                assembler.refreshFixedDofs();
                instabilityOnsetStored = true;
                skipStepLengthAdjustment = true;
                std::cout << "Restarting from instability onset with eigenvector perturbation.\n";
                std::cout << "Using post-bisection step length " << deltaDisp
                          << " for the next step.\n";
            }
        }

        const double appliedStrain = appliedDisp / L;
        const double outputProgress = std::abs(appliedStrain);
        const bool finalStep = std::abs(appliedDisp - maxDisp) <= 1e-6;
        bool writeResultOutput = resultOutputStrainInterval <= 0.0 || finalStep;
        if (!writeResultOutput)
            writeResultOutput = outputProgress + outputStrainTolerance >= nextOutputStrain;

        if (writeResultOutput)
        {
            assembler.constructSolution(solver.solutionView(), solver.allFixedDofsView(), displacementFunction);
            assembler.constructStrainGradientStressFunctions(displacementFunction, firstPiolaFunction, cauchyStressFunction);

		    postProcessor.outputToParaview(fileNameWithPath, outputStep, collection);
            outputSectionStress11(outputStep);
            std::cout << "Wrote result output " << outputStep
                      << " at applied strain " << appliedStrain << ".\n";
            outputStep++;

            if (resultOutputStrainInterval > 0.0)
            {
                const double completedIntervals =
                    std::floor((outputProgress + outputStrainTolerance) / resultOutputStrainInterval);
                nextOutputStrain = (completedIntervals + 1.0) * resultOutputStrainInterval;
            }
        }

        solver.solutionToHost(previousSolution);
        solver.fixedDofsToHost(previousFixedDofs);
        previousStability = currentStability;

        if (!skipStepLengthAdjustment && targetNumIterations > 0 && abs(appliedDisp - maxDisp) > 1e-6)
        {
            const double oldDeltaDisp = deltaDisp;
            if (stepNumIterations > targetNumIterations)
            {
                const double factor = std::max(0.5, sqrt(static_cast<double>(targetNumIterations) /
                                                         static_cast<double>(stepNumIterations)));
                deltaDisp *= factor;
                std::cout << "Reducing next step length to " << deltaDisp
                          << " because " << stepNumIterations
                          << " iterations exceeded target " << targetNumIterations << ".\n";
            }
            else if (stepNumIterations < targetNumIterations)
            {
                const double denominator = std::max(1, stepNumIterations);
                const double factor = std::min(2.0, sqrt(static_cast<double>(targetNumIterations) /
                                                         static_cast<double>(denominator)));
                deltaDisp *= factor;
                std::cout << "Increasing next step length to " << deltaDisp
                          << " because " << stepNumIterations
                          << " iterations were below target " << targetNumIterations << ".\n";
            }
            else
            {
                std::cout << "Maintaining next step length " << deltaDisp
                          << " because solver iterations matched target "
                          << targetNumIterations << ".\n";
            }

            double maxNextStepMagnitude = abs(maxDisp - appliedDisp);
            if (resultOutputStrainInterval > 0.0)
            {
                const double distanceToNextOutput =
                    std::max(0.0, nextOutputStrain * L - abs(appliedDisp));
                maxNextStepMagnitude = std::min(maxNextStepMagnitude, distanceToNextOutput);
            }

            if (abs(deltaDisp) > maxNextStepMagnitude)
            {
                deltaDisp = (deltaDisp >= 0.0 ? 1.0 : -1.0) * maxNextStepMagnitude;
                std::cout << "Capping next step length to output interval displacement "
                          << deltaDisp << ".\n";
            }

            if (deltaDisp != oldDeltaDisp)
            {
                disp[0] = deltaDisp;
                assembler.refreshFixedDofs();
            }
        }

        step++;
    }

	collection.save();
	auto solveend = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = solveend - solvestart;
    std::cout << "Total solve time: " << elapsed.count() << " seconds\n";

    return 0;
}
