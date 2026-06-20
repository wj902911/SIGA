#include <GPUPostProcessor.h>
#include <GPUSolver.h>
#include <TeeLogger.h>

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 9)
    {
        std::cerr
            << "Usage: " << argv[0]
            << " <L> <H> <numEle_L> <numEle_H>"
            << " <numRefinements> <numDegElev> <initialDeltaDisp> <maxStrain>"
            << " [numBisectionIterations] [targetNumIterations]"
            << " [bucklingPerturbationAmplitude] [postBisectionDeltaDisp]"
            << " [resultOutputStrainInterval] [materialLaw]\n"
            << "Example: " << argv[0]
            << " 1000 10 10 1 0 1 -0.1 -0.001 20\n";
        return 1;
    }

    const double YM = 1.725;
    const double PR = 0.3;
    const double L = std::stod(argv[1]);
    const double H = std::stod(argv[2]);
    const int numEle_L = std::stoi(argv[3]);
    const int numEle_H = std::stoi(argv[4]);
    const int numRefinements = std::stoi(argv[5]);
    const int numDegElev = std::stoi(argv[6]);
    const double initialDeltaDispInput = std::stod(argv[7]);
    const double maxStrain = std::stod(argv[8]);
    const int numBisectionIterations = argc > 9 ? std::stoi(argv[9]) : 20;
    const int targetNumIterations = argc > 10 ? std::stoi(argv[10]) : 5;
    const double bucklingPerturbationAmplitude =
        argc > 11 ? std::stod(argv[11]) : 1e-3 * std::abs(initialDeltaDispInput);
    const double postBisectionDeltaDisp =
        argc > 12 ? std::stod(argv[12]) : 0.1 * initialDeltaDispInput;
    const double resultOutputStrainInterval =
        argc > 13 ? std::stod(argv[13]) : 0.0;
    const int materialLaw = argc > 14 ? std::stoi(argv[14]) : 1;
    if (materialLaw != 0 && materialLaw != 1)
        throw std::invalid_argument("materialLaw must be 0 (StVK) or 1 (neo-Hookean).");

    const double maxDisp = maxStrain * L;

    const std::string rootFolder = "./nonlinearElasticity_2DBeamBuckling_output";
    const std::string filenameParaview = "nonlinearElasticity_2DBeamBuckling_";
    std::string outputFolder = rootFolder + "/" + filenameParaview + "output";
    for (int i = 1; i < argc; ++i)
        outputFolder += "_" + std::string(argv[i]);
    std::filesystem::create_directories(outputFolder);
    TeeLogger log(outputFolder + "/log.txt");

    const std::string fileNameWithPath = outputFolder + "/" + filenameParaview;
    ParaviewCollection collection(fileNameWithPath);
    const std::string sectionDataFolder = outputFolder + "/SectionData";
    std::filesystem::create_directories(sectionDataFolder);
    const std::string onsetFile = outputFolder + "/instability_onset.txt";
    const std::string eigenvalueFile =
        outputFolder + "/instability_smallest_eigenvalue.txt";
    const std::string eigenvectorFile =
        outputFolder + "/instability_smallest_eigenvector.txt";

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

    MultiBasis basisDisplacement(geometry);
    for (int i = 0; i < numDegElev; ++i)
        basisDisplacement.degreeElevate();
    basisDisplacement.uniformRefine(0, numEle_L - 1);
    basisDisplacement.uniformRefine(1, numEle_H - 1);
    for (int i = 0; i < numRefinements; ++i)
        basisDisplacement.uniformRefine();

    std::cout << "Nonlinear elasticity 2D beam buckling\n";
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
    std::cout << "Max displacement: " << maxDisp << "\n";
    std::cout << "Bisection iterations: " << numBisectionIterations << "\n";
    std::cout << "Target solver iterations per step: " << targetNumIterations << "\n";
    std::cout << "Buckling perturbation amplitude: "
              << bucklingPerturbationAmplitude << "\n";
    std::cout << "Post-bisection step length: " << postBisectionDeltaDisp << "\n";
    if (resultOutputStrainInterval > 0.0)
        std::cout << "Result output strain interval: "
                  << resultOutputStrainInterval << "\n";
    else
        std::cout << "Result output strain interval: every converged step\n";

    BoundaryConditions bcInfo;
    const std::vector<double> zeros{0.0, 0.0};
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, zeros, 0);
    bcInfo.addCondition(0, boundary::southwest, condition_type::dirichlet, zeros, 1);
    bcInfo.addCondition(0, boundary::southeast, condition_type::dirichlet, zeros, 1);
    std::vector<double> disp{initialDeltaDispInput, 0.0};
    bcInfo.addCondition(0, boundary::east, condition_type::dirichlet, disp, 0);

    Eigen::VectorXd bodyForce(2);
    bodyForce << 0.0, 0.0;

    GPUAssembler assembler(geometry, basisDisplacement, bcInfo, bodyForce);
    assembler.options().setReal("youngs_modulus", YM);
    assembler.options().setReal("poissons_ratio", PR);
    assembler.options().setInt("material_law", materialLaw);
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
                      << storedCriticalEigenvector.size()
                      << " entries but the model has "
                      << assembler.numDofs() << " dofs.\n";
        }
    }

    std::vector<int> numPointsPerPatch{1000};
    GPUPostProcessor postProcessor(assembler, numPointsPerPatch, true, 2);

    MultiPatch displacementHost;
    basisDisplacement.giveBasis(displacementHost, 2);
    GPUFunction displacementFunction(displacementHost);
    postProcessor.addFunction("displacement", &displacementFunction);

    MultiPatch cauchyStressHost;
    basisDisplacement.giveBasis(cauchyStressHost, assembler.dimTensor());
    GPUFunction cauchyStressFunction(cauchyStressHost);
    postProcessor.addFunction("stress_cauchy", &cauchyStressFunction);

    auto outputSectionCauchyStress11 = [&](int outputStepIndex)
    {
        constexpr int numSectionPoints = 101;
        Eigen::MatrixXd sectionPoints(2, numSectionPoints);
        for (int i = 0; i < numSectionPoints; ++i)
        {
            sectionPoints(0, i) = 0.5;
            sectionPoints(1, i) =
                static_cast<double>(i) / static_cast<double>(numSectionPoints - 1);
        }

        const Eigen::MatrixXd sectionCauchyStress =
            cauchyStressFunction.eval(0, sectionPoints);
        const std::string stepFolder =
            sectionDataFolder + "/step_" + std::to_string(outputStepIndex);
        std::filesystem::create_directories(stepFolder);

        std::ofstream cauchyStressOut(stepFolder + "/SecCauStress11.txt");
        cauchyStressOut << std::setprecision(16);
        for (int i = 0; i < numSectionPoints; ++i)
            cauchyStressOut << sectionCauchyStress(0, i) << "\n";
    };

    auto writeOutput = [&](int outputStepIndex, double appliedDisp)
    {
        assembler.constructSolution(solver.solutionView(),
                                    solver.allFixedDofsView(),
                                    displacementFunction);
        assembler.constructCauchyStressFunction(displacementFunction,
                                                cauchyStressFunction);
        postProcessor.outputToParaview(fileNameWithPath, outputStepIndex, collection);
        outputSectionCauchyStress11(outputStepIndex);
        std::cout << "Wrote result output " << outputStepIndex
                  << " at applied strain " << appliedDisp / L << ".\n";
    };

    int outputStep = 0;
    writeOutput(outputStep++, 0.0);
    double nextOutputStrain = resultOutputStrainInterval;
    const double outputStrainTolerance =
        std::max(1e-12, 1e-8 * std::abs(maxStrain));

    int step = 1;
    double appliedDisp = 0.0;
    double deltaDisp = initialDeltaDispInput;
    bool instabilityOnsetStored = false;
    double previousAppliedDisp = appliedDisp;
    double previousStability = std::numeric_limits<double>::infinity();
    Eigen::VectorXd previousSolution;
    Eigen::VectorXd previousFixedDofs;
    solver.solutionToHost(previousSolution);
    solver.fixedDofsToHost(previousFixedDofs);

    auto solvestart = std::chrono::high_resolution_clock::now();
    while (std::abs(appliedDisp - maxDisp) > 1e-6)
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
            if (std::abs(deltaToNextOutput) > outputStrainTolerance * L &&
                std::abs(deltaDisp) > std::abs(deltaToNextOutput))
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
            const bool crosses =
                (deltaDisp > 0.0 && appliedDisp < storedCriticalDisp &&
                 nextAppliedDisp > storedCriticalDisp) ||
                (deltaDisp < 0.0 && appliedDisp > storedCriticalDisp &&
                 nextAppliedDisp < storedCriticalDisp);
            const bool hits =
                std::abs(nextAppliedDisp - storedCriticalDisp) <= 1e-12;

            if (crosses || hits)
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

        if (!instabilityOnsetStored &&
            (currentStability < 0.0 || reachedStoredCriticalDisp))
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
                    criticalSolution +=
                        bucklingPerturbationAmplitude * storedCriticalEigenvector;
                else
                    std::cout << "Skipping buckling perturbation because solution "
                              << "and eigenvector sizes differ.\n";

                solver.setSolutionFromHost(criticalSolution);
                solver.setFixedDofsFromHost(criticalFixedDofs);
                appliedDisp = storedCriticalDisp;
                deltaDisp = postBisectionDeltaDisp;
                disp[0] = deltaDisp;
                assembler.refreshFixedDofs();
                instabilityOnsetStored = true;
                skipStepLengthAdjustment = true;
                std::cout << "Restarting from stored instability onset with "
                          << "eigenvector perturbation.\n";
            }
            else
            {
                std::cout << "Instability detected. Starting bisection between "
                          << previousAppliedDisp << " and " << appliedDisp << ".\n";

                double stableDisp = previousAppliedDisp;
                double unstableDisp = appliedDisp;
                double stableStability = previousStability;
                double unstableStability = currentStability;

                for (int bisectionIter = 0;
                     bisectionIter < numBisectionIterations; ++bisectionIter)
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
                const double criticalEigenvalue =
                    solver.smallestEigenValue(criticalEigenvector);
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
                         << "buckling_perturbation_amplitude "
                         << bucklingPerturbationAmplitude << "\n"
                         << "bisection_iterations "
                         << numBisectionIterations << "\n";

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
                    criticalSolution +=
                        bucklingPerturbationAmplitude * criticalEigenvector;
                else
                    std::cout << "Skipping buckling perturbation because solution "
                              << "and eigenvector sizes differ.\n";

                solver.setSolutionFromHost(criticalSolution);
                solver.setFixedDofsFromHost(criticalFixedDofs);
                appliedDisp = unstableDisp;
                currentStability = criticalStability;
                deltaDisp = postBisectionDeltaDisp;
                disp[0] = deltaDisp;
                assembler.refreshFixedDofs();
                instabilityOnsetStored = true;
                skipStepLengthAdjustment = true;
                std::cout << "Restarting from instability onset with "
                          << "eigenvector perturbation.\n";
            }
        }

        const double appliedStrain = appliedDisp / L;
        const double outputProgress = std::abs(appliedStrain);
        const bool finalStep = std::abs(appliedDisp - maxDisp) <= 1e-6;
        bool writeResultOutput = resultOutputStrainInterval <= 0.0 || finalStep;
        if (!writeResultOutput)
            writeResultOutput =
                outputProgress + outputStrainTolerance >= nextOutputStrain;

        if (writeResultOutput)
        {
            writeOutput(outputStep, appliedDisp);
            ++outputStep;

            if (resultOutputStrainInterval > 0.0)
            {
                const double completedIntervals =
                    std::floor((outputProgress + outputStrainTolerance) /
                               resultOutputStrainInterval);
                nextOutputStrain =
                    (completedIntervals + 1.0) * resultOutputStrainInterval;
            }
        }

        solver.solutionToHost(previousSolution);
        solver.fixedDofsToHost(previousFixedDofs);
        previousStability = currentStability;

        if (!skipStepLengthAdjustment && targetNumIterations > 0 &&
            std::abs(appliedDisp - maxDisp) > 1e-6)
        {
            const double oldDeltaDisp = deltaDisp;
            if (stepNumIterations > targetNumIterations)
            {
                const double factor = std::max(
                    0.5,
                    std::sqrt(static_cast<double>(targetNumIterations) /
                              static_cast<double>(stepNumIterations)));
                deltaDisp *= factor;
                std::cout << "Reducing next step length to " << deltaDisp
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
                deltaDisp *= factor;
                std::cout << "Increasing next step length to " << deltaDisp
                          << " because " << stepNumIterations
                          << " iterations were below target "
                          << targetNumIterations << ".\n";
            }
            else
            {
                std::cout << "Maintaining next step length " << deltaDisp
                          << " because solver iterations matched target "
                          << targetNumIterations << ".\n";
            }

            double maxNextStepMagnitude = std::abs(maxDisp - appliedDisp);
            if (resultOutputStrainInterval > 0.0)
            {
                const double distanceToNextOutput =
                    std::max(0.0, nextOutputStrain * L - std::abs(appliedDisp));
                maxNextStepMagnitude =
                    std::min(maxNextStepMagnitude, distanceToNextOutput);
            }

            if (std::abs(deltaDisp) > maxNextStepMagnitude)
            {
                deltaDisp = (deltaDisp >= 0.0 ? 1.0 : -1.0) *
                            maxNextStepMagnitude;
                if (resultOutputStrainInterval > 0.0)
                    std::cout << "Capping next step length to output interval "
                              << "displacement " << deltaDisp << ".\n";
                else
                    std::cout << "Capping next step length to remaining "
                              << "displacement " << deltaDisp << ".\n";
            }

            if (deltaDisp != oldDeltaDisp)
            {
                disp[0] = deltaDisp;
                assembler.refreshFixedDofs();
            }
        }

        ++step;
    }

    collection.save();
    auto solveend = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = solveend - solvestart;
    std::cout << "Total solve time: " << elapsed.count() << " seconds\n";
    std::cout << "Paraview output: " << outputFolder << "\n";

    return 0;
}
