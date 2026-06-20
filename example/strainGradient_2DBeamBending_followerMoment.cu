#include "device_launch_parameters.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <GPUPostProcessor.h>
#include <GPUSolver.h>
#include <GPUStrainGradientElasticityAssembler.h>
#include <SparseSystem.h>
#include <TeeLogger.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <sstream>
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

bool parseBool(const std::string& value, const std::string& key)
{
    std::string text = trim(value);
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (text == "1" || text == "true" || text == "on" || text == "yes")
        return true;
    if (text == "0" || text == "false" || text == "off" || text == "no")
        return false;

    throw std::invalid_argument("Expected boolean value for " + key +
                                " (true/false, on/off, yes/no, or 1/0).");
}

bool parameterBool(const std::map<std::string, std::string>& parameters,
                   const std::string& key,
                   bool defaultValue)
{
    const auto it = parameters.find(key);
    if (it == parameters.end())
        return defaultValue;
    return parseBool(it->second, key);
}

std::vector<double> solveGradedHalfThicknesses(int numHalfElements,
                                               double surfaceRatio,
                                               double centerRatio,
                                               double targetSum)
{
    constexpr double tolerance = 1e-12;
    if (numHalfElements < 1)
        return {};

    if (surfaceRatio <= 0.0 || centerRatio <= 0.0)
        throw std::invalid_argument(
            "H-direction element thickness ratios must be positive.");
    if (surfaceRatio > centerRatio)
        throw std::invalid_argument(
            "surfaceElementThicknessRatio must be no larger than centerElementThicknessRatio.");

    if (numHalfElements == 1)
    {
        if (std::abs(surfaceRatio - targetSum) > tolerance)
            throw std::invalid_argument(
                "The requested H-direction element thickness ratios cannot sum to H for this numEle_H.");
        return {surfaceRatio};
    }

    const double minSum =
        static_cast<double>(numHalfElements - 1) * surfaceRatio +
        centerRatio;
    const double maxSum =
        surfaceRatio +
        static_cast<double>(numHalfElements - 1) * centerRatio;
    if (targetSum < minSum - tolerance || targetSum > maxSum + tolerance)
    {
        std::ostringstream message;
        message << "The requested H-direction element thickness ratios cannot "
                << "sum to H for numEle_H. Required half-thickness sum is "
                << targetSum << ", but feasible range is [" << minSum
                << ", " << maxSum << "].";
        throw std::invalid_argument(message.str());
    }

    auto sumForExponent = [&](double exponent)
    {
        double sum = surfaceRatio + centerRatio;
        for (int i = 1; i < numHalfElements - 1; ++i)
        {
            const double r =
                static_cast<double>(i) /
                static_cast<double>(numHalfElements - 1);
            sum += surfaceRatio +
                   (centerRatio - surfaceRatio) * std::pow(r, exponent);
        }
        return sum;
    };

    std::vector<double> thicknesses(numHalfElements, surfaceRatio);
    thicknesses.back() = centerRatio;
    if (std::abs(targetSum - minSum) <= tolerance)
        return thicknesses;

    if (std::abs(targetSum - maxSum) <= tolerance)
    {
        std::fill(thicknesses.begin() + 1, thicknesses.end(), centerRatio);
        return thicknesses;
    }

    double low = 1e-12;
    double high = 1.0;
    while (sumForExponent(high) > targetSum)
    {
        high *= 2.0;
        if (high > 1e12)
            throw std::runtime_error(
                "Failed to solve H-direction graded mesh exponent.");
    }

    for (int iter = 0; iter < 100; ++iter)
    {
        const double mid = 0.5 * (low + high);
        if (sumForExponent(mid) > targetSum)
            low = mid;
        else
            high = mid;
    }

    const double exponent = 0.5 * (low + high);
    for (int i = 1; i < numHalfElements - 1; ++i)
    {
        const double r =
            static_cast<double>(i) /
            static_cast<double>(numHalfElements - 1);
        thicknesses[i] =
            surfaceRatio +
            (centerRatio - surfaceRatio) * std::pow(r, exponent);
    }
    return thicknesses;
}

std::vector<double> hDirectionGradedInternalKnots(int numElements,
                                                  double surfaceRatio,
                                                  double centerRatio)
{
    constexpr double tolerance = 1e-12;
    if (numElements < 3)
        throw std::invalid_argument(
            "H-direction graded mesh requires numEle_H >= 3.");
    if (surfaceRatio <= 0.0 || centerRatio <= 0.0 ||
        surfaceRatio >= 1.0 || centerRatio >= 1.0)
        throw std::invalid_argument(
            "H-direction element thickness ratios must be in (0, 1).");
    if (centerRatio < surfaceRatio)
        throw std::invalid_argument(
            "centerElementThicknessRatio must be at least surfaceElementThicknessRatio.");

    std::vector<double> thicknesses;
    thicknesses.reserve(static_cast<std::size_t>(numElements));

    if (numElements % 2 == 0)
    {
        const int numHalfElements = numElements / 2;
        const std::vector<double> half =
            solveGradedHalfThicknesses(numHalfElements,
                                       surfaceRatio,
                                       centerRatio,
                                       0.5);
        thicknesses.insert(thicknesses.end(), half.begin(), half.end());
        thicknesses.insert(thicknesses.end(), half.rbegin(), half.rend());
    }
    else
    {
        const int numSideElements = numElements / 2;
        const double sideTargetSum = 0.5 * (1.0 - centerRatio);
        const std::vector<double> side =
            solveGradedHalfThicknesses(numSideElements,
                                       surfaceRatio,
                                       centerRatio,
                                       sideTargetSum);
        thicknesses.insert(thicknesses.end(), side.begin(), side.end());
        thicknesses.push_back(centerRatio);
        thicknesses.insert(thicknesses.end(), side.rbegin(), side.rend());
    }

    double sum = 0.0;
    for (double thickness : thicknesses)
        sum += thickness;
    if (std::abs(sum - 1.0) > 1e-10)
        throw std::runtime_error("Internal error: H-direction thickness ratios do not sum to 1.");

    std::vector<double> knots;
    knots.reserve(static_cast<std::size_t>(numElements - 1));
    double knot = 0.0;
    for (int i = 0; i < numElements - 1; ++i)
    {
        knot += thicknesses[i];
        if (knot <= tolerance || knot >= 1.0 - tolerance)
            throw std::runtime_error("Internal error: generated H-direction knot is outside (0, 1).");
        knots.push_back(knot);
    }
    return knots;
}

bool envFlag(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
        return false;

    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text != "0" && text != "false" && text != "off";
}

int activeDofIndex(const DofMapper& mapper, int localDof, int patchIdx)
{
    const std::vector<int>& dofs = mapper.getDofs(0);
    const std::vector<int> offsets = mapper.getOffset();
    return dofs[offsets[patchIdx] + localDof] + mapper.getShift();
}

bool isFreeDof(const DofMapper& mapper, int activeIndex)
{
    return activeIndex < mapper.getCurElimId() + mapper.getShift();
}

double grevilleCoordinate(const TensorBsplineBasis& basis,
                          int direction,
                          int basisIndex)
{
    const int degree = basis.getOrder(direction);
    const std::vector<double>& knots = basis.getKnots(direction);
    if (degree <= 0)
        return knots[basisIndex];

    double coordinate = 0.0;
    for (int i = 1; i <= degree; ++i)
        coordinate += knots[basisIndex + i];
    return coordinate / static_cast<double>(degree);
}

Eigen::MatrixXd grevillePoints(const TensorBsplineBasis& basis)
{
    const int dim = basis.getDim();
    const int numControlPoints = basis.getNumControlPoints();
    Eigen::MatrixXd points(dim, numControlPoints);

    for (int cp = 0; cp < numControlPoints; ++cp)
    {
        int remainder = cp;
        for (int d = 0; d < dim; ++d)
        {
            const int basisIndex = remainder % basis.size(d);
            remainder /= basis.size(d);
            points(d, cp) = grevilleCoordinate(basis, d, basisIndex);
        }
    }

    return points;
}

MultiPatch buildRefinedGeometry(const MultiPatch& geometry,
                                const MultiBasis& basis)
{
    MultiPatch refinedGeometry;
    for (int patchIndex = 0; patchIndex < geometry.getNumPatches(); ++patchIndex)
    {
        const TensorBsplineBasis& refinedBasis = basis.basis(patchIndex);
        const Eigen::MatrixXd parametricPoints = grevillePoints(refinedBasis);

        Eigen::MatrixXd physicalPoints;
        geometry.patch(patchIndex).eval_into(parametricPoints, physicalPoints);

        const Eigen::MatrixXd refinedControlPoints = physicalPoints.transpose();
        Patch refinedPatch(refinedBasis, refinedControlPoints);
        refinedGeometry.addPatch(refinedPatch);
    }
    refinedGeometry.computeTopology();
    return refinedGeometry;
}

std::string controlPointLocationStepFilename(int outputStep)
{
    std::ostringstream name;
    name << "step_" << std::setw(4) << std::setfill('0') << outputStep << ".txt";
    return name.str();
}

void writeControlPointLocations(const GPUFunction& displacementFunction,
                                const MultiPatch& refinedGeometry,
                                const std::string& outputFolder,
                                int outputStep)
{
    if (outputStep < 0)
        throw std::invalid_argument("outputStep must be nonnegative.");
    if (displacementFunction.numPatches() != refinedGeometry.getNumPatches())
        throw std::invalid_argument("Displacement and refined geometry patch counts do not match.");

    std::filesystem::create_directories(outputFolder);
    std::ofstream out(outputFolder + "/" +
                      controlPointLocationStepFilename(outputStep));
    if (!out)
        throw std::runtime_error("Cannot open control-point location output file.");

    out << std::scientific << std::setprecision(16);
    for (int patchIndex = 0; patchIndex < refinedGeometry.getNumPatches();
         ++patchIndex)
    {
        const Eigen::MatrixXd referenceControlPoints =
            refinedGeometry.patch(patchIndex).getControlPoints();
        const Eigen::MatrixXd displacementControlPoints =
            displacementFunction.controlPoints(patchIndex);

        if (referenceControlPoints.rows() != displacementControlPoints.rows() ||
            referenceControlPoints.cols() != displacementControlPoints.cols())
        {
            throw std::runtime_error(
                "Refined geometry and displacement control points have incompatible sizes.");
        }

        const Eigen::MatrixXd currentControlPoints =
            referenceControlPoints + displacementControlPoints;
        for (Eigen::Index i = 0; i < currentControlPoints.rows(); ++i)
        {
            for (Eigen::Index j = 0; j < currentControlPoints.cols(); ++j)
            {
                if (j > 0)
                    out << '\t';
                out << currentControlPoints(i, j);
            }
            out << '\n';
        }
    }
}

void runFollowerMomentTangentCheck(GPUStrainGradientElasticityAssembler& assembler,
                                   GPUSolver& solver,
                                   const MultiBasis& basis,
                                   const BoundaryConditions& bcInfo,
                                   double loadFactor,
                                   double H)
{
    std::vector<DofMapper> dofMappers(2);
    basis.getMappers(true, bcInfo, dofMappers, true);
    SparseSystem sparseSystem(dofMappers, Eigen::VectorXi::Ones(2));

    Eigen::VectorXd solution;
    solver.solutionToHost(solution);

    Eigen::VectorXd direction = Eigen::VectorXd::Zero(solution.size());
    const char* componentFilter = std::getenv("SIGA_FOLLOWER_MOMENT_TANGENT_COMPONENT");
    const bool xOnly = componentFilter != nullptr && std::string(componentFilter) == "x";
    const bool yOnly = componentFilter != nullptr && std::string(componentFilter) == "y";
    const char* modeFilter = std::getenv("SIGA_FOLLOWER_MOMENT_TANGENT_MODE");
    const std::string perturbationMode =
        modeFilter == nullptr ? "sin" : std::string(modeFilter);
    const char* onlyBoundaryIndexText =
        std::getenv("SIGA_FOLLOWER_MOMENT_TANGENT_ONLY_BOUNDARY_INDEX");
    const int onlyBoundaryIndex =
        onlyBoundaryIndexText == nullptr || onlyBoundaryIndexText[0] == '\0'
            ? -1
            : std::atoi(onlyBoundaryIndexText);
    const Eigen::VectorXi eastDofs = basis.basis(0).boundary(BoxSide(boundary::east));
    const TensorBsplineBasis& patchBasis = basis.basis(0);
    const int sizeU = patchBasis.size(0);
    const int degreeV = patchBasis.getOrder(1);
    const std::vector<double>& knotsV = patchBasis.getKnots(1);
    auto grevilleV = [&](int basisIndex)
    {
        if (degreeV <= 0)
            return knotsV[basisIndex];

        double sum = 0.0;
        for (int k = 1; k <= degreeV; ++k)
            sum += knotsV[basisIndex + k];
        return sum / static_cast<double>(degreeV);
    };

    for (int i = 0; i < eastDofs.size(); ++i)
    {
        if (onlyBoundaryIndex >= 0 && i != onlyBoundaryIndex)
            continue;

        for (int comp = 0; comp < 2; ++comp)
        {
            if ((xOnly && comp != 0) || (yOnly && comp != 1))
                continue;

            const DofMapper& mapper = dofMappers[comp];
            const int active = activeDofIndex(mapper, eastDofs[i], 0);
            if (!isFreeDof(mapper, active))
                continue;

            const int global = sparseSystem.rowBlockOffset(comp) + active;
            if (global < 0 || global >= direction.size())
                continue;

            if (perturbationMode == "constant")
            {
                direction[global] = 1.0;
            }
            else if (perturbationMode == "linear")
            {
                const double denom =
                    static_cast<double>(eastDofs.size() > 1
                                            ? eastDofs.size() - 1
                                            : 1);
                direction[global] = static_cast<double>(i) / denom;
            }
            else if (perturbationMode == "greville_y")
            {
                const int vIndex = eastDofs[i] / sizeU;
                direction[global] = H * grevilleV(vIndex);
            }
            else
            {
                direction[global] = std::sin(0.37 * static_cast<double>(i + 1) +
                                             0.53 * static_cast<double>(comp + 1));
            }
        }
    }

    const double directionNorm = direction.norm();
    if (directionNorm == 0.0)
    {
        std::cout << "Follower moment tangent check skipped: empty east-boundary perturbation.\n";
        return;
    }
    direction /= directionNorm;
    const double affineStretchScale = 1.0 / directionNorm;

    const double previousScaling = assembler.options().getReal("neumann_load_scaling");
    assembler.options().setReal("neumann_load_scaling", loadFactor);

    auto assembleFollowerOnly = [&]()
    {
        assembler.constructDispSolution(solver.solutionView(),
                                        solver.allFixedDofsView());
        assembler.setMatrixAndRHSZeros();
        assembler.assembleFollowerMomentBoundaryCondition(
            solver.allFixedDofsView());
    };

    assembleFollowerOnly();
    using RowSparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
    RowSparseMatrix tangent = assembler.csrMatrix().toEigenCSR();
    const Eigen::VectorXd rhs0 = assembler.hostRHS();
    const Eigen::VectorXd tangentDirection = tangent * direction;

    std::cout << "Follower moment tangent finite-difference check at load factor "
              << loadFactor << "\n";
    if (xOnly || yOnly)
        std::cout << "  perturbation component: " << (xOnly ? "x" : "y") << "\n";
    std::cout << "  perturbation mode: " << perturbationMode << "\n";
    if (perturbationMode == "greville_y")
    {
        const Eigen::VectorXd expectedTangentDirection =
            affineStretchScale * rhs0;
        const double denominator = std::max(
            std::max(expectedTangentDirection.norm(),
                     tangentDirection.norm()),
            1e-30);
        std::cout << "  affine stretch scale: " << affineStretchScale
                  << ", rel||Kp - s*rhs||="
                  << (tangentDirection - expectedTangentDirection).norm() /
                         denominator
                  << "\n";
    }
    for (const double eps : {1e-4, 1e-5, 1e-6, 1e-7})
    {
        solver.setSolutionFromHost(solution + eps * direction);
        assembleFollowerOnly();
        const Eigen::VectorXd rhs1 = assembler.hostRHS();
        const Eigen::VectorXd finiteDifference = (rhs1 - rhs0) / eps;
        const double denominator = std::max(
            std::max(finiteDifference.norm(), tangentDirection.norm()), 1e-30);
        const Eigen::VectorXd negativeSignMismatch =
            finiteDifference + tangentDirection;
        const Eigen::VectorXd positiveSignMismatch =
            finiteDifference - tangentDirection;
        Eigen::Index maxMismatchIndex = 0;
        const double maxMismatch =
            negativeSignMismatch.cwiseAbs().maxCoeff(&maxMismatchIndex);
        const double errorForNegativeLoadDerivative =
            negativeSignMismatch.norm() / denominator;
        const double errorForPositiveLoadDerivative =
            positiveSignMismatch.norm() / denominator;

        std::cout << "  eps " << eps
                  << ": ||fd||=" << finiteDifference.norm()
                  << ", ||Kp||=" << tangentDirection.norm()
                  << ", rel||fd+Kp||=" << errorForNegativeLoadDerivative
                  << ", rel||fd-Kp||=" << errorForPositiveLoadDerivative
                  << ", max(fd+Kp)=" << maxMismatch
                  << " at dof " << maxMismatchIndex
                  << " (fd=" << finiteDifference[maxMismatchIndex]
                  << ", Kp=" << tangentDirection[maxMismatchIndex] << ")"
                  << "\n";
        const char* probeIndexText =
            std::getenv("SIGA_FOLLOWER_MOMENT_TANGENT_PROBE_DOF");
        if (probeIndexText != nullptr && probeIndexText[0] != '\0')
        {
            const Eigen::Index probeIndex =
                static_cast<Eigen::Index>(std::atoi(probeIndexText));
            if (probeIndex >= 0 && probeIndex < finiteDifference.size())
                std::cout << "    probe dof " << probeIndex
                          << ": fd=" << finiteDifference[probeIndex]
                          << ", Kp=" << tangentDirection[probeIndex]
                          << ", rhs=" << rhs0[probeIndex]
                          << "\n";
        }
    }

    solver.setSolutionFromHost(solution);
    assembler.options().setReal("neumann_load_scaling", previousScaling);
    assembleFollowerOnly();
}
} // namespace

int main(int argc, char* argv[])
{
    double L = 150.0;
    double H = 10.0;
    double YM = 1.0;
    double PR = 0.495;
    double lengthScale = 5.0;
    double followerMoment = -0.05;
    double centerElementThicknessRatio = 0.0;
    double surfaceElementThicknessRatio = 0.0;
    int numEle_L = 160;
    int numEle_H = 16;
    int numDegElev = 2;
    int numPointsPerPatchValue = 1000;
    int materialLaw = 1;
    double initialLoadStep = 0.005;
    int maxNewtonIterations = 50;
    int targetNumIterations = 5;
    bool outputGaussPointData = true;
    bool outputControlPointLocations = true;
    bool printTiming = false;
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
        followerMoment = parameterDouble(parameters, "followerMoment",
                                         followerMoment);
        centerElementThicknessRatio =
            parameterDouble(parameters, "centerElementThicknessRatio",
                            centerElementThicknessRatio);
        surfaceElementThicknessRatio =
            parameterDouble(parameters, "surfaceElementThicknessRatio",
                            surfaceElementThicknessRatio);
        numEle_L = parameterInt(parameters, "numEle_L", numEle_L);
        numEle_H = parameterInt(parameters, "numEle_H", numEle_H);
        numDegElev = parameterInt(parameters, "numDegElev", numDegElev);
        numPointsPerPatchValue = parameterInt(parameters, "numPointsPerPatch",
                                             numPointsPerPatchValue);
        materialLaw = parameterInt(parameters, "materialLaw", materialLaw);
        initialLoadStep = parameterDouble(parameters, "initialLoadStep",
                                          initialLoadStep);
        maxNewtonIterations = parameterInt(parameters, "maxNewtonIterations",
                                           maxNewtonIterations);
        targetNumIterations = parameterInt(parameters, "targetNumIterations",
                                           targetNumIterations);
        outputGaussPointData =
            parameterBool(parameters, "outputGaussPointData",
                          outputGaussPointData);
        outputControlPointLocations =
            parameterBool(parameters, "outputControlPointLocations",
                          outputControlPointLocations);
        printTiming = parameterBool(parameters, "printTiming", printTiming);
        outputPostfix = parameterString(parameters, "outputPostfix", outputPostfix);
    }
    else
    {
        if (argc > 1) numEle_L = std::stoi(argv[1]);
        if (argc > 2) numEle_H = std::stoi(argv[2]);
        if (argc > 3) numDegElev = std::stoi(argv[3]);
        if (argc > 4) followerMoment = std::stod(argv[4]);
        if (argc > 5) lengthScale = std::stod(argv[5]);
        if (argc > 6) L = std::stod(argv[6]);
        if (argc > 7) H = std::stod(argv[7]);
        if (argc > 8) YM = std::stod(argv[8]);
        if (argc > 9) PR = std::stod(argv[9]);
        if (argc > 10) materialLaw = std::stoi(argv[10]);
        if (argc > 11) initialLoadStep = std::stod(argv[11]);
        if (argc > 12) maxNewtonIterations = std::stoi(argv[12]);
        if (argc > 13) targetNumIterations = std::stoi(argv[13]);
        if (argc > 14) outputGaussPointData =
            parseBool(argv[14], "outputGaussPointData");
        if (argc > 15) outputControlPointLocations =
            parseBool(argv[15], "outputControlPointLocations");
        if (argc > 16) printTiming = parseBool(argv[16], "printTiming");
        if (argc > 17) centerElementThicknessRatio = std::stod(argv[17]);
        if (argc > 18) surfaceElementThicknessRatio = std::stod(argv[18]);
        outputPostfix = "manual";
        for (int i = 1; i < argc; ++i)
            outputPostfix += "_" + std::string(argv[i]);
    }

    if (L <= 0.0 || H <= 0.0)
        throw std::invalid_argument("L and H must be positive.");
    if (numEle_L < 1 || numEle_H < 1)
        throw std::invalid_argument("numEle_L and numEle_H must be positive.");
    if (numDegElev < 0)
        throw std::invalid_argument("numDegElev must be nonnegative.");
    if (materialLaw != 0 && materialLaw != 1)
        throw std::invalid_argument("materialLaw must be 0 (StVK) or 1 (neo-Hookean).");
    if (initialLoadStep <= 0.0 || initialLoadStep > 1.0)
        throw std::invalid_argument("initialLoadStep must be in (0, 1].");
    if (maxNewtonIterations < 1)
        throw std::invalid_argument("maxNewtonIterations must be positive.");
    const bool useGradedHMesh =
        centerElementThicknessRatio > 0.0 ||
        surfaceElementThicknessRatio > 0.0;
    if (useGradedHMesh &&
        (centerElementThicknessRatio <= 0.0 ||
         surfaceElementThicknessRatio <= 0.0))
    {
        throw std::invalid_argument(
            "Both centerElementThicknessRatio and surfaceElementThicknessRatio must be positive for graded H mesh.");
    }

    const std::string rootFolder = "./strainGradient_2DBeamBending_followerMoment_output";
    const std::string outputFolderName =
        "strainGradient_2DBeamBending_followerMoment_output_" + outputPostfix;
    const std::string filenameParaview = "sg_bend_fm_";
    const std::string outputFolder =
        rootFolder + "/" + outputFolderName;
    std::filesystem::create_directories(outputFolder);
    const std::string sectionDataFolder = outputFolder + "/SectionData";
    std::filesystem::create_directories(sectionDataFolder);
    const std::string gaussPointDataFolder = outputFolder + "/GaussPointData";
    if (outputGaussPointData)
        std::filesystem::create_directories(gaussPointDataFolder);
    const std::string controlPointLocationFolder =
        outputFolder + "/ControlPointLocations";
    if (outputControlPointLocations)
        std::filesystem::create_directories(controlPointLocationFolder);
    TeeLogger log(outputFolder + "/log.txt");

    std::cout << "2D strain-gradient cantilever beam bending with follower moment\n";
    std::cout << "Beam length: " << L << "\n";
    std::cout << "Beam height: " << H << "\n";
    std::cout << "Number of elements: "
              << numEle_L << " x " << numEle_H << "\n";
    if (useGradedHMesh)
    {
        std::cout << "H-direction graded mesh thickness/H: center = "
                  << centerElementThicknessRatio
                  << ", top/bottom = "
                  << surfaceElementThicknessRatio << "\n";
    }
    else
    {
        std::cout << "H-direction mesh: uniform\n";
    }
    std::cout << "Degree elevations: " << numDegElev << "\n";
    std::cout << "Material: Y = " << YM << ", nu = " << PR
              << ", length scale = " << lengthScale << "\n";
    std::cout << "Right-end follower moment: " << followerMoment << "\n";
    std::cout << "Initial load step factor: " << initialLoadStep << "\n";
    std::cout << "Max Newton iterations per load step: "
              << maxNewtonIterations << "\n";
    std::cout << "Target solver iterations per load step: "
              << targetNumIterations << "\n";
    std::cout << "Gauss-point data output: "
              << (outputGaussPointData ? "on" : "off") << "\n";
    std::cout << "Control-point location output: "
              << (outputControlPointLocations ? "on" : "off") << "\n";
    std::cout << "Timing output: "
              << (printTiming ? "on" : "off") << "\n";

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
    if (useGradedHMesh)
    {
        const std::vector<double> hKnots =
            hDirectionGradedInternalKnots(numEle_H,
                                          surfaceElementThicknessRatio,
                                          centerElementThicknessRatio);
        basis.insertKnots(1, hKnots);
    }
    else
    {
        basis.uniformRefine(1, numEle_H - 1);
    }
    const MultiPatch refinedGeometry =
        outputControlPointLocations ? buildRefinedGeometry(geometry, basis)
                                    : MultiPatch();

    BoundaryConditions bcInfo;
    const std::vector<double> zeros{0.0, 0.0};
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, zeros, 0);
    bcInfo.addCondition(0, boundary::southwest, condition_type::dirichlet, zeros, 1);

    const std::vector<double> moment{followerMoment};
    bcInfo.addCondition(0, boundary::east, condition_type::follower_moment,
                        moment, 0);

    Eigen::VectorXd bodyForce(2);
    bodyForce << 0.0, 0.0;

    GPUStrainGradientElasticityAssembler assembler(geometry, basis, bcInfo, bodyForce);
    assembler.options().setReal("youngs_modulus", YM);
    assembler.options().setReal("poissons_ratio", PR);
    assembler.options().setReal("length_scale", lengthScale);
    assembler.options().setReal("neumann_load_scaling", 0.0);
    assembler.options().setInt("material_law", materialLaw);
    assembler.options().setSwitch("use_nonsymmetric_newton_solver", true);
    assembler.options().setSwitch("print_timing", printTiming);
    std::cout << "Initialized system with " << assembler.numDofs() << " dofs.\n";

    GPUSolver solver(assembler);
    solver.setPrintTiming(printTiming);
    if (printTiming)
        solver.resetTimingStats();
    solver.setTolerance(1e-10, 1e-10);
    solver.setMaxIterations(maxNewtonIterations);

    if (envFlag("SIGA_FOLLOWER_MOMENT_TANGENT_CHECK_INITIAL") ||
        envFlag("SIGA_FOLLOWER_MOMENT_TANGENT_CHECK_ONLY"))
    {
        runFollowerMomentTangentCheck(assembler, solver, basis, bcInfo, 1.0,
                                      H);
        if (envFlag("SIGA_FOLLOWER_MOMENT_TANGENT_CHECK_ONLY"))
            return 0;
    }

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

    MultiPatch deformationGradientHost;
    basis.giveBasis(deformationGradientHost, 4);
    GPUFunction deformationGradientFunction(deformationGradientHost);
    assembler.constructDeformationGradientFunction(displacementFunction,
                                                   deformationGradientFunction);

    MultiPatch deformationGradientGradientHost;
    basis.giveBasis(deformationGradientGradientHost, 8);
    GPUFunction deformationGradientGradientFunction(
        deformationGradientGradientHost);

    MultiPatch greenLagrangeStrainGradientHost;
    basis.giveBasis(greenLagrangeStrainGradientHost, 8);
    GPUFunction greenLagrangeStrainGradientFunction(
        greenLagrangeStrainGradientHost);
    assembler.constructKinematicGradientFunctions(
        displacementFunction,
        deformationGradientGradientFunction,
        greenLagrangeStrainGradientFunction);

    const std::string filePrefix = outputFolder + "/" + filenameParaview;
    ParaviewCollection collection(filePrefix);

    std::vector<int> numPointsPerPatch{numPointsPerPatchValue};
    GPUPostProcessor postProcessor(assembler, numPointsPerPatch, true, 2);
    postProcessor.addFunction("displacement", &displacementFunction);
    postProcessor.addFunction("first_piola", &firstPiolaFunction);
    postProcessor.addFunction("stress_cauchy", &cauchyStressFunction);
    postProcessor.addFunction("grad_F", &deformationGradientGradientFunction);
    postProcessor.addFunction("grad_green_lagrange_strain",
                              &greenLagrangeStrainGradientFunction);

    auto outputFixedEndSectionStress11 = [&](int outputStep)
    {
        constexpr int numSectionPoints = 101;
        Eigen::MatrixXd sectionPoints(2, numSectionPoints);
        for (int i = 0; i < numSectionPoints; ++i)
        {
            sectionPoints(0, i) = 0.0;
            sectionPoints(1, i) =
                static_cast<double>(i) /
                static_cast<double>(numSectionPoints - 1);
        }

        const Eigen::MatrixXd sectionStress =
            firstPiolaFunction.eval(0, sectionPoints);
        const Eigen::MatrixXd sectionCauchyStress =
            cauchyStressFunction.eval(0, sectionPoints);
        const Eigen::MatrixXd sectionDeformationGradient =
            deformationGradientFunction.eval(0, sectionPoints);
        const std::string stepFolder =
            sectionDataFolder + "/step_" + std::to_string(outputStep);
        std::filesystem::create_directories(stepFolder);

        std::ofstream stressOut(stepFolder + "/SecStress11.txt");
        stressOut << std::setprecision(16);
        for (int i = 0; i < numSectionPoints; ++i)
            stressOut << sectionStress(0, i) << "\n";

        std::ofstream cauchyStressOut(stepFolder + "/SecCauStress11.txt");
        cauchyStressOut << std::setprecision(16);
        for (int i = 0; i < numSectionPoints; ++i)
            cauchyStressOut << sectionCauchyStress(0, i) << "\n";

        const auto writeDeformationGradientComponent =
            [&](const std::string& filename, int component)
        {
            std::ofstream out(stepFolder + "/" + filename);
            out << std::setprecision(16);
            for (int i = 0; i < numSectionPoints; ++i)
                out << sectionDeformationGradient(component, i) << "\n";
        };
        writeDeformationGradientComponent("SecF11.txt", 0);
        writeDeformationGradientComponent("SecF12.txt", 1);
        writeDeformationGradientComponent("SecF21.txt", 2);
        writeDeformationGradientComponent("SecF22.txt", 3);
    };

    auto writeParaviewOutput = [&](int outputStep, double loadFactor)
    {
        assembler.constructSolution(solver.solutionView(),
                                    solver.allFixedDofsView(),
                                    displacementFunction);
        if (outputControlPointLocations)
        {
            writeControlPointLocations(displacementFunction,
                                       refinedGeometry,
                                       controlPointLocationFolder,
                                       outputStep);
        }
        assembler.constructStrainGradientStressFunctions(displacementFunction,
                                                         firstPiolaFunction,
                                                         cauchyStressFunction);
        assembler.constructDeformationGradientFunction(displacementFunction,
                                                       deformationGradientFunction);
        assembler.constructKinematicGradientFunctions(
            displacementFunction,
            deformationGradientGradientFunction,
            greenLagrangeStrainGradientFunction);
        if (outputGaussPointData)
        {
            assembler.writeGaussPointKinematicsTSV(displacementFunction,
                                                   gaussPointDataFolder,
                                                   outputStep);
        }
        postProcessor.outputToParaview(filePrefix, outputStep, collection);
        collection.saveStep();
        outputFixedEndSectionStress11(outputStep);
        std::cout << "Wrote result output " << outputStep
                  << " at load factor " << loadFactor
                  << " and follower moment "
                  << loadFactor * followerMoment << ".\n";
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
                  << ", follower moment " << loadFactor * followerMoment
                  << ", step length " << loadStep << "\n";

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
    if (printTiming)
        solver.printTimingSummary("Total Newton solver timing");
    if (envFlag("SIGA_FOLLOWER_MOMENT_TANGENT_CHECK"))
        runFollowerMomentTangentCheck(assembler, solver, basis, bcInfo,
                                      loadFactor, H);
    std::cout << "Paraview output: " << outputFolder << "\n";
    return solver.isConverged() ? 0 : 2;
}
