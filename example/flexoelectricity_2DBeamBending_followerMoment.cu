#include "device_launch_parameters.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <GPUFlexoelectriciyAssembler.h>
#include <GPUPostProcessor.h>
#include <GPUSolver.h>
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

std::string normalizedText(const std::string& value)
{
    std::string text = trim(value);
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
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

enum class BottomElectricGrounding
{
    SouthSide,
    SouthwestCorner
};

BottomElectricGrounding parseBottomElectricGrounding(const std::string& value,
                                                     const std::string& key)
{
    const std::string text = normalizedText(value);
    if (text == "south" || text == "bottom" || text == "side" ||
        text == "surface" || text == "bottom-side" ||
        text == "bottom_side" || text == "south-side" ||
        text == "south_side")
        return BottomElectricGrounding::SouthSide;

    if (text == "southwest" || text == "sw" || text == "bottom-left" ||
        text == "bottom_left" || text == "corner" ||
        text == "southwest-corner" || text == "southwest_corner")
        return BottomElectricGrounding::SouthwestCorner;

    throw std::invalid_argument(
        "Expected " + key +
        " to be south/bottom/side or southwest/bottom-left/corner.");
}

BottomElectricGrounding parameterBottomElectricGrounding(
    const std::map<std::string, std::string>& parameters,
    const std::string& key,
    BottomElectricGrounding defaultValue)
{
    const auto it = parameters.find(key);
    if (it == parameters.end())
        return defaultValue;
    return parseBottomElectricGrounding(it->second, key);
}

std::string bottomElectricGroundingName(BottomElectricGrounding grounding)
{
    switch (grounding)
    {
    case BottomElectricGrounding::SouthSide:
        return "south side";
    case BottomElectricGrounding::SouthwestCorner:
        return "southwest corner";
    }
    return "unknown";
}

std::string boundarySideName(const BoxSide& side)
{
    switch (side.index())
    {
    case boundary::west: return "west";
    case boundary::east: return "east";
    case boundary::south: return "south";
    case boundary::north: return "north";
    case boundary::front: return "front";
    case boundary::back: return "back";
    default: return "none";
    }
}

Eigen::MatrixXd boundarySamplePoints(const BoxSide& side, int numPoints)
{
    if (numPoints < 1)
        throw std::invalid_argument("numPoints must be positive.");

    const int fixedDirection = side.direction();
    if (fixedDirection < 0 || fixedDirection >= 2)
        throw std::invalid_argument(
            "Electrode boundary output is only implemented for 2D sides.");

    const int freeDirection = fixedDirection == 0 ? 1 : 0;
    const double fixedValue = side.parameter() ? 1.0 : 0.0;

    Eigen::MatrixXd points(2, numPoints);
    for (int i = 0; i < numPoints; ++i)
    {
        points(fixedDirection, i) = fixedValue;
        points(freeDirection, i) =
            numPoints == 1
                ? 0.0
                : static_cast<double>(i) /
                      static_cast<double>(numPoints - 1);
    }
    return points;
}

bool parseOptionalBoundarySide(const std::string& value,
                               BoxSide& side,
                               const std::string& key)
{
    const std::string text = normalizedText(value);
    if (text.empty() || text == "none" || text == "off" ||
        text == "false" || text == "0")
        return false;

    if (text == "west" || text == "left")
    {
        side = BoxSide(boundary::west);
        return true;
    }
    if (text == "east" || text == "right")
    {
        side = BoxSide(boundary::east);
        return true;
    }
    if (text == "south" || text == "bottom" || text == "down")
    {
        side = BoxSide(boundary::south);
        return true;
    }
    if (text == "north" || text == "top" || text == "up")
    {
        side = BoxSide(boundary::north);
        return true;
    }
    if (text == "front")
    {
        side = BoxSide(boundary::front);
        return true;
    }
    if (text == "back")
    {
        side = BoxSide(boundary::back);
        return true;
    }

    throw std::invalid_argument(
        "Expected boundary side for " + key +
        " (none, west/east/south/north, or left/right/bottom/top).");
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

double envDouble(const char* name, double defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
        return defaultValue;
    return std::atof(value);
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

void runFollowerMomentTangentCheck(GPUFlexoelectriciyAssembler& assembler,
                                   GPUSolver& solver,
                                   const MultiBasis& basisDisplacement,
                                   const MultiBasis& basisElectricPotential,
                                   const BoundaryConditions& bcInfo,
                                   double loadFactor,
                                   double H)
{
    std::vector<DofMapper> dofMappers(3);
    basisDisplacement.getMappers(true, bcInfo, dofMappers, true);
    basisElectricPotential.getMapper(true, bcInfo, 2, dofMappers.back(), true);
    SparseSystem sparseSystem(dofMappers, Eigen::VectorXi::Ones(3));

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
    const Eigen::VectorXi eastDofs =
        basisDisplacement.basis(0).boundary(BoxSide(boundary::east));
    const TensorBsplineBasis& patchBasis = basisDisplacement.basis(0);
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
    double dielectricPermittivity = 0.092;
    double muL = 0.0;
    double muT = 10.0;
    double muS = 0.0;
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
    int includeHbarFlexoCorrection = 0;
    bool outputGaussPointData = false;
    bool outputControlPointLocations = true;
    bool outputMesh = true;
    bool printTiming = false;
    std::string outputPostfix = "default";
    std::string electrodeBoundary = "none";
    BottomElectricGrounding bottomElectricGrounding =
        BottomElectricGrounding::SouthSide;
    bool useTopElectrode = false;
    bool useTopElectrodeOptionSet = false;

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
        dielectricPermittivity =
            parameterDouble(parameters, "dielectricPermittivity",
                            dielectricPermittivity);
        muL = parameterDouble(parameters, "muL", muL);
        muT = parameterDouble(parameters, "muT", muT);
        muS = parameterDouble(parameters, "muS", muS);
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
        includeHbarFlexoCorrection =
            parameterInt(parameters, "includeHbarFlexoCorrection",
                         includeHbarFlexoCorrection);
        outputGaussPointData =
            parameterBool(parameters, "outputGaussPointData",
                          outputGaussPointData);
        outputControlPointLocations =
            parameterBool(parameters, "outputControlPointLocations",
                          outputControlPointLocations);
        outputMesh = parameterBool(parameters, "outputMesh", outputMesh);
        printTiming = parameterBool(parameters, "printTiming", printTiming);
        electrodeBoundary =
            parameterString(parameters, "electrodeBoundary", electrodeBoundary);
        bottomElectricGrounding =
            parameterBottomElectricGrounding(parameters,
                                            "bottomElectricGrounding",
                                            bottomElectricGrounding);
        const auto useTopElectrodeIt = parameters.find("useTopElectrode");
        if (useTopElectrodeIt != parameters.end())
        {
            useTopElectrodeOptionSet = true;
            useTopElectrode =
                parameterBool(parameters, "useTopElectrode", useTopElectrode);
        }
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
        if (argc > 17) outputMesh = parseBool(argv[17], "outputMesh");
        if (argc > 18) centerElementThicknessRatio = std::stod(argv[18]);
        if (argc > 19) surfaceElementThicknessRatio = std::stod(argv[19]);
        if (argc > 20) dielectricPermittivity = std::stod(argv[20]);
        if (argc > 21) muL = std::stod(argv[21]);
        if (argc > 22) muT = std::stod(argv[22]);
        if (argc > 23) muS = std::stod(argv[23]);
        if (argc > 24) includeHbarFlexoCorrection = std::stoi(argv[24]);
        if (argc > 25) electrodeBoundary = argv[25];
        if (argc > 26)
        {
            useTopElectrodeOptionSet = true;
            useTopElectrode = parseBool(argv[26], "useTopElectrode");
        }
        if (argc > 27)
            bottomElectricGrounding =
                parseBottomElectricGrounding(argv[27],
                                             "bottomElectricGrounding");
        outputPostfix = "manual";
        for (int i = 1; i < argc; ++i)
            outputPostfix += "_" + std::string(argv[i]);
    }

    if (useTopElectrodeOptionSet)
        electrodeBoundary = useTopElectrode ? "north" : "none";

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
    if (outputGaussPointData)
        throw std::invalid_argument(
            "outputGaussPointData is not available for the flexoelectric follower-moment example.");
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
    BoxSide electrodeBoundarySide;
    const bool useElectrodeBoundary =
        parseOptionalBoundarySide(electrodeBoundary, electrodeBoundarySide,
                                  "electrodeBoundary");

    const std::string rootFolder =
        "./flexoelectricity_2DBeamBending_followerMoment_output";
    const std::string outputFolderName =
        "flexoelectricity_2DBeamBending_followerMoment_output_" +
        outputPostfix;
    const std::string filenameParaview = "flexo_bend_fm_";
    const std::string outputFolder =
        rootFolder + "/" + outputFolderName;
    std::filesystem::create_directories(outputFolder);
    const std::string sectionDataFolder = outputFolder + "/SectionData";
    std::filesystem::create_directories(sectionDataFolder);
    const std::string controlPointLocationFolder =
        outputFolder + "/ControlPointLocations";
    if (outputControlPointLocations)
        std::filesystem::create_directories(controlPointLocationFolder);
    TeeLogger log(outputFolder + "/log.txt");

    std::cout << "2D flexoelectric cantilever beam bending with follower moment\n";
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
    std::cout << "Mechanical material: Y = " << YM << ", nu = " << PR
              << ", length scale = " << lengthScale << "\n";
    std::cout << "Dielectric permittivity: "
              << dielectricPermittivity << "\n";
    std::cout << "Flexoelectric tensor: mu_L = " << muL
              << ", mu_T = " << muT
              << ", mu_S = " << muS << "\n";
    std::cout << "hbar flexoelectric correction: "
              << (includeHbarFlexoCorrection ? "on" : "off") << "\n";
    std::cout << "Electrical BC: "
              << bottomElectricGroundingName(bottomElectricGrounding)
              << " grounded";
    if (useElectrodeBoundary)
        std::cout << ", " << boundarySideName(electrodeBoundarySide)
                  << " side equipotential electrode";
    else
        std::cout << ", remaining boundaries open circuit";
    std::cout << "\n";
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
    std::cout << "Mesh output: "
              << (outputMesh ? "on" : "off") << "\n";
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

    MultiBasis basisDisplacement(geometry);
    MultiBasis basisElectricPotential(geometry);
    for (int i = 0; i < numDegElev; ++i)
    {
        basisDisplacement.degreeElevate(false);
        basisElectricPotential.degreeElevate(false);
    }
    basisDisplacement.uniformRefine(0, numEle_L - 1);
    basisElectricPotential.uniformRefine(0, numEle_L - 1);
    if (useGradedHMesh)
    {
        const std::vector<double> hKnots =
            hDirectionGradedInternalKnots(numEle_H,
                                          surfaceElementThicknessRatio,
                                          centerElementThicknessRatio);
        basisDisplacement.insertKnots(1, hKnots);
        basisElectricPotential.insertKnots(1, hKnots);
    }
    else
    {
        basisDisplacement.uniformRefine(1, numEle_H - 1);
        basisElectricPotential.uniformRefine(1, numEle_H - 1);
    }
    const MultiPatch refinedGeometry =
        outputControlPointLocations
            ? buildRefinedGeometry(geometry, basisDisplacement)
            : MultiPatch();

    BoundaryConditions bcInfo;
    const std::vector<double> zeros{0.0, 0.0};
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, zeros, 0);
    bcInfo.addCondition(0, boundary::southwest, condition_type::dirichlet, zeros, 1);
    if (bottomElectricGrounding == BottomElectricGrounding::SouthSide)
    {
        bcInfo.addCondition(0, boundary::south, condition_type::dirichlet,
                            zeros, 2);
    }
    else
    {
        bcInfo.addCondition(0, boundary::southwest, condition_type::dirichlet,
                            zeros, 2);
    }
    if (useElectrodeBoundary)
        bcInfo.addElectrodeBoundary(0, electrodeBoundarySide, 2);

    const std::vector<double> moment{followerMoment};
    bcInfo.addCondition(0, boundary::east, condition_type::follower_moment,
                        moment, 0);

    Eigen::VectorXd bodyForce(2);
    bodyForce << 0.0, 0.0;

    GPUFlexoelectriciyAssembler assembler(geometry, basisDisplacement,
                                          basisElectricPotential, bcInfo,
                                          bodyForce);
    assembler.options().setReal("youngs_modulus", YM);
    assembler.options().setReal("poissons_ratio", PR);
    assembler.options().setReal("length_scale", lengthScale);
    assembler.options().setReal("dielectric_permittivity",
                                dielectricPermittivity);
    assembler.options().setReal("flexoelectric_mu_L", muL);
    assembler.options().setReal("flexoelectric_mu_T", muT);
    assembler.options().setReal("flexoelectric_mu_S", muS);
    assembler.options().setReal("neumann_load_scaling", 0.0);
    assembler.options().setInt("material_law", materialLaw);
    assembler.options().setInt("include_hbar_flexo_correction",
                               includeHbarFlexoCorrection);
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
        runFollowerMomentTangentCheck(assembler, solver, basisDisplacement,
                                      basisElectricPotential, bcInfo, 1.0, H);
        if (envFlag("SIGA_FOLLOWER_MOMENT_TANGENT_CHECK_ONLY"))
            return 0;
    }

    MultiPatch displacementHost;
    basisDisplacement.giveBasis(displacementHost, 2);
    GPUFunction displacementFunction(displacementHost);
    assembler.constructSolution(solver.solutionView(),
                                solver.allFixedDofsView(),
                                displacementFunction);

    MultiPatch electricPotentialHost;
    basisElectricPotential.giveBasis(electricPotentialHost, 1);
    GPUFunction electricPotentialFunction(electricPotentialHost);

    MultiPatch electricFieldHost;
    basisElectricPotential.giveBasis(electricFieldHost, 2);
    GPUFunction electricFieldFunction(electricFieldHost);

    MultiPatch cauchyStressHost;
    basisDisplacement.giveBasis(cauchyStressHost, assembler.dimTensor());
    GPUFunction cauchyStressFunction(cauchyStressHost);
    assembler.constructCauchyStressFunction(displacementFunction,
                                            cauchyStressFunction);

    MultiPatch deformationGradientHost;
    basisDisplacement.giveBasis(deformationGradientHost, 4);
    GPUFunction deformationGradientFunction(deformationGradientHost);
    assembler.constructDeformationGradientFunction(displacementFunction,
                                                   deformationGradientFunction);

    MultiPatch deformationGradientGradientHost;
    basisDisplacement.giveBasis(deformationGradientGradientHost, 8);
    GPUFunction deformationGradientGradientFunction(
        deformationGradientGradientHost);

    MultiPatch greenLagrangeStrainGradientHost;
    basisDisplacement.giveBasis(greenLagrangeStrainGradientHost, 8);
    GPUFunction greenLagrangeStrainGradientFunction(
        greenLagrangeStrainGradientHost);
    assembler.constructKinematicGradientFunctions(
        displacementFunction,
        deformationGradientGradientFunction,
        greenLagrangeStrainGradientFunction);

    const std::string filePrefix = outputFolder + "/" + filenameParaview;
    ParaviewCollection collection(filePrefix);

    std::vector<int> numPointsPerPatch{numPointsPerPatchValue};
    GPUPostProcessor postProcessor(assembler, numPointsPerPatch, outputMesh, 2);
    postProcessor.addFunction("displacement", &displacementFunction);
    postProcessor.addFunction("stress_cauchy", &cauchyStressFunction);
    postProcessor.addFunction("electric_potential", &electricPotentialFunction);
    postProcessor.addFunction("electric_field", &electricFieldFunction);
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

        const Eigen::MatrixXd sectionCauchyStress =
            cauchyStressFunction.eval(0, sectionPoints);
        const Eigen::MatrixXd sectionDeformationGradient =
            deformationGradientFunction.eval(0, sectionPoints);
        const Eigen::MatrixXd sectionPotential =
            electricPotentialFunction.eval(0, sectionPoints);
        const Eigen::MatrixXd sectionElectricField =
            electricFieldFunction.eval(0, sectionPoints);
        const std::string stepFolder =
            sectionDataFolder + "/step_" + std::to_string(outputStep);
        std::filesystem::create_directories(stepFolder);
        const auto openSectionFile = [&](const std::string& filename)
        {
            const std::string path = stepFolder + "/" + filename;
            std::ofstream out(path);
            if (!out)
                throw std::runtime_error("Cannot open section output file: " +
                                         path);
            out << std::setprecision(16);
            return out;
        };

        std::ofstream stressOut = openSectionFile("SecStress11.txt");
        for (int i = 0; i < numSectionPoints; ++i)
        {
            const double sigma11 = sectionCauchyStress(0, i);
            const double sigma12 = sectionCauchyStress(2, i);
            const double f12 = sectionDeformationGradient(1, i);
            const double f22 = sectionDeformationGradient(3, i);
            stressOut << sigma11 * f22 - sigma12 * f12 << "\n";
        }

        std::ofstream cauchyStressOut =
            openSectionFile("SecCauStress11.txt");
        for (int i = 0; i < numSectionPoints; ++i)
            cauchyStressOut << sectionCauchyStress(0, i) << "\n";

        std::ofstream potentialOut =
            openSectionFile("SecElectricPotential.txt");
        for (int i = 0; i < numSectionPoints; ++i)
            potentialOut << sectionPotential(0, i) << "\n";

        std::ofstream electricFieldOut =
            openSectionFile("SecElectricFieldY.txt");
        for (int i = 0; i < numSectionPoints; ++i)
            electricFieldOut << sectionElectricField(1, i) << "\n";

        const auto writeDeformationGradientComponent =
            [&](const std::string& filename, int component)
        {
            std::ofstream out = openSectionFile(filename);
            for (int i = 0; i < numSectionPoints; ++i)
                out << sectionDeformationGradient(component, i) << "\n";
        };
        writeDeformationGradientComponent("SecF11.txt", 0);
        writeDeformationGradientComponent("SecF12.txt", 1);
        writeDeformationGradientComponent("SecF21.txt", 2);
        writeDeformationGradientComponent("SecF22.txt", 3);
    };

    auto outputFinalElectrodePotential = [&](int lastOutputStep)
    {
        if (!useElectrodeBoundary)
            return;

        constexpr int numBoundaryPoints = 101;
        const Eigen::MatrixXd boundaryPoints =
            boundarySamplePoints(electrodeBoundarySide, numBoundaryPoints);
        const Eigen::MatrixXd boundaryPotential =
            electricPotentialFunction.eval(0, boundaryPoints);

        const double electrodePotential = boundaryPotential.row(0).mean();
        const double minPotential = boundaryPotential.row(0).minCoeff();
        const double maxPotential = boundaryPotential.row(0).maxCoeff();
        const double spread = maxPotential - minPotential;

        const std::string stepFolder =
            sectionDataFolder + "/step_" + std::to_string(lastOutputStep);
        std::filesystem::create_directories(stepFolder);
        const std::string path =
            stepFolder + "/ElectrodeElectricPotential.txt";
        std::ofstream out(path);
        if (!out)
            throw std::runtime_error("Cannot open electrode potential output file: " +
                                     path);
        out << std::setprecision(16) << electrodePotential << "\n";

        std::cout << "Wrote final electrode electric potential to "
                  << path << " (boundary: "
                  << boundarySideName(electrodeBoundarySide)
                  << ", sampled spread: " << spread << ").\n";
    };

    auto writeParaviewOutput = [&](int outputStep, double loadFactor)
    {
        assembler.constructSolution(solver.solutionView(),
                                    solver.allFixedDofsView(),
                                    displacementFunction);
        assembler.constructElecSolution(solver.solutionView(),
                                        solver.allFixedDofsView(),
                                        electricPotentialFunction);
        assembler.constructElectricFieldFunction(electricPotentialFunction,
                                                 electricFieldFunction);
        if (outputControlPointLocations)
        {
            writeControlPointLocations(displacementFunction,
                                       refinedGeometry,
                                       controlPointLocationFolder,
                                       outputStep);
        }
        assembler.constructCauchyStressFunction(displacementFunction,
                                                cauchyStressFunction);
        assembler.constructDeformationGradientFunction(displacementFunction,
                                                       deformationGradientFunction);
        assembler.constructKinematicGradientFunctions(
            displacementFunction,
            deformationGradientGradientFunction,
            greenLagrangeStrainGradientFunction);
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

        if (envFlag("SIGA_FLEXO_NUMERIC_JACOBIAN_CHECK"))
        {
            Eigen::VectorXd jacobianCheckSolution;
            solver.solutionToHost(jacobianCheckSolution);
            const double relativeStep = envDouble(
                "SIGA_FLEXO_NUMERIC_JACOBIAN_REL_STEP", 1.0e-6);
            std::cout << "Checking flexoelectric numerical Jacobian at step "
                      << step << ", load factor " << loadFactor << ".\n";
            assembler.checkNumericalJacobian(jacobianCheckSolution,
                                             solver.allFixedDofsView(),
                                             relativeStep,
                                             stepNumIterations);
            if (envFlag("SIGA_FLEXO_NUMERIC_JACOBIAN_CHECK_ONLY"))
                return 0;
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
    outputFinalElectrodePotential(outputStep - 1);

    Eigen::VectorXd solution;
    solver.solutionToHost(solution);
    std::cout << "Solved at full load. Solution norm: " << solution.norm() << "\n";
    if (printTiming)
        solver.printTimingSummary("Total Newton solver timing");
    if (envFlag("SIGA_FOLLOWER_MOMENT_TANGENT_CHECK"))
        runFollowerMomentTangentCheck(assembler, solver, basisDisplacement,
                                      basisElectricPotential, bcInfo,
                                      loadFactor, H);
    std::cout << "Paraview output: " << outputFolder << "\n";
    return solver.isConverged() ? 0 : 2;
}
