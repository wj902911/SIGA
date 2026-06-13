#include <GPUSolver.h>
#include <GPUPostProcessor.h>
#include <TeeLogger.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <GPUFlexoelectriciyAssembler.h>
#include <fstream>
#include <iomanip>
#include <limits>
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
            throw std::runtime_error("Expected 'key: value' in " + path +
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

double solveCommonRatio(double firstTerm, double sum, int n)
{
    if (n <= 1)
        return 1.0;

    double x = 1.1;
    double xOld = 0.0;
    constexpr double tol = 1e-12;
    for (int iter = 0; iter < 100 && std::abs(x - xOld) > tol; ++iter)
    {
        xOld = x;
        const double xToN = std::pow(x, n);
        const double f = firstTerm * (1.0 - xToN) / (1.0 - x) - sum;
        const double df =
            (-firstTerm * n * std::pow(x, n - 1) * (1.0 - x) +
             firstTerm * (1.0 - xToN)) /
            std::pow(1.0 - x, 2);
        x -= f / df;
    }

    return x;
}

std::vector<double> gradedSubstrateInternalKnots(int numElements,
                                                 double substrateHeight,
                                                 double filmHeight,
                                                 int numFilmElements)
{
    std::vector<double> knots;
    if (numElements <= 1)
        return knots;

    const double filmElementHeight = filmHeight / static_cast<double>(numFilmElements);
    double topElementFraction = filmElementHeight / substrateHeight;
    if (topElementFraction <= 0.0 || topElementFraction * numElements >= 1.0)
        topElementFraction = 1.0 / static_cast<double>(numElements);

    const double q = solveCommonRatio(topElementFraction, 1.0, numElements);
    double knot = 0.0;
    knots.reserve(static_cast<std::size_t>(numElements - 1));
    for (int i = 0; i < numElements - 1; ++i)
    {
        knot += std::pow(q, numElements - 1 - i) * topElementFraction;
        knots.push_back(knot);
    }
    return knots;
}

} // namespace

int main(int argc, char* argv[])
{
    const bool useParameterFile = argc == 2 &&
        std::filesystem::path(argv[1]).extension() == ".txt";
    if (!useParameterFile && argc < 12)
    {
        std::cerr
            << "Usage: " << argv[0]
            << " <parameterFile.txt>\n"
            << "   or: " << argv[0]
            << " <lengthScale> <substrateLengthScale>"
            << " <length> <filmHeight> <substrateHeight>"
            << " <numEle_L> <numEle_Hf> <numEle_Hs> <numDegElev>"
            << " <initialDeltaDisp> <maxStrain>"
            << " [numBisectionIterations] [targetNumIterations]"
            << " [bucklingPerturbationAmplitude] [postBisectionDeltaDisp]"
            << " [resultOutputStrainInterval]"
            << " [muL] [substrateMuL] [muT] [substrateMuT]"
            << " [muS] [substrateMuS]"
            << " [dielectricPermittivity] [substrateDielectricPermittivity]"
            << " [YM] [substrateYM] [PR] [substratePR]"
            << " [materialLaw] [substrateMaterialLaw]"
            << " [includeHbarFlexoCorrection]"
            << " [substrateIncludeHbarFlexoCorrection]"
            << " [condensedEigenMaxIterations]"
            << " [numPointsPerPatch] [numPointsFilm] [numPointsSubstrate]"
            << " [outputPostfix] [detectSecondInstability]"
            << " [numNearZeroEigenvalues]\n"
            << "Use condensedEigenMaxIterations <= 0 for exact reduced-matrix LDLT stability.\n"
            << "Each omitted substrate material value defaults to the corresponding film value.\n"
            << "Example: " << argv[0]
            << " 0.4 0.4 32.05 0.4 19 24 1 8 2 -0.05 -0.05 20 5\n";
        return 1;
    }

    double YM = 1.725;
    double substrateYM = YM;
    double PR = 0.3;
    double substratePR = PR;
    double lengthScale = 0.0;
    double substrateLengthScale = 0.0;

    double L = 0.0;
    double Hf = 0.0;
    double Hs = 0.0;

    int numEle_L = 0;
    int numEle_Hf = 0;
    int numEle_Hs = 0;

    int numDegElev = 0;

    double initialDeltaDisp = 0.0;
    double maxStrain = 0.0;
    int numBisectionIterations = 20;
    int targetNumIterations = 5;
    double bucklingPerturbationAmplitude = 0.0;
    double postBisectionDeltaDisp = 0.0;
    double resultOutputStrainInterval = 0.0;
    double muL = 0.0;
    double substrateMuL = muL;
    double muT = 0.0;
    double substrateMuT = muT;
    double muS = 0.0;
    double substrateMuS = muS;
    double dielectricPermittivity = 0.092;
    double substrateDielectricPermittivity = dielectricPermittivity;
    int materialLaw = 1;
    int substrateMaterialLaw = materialLaw;
    int includeHbarFlexoCorrection = 0;
    int substrateIncludeHbarFlexoCorrection = includeHbarFlexoCorrection;
    int condensedEigenMaxIterations = 60;
    int numPointsPerPatchValue = 1000;
    int numPointsFilm = numPointsPerPatchValue;
    int numPointsSubstrate = numPointsPerPatchValue;
    int detectSecondInstability = 1;
    int numNearZeroEigenvalues = 1;
    std::string outputSuffix;

    if (useParameterFile)
    {
        const std::map<std::string, std::string> parameters =
            readParameterFile(argv[1]);
        lengthScale = parameterDouble(parameters, "lengthScale");
        substrateLengthScale =
            parameterDouble(parameters, "substrateLengthScale", lengthScale);
        L = parameterDouble(parameters, "L");
        Hf = parameterDouble(parameters, "Hf");
        Hs = parameterDouble(parameters, "Hs");
        numEle_L = parameterInt(parameters, "numEle_L");
        numEle_Hf = parameterInt(parameters, "numEle_Hf");
        numEle_Hs = parameterInt(parameters, "numEle_Hs");
        numDegElev = parameterInt(parameters, "numDegElev");
        initialDeltaDisp = parameterDouble(parameters, "initialDeltaDisp");
        maxStrain = parameterDouble(parameters, "maxStrain");
        numBisectionIterations =
            parameterInt(parameters, "numBisectionIterations",
                         numBisectionIterations);
        targetNumIterations =
            parameterInt(parameters, "targetNumIterations",
                         targetNumIterations);
        bucklingPerturbationAmplitude = parameterDouble(parameters,
            "bucklingPerturbationAmplitude", 1e-3 * abs(initialDeltaDisp));
        postBisectionDeltaDisp = parameterDouble(parameters,
            "postBisectionDeltaDisp", 0.1 * initialDeltaDisp);
        resultOutputStrainInterval = parameterDouble(parameters,
            "resultOutputStrainInterval", resultOutputStrainInterval);
        muL = parameterDouble(parameters, "muL", muL);
        substrateMuL = parameterDouble(parameters, "substrateMuL", muL);
        muT = parameterDouble(parameters, "muT", muT);
        substrateMuT = parameterDouble(parameters, "substrateMuT", muT);
        muS = parameterDouble(parameters, "muS", muS);
        substrateMuS = parameterDouble(parameters, "substrateMuS", muS);
        dielectricPermittivity = parameterDouble(parameters,
            "dielectricPermittivity", dielectricPermittivity);
        substrateDielectricPermittivity = parameterDouble(parameters,
            "substrateDielectricPermittivity", dielectricPermittivity);
        YM = parameterDouble(parameters, "YM", YM);
        substrateYM = parameterDouble(parameters, "substrateYM", YM);
        PR = parameterDouble(parameters, "PR", PR);
        substratePR = parameterDouble(parameters, "substratePR", PR);
        materialLaw = parameterInt(parameters, "materialLaw", materialLaw);
        substrateMaterialLaw = parameterInt(parameters, "substrateMaterialLaw",
                                            materialLaw);
        includeHbarFlexoCorrection = parameterInt(parameters,
            "includeHbarFlexoCorrection", includeHbarFlexoCorrection);
        substrateIncludeHbarFlexoCorrection = parameterInt(parameters,
            "substrateIncludeHbarFlexoCorrection",
            includeHbarFlexoCorrection);
        condensedEigenMaxIterations = parameterInt(parameters,
            "condensedEigenMaxIterations", condensedEigenMaxIterations);
        numPointsPerPatchValue = parameterInt(parameters, "numPointsPerPatch",
                                              numPointsPerPatchValue);
        numPointsFilm = parameterInt(parameters, "numPointsFilm",
                                     numPointsPerPatchValue);
        numPointsSubstrate = parameterInt(parameters, "numPointsSubstrate",
                                           numPointsPerPatchValue);
        detectSecondInstability = parameterInt(parameters,
            "detectSecondInstability", detectSecondInstability);
        numNearZeroEigenvalues = parameterInt(parameters,
            "numNearZeroEigenvalues", numNearZeroEigenvalues);
        const std::string outputPostfix = parameterString(
            parameters, "outputPostfix",
            std::filesystem::path(argv[1]).stem().string());
        if (!outputPostfix.empty())
            outputSuffix = "_" + outputPostfix;
    }
    else
    {
        lengthScale = std::stod(argv[1]);
        substrateLengthScale = std::stod(argv[2]);
        L = std::stod(argv[3]);
        Hf = std::stod(argv[4]);
        Hs = std::stod(argv[5]);
        numEle_L = std::stoi(argv[6]);
        numEle_Hf = std::stoi(argv[7]);
        numEle_Hs = std::stoi(argv[8]);
        numDegElev = std::stoi(argv[9]);
        initialDeltaDisp = std::stod(argv[10]);
        maxStrain = std::stod(argv[11]);
        if (argc > 12)
            numBisectionIterations = std::stoi(argv[12]);
        if (argc > 13)
            targetNumIterations = std::stoi(argv[13]);
        bucklingPerturbationAmplitude = 1e-3 * abs(initialDeltaDisp);
        if (argc > 14)
            bucklingPerturbationAmplitude = std::stod(argv[14]);
        postBisectionDeltaDisp = 0.1 * initialDeltaDisp;
        if (argc > 15)
            postBisectionDeltaDisp = std::stod(argv[15]);
        if (argc > 16)
            resultOutputStrainInterval = std::stod(argv[16]);
        if (argc > 17)
            muL = std::stod(argv[17]);
        substrateMuL = muL;
        if (argc > 18)
            substrateMuL = std::stod(argv[18]);
        if (argc > 19)
            muT = std::stod(argv[19]);
        substrateMuT = muT;
        if (argc > 20)
            substrateMuT = std::stod(argv[20]);
        if (argc > 21)
            muS = std::stod(argv[21]);
        substrateMuS = muS;
        if (argc > 22)
            substrateMuS = std::stod(argv[22]);
        if (argc > 23)
            dielectricPermittivity = std::stod(argv[23]);
        substrateDielectricPermittivity = dielectricPermittivity;
        if (argc > 24)
            substrateDielectricPermittivity = std::stod(argv[24]);
        if (argc > 25)
            YM = std::stod(argv[25]);
        substrateYM = YM;
        if (argc > 26)
            substrateYM = std::stod(argv[26]);
        if (argc > 27)
            PR = std::stod(argv[27]);
        substratePR = PR;
        if (argc > 28)
            substratePR = std::stod(argv[28]);
        if (argc > 29)
            materialLaw = std::stoi(argv[29]);
        substrateMaterialLaw = materialLaw;
        if (argc > 30)
            substrateMaterialLaw = std::stoi(argv[30]);
        if (argc > 31)
            includeHbarFlexoCorrection = std::stoi(argv[31]);
        substrateIncludeHbarFlexoCorrection = includeHbarFlexoCorrection;
        if (argc > 32)
            substrateIncludeHbarFlexoCorrection = std::stoi(argv[32]);
        if (argc > 33)
            condensedEigenMaxIterations = std::stoi(argv[33]);
        if (argc > 34)
            numPointsPerPatchValue = std::stoi(argv[34]);
        numPointsFilm = numPointsPerPatchValue;
        numPointsSubstrate = numPointsPerPatchValue;
        if (argc > 35)
            numPointsFilm = std::stoi(argv[35]);
        if (argc > 36)
            numPointsSubstrate = std::stoi(argv[36]);

        const std::string outputPostfix = argc > 37 ? std::string(argv[37]) : "";
        if (argc > 38)
            detectSecondInstability = std::stoi(argv[38]);
        if (argc > 39)
            numNearZeroEigenvalues = std::stoi(argv[39]);
        if (!outputPostfix.empty())
        {
            outputSuffix = "_" + outputPostfix;
        }
        else
        {
            for (int i = 1; i < argc; ++i)
            {
                if (i != 14 && i != 15)
                    outputSuffix += "_" + std::string(argv[i]);
            }
        }
    }
    if (numEle_L < 1 || numEle_Hf < 1 || numEle_Hs < 1)
        throw std::invalid_argument("Element counts must be positive.");

    if (numDegElev < 2)
        throw std::invalid_argument(
            "numDegElev must be at least 2 because this wrinkling example "
            "uses third-derivative boundary coupling.");
    if (numPointsFilm < 2 || numPointsSubstrate < 2)
        throw std::invalid_argument(
            "numPointsFilm and numPointsSubstrate must be at least 2 for Paraview output.");
    if (numNearZeroEigenvalues < 1)
        throw std::invalid_argument(
            "numNearZeroEigenvalues must be at least 1.");
    const bool useExactCondensedStability = condensedEigenMaxIterations <= 0;
    const int eigenvectorMaxIterations =
        condensedEigenMaxIterations > 0 ? condensedEigenMaxIterations : 60;
    const double meterToMicrometer = 1.0e-6;
    const double dielectricPermittivityModel = dielectricPermittivity * meterToMicrometer;
    const double muLModel = muL * meterToMicrometer;
    const double muTModel = muT * meterToMicrometer;
    const double muSModel = muS * meterToMicrometer;
    const double substrateDielectricPermittivityModel =
        substrateDielectricPermittivity * meterToMicrometer;
    const double substrateMuLModel = substrateMuL * meterToMicrometer;
    const double substrateMuTModel = substrateMuT * meterToMicrometer;
    const double substrateMuSModel = substrateMuS * meterToMicrometer;
    if (materialLaw != 0 && materialLaw != 1)
        throw std::invalid_argument("materialLaw must be 0 (StVK) or 1 (neo-Hookean).");
    if (substrateMaterialLaw != 0 && substrateMaterialLaw != 1)
        throw std::invalid_argument(
            "substrateMaterialLaw must be 0 (StVK) or 1 (neo-Hookean).");
    std::vector<int> numPointsPerPatch{ numPointsFilm, numPointsSubstrate };

    if (!std::filesystem::exists("./flexoelectricity_2DFilmWrinkling"))
		std::filesystem::create_directory("./flexoelectricity_2DFilmWrinkling");
	std::string filenameParaview = "flexoelectricity_2DFilmWrinkling_";
    std::string outputFolder =
        "./flexoelectricity_2DFilmWrinkling/" + filenameParaview +
        "output" + outputSuffix;
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
    const std::string nearZeroEigenvaluesFile =
        outputFolder + "/instability_near_zero_eigenvalues.txt";
    const std::string nearZeroEigenvaluesStepFile =
        outputFolder + "/near_zero_eigenvalues_steps.txt";

    double maxDisp = maxStrain * L;
    constexpr int filmPatchIndex = 0;
    constexpr int substratePatchIndex = 1;

    int knot_u_order = 1;
    int knot_v_order = 1;
    std::vector<double> knot_u{ 0., 0., 1., 1. };
    std::vector<double> knot_v{ 0., 0., 1., 1. };
    const std::vector<double> substrateVerticalAnalysisKnots =
        gradedSubstrateInternalKnots(numEle_Hs, Hs, Hf, numEle_Hf);

	Eigen::MatrixXd filmControlPoints(4, 2);
    filmControlPoints << 0., Hs,
                         L, Hs,
                         0., Hs + Hf,
                         L, Hs + Hf;
	Eigen::MatrixXd substrateControlPoints(4, 2);
    substrateControlPoints << 0., 0.,
                              L, 0.,
                              0., Hs,
                              L, Hs;
    KnotVector uFilm(knot_u_order, knot_u);
	KnotVector vFilm(knot_v_order, knot_v);
    KnotVector uSubstrate(knot_u_order, knot_u);
	KnotVector vSubstrate(knot_v_order, knot_v);
    Patch filmPatch(uFilm, vFilm, filmControlPoints);
    Patch substratePatch(uSubstrate, vSubstrate, substrateControlPoints);
    MultiPatch geometry;
	geometry.addPatch(filmPatch);
	geometry.addPatch(substratePatch);
	geometry.computeTopology();
	MultiBasis basisDisplacement(geometry);
	MultiBasis basisElectricPotential(geometry);

    for (int i = 0; i < numDegElev; ++i)
        basisElectricPotential.degreeElevate();
    for (int i = 0; i < numDegElev; ++i)
        basisDisplacement.degreeElevate();

    basisDisplacement.uniformRefine(0, numEle_L - 1);
    basisElectricPotential.uniformRefine(0, numEle_L - 1);
    basisDisplacement.uniformRefine(filmPatchIndex, 1, numEle_Hf - 1);
    basisElectricPotential.uniformRefine(filmPatchIndex, 1, numEle_Hf - 1);
    basisDisplacement.insertKnots(substratePatchIndex, 1, substrateVerticalAnalysisKnots);
    basisElectricPotential.insertKnots(substratePatchIndex, 1, substrateVerticalAnalysisKnots);

    std::cout << "Film Young's Modulus: " << YM << "\n";
    std::cout << "Substrate Young's Modulus: " << substrateYM << "\n";
    std::cout << "Film Poisson's Ratio: " << PR << "\n";
    std::cout << "Substrate Poisson's Ratio: " << substratePR << "\n";
    std::cout << "Film material law: " << (materialLaw == 0 ? "StVK" : "neo-Hookean")
              << " (" << materialLaw << ")\n";
    std::cout << "Substrate material law: "
              << (substrateMaterialLaw == 0 ? "StVK" : "neo-Hookean")
              << " (" << substrateMaterialLaw << ")\n";
    std::cout << "Film length scale: " << lengthScale << "\n";
    std::cout << "Substrate length scale: " << substrateLengthScale << "\n";
    std::cout << "Film dielectric permittivity input: " << dielectricPermittivity
              << " nJ/(V^2 m), model units: "
              << dielectricPermittivityModel << " nJ/(V^2 um)\n";
    std::cout << "Substrate dielectric permittivity input: "
              << substrateDielectricPermittivity
              << " nJ/(V^2 m), model units: "
              << substrateDielectricPermittivityModel << " nJ/(V^2 um)\n";
    std::cout << "Film flexoelectric tensor input: mu_L = " << muL
              << ", mu_T = " << muT
              << ", mu_S = " << muS << " nJ/(V m)\n";
    std::cout << "Film flexoelectric tensor model units: mu_L = " << muLModel
              << ", mu_T = " << muTModel
              << ", mu_S = " << muSModel << " nJ/(V um)\n";
    std::cout << "Substrate flexoelectric tensor input: mu_L = "
              << substrateMuL
              << ", mu_T = " << substrateMuT
              << ", mu_S = " << substrateMuS << " nJ/(V m)\n";
    std::cout << "Substrate flexoelectric tensor model units: mu_L = "
              << substrateMuLModel
              << ", mu_T = " << substrateMuTModel
              << ", mu_S = " << substrateMuSModel << " nJ/(V um)\n";
    std::cout << "Film hbar flexoelectric correction: "
              << (includeHbarFlexoCorrection ? "on" : "off") << "\n";
    std::cout << "Substrate hbar flexoelectric correction: "
              << (substrateIncludeHbarFlexoCorrection ? "on" : "off") << "\n";
    if (useExactCondensedStability)
        std::cout << "Condensed stability: exact reduced matrix with SimplicialLDLT\n";
    else
        std::cout << "Condensed eigen max iterations: "
                  << condensedEigenMaxIterations << "\n";
    std::cout << "Film-substrate length: " << L << "\n";
    std::cout << "Film height: " << Hf << "\n";
    std::cout << "Substrate height: " << Hs << "\n";
    std::cout << "Film elements: " << numEle_L << " x " << numEle_Hf << "\n";
    std::cout << "Substrate elements: " << numEle_L << " x " << numEle_Hs
              << " (graded in vertical direction)\n";
    std::cout << "Basis function degree: " << numDegElev + 1 << "\n";
    std::cout << "Paraview points in film patch: " << numPointsFilm << "\n";
    std::cout << "Paraview points in substrate patch: "
              << numPointsSubstrate << "\n";
    std::cout << "Max displacement: " << maxDisp << "\n";
    std::cout << "Bisection iterations: " << numBisectionIterations << "\n";
    std::cout << "Target solver iterations per step: " << targetNumIterations << "\n";
    std::cout << "Buckling perturbation amplitude: " << bucklingPerturbationAmplitude << "\n";
    std::cout << "Post-bisection step length: " << postBisectionDeltaDisp << "\n";
    std::cout << "Detect second instability: "
              << (detectSecondInstability ? "on" : "off") << "\n";
    if (resultOutputStrainInterval > 0.0)
        std::cout << "Result output strain interval: " << resultOutputStrainInterval << "\n";
    else
        std::cout << "Result output strain interval: every converged step\n";

    BoundaryConditions bcInfo;
    std::vector<double> zeros{0.0, 0.0};
    bcInfo.addCondition(filmPatchIndex, boundary::west, condition_type::dirichlet, zeros, 0);
    bcInfo.addCondition(substratePatchIndex, boundary::west, condition_type::dirichlet, zeros, 0);
    bcInfo.addCondition(substratePatchIndex, boundary::south, condition_type::dirichlet, zeros, 1);
    std::vector<double> disp{initialDeltaDisp, 0.0};
    bcInfo.addCondition(filmPatchIndex, boundary::east, condition_type::dirichlet, disp, 0);
    bcInfo.addCondition(substratePatchIndex, boundary::east, condition_type::dirichlet, disp, 0);
    //bcInfo.addCondition(filmPatchIndex, boundary::east, condition_type::dirichlet, zeros, 2);
    //bcInfo.addCondition(substratePatchIndex, boundary::east, condition_type::dirichlet, zeros, 2);
    bcInfo.addCondition(substratePatchIndex, boundary::south, condition_type::dirichlet, zeros, 2);
    for (int patchIndex : {filmPatchIndex, substratePatchIndex})
    {
        bcInfo.addBoundaryCoupling(patchIndex, boundary::west, 1, 1);
        //bcInfo.addBoundaryCoupling(patchIndex, boundary::west, 2, 1);
        //bcInfo.addBoundaryCoupling(patchIndex, boundary::west, 3, 1);
        bcInfo.addBoundaryCoupling(patchIndex, boundary::east, 1, 1);
        //bcInfo.addBoundaryCoupling(patchIndex, boundary::east, 2, 1);
        //bcInfo.addBoundaryCoupling(patchIndex, boundary::east, 3, 1);
    }

    Eigen::VectorXd bodyForce(2);
	bodyForce << 0.0, 0.0;

    GPUFlexoelectriciyAssembler assembler(geometry, basisDisplacement,
                                          basisElectricPotential, bcInfo,
                                          bodyForce);
    assembler.options().setReal("youngs_modulus", YM);
	assembler.options().setReal("poissons_ratio", PR);
    assembler.options().setReal("length_scale", lengthScale);
    assembler.options().setReal("dielectric_permittivity", dielectricPermittivityModel);
    assembler.options().setReal("flexoelectric_mu_L", muLModel);
    assembler.options().setReal("flexoelectric_mu_T", muTModel);
    assembler.options().setReal("flexoelectric_mu_S", muSModel);
    assembler.options().setInt("material_law", materialLaw);
    assembler.options().setInt("include_hbar_flexo_correction",
                               includeHbarFlexoCorrection);
    assembler.setPatchRealOption("youngs_modulus", {YM, substrateYM});
	assembler.setPatchRealOption("poissons_ratio", {PR, substratePR});
    assembler.setPatchRealOption("length_scale",
                                 {lengthScale, substrateLengthScale});
    assembler.setPatchRealOption("dielectric_permittivity",
        {dielectricPermittivityModel, substrateDielectricPermittivityModel});
    assembler.setPatchRealOption("flexoelectric_mu_L",
                                 {muLModel, substrateMuLModel});
    assembler.setPatchRealOption("flexoelectric_mu_T",
                                 {muTModel, substrateMuTModel});
    assembler.setPatchRealOption("flexoelectric_mu_S",
                                 {muSModel, substrateMuSModel});
    assembler.setPatchIntOption("material_law",
                                {materialLaw, substrateMaterialLaw});
    assembler.setPatchIntOption("include_hbar_flexo_correction",
        {includeHbarFlexoCorrection, substrateIncludeHbarFlexoCorrection});
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

    MultiPatch electricPotentialHost;
    basisElectricPotential.giveBasis(electricPotentialHost, 1);
    GPUFunction electricPotentialFunction(electricPotentialHost);
    postProcessor.addFunction("electric_potential", &electricPotentialFunction);

    MultiPatch electricFieldHost;
    basisElectricPotential.giveBasis(electricFieldHost, 2);
    GPUFunction electricFieldFunction(electricFieldHost);
    postProcessor.addFunction("electric_field", &electricFieldFunction);

    MultiPatch cauchyStressHost;
    const int cauchyStressDim = assembler.domainDim() * (assembler.domainDim() + 1) / 2;
    basisDisplacement.giveBasis(cauchyStressHost, cauchyStressDim);
    GPUFunction cauchyStressFunction(cauchyStressHost);
    postProcessor.addFunction("stress_cauchy", &cauchyStressFunction);


    int outputStep = 0;
    auto updateOutputFunctions = [&]()
    {
        assembler.constructSolution(solver.solutionView(), solver.allFixedDofsView(),
                                    displacementFunction);
        assembler.constructElecSolution(solver.solutionView(), solver.allFixedDofsView(),
                                        electricPotentialFunction);
        assembler.constructElectricFieldFunction(electricPotentialFunction,
                                                 electricFieldFunction);
        assembler.constructCauchyStressFunction(displacementFunction,
                                                cauchyStressFunction);
    };

    auto condensedStability = [&]()
    {
        if (useExactCondensedStability)
            return solver.condensedMechanicalStabilityLDLT();

        Eigen::VectorXd displacementEigenvector;
        Eigen::VectorXd electricEigenvector;
        return solver.smallestCondensedMechanicalEigenpair(
            displacementEigenvector, &electricEigenvector,
            condensedEigenMaxIterations);
    };

    auto outputSectionData = [&](int outputStepIndex)
    {
        constexpr int numSectionPoints = 101;
        Eigen::MatrixXd topSurfacePoints(2, numSectionPoints);
        for (int i = 0; i < numSectionPoints; ++i)
        {
            topSurfacePoints(0, i) =
                static_cast<double>(i) / static_cast<double>(numSectionPoints - 1);
            topSurfacePoints(1, i) = 1.0;
        }

        const Eigen::MatrixXd topDisplacement =
            displacementFunction.eval(filmPatchIndex, topSurfacePoints);
        const Eigen::MatrixXd topCauchyStress =
            cauchyStressFunction.eval(filmPatchIndex, topSurfacePoints);
        const Eigen::MatrixXd topPotential =
            electricPotentialFunction.eval(filmPatchIndex, topSurfacePoints);
        const Eigen::MatrixXd topElectricField =
            electricFieldFunction.eval(filmPatchIndex, topSurfacePoints);
        const std::string stepFolder = sectionDataFolder + "/step_" + std::to_string(outputStepIndex);
        if (!std::filesystem::exists(stepFolder))
            std::filesystem::create_directory(stepFolder);

        std::ofstream displacementOut(stepFolder + "/TopDisplacementY.txt");
        displacementOut << std::setprecision(16);
        for (int i = 0; i < numSectionPoints; ++i)
            displacementOut << topDisplacement(1, i) << "\n";

        std::ofstream cauchyStressOut(stepFolder + "/SecCauStress11.txt");
        cauchyStressOut << std::setprecision(16);
        for (int i = 0; i < numSectionPoints; ++i)
            cauchyStressOut << topCauchyStress(0, i) << "\n";

        std::ofstream potentialOut(stepFolder + "/SecElectricPotential.txt");
        potentialOut << std::setprecision(16);
        for (int i = 0; i < numSectionPoints; ++i)
            potentialOut << topPotential(0, i) << "\n";

        std::ofstream electricFieldOut(stepFolder + "/SecElectricFieldY.txt");
        electricFieldOut << std::setprecision(16);
        for (int i = 0; i < numSectionPoints; ++i)
            electricFieldOut << topElectricField(1, i) << "\n";
    };

    bool nearZeroEigenvaluesStepHeaderWritten = false;
    auto outputNearZeroEigenvaluesForStep =
        [&](int stepIndex)
    {
        if (numNearZeroEigenvalues <= 1)
            return;

        std::vector<double> rowEigenvalues;
        std::vector<std::string> rowLabels;

        std::cout << "Computing " << numNearZeroEigenvalues
                  << " full-system eigenvalues closest to zero for step "
                  << stepIndex << " using Spectra.\n";

        const Eigen::VectorXd nearZeroEigenvalues =
            solver.smallestEigenValue(numNearZeroEigenvalues);
        if (nearZeroEigenvalues.size() == 0)
        {
            std::cout << "No near-zero eigenvalues were returned. "
                      << "Make sure ENABLE_SPECTRA=ON for this calculation.\n";
            return;
        }

        for (int i = 0; i < numNearZeroEigenvalues; ++i)
        {
            rowLabels.push_back("eigenvalue_" + std::to_string(i + 1));
            rowEigenvalues.push_back(
                i < nearZeroEigenvalues.size()
                ? nearZeroEigenvalues[i]
                : std::numeric_limits<double>::quiet_NaN());
        }

        std::ofstream nearZeroOut(
            nearZeroEigenvaluesStepFile,
            nearZeroEigenvaluesStepHeaderWritten
                ? std::ios::app
                : std::ios::out);
        nearZeroOut << std::setprecision(16);
        if (!nearZeroEigenvaluesStepHeaderWritten)
        {
            nearZeroOut << "# step";
            for (const std::string& label : rowLabels)
                nearZeroOut << " " << label;
            nearZeroOut << "\n";
            nearZeroEigenvaluesStepHeaderWritten = true;
        }

        nearZeroOut << stepIndex;
        for (double value : rowEigenvalues)
            nearZeroOut << " " << value;
        nearZeroOut << "\n";

        std::cout << "Near-zero eigenvalues at step " << stepIndex
                  << ":\n";
        for (std::size_t i = 0; i < rowEigenvalues.size(); ++i)
        {
            std::cout << "  " << rowLabels[i] << ": "
                      << rowEigenvalues[i] << "\n";
        }
    };

    updateOutputFunctions();
	postProcessor.outputToParaview(fileNameWithPath, outputStep++, collection);
    outputSectionData(outputStep - 1);
    double nextOutputStrain = resultOutputStrainInterval;
    const double outputStrainTolerance = std::max(1e-12, 1e-8 * std::abs(maxStrain));

    int step = 1;
    double appliedDisp = 0.0;
    double deltaDisp = initialDeltaDisp;
    bool instabilityOnsetStored = false;
    bool bucklingPerturbationApplied = false;
    int instabilityRecordIndex = 0;
    bool havePerturbationRetryState = false;
    bool perturbationRetryPending = false;
    bool retryUsesFullEigenvector = false;
    Eigen::VectorXd retryBaseSolution;
    Eigen::VectorXd retryBaseFixedDofs;
    Eigen::VectorXd retryFullEigenvector;
    Eigen::VectorXd retryDisplacementEigenvector;
    Eigen::VectorXd retryElectricEigenvector;
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

        if (hasStoredCriticalData && !bucklingPerturbationApplied)
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
        if (!solver.isConverged() && havePerturbationRetryState)
        {
            const double oldAmplitude = bucklingPerturbationAmplitude;
            bucklingPerturbationAmplitude *= 0.5;
            std::cout << "The step using the perturbed solution did not "
                      << "converge. Reducing buckling perturbation amplitude "
                      << "from " << oldAmplitude << " to "
                      << bucklingPerturbationAmplitude
                      << " and retrying the same load step.\n";

            solver.setSolutionFromHost(retryBaseSolution);
            solver.setFixedDofsFromHost(retryBaseFixedDofs);
            if (retryUsesFullEigenvector)
            {
                Eigen::VectorXd perturbedSolution = retryBaseSolution;
                if (perturbedSolution.size() == retryFullEigenvector.size())
                {
                    perturbedSolution +=
                        bucklingPerturbationAmplitude * retryFullEigenvector;
                    solver.setSolutionFromHost(perturbedSolution);
                }
                else
                {
                    std::cout << "Cannot retry full-vector perturbation because "
                              << "solution and eigenvector sizes differ.\n";
                }
            }
            else if (!solver.perturbWithCondensedEigenvectors(
                         retryDisplacementEigenvector,
                         retryElectricEigenvector,
                         bucklingPerturbationAmplitude, 1))
            {
                std::cout << "Cannot retry condensed eigenvector perturbation "
                          << "because the stored eigenvectors are invalid.\n";
            }

            appliedDisp = previousAppliedDisp;
            disp[0] = deltaDisp;
            assembler.refreshFixedDofs();
            continue;
        }
        if (solver.isConverged() && perturbationRetryPending)
        {
            havePerturbationRetryState = false;
            perturbationRetryPending = false;
        }

        double currentStability = condensedStability();
        std::cout << "Stability: " << currentStability << "\n";
        outputNearZeroEigenvaluesForStep(step);

        const bool reachedStoredCriticalDisp =
            hasStoredCriticalData && !bucklingPerturbationApplied &&
            std::abs(appliedDisp - storedCriticalDisp) <= 1e-10;

        const bool shouldHandleInstability =
            reachedStoredCriticalDisp ||
            (currentStability <= 0.0 &&
             (!bucklingPerturbationApplied || detectSecondInstability));

        if (shouldHandleInstability)
        {
            if (reachedStoredCriticalDisp && !bucklingPerturbationApplied)
            {
                std::cout << "Reached stored instability onset. Skipping bisection and "
                          << "using stored smallest eigenvalue "
                          << storedCriticalEigenvalue << ".\n";

                Eigen::VectorXd criticalSolution;
                Eigen::VectorXd criticalFixedDofs;
                solver.solutionToHost(criticalSolution);
                solver.fixedDofsToHost(criticalFixedDofs);

                retryBaseSolution = criticalSolution;
                retryBaseFixedDofs = criticalFixedDofs;
                retryFullEigenvector = storedCriticalEigenvector;
                retryDisplacementEigenvector.resize(0);
                retryElectricEigenvector.resize(0);
                retryUsesFullEigenvector = true;
                havePerturbationRetryState = true;
                perturbationRetryPending = true;

                if (criticalSolution.size() == storedCriticalEigenvector.size())
                    criticalSolution +=
                        bucklingPerturbationAmplitude * storedCriticalEigenvector;
                else
                    std::cout << "Skipping buckling perturbation because "
                              << "solution and eigenvector sizes differ.\n";

                solver.setSolutionFromHost(criticalSolution);
                solver.setFixedDofsFromHost(criticalFixedDofs);
                appliedDisp = storedCriticalDisp;
                currentStability = storedCriticalEigenvalue;
                deltaDisp = postBisectionDeltaDisp;
                disp[0] = deltaDisp;
                assembler.refreshFixedDofs();
                instabilityRecordIndex = std::max(instabilityRecordIndex, 1);
                instabilityOnsetStored = true;
                bucklingPerturbationApplied = true;
                skipStepLengthAdjustment = true;
                std::cout << "Restarting from stored instability onset "
                          << "with eigenvector perturbation.\n";
                std::cout << "Using post-bisection step length "
                          << deltaDisp << " for the next step.\n";
            }
            else
            {
                std::cout << "Instability detected. Starting bisection between "
                          << previousAppliedDisp << " and " << appliedDisp
                          << ".\n";

                double stableDisp = previousAppliedDisp;
                double unstableDisp = appliedDisp;
                double stableStability = previousStability;
                double unstableStability = currentStability;

                for (int bisectionIter = 0;
                     bisectionIter < numBisectionIterations;
                     ++bisectionIter)
                {
                    const double trialDisp =
                        0.5 * (stableDisp + unstableDisp);
                    disp[0] = trialDisp - previousAppliedDisp;
                    assembler.refreshFixedDofs();
                    solver.setSolutionFromHost(previousSolution);
                    solver.setFixedDofsFromHost(previousFixedDofs);

                    std::cout << "Bisection " << bisectionIter + 1
                              << "/" << numBisectionIterations
                              << ": trial displacement " << trialDisp
                              << "\n";

                    solver.solve();
                    const double trialStability = condensedStability();
                    std::cout << "Trial stability: " << trialStability
                              << "\n";

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

                Eigen::VectorXd criticalDisplacementEigenvector;
                Eigen::VectorXd criticalElectricEigenvector;
                std::cout << "Computing condensed smallest eigenpair "
                          << "at bisection onset.\n";
                const double criticalEigenvalue =
                    solver.smallestCondensedMechanicalEigenpair(
                        criticalDisplacementEigenvector,
                        &criticalElectricEigenvector,
                        eigenvectorMaxIterations);
                const double criticalStability =
                    useExactCondensedStability
                    ? solver.condensedMechanicalStabilityLDLT()
                    : criticalEigenvalue;

                Eigen::VectorXd criticalEigenvector(
                    criticalDisplacementEigenvector.size() +
                    criticalElectricEigenvector.size());
                criticalEigenvector << criticalDisplacementEigenvector,
                                       criticalElectricEigenvector;

                Eigen::VectorXd criticalSolution;
                Eigen::VectorXd criticalFixedDofs;
                solver.solutionToHost(criticalSolution);
                solver.fixedDofsToHost(criticalFixedDofs);
                retryBaseSolution = criticalSolution;
                retryBaseFixedDofs = criticalFixedDofs;
                retryFullEigenvector.resize(0);
                retryDisplacementEigenvector = criticalDisplacementEigenvector;
                retryElectricEigenvector = criticalElectricEigenvector;
                retryUsesFullEigenvector = false;
                havePerturbationRetryState = true;
                perturbationRetryPending = true;

                const int currentInstabilityRecord = ++instabilityRecordIndex;
                const std::string recordSuffix =
                    currentInstabilityRecord == 1
                    ? std::string()
                    : "_" + std::to_string(currentInstabilityRecord);
                const std::string currentOnsetFile =
                    currentInstabilityRecord == 1
                    ? onsetFile
                    : outputFolder + "/instability_onset" + recordSuffix + ".txt";
                const std::string currentEigenvalueFile =
                    currentInstabilityRecord == 1
                    ? eigenvalueFile
                    : outputFolder + "/instability_smallest_eigenvalue" +
                          recordSuffix + ".txt";
                const std::string currentEigenvectorFile =
                    currentInstabilityRecord == 1
                    ? eigenvectorFile
                    : outputFolder + "/instability_smallest_eigenvector" +
                          recordSuffix + ".txt";
                const std::string currentNearZeroEigenvaluesFile =
                    currentInstabilityRecord == 1
                    ? nearZeroEigenvaluesFile
                    : outputFolder + "/instability_near_zero_eigenvalues" +
                          recordSuffix + ".txt";

                std::ofstream onsetOut(currentOnsetFile);
                onsetOut << std::setprecision(16)
                         << "instability_index "
                         << currentInstabilityRecord << "\n"
                         << "critical_applied_displacement "
                         << unstableDisp << "\n"
                         << "lower_stable_applied_displacement "
                         << stableDisp << "\n"
                         << "upper_unstable_applied_displacement "
                         << unstableDisp << "\n"
                         << "lower_stability " << stableStability << "\n"
                         << "upper_stability " << unstableStability << "\n"
                         << "critical_stability " << criticalStability
                         << "\n"
                         << "buckling_perturbation_amplitude "
                         << bucklingPerturbationAmplitude << "\n"
                         << "bisection_iterations "
                         << numBisectionIterations << "\n";

                std::ofstream eigenvalueOut(currentEigenvalueFile);
                eigenvalueOut << std::setprecision(16)
                              << criticalEigenvalue << "\n";

                std::ofstream eigenvectorOut(currentEigenvectorFile);
                eigenvectorOut << std::setprecision(16);
                for (Eigen::Index i = 0;
                     i < criticalEigenvector.size();
                     ++i)
                    eigenvectorOut << criticalEigenvector[i] << "\n";

                if (numNearZeroEigenvalues > 1)
                {
                    std::cout << "Computing " << numNearZeroEigenvalues
                              << " full-system eigenvalues closest to zero "
                              << "using Spectra.\n";
                    const Eigen::VectorXd nearZeroEigenvalues =
                        solver.smallestEigenValue(numNearZeroEigenvalues);
                    if (nearZeroEigenvalues.size() == 0)
                    {
                        std::cout << "No near-zero eigenvalues were returned. "
                                  << "Make sure ENABLE_SPECTRA=ON for this "
                                  << "calculation.\n";
                    }
                    else
                    {
                        std::ofstream nearZeroOut(
                            currentNearZeroEigenvaluesFile);
                        nearZeroOut << std::setprecision(16);
                        std::cout << "Near-zero eigenvalues:\n";
                        for (Eigen::Index i = 0;
                             i < nearZeroEigenvalues.size();
                             ++i)
                        {
                            nearZeroOut << nearZeroEigenvalues[i] << "\n";
                            std::cout << "  " << i + 1 << ": "
                                      << nearZeroEigenvalues[i] << "\n";
                        }

                        std::cout << "Stored "
                                  << nearZeroEigenvalues.size()
                                  << " near-zero eigenvalues in "
                                  << currentNearZeroEigenvaluesFile
                                  << ".\n";
                    }
                }

                std::cout << "Stored instability onset record "
                          << currentInstabilityRecord
                          << " at displacement " << unstableDisp
                          << " with smallest eigenvalue "
                          << criticalEigenvalue << ".\n";

                solver.setSolutionFromHost(criticalSolution);
                solver.setFixedDofsFromHost(criticalFixedDofs);
                if (!solver.perturbWithCondensedEigenvectors(
                        criticalDisplacementEigenvector,
                        criticalElectricEigenvector,
                        bucklingPerturbationAmplitude, 1))
                    std::cout << "Skipping buckling perturbation because "
                              << "condensed eigenvectors are invalid.\n";

                appliedDisp = unstableDisp;
                currentStability = criticalStability;
                deltaDisp = postBisectionDeltaDisp;
                disp[0] = deltaDisp;
                assembler.refreshFixedDofs();
                instabilityOnsetStored = true;
                bucklingPerturbationApplied = true;
                skipStepLengthAdjustment = true;
                std::cout << "Restarting from instability onset "
                          << "with eigenvector perturbation.\n";
                std::cout << "Using post-bisection step length "
                          << deltaDisp << " for the next step.\n";
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
            updateOutputFunctions();

		    postProcessor.outputToParaview(fileNameWithPath, outputStep, collection);
            outputSectionData(outputStep);
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
