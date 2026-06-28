#pragma once

#include <DeviceArray.h>
#include <DeviceCSRMatrix.h>
#include <MultiPatchDeviceView.h>
#include <SparseSystemDeviceView.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace siga::gpuasm
{

inline bool envFlag(const char* name, bool defaultValue)
{
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0')
        return defaultValue;
    return value[0] != '0' && value[0] != 'f' && value[0] != 'F' &&
           value[0] != 'n' && value[0] != 'N';
}

inline int envInt(const char* name, int defaultValue)
{
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0')
        return defaultValue;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value)
        return defaultValue;
    if (parsed < std::numeric_limits<int>::min())
        return std::numeric_limits<int>::min();
    if (parsed > std::numeric_limits<int>::max())
        return std::numeric_limits<int>::max();
    return static_cast<int>(parsed);
}

inline double envDouble(const char* name, double defaultValue)
{
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0')
        return defaultValue;
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || !std::isfinite(parsed))
        return defaultValue;
    return parsed;
}

inline std::string gibString(unsigned long long bytes)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(2)
        << static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)
        << " GiB";
    return out.str();
}

inline unsigned long long bytesForCount(long long count,
                                        unsigned long long elementSize)
{
    if (count <= 0)
        return 0;
    return static_cast<unsigned long long>(count) * elementSize;
}

template <typename T>
unsigned long long vectorBytes(DeviceVectorView<T> view)
{
    return bytesForCount(view.size(), sizeof(T));
}

template <typename T>
unsigned long long nestedArrayBytes(DeviceNestedArrayView<T> view)
{
    return vectorBytes(view.offsetsView()) + vectorBytes(view.wholeView());
}

inline void printCudaMemoryReport(
    const std::string& label, unsigned long long requiredBytes,
    bool enabled,
    const std::vector<std::pair<std::string, unsigned long long>>& parts = {})
{
    if (!enabled)
        return;

    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        std::cout << "Assembly memory report [" << label
                  << "]: cudaGetDevice failed: " << cudaGetErrorString(err)
                  << ", required " << gibString(requiredBytes)
                  << " (" << requiredBytes << " bytes)\n";
        return;
    }

    size_t freeMem = 0;
    size_t totalMem = 0;
    err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess)
    {
        std::cout << "Assembly memory report [" << label << "] device "
                  << device << ": cudaMemGetInfo failed: "
                  << cudaGetErrorString(err) << ", required "
                  << gibString(requiredBytes) << " (" << requiredBytes
                  << " bytes)\n";
        return;
    }

    const unsigned long long freeBytes =
        static_cast<unsigned long long>(freeMem);
    const unsigned long long totalBytes =
        static_cast<unsigned long long>(totalMem);
    std::cout << "Assembly memory report [" << label << "] device " << device
              << ": required " << gibString(requiredBytes)
              << " (" << requiredBytes << " bytes), available/free "
              << gibString(freeBytes) << " (" << freeBytes
              << " bytes), total " << gibString(totalBytes)
              << " (" << totalBytes << " bytes)";
    if (requiredBytes > freeBytes)
        std::cout << " -- required exceeds currently available memory";
    std::cout << "\n";

    for (const auto& part : parts)
    {
        if (part.second == 0)
            continue;
        std::cout << "  " << part.first << ": " << gibString(part.second)
                  << " (" << part.second << " bytes)\n";
    }
}

inline std::vector<int> usableAssemblyDevices(bool enableMultiGPU)
{
    int primaryDevice = 0;
    cudaError_t err = cudaGetDevice(&primaryDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("cudaGetDevice failed before GPU assembly");

    std::vector<int> devices{primaryDevice};
    int deviceCount = 0;
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
        deviceCount = 1;

    if (enableMultiGPU && deviceCount > 1)
    {
        for (int device = 0; device < deviceCount; ++device)
        {
            if (device == primaryDevice)
                continue;

            int canDeviceReadPrimary = 0;
            int canPrimaryReadDevice = 0;
            cudaDeviceCanAccessPeer(&canDeviceReadPrimary, device,
                                    primaryDevice);
            cudaDeviceCanAccessPeer(&canPrimaryReadDevice, primaryDevice,
                                    device);
            if (!canDeviceReadPrimary || !canPrimaryReadDevice)
                continue;

            cudaSetDevice(device);
            cudaError_t peerErr = cudaDeviceEnablePeerAccess(primaryDevice, 0);
            if (peerErr == cudaErrorPeerAccessAlreadyEnabled)
                cudaGetLastError();
            else if (peerErr != cudaSuccess)
                continue;

            cudaSetDevice(primaryDevice);
            peerErr = cudaDeviceEnablePeerAccess(device, 0);
            if (peerErr == cudaErrorPeerAccessAlreadyEnabled)
                cudaGetLastError();
            else if (peerErr != cudaSuccess)
                continue;

            devices.push_back(device);
        }
        cudaSetDevice(primaryDevice);
    }

    return devices;
}

inline void printDeviceSelection(const char* label,
                                 const std::vector<int>& devices,
                                 bool enableMultiGPU)
{
    static bool reported = false;
    if (reported)
        return;

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
        deviceCount = static_cast<int>(devices.size());

    std::cout << label << " CUDA devices: " << devices.size()
              << " usable of " << deviceCount << " visible";
    if (!enableMultiGPU)
        std::cout << " (multi-GPU disabled)";
    else if (devices.size() < static_cast<std::size_t>(deviceCount))
        std::cout << " (non-peer-accessible devices skipped)";
    std::cout << "\n";
    reported = true;
}

template <typename T>
DeviceArray<T> peerCopy(DeviceVectorView<T> source, int sourceDevice,
                        int targetDevice, const char* label)
{
    DeviceArray<T> target(source.size());
    if (source.size() == 0)
        return target;

    cudaError_t err = cudaMemcpyPeer(target.data(), targetDevice,
                                     source.data(), sourceDevice,
                                     source.size() * sizeof(T));
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cudaMemcpyPeer failed for ") +
                                 label + ": " + cudaGetErrorString(err));
    return target;
}

template <typename T>
void peerCopyInto(DeviceArray<T>& target, DeviceVectorView<T> source,
                  int sourceDevice, int targetDevice, const char* label)
{
    if (target.size() != source.size())
        target.resize(source.size());
    if (source.size() == 0)
        return;

    cudaError_t err = cudaMemcpyPeer(target.data(), targetDevice,
                                     source.data(), sourceDevice,
                                     source.size() * sizeof(T));
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cudaMemcpyPeer failed for ") +
                                 label + ": " + cudaGetErrorString(err));
}

template <typename T>
void peerCopySliceInto(DeviceArray<T>& target, const T* sourceData,
                       int sourceSize, int sourceDevice, int targetDevice,
                       const char* label)
{
    if (target.size() != sourceSize)
        target.resize(sourceSize);
    if (sourceSize == 0)
        return;

    cudaError_t err = cudaMemcpyPeer(target.data(), targetDevice, sourceData,
                                     sourceDevice,
                                     static_cast<std::size_t>(sourceSize) *
                                         sizeof(T));
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cudaMemcpyPeer failed for ") +
                                 label + ": " + cudaGetErrorString(err));
}

template <typename T>
struct NestedArrayReplica
{
    DeviceArray<int> offsets;
    DeviceArray<T> data;

    void update(DeviceNestedArrayView<T> source, int sourceDevice,
                int targetDevice, const char* label)
    {
        const std::string offsetLabel = std::string(label) + " offsets";
        const std::string dataLabel = std::string(label) + " data";
        peerCopyInto(offsets, source.offsetsView(), sourceDevice,
                     targetDevice, offsetLabel.c_str());
        peerCopyInto(data, source.wholeView(), sourceDevice, targetDevice,
                     dataLabel.c_str());
    }

    DeviceNestedArrayView<T> view() const
    {
        return DeviceNestedArrayView<T>(offsets.vectorView(),
                                        data.vectorView());
    }
};

struct MultiPatchReplica
{
    int numPatches = 0;
    int domainDim = 0;
    int targetDim = 0;
    DeviceArray<int> patchIntDataOffsets;
    DeviceArray<int> patchKnotsPoolOffsets;
    DeviceArray<int> patchControlPointsPoolOffsets;
    DeviceArray<int> intData;
    DeviceArray<double> knotsPools;
    DeviceArray<double> controlPointsPools;
    NestedArrayReplica<int> multSumsOffsets;
    NestedArrayReplica<int> multSums;

    void updateStaticData(MultiPatchDeviceView source, int sourceDevice,
                          int targetDevice, const char* label)
    {
        numPatches = source.numPatches();
        domainDim = source.domainDim();
        targetDim = source.targetDim();
        const std::string prefix(label);
        peerCopyInto(patchIntDataOffsets, source.patchIntDataOffsets(),
                     sourceDevice, targetDevice,
                     (prefix + " patch int offsets").c_str());
        peerCopyInto(patchKnotsPoolOffsets, source.patchKnotsPoolOffsets(),
                     sourceDevice, targetDevice,
                     (prefix + " patch knot offsets").c_str());
        peerCopyInto(patchControlPointsPoolOffsets,
                     source.patchControlPointsPoolOffsets(), sourceDevice,
                     targetDevice, (prefix + " control-point offsets").c_str());
        peerCopyInto(intData, source.intData(), sourceDevice, targetDevice,
                     (prefix + " int data").c_str());
        peerCopyInto(knotsPools, source.knotsPools(), sourceDevice,
                     targetDevice, (prefix + " knots").c_str());
        multSumsOffsets.update(source.multSumsOffsets(), sourceDevice,
                               targetDevice,
                               (prefix + " mult sums offsets").c_str());
        multSums.update(source.multSums(), sourceDevice, targetDevice,
                        (prefix + " mult sums").c_str());
        updateControlPoints(source, sourceDevice, targetDevice, label);
    }

    void updateControlPoints(MultiPatchDeviceView source, int sourceDevice,
                             int targetDevice, const char* label)
    {
        const std::string dataLabel = std::string(label) + " control points";
        peerCopyInto(controlPointsPools, source.controlPointsPools(),
                     sourceDevice, targetDevice, dataLabel.c_str());
    }

    MultiPatchDeviceView view() const
    {
        return MultiPatchDeviceView(
            numPatches, domainDim, targetDim,
            patchIntDataOffsets.vectorView(),
            patchKnotsPoolOffsets.vectorView(),
            patchControlPointsPoolOffsets.vectorView(), intData.vectorView(),
            knotsPools.vectorView(), controlPointsPools.vectorView(),
            multSumsOffsets.view(), multSums.view());
    }
};

inline unsigned long long multiPatchReplicaBytes(MultiPatchDeviceView view)
{
    return vectorBytes(view.patchIntDataOffsets()) +
           vectorBytes(view.patchKnotsPoolOffsets()) +
           vectorBytes(view.patchControlPointsPoolOffsets()) +
           vectorBytes(view.intData()) + vectorBytes(view.knotsPools()) +
           vectorBytes(view.controlPointsPools()) +
           nestedArrayBytes(view.multSumsOffsets()) +
           nestedArrayBytes(view.multSums());
}

inline unsigned long long sparseMetadataBytes(
    const SparseSystemDeviceView& system, int localRowPtrSize, int localNnz)
{
    return vectorBytes(system.mappersData()) +
           vectorBytes(system.rowBlocks()) +
           vectorBytes(system.colBlocks()) +
           vectorBytes(system.rowStrides()) +
           vectorBytes(system.colStrides()) +
           vectorBytes(system.colVars()) +
           vectorBytes(system.dims()) +
           vectorBytes(system.permOldToNew()) +
           vectorBytes(system.permNewToOld()) +
           bytesForCount(localRowPtrSize, sizeof(int)) +
           bytesForCount(localNnz, sizeof(int));
}

struct SparseOutputBuffer
{
    int outputRowStart = 0;
    int outputRowCount = 0;
    int matrixValueStart = 0;
    int matrixValueCount = 0;
    DeviceArray<double> matrixValues;
    DeviceArray<double> rhs;
    DeviceArray<int> sparseMappersData;
    DeviceArray<int> sparseRow;
    DeviceArray<int> sparseCol;
    DeviceArray<int> sparseRstr;
    DeviceArray<int> sparseCstr;
    DeviceArray<int> sparseCvar;
    DeviceArray<int> sparseDims;
    DeviceArray<int> sparsePermOld2New;
    DeviceArray<int> sparsePermNew2Old;
    DeviceArray<int> csrRowPtr;
    DeviceArray<int> csrColInd;

    SparseOutputBuffer() = default;

    SparseOutputBuffer(int matrixSize, int rhsSize)
        : matrixValues(matrixSize), rhs(rhsSize)
    {
    }

    void copySparseBaseMetadata(const SparseSystemDeviceView& system,
                                int sourceDevice, int targetDevice)
    {
        sparseMappersData = peerCopy(system.mappersData(), sourceDevice,
                                     targetDevice, "sparse mappers");
        sparseRow = peerCopy(system.rowBlocks(), sourceDevice, targetDevice,
                             "sparse row blocks");
        sparseCol = peerCopy(system.colBlocks(), sourceDevice, targetDevice,
                             "sparse col blocks");
        sparseRstr = peerCopy(system.rowStrides(), sourceDevice, targetDevice,
                              "sparse row strides");
        sparseCstr = peerCopy(system.colStrides(), sourceDevice, targetDevice,
                              "sparse col strides");
        sparseCvar = peerCopy(system.colVars(), sourceDevice, targetDevice,
                              "sparse col vars");
        sparseDims = peerCopy(system.dims(), sourceDevice, targetDevice,
                              "sparse dims");
        sparsePermOld2New = peerCopy(system.permOldToNew(), sourceDevice,
                                     targetDevice,
                                     "sparse old-to-new permutation");
        sparsePermNew2Old = peerCopy(system.permNewToOld(), sourceDevice,
                                     targetDevice,
                                     "sparse new-to-old permutation");
    }

    void updateLocalSparseWindow(const SparseSystemDeviceView& system,
                                 const std::vector<int>& localRowPtrHost,
                                 int localRowStart,
                                 int localMatrixValueStart,
                                 int sourceDevice, int targetDevice)
    {
        outputRowStart = localRowStart;
        outputRowCount = static_cast<int>(localRowPtrHost.size()) - 1;
        matrixValueStart = localMatrixValueStart;
        csrRowPtr = localRowPtrHost;
        matrixValueCount = localRowPtrHost.empty() ? 0 : localRowPtrHost.back();
        if (matrixValueCount > matrixValues.size() ||
            outputRowCount > rhs.size())
            throw std::runtime_error(
                "Local CSR window exceeds the allocated streaming output buffer.");
        peerCopySliceInto(
            csrColInd, system.csrMatrix().colInd().data() +
                           localMatrixValueStart,
            matrixValueCount, sourceDevice, targetDevice,
            "local CSR column indices");
    }

    void clearActiveOutput()
    {
        if (matrixValues.data() && matrixValueCount > 0)
        {
            cudaError_t err =
                cudaMemset(matrixValues.data(), 0,
                           static_cast<std::size_t>(matrixValueCount) *
                               sizeof(double));
            if (err != cudaSuccess)
                throw std::runtime_error(
                    std::string("CUDA memset failed for active matrix output: ") +
                    cudaGetErrorString(err));
        }
        if (rhs.data() && outputRowCount > 0)
        {
            cudaError_t err =
                cudaMemset(rhs.data(), 0,
                           static_cast<std::size_t>(outputRowCount) *
                               sizeof(double));
            if (err != cudaSuccess)
                throw std::runtime_error(
                    std::string("CUDA memset failed for active RHS output: ") +
                    cudaGetErrorString(err));
        }
    }

    bool hasLocalSparseMetadata() const
    {
        return sparseMappersData.size() != 0;
    }

    SparseSystemDeviceView sparseSystemView(
        const SparseSystemDeviceView& primarySystemView) const
    {
        if (!hasLocalSparseMetadata())
            return primarySystemView;

        DeviceCSRMatrixView localCSR(
            primarySystemView.csrMatrix().numCols(), outputRowStart,
            csrRowPtr.vectorView(), csrColInd.vectorView(),
            DeviceVectorView<double>(matrixValues.data(), matrixValueCount));
        return SparseSystemDeviceView(
            sparseMappersData.vectorView(), sparseRow.vectorView(),
            sparseCol.vectorView(), sparseRstr.vectorView(),
            sparseCstr.vectorView(), sparseCvar.vectorView(),
            sparseDims.vectorView(),
            DeviceVectorView<double>(rhs.data(), outputRowCount), localCSR,
            sparsePermOld2New.vectorView(), sparsePermNew2Old.vectorView(),
            outputRowStart);
    }
};

static __global__
void addLocalAssemblyBufferKernel(DeviceVectorView<double> dstMatrixValues,
                                  DeviceVectorView<double> srcMatrixValues,
                                  DeviceVectorView<double> dstRHS,
                                  DeviceVectorView<double> srcRHS,
                                  int matrixValueStart,
                                  int rowStart)
{
    const int matrixSize = srcMatrixValues.size();
    const int rhsSize = srcRHS.size();
    const int totalSize = matrixSize > rhsSize ? matrixSize : rhsSize;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < totalSize;
         idx += blockDim.x * gridDim.x)
    {
        if (idx < matrixSize)
            dstMatrixValues[matrixValueStart + idx] += srcMatrixValues[idx];
        if (idx < rhsSize)
            dstRHS[rowStart + idx] += srcRHS[idx];
    }
}

inline void reduceSparseOutputBuffer(DeviceVectorView<double> dstMatrixValues,
                                     DeviceVectorView<double> dstRHS,
                                     const SparseOutputBuffer& buffer)
{
    const int totalSize =
        std::max(buffer.matrixValueCount, buffer.outputRowCount);
    if (totalSize <= 0)
        return;

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (totalSize + threadsPerBlock - 1) /
                              threadsPerBlock;
    addLocalAssemblyBufferKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dstMatrixValues,
        DeviceVectorView<double>(buffer.matrixValues.data(),
                                 buffer.matrixValueCount),
        dstRHS, DeviceVectorView<double>(buffer.rhs.data(),
                                         buffer.outputRowCount),
        buffer.matrixValueStart, buffer.outputRowStart);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error(
            "CUDA synchronize failed while reducing multi-GPU assembly buffers");
}

struct AssemblyChunk
{
    int sequence = 0;
    int elementStart = 0;
    int elementCount = 0;
    int gpStart = 0;
    int gpCount = 0;
    int rowStart = 0;
    int rowCount = 0;
    int matrixValueStart = 0;
    int matrixValueCount = 0;
    std::vector<int> rowPtrHost;
};

struct AssemblySchedule
{
    int chunkElementLimit = 0;
    int rounds = 0;
    std::vector<std::vector<AssemblyChunk>> chunksByDevice;
    std::vector<int> maxElementCounts;
    std::vector<int> maxGPCounts;
    std::vector<int> maxRowCounts;
    std::vector<int> maxMatrixValueCounts;
    std::vector<unsigned long long> requiredBytes;
};

static __global__
void chunkRowBoundsKernel(int elementStart, int numElements, int N_D,
                          int numFields, MultiPatchDeviceView displacement,
                          MultiPatchDeviceView electricPotential,
                          SparseSystemDeviceView system,
                          DeviceMatrixView<double> pts,
                          int* bounds)
{
    const int dim = displacement.domainDim();
    const int totalEntries = numElements * N_D * numFields;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < totalEntries;
         idx += blockDim.x * gridDim.x)
    {
        int entry = idx;
        const int row = entry % numFields; entry /= numFields;
        const int i = entry % N_D; entry /= N_D;
        const int elementGlobal = elementStart + entry;

        int patchIdx = 0;
        displacement.threadPatch_element(elementGlobal, patchIdx);
        TensorBsplineBasisDeviceView dispBasis = displacement.basis(patchIdx);
        const int numGPsInElement = dispBasis.numGPsInElement();

        double ptForIndexData[3] = {0.0};
        DeviceVectorView<double> ptForIndex(ptForIndexData, dim);
        for (int a = 0; a < dim; ++a)
            ptForIndex[a] = pts(a, elementGlobal * numGPsInElement);

        int activeIndex = 0;
        if (row < displacement.targetDim())
        {
            activeIndex = dispBasis.activeIndex(ptForIndex, i);
        }
        else
        {
            TensorBsplineBasisDeviceView electricBasis =
                electricPotential.basis(patchIdx);
            activeIndex = electricBasis.activeIndex(ptForIndex, i);
        }

        const int mappedRow = system.mapRowIndex(activeIndex, patchIdx, row);
        if (system.isRowEntry(mappedRow, row))
        {
            const int matrixRow = system.matrixRowIndex(mappedRow, row);
            atomicMin(bounds, matrixRow);
            atomicMax(bounds + 1, matrixRow);
        }
    }
}

inline std::pair<int, int> chunkRowRange(
    int elementStart, int elementCount, int N_D, int numFields,
    MultiPatchDeviceView displacement, MultiPatchDeviceView electricPotential,
    SparseSystemDeviceView system, DeviceMatrixView<double> pts)
{
    if (elementCount <= 0)
        return {0, 0};

    const long long totalEntriesLong =
        static_cast<long long>(elementCount) * N_D * numFields;
    if (totalEntriesLong > std::numeric_limits<int>::max())
        throw std::runtime_error(
            "Assembly chunk row scan is too large for 32-bit CUDA indexing.");
    const int totalEntries = static_cast<int>(totalEntriesLong);

    int* boundsDevice = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&boundsDevice),
                                 2 * sizeof(int));
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(
            "CUDA malloc failed for assembly chunk row bounds: ") +
                                 cudaGetErrorString(err));

    const int boundsHostInit[2] = {
        std::numeric_limits<int>::max(),
        -1
    };
    err = cudaMemcpy(boundsDevice, boundsHostInit, 2 * sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cudaFree(boundsDevice);
        throw std::runtime_error(std::string(
            "CUDA memcpy failed while initializing assembly chunk row bounds: ") +
                                 cudaGetErrorString(err));
    }

    int minGrid = 0;
    int blockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        chunkRowBoundsKernel, 0, totalEntries);
    if (blockSize <= 0)
        blockSize = 128;
    const int gridSize = (totalEntries + blockSize - 1) / blockSize;
    chunkRowBoundsKernel<<<gridSize, blockSize>>>(
        elementStart, elementCount, N_D, numFields, displacement,
        electricPotential, system, pts, boundsDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFree(boundsDevice);
        throw std::runtime_error(std::string(
            "CUDA launch failed during assembly chunk row scan: ") +
                                 cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        cudaFree(boundsDevice);
        throw std::runtime_error(std::string(
            "CUDA synchronize failed during assembly chunk row scan: ") +
                                 cudaGetErrorString(err));
    }

    int boundsHost[2] = {0, -1};
    err = cudaMemcpy(boundsHost, boundsDevice, 2 * sizeof(int),
                     cudaMemcpyDeviceToHost);
    cudaFree(boundsDevice);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(
            "CUDA memcpy failed while reading assembly chunk row bounds: ") +
                                 cudaGetErrorString(err));

    if (boundsHost[1] < boundsHost[0])
        return {0, 0};
    return {boundsHost[0], boundsHost[1] + 1};
}

inline std::vector<int> copyRowPtrWindow(DeviceVectorView<int> rowPtr,
                                         int rowStart, int rowCount,
                                         int& matrixValueStart,
                                         const char* label)
{
    std::vector<int> host(static_cast<std::size_t>(rowCount) + 1);
    if (host.empty())
    {
        matrixValueStart = 0;
        return host;
    }
    cudaError_t err = cudaMemcpy(host.data(), rowPtr.data() + rowStart,
                                 host.size() * sizeof(int),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA memcpy failed for ") +
                                 label + ": " + cudaGetErrorString(err));
    const int start = host.front();
    matrixValueStart = start;
    for (int& value : host)
        value -= start;
    return host;
}

inline AssemblySchedule buildAssemblySchedule(
    int chunkElementLimit, int numElements, int gpsPerElement,
    int numAssemblyDevices, int numBasisFunctions, int numFields, int rhsSize,
    int matrixValuesSize, bool useLocalCSRAssembly,
    const SparseSystemDeviceView& primarySystemView,
    MultiPatchDeviceView primaryDisplacementView,
    MultiPatchDeviceView primaryElectricPotentialView,
    DeviceMatrixView<double> primaryGPTableView)
{
    AssemblySchedule schedule;
    schedule.chunkElementLimit = std::max(1, chunkElementLimit);
    schedule.chunksByDevice.resize(numAssemblyDevices);
    schedule.maxElementCounts.assign(numAssemblyDevices, 0);
    schedule.maxGPCounts.assign(numAssemblyDevices, 0);
    schedule.maxRowCounts.assign(numAssemblyDevices, 0);
    schedule.maxMatrixValueCounts.assign(numAssemblyDevices, 0);
    schedule.requiredBytes.assign(numAssemblyDevices, 0);

    int sequence = 0;
    for (int elementStart = 0; elementStart < numElements;
         elementStart += schedule.chunkElementLimit, ++sequence)
    {
        AssemblyChunk chunk;
        chunk.sequence = sequence;
        chunk.elementStart = elementStart;
        chunk.elementCount =
            std::min(schedule.chunkElementLimit, numElements - elementStart);
        chunk.gpStart = elementStart * gpsPerElement;
        chunk.gpCount = chunk.elementCount * gpsPerElement;

        const int idx = sequence % numAssemblyDevices;
        if (idx == 0)
        {
            chunk.rowStart = 0;
            chunk.rowCount = rhsSize;
            chunk.matrixValueStart = 0;
            chunk.matrixValueCount = matrixValuesSize;
        }
        else if (useLocalCSRAssembly)
        {
            const auto rowRange = chunkRowRange(
                chunk.elementStart, chunk.elementCount, numBasisFunctions,
                numFields, primaryDisplacementView,
                primaryElectricPotentialView, primarySystemView,
                primaryGPTableView);
            chunk.rowStart = rowRange.first;
            chunk.rowCount = rowRange.second - rowRange.first;
            chunk.rowPtrHost = copyRowPtrWindow(
                primarySystemView.csrMatrix().rowPtr(), chunk.rowStart,
                chunk.rowCount, chunk.matrixValueStart,
                "local assembly CSR row pointer");
            chunk.matrixValueCount =
                chunk.rowPtrHost.empty() ? 0 : chunk.rowPtrHost.back();
        }
        else
        {
            chunk.rowStart = 0;
            chunk.rowCount = rhsSize;
            chunk.rowPtrHost = copyRowPtrWindow(
                primarySystemView.csrMatrix().rowPtr(), 0, rhsSize,
                chunk.matrixValueStart, "full assembly CSR row pointer");
            chunk.matrixValueCount =
                chunk.rowPtrHost.empty() ? 0 : chunk.rowPtrHost.back();
        }

        schedule.maxElementCounts[idx] =
            std::max(schedule.maxElementCounts[idx], chunk.elementCount);
        schedule.maxGPCounts[idx] =
            std::max(schedule.maxGPCounts[idx], chunk.gpCount);
        schedule.maxRowCounts[idx] =
            std::max(schedule.maxRowCounts[idx], chunk.rowCount);
        schedule.maxMatrixValueCounts[idx] =
            std::max(schedule.maxMatrixValueCounts[idx],
                     chunk.matrixValueCount);
        schedule.chunksByDevice[idx].push_back(std::move(chunk));
    }

    for (const auto& chunks : schedule.chunksByDevice)
        schedule.rounds =
            std::max(schedule.rounds, static_cast<int>(chunks.size()));

    return schedule;
}

} // namespace siga::gpuasm
