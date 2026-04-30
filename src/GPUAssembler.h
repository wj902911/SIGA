#pragma once

#include <MultiPatchDeviceData.h>
#include <MultiBasisDeviceData.h>
#include <SparseSystemDeviceData.h>
#include <BoundaryCondition.h>
#include <MultiGaussPointsDeviceData.h>
#include <OptionList.h>
#include <GPUFunction.h>

class GPUAssembler
{
private:
    int m_targetDim = 2;
    int m_domainDim = 2;
    int m_dimTensor = 0;
    int m_numElements = 0;
    int m_N_D = 0;
    int m_totalGPs = 0;
    int m_numDerivatives = 1;
    int m_geoP1 = 1;
    int m_dispP1 = 1;
    //int m_numBatches = 1;
    //size_t m_batchElements = 0;
    //size_t m_batchSize = 0;
    DeviceArray<double> m_GPData;
    DeviceArray<double> m_GPTable;
    DeviceArray<double> m_wts;
    DeviceArray<double> m_geoValuesAndDerss;
    DeviceArray<double> m_dispValuesAndDerss;
    MultiPatchDeviceData m_multiPatch;
    MultiPatchDeviceData m_displacement;
    MultiPatch m_displacementHost;
    const MultiPatch& m_multiPatchHost;
    MultiBasisDeviceData m_multiBasis;
    const MultiBasis& m_multiBasisHost;
    BoundaryConditions m_boundaryConditions;
    DeviceNestedArray<double> m_ddof;
    DeviceNestedArray<double> m_ddof_zero;
    SparseSystemDeviceData m_sparseSystem;
    //SparseSystem m_sparseSystemHost;
    DeviceArray<double> m_bodyForce;
    MultiGaussPointsDeviceData m_multiGaussPoints;
    bool m_initialAssemble = true;
    OptionList m_options;
public:
    __host__
    GPUAssembler(const MultiPatch& multiPatch,
                  const MultiBasis& multiBasis,
                  const BoundaryConditions& bc,
                  const Eigen::VectorXd& bodyForce,
                  bool baseInitial = false);
    __host__
    OptionList defaultOptions(); 
    __host__
    void setDefaultOptions(const OptionList &opt);

    __host__
    void computeDirichletDofs(int unk_, 
                              const std::vector<DofMapper> &mappers,
                              std::vector<Eigen::VectorXd> &ddof,
                              const MultiBasis &multiBasis);

    __host__
    int dim() const { return m_targetDim; }
    
    __host__
    void print() const;

    __host__
    void printMultiPatch() const;

    __host__
    void printMultiBasis() const;

    __host__
    int numDofs() const;

    __host__
    int numElements() const { return m_numElements; }

    __host__
    const DeviceNestedArray<double>& allFixedDofs() const { return m_ddof; }

    __host__
    void constructSolution(const DeviceVectorView<double>& solVector,
                           const DeviceNestedArrayView<double>& fixedDoFs,
                           GPUDisplacementFunction& displacementFunction) const;

    __host__
    void constructDispSolution(const DeviceVectorView<double>& solVector,
                               const DeviceNestedArrayView<double>& fixedDoFs) const;
    
    
    __host__
    virtual void assemble(const DeviceVectorView<double>& solVector, int numIter,
                  const DeviceNestedArrayView<double>& fixedDoFs);
    __host__
    virtual void refreshFixedDofs();

    
    __host__
    void assembleMatrix(
        const DeviceNestedArrayView<double>& fixedDofs_assemble,
        const DeviceMatrixView<double>& geoJacobianInvs,
        const DeviceVectorView<double>& measures,
        const DeviceVectorView<double>& weightForces,
        const DeviceVectorView<double>& weightBodys,
        const DeviceMatrixView<double>& Fs,
        const DeviceMatrixView<double>& Ss,
        const DeviceMatrixView<double>& Cs);

    //__host__
    //DeviceMatrixView<double> matrix() const
    //{ return m_sparseSystem.deviceView().matrix(); }

    __host__
    DeviceVectorView<double> rhs() const
    { return m_sparseSystem.deviceView().rhs(); }

#if 0
    __host__
    DeviceVectorView<int> rows() const
    { return m_sparseSystem.deviceView().rows(); }

    __host__
    DeviceVectorView<int> cols() const
    { return m_sparseSystem.deviceView().cols(); }

    __host__
    DeviceVectorView<double> values() const
    { return m_sparseSystem.deviceView().values(); }  
#endif

    __host__
    void denseMatrix(DeviceMatrixView<double> denseMat) const;

    __host__
    const DeviceCSRMatrix& csrMatrix() const 
    { return m_sparseSystem.csrMatrix(); }

    __host__
    Eigen::VectorXd hostRHS() const { return m_sparseSystem.hostRHS(); }

#if 0
    __host__
    int numDofs() const
    {
        int sum = 0;
        for (int c = 0; c < m_sparseSystemHost.numColBlocks(); c++)
            sum += m_sparseSystemHost.colMapper(c).freeSize();
        return sum;
    }
#endif

    __host__
    const MultiPatchDeviceData& geometryData() const { return m_multiPatch; }

    __host__
    const MultiBasis& basisHost() const { return m_multiBasisHost; }

    __host__
    const MultiPatch& geometryHost() const { return m_multiPatchHost; }

    __host__
    const MultiPatchDeviceView& geometryView() const
    { return m_multiPatch.deviceView(); }

    __host__
    const MultiGaussPointsDeviceView& gaussPointsView() const
    { return m_multiGaussPoints.view(); }

    __host__
    const DeviceMatrixView<double>& gpTable() const 
    { return m_GPTable.matrixView(m_domainDim, m_totalGPs); }

    __host__
    const DeviceArray<double>& wts() const { return m_wts; }

    __host__
    int numPatches() const { return m_multiPatch.numPatches(); }

    __host__
    int domainDim() const { return m_multiPatch.domainDim(); }

    __host__
    int targetDim() const { return m_targetDim; }

    __host__
    OptionList& options() { return m_options; }

    __host__
    const BoundaryConditions& boundaryConditions() const { return m_boundaryConditions; }

    __host__
    void setMatrixRows(int rows) { m_sparseSystem.setMatrixRows(rows); }

    __host__
    void setMatrixCols(int cols) { m_sparseSystem.setMatrixCols(cols); }

    __host__
    void setSparseSystemIntDataOffsets(const std::vector<int> & offsets) 
    { m_sparseSystem.setIntDataOffsets(offsets); }

    __host__
    void setSparseSystemIntData(const std::vector<int> & intData) 
    { m_sparseSystem.setIntData(intData); }

    __host__
    void setupSparseSystem(const SparseSystem& sparseSystem)
    {
        m_sparseSystem.setMatrixRows(sparseSystem.matrixRows());
        m_sparseSystem.setMatrixCols(sparseSystem.matrixCols());
        std::vector<int> intDataOffsets;
        std::vector<int> intData;
        sparseSystem.getDataVector(intDataOffsets, intData);
        m_sparseSystem.setIntDataOffsets(intDataOffsets);
        m_sparseSystem.setIntData(intData);
        m_sparseSystem.resizeRHS(sparseSystem.matrixRows());
        m_sparseSystem.setPermVectors(sparseSystem.permOld2New(), sparseSystem.permNew2Old());
    }

    __host__
    void setDdofZero(const std::vector<Eigen::VectorXd>& ddof_zero) { m_ddof_zero.setData(ddof_zero); }
    __host__
    void setDdof(const std::vector<Eigen::VectorXd>& ddof) { m_ddof.setData(ddof); }
    __host__
    void setDisplacementPatches(const MultiBasis& displacementBasis) { 
        displacementBasis.giveBasis(m_displacementHost, m_targetDim);
        m_displacement = MultiPatchDeviceData(m_displacementHost);
    }
    __host__
    int N_D() const { return m_N_D; }

    __host__
    int numDispMatrixEntries() const;

    __host__
    const MultiPatchDeviceView& displacementView() const { return m_displacement.deviceView(); }

    __host__
    const SparseSystemDeviceView& sparseSystemDeviceView() const { return m_sparseSystem.deviceView(); }

    __host__
    void setCSRMatrixFromCOO(int numRows, int numCols,
                             DeviceVectorView<int> cooR, 
                             DeviceVectorView<int> cooC)
    { m_sparseSystem.setCSRMatrixFromCOO(numRows, numCols, cooR, cooC); }

    __host__
    void computeCOO(DeviceVectorView<int> cooRows, 
                    DeviceVectorView<int> cooCols) const;

    __host__
    int numDoublesPerGP() const
    { return m_domainDim * m_domainDim * 3 + 3 + m_dimTensor * m_dimTensor; }

    __host__
    void computeGPTable();

    __host__
    void evaluateBasisValuesAndDerivativesAtGPs();

    __host__
    int numGPs() const { return m_totalGPs; }

    __host__    
    int dimTensor() const { return m_dimTensor; }

    __host__
    void allocateGPData();

    __host__
    void printDispValuesAndDerss() const
    { 
        m_dispValuesAndDerss.matrixView(m_dispP1, 
        numGPs() * (m_numDerivatives + 1) * domainDim()).print(); 
    }

    __host__
    void setMatrixAndRHSZeros()
    {
        m_sparseSystem.matrixSetZero();
        m_sparseSystem.RHSSetZero();
    }

    __host__
    const MultiBasisDeviceView& multiBasisDeviceView() const 
    { return m_multiBasis.deviceView(); }

    __host__
    int totalNumControlPoints() const
    { return m_displacementHost.getTotalNumControlPoints(); }

    __host__
    void setBasisPatches();

    __host__
    void getFixedDofsForAssemble(int numIter, DeviceNestedArrayView<double>& fixedDofs_assemble) const;

    __host__
    const OptionList& options() const { return m_options; }

    __host__
    double* GPDataPtr() { return m_GPData.data(); }

    __host__
    void setGPDataZeros()
    { m_GPData.setZero(); }

    __host__
    const DeviceMatrixView<double> geoValuesAndDerssView() const 
    { return m_geoValuesAndDerss.matrixView(m_geoP1, numGPs() * (m_numDerivatives + 1) * domainDim()); }

    __host__
    const DeviceMatrixView<double> dispValuesAndDerssView() const
    { return m_dispValuesAndDerss.matrixView(m_dispP1, numGPs() * (m_numDerivatives + 1) * domainDim()); }

    __host__
    int numDerivatives() const { return m_numDerivatives; }

    __host__
    void printCSRMatrix() const
    { m_sparseSystem.csrMatrix().print_host(); }

    __host__
    const DeviceArray<double>& bodyForce() const { return m_bodyForce; }
};