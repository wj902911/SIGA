#pragma once

#include <MultiPatchDeviceData.h>
#include <MultiBasisDeviceData.h>
#include <SparseSystemDeviceData.h>
#include <BoundaryCondition.h>
#include <MultiGaussPointsDeviceData.h>

class GPUAssembler
{
private:
    int m_problemDim = 2;
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
public:
    __host__
    GPUAssembler(const MultiPatch& multiPatch,
                  const MultiBasis& multiBasis,
                  const BoundaryConditions& bc,
                  const Eigen::VectorXd& bodyForce);
        

    __host__
    void computeDirichletDofs(int unk_, 
                              const std::vector<DofMapper> &mappers,
                              std::vector<Eigen::VectorXd> &ddof,
                              const MultiBasis &multiBasis);

    __host__
    int dim() const { return m_problemDim; }
    
    __host__
    void print() const;

    __host__
    void printMultiPatch() const;

    __host__
    void printMultiBasis() const;

    __host__
    int numDofs() const;

    __host__
    const DeviceNestedArray<double>& allFixedDofs() const { return m_ddof; }

    __host__
    void constructSolution(const DeviceVectorView<double>& solVector,
                           const DeviceNestedArrayView<double>& fixedDoFs,
                           MultiPatchDeviceView& displacementDeviceView) const;
    
    __host__
    void assemble(const DeviceVectorView<double>& solVector, int numIter,
                  const DeviceNestedArrayView<double>& fixedDoFs);

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
    const MultiPatchDeviceView geometryView() const
    { return m_multiPatch.deviceView(); }

    __host__
    const MultiGaussPointsDeviceView gaussPointsView() const
    { return m_multiGaussPoints.view(); }

    __host__
    int numPatches() const { return m_multiPatch.numPatches(); }

    __host__
    int domainDim() const { return m_multiPatch.domainDim(); }

    __host__
    int targetDim() const { return m_multiPatch.targetDim(); }
};