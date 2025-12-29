#pragma once

#include "MultiPatch.h"
#include "MultiPatchData.h"
#include "MultiBasis.h"
#include "BoundaryCondition.h"
#include "DeviceVector.h"
#include "SparseSystem.h"
#include "MultiPatch_d.h"

class Assembler
{
public:
    Assembler(const MultiPatch& multiPatch, const MultiBasis& multiBasis, 
              const BoundaryConditions& bc, const Eigen::VectorXd& bodyForce);
    ~Assembler()=default;

    int getNumPatches() const;
    int numDofs() const;

    void assemble(const DeviceVector<double>& solVector, int numIter);
    void assembleNeumannBCs();

    void computeDirichletDofs(int unk_, const std::vector<DofMapper> &mappers);

#if 1
    void constructSolution(const DeviceVector<double>& solVector, MultiPatch &displacement) const;
#endif

    int getBoundaryData_Neumann(thrust::device_vector<int>& sizes, 
                                thrust::device_vector<int>& starts,
                                thrust::device_vector<int> &PatchIdxs,
                                thrust::device_vector<int> &sides,
                                thrust::device_vector<double> &values) const;

    const DeviceMatrix<double> & matrix() const { return m_sparseSystem.matrix(); }
    const DeviceVector<double> & rhs() const { return m_sparseSystem.rhs(); }

    void fixedDofs(DeviceObjectArray<DeviceVector<double>> &fixedDoFs_d) const;

private:
    //MultiPatchData m_multiPatchData;
    MultiPatch m_multiPatch;
    MultiBasis m_multiBasis;
    //std::vector<DofMapper> m_dofMappers;
    BoundaryConditions m_boundaryConditions;
    std::vector<std::vector<double>> m_ddof;
    SparseSystem m_sparseSystem;
    Eigen::VectorXd m_bodyForce;

#if 0
    thrust::device_vector<double> m_matrix;
    int m_matRows;
    int m_matCols;
    thrust::device_vector<double> m_RHS;

    thrust::device_vector<int> m_row;
    thrust::device_vector<int> m_col;
    thrust::device_vector<int> m_rstr;
    thrust::device_vector<int> m_cstr;
    thrust::device_vector<int> m_cvar;
    thrust::device_vector<int> m_dims;
#endif
};