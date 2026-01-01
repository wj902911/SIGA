#include "Function.h"
#include "MultiPatch_d.h"
#include "DeviceObjectPointer.h"

void DisplacementFunction::eval_into(const Eigen::MatrixXi &numPointsPerDir, 
                                           Eigen::MatrixXd &values) const
{
    int dim = m_displacement.getCPDim();
    int numPoints = 0;
    int numPatches = m_displacement.getNumPatches();
    for (int p = 0; p < numPatches; p++)
        numPoints += numPointsPerDir.col(p).prod();
    values.resize(dim, numPoints);
    values.setZero();
    MultiPatch_d displacement_d(m_displacement);
    DeviceObjectPointer<MultiPatch_d> d_displacement(displacement_d);
    d_displacement.pointer()->eval_into(numPointsPerDir, values);
}