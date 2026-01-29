#pragma once

#include <vector>
#include <DofMapper.h>
#include <Eigen/Core>
#include <DeviceArray.h>

class SparseSystem
{
private:
std::vector<DofMapper> m_mappers;
Eigen::VectorXi m_row;
Eigen::VectorXi m_col;
Eigen::VectorXi m_rstr;
Eigen::VectorXi m_cstr;
Eigen::VectorXi m_cvar;
Eigen::VectorXi m_dims;

Eigen::MatrixXd m_matrix;
Eigen::VectorXd m_RHS;

public:
    SparseSystem() = default;
    SparseSystem(std::vector<DofMapper>& mappers, 
                 const Eigen::VectorXi& dims);

    void getDataVector(std::vector<int>& intDataOffsets, 
                       std::vector<int>& data_int,
                       std::vector<double>& data_double) const;
    void getDataVector(std::vector<int>& intDataOffsets, 
                       DeviceArray<int>& data_int,
                       DeviceArray<double>& data_double) const
    {
        std::vector<int> data_int_vec;
        std::vector<double> data_double_vec;
        getDataVector(intDataOffsets, data_int_vec, data_double_vec);
        data_int = data_int_vec;
        data_double = data_double_vec;
    }

    Eigen::MatrixXd& matrix() { return m_matrix; }
    const Eigen::MatrixXd& matrix() const { return m_matrix; }

    Eigen::VectorXd& rhs() { return m_RHS; }
    const Eigen::VectorXd& rhs() const { return m_RHS; }
};