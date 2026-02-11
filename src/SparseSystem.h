#pragma once

#include <vector>
#include <DofMapper.h>
#include <Eigen/Core>
#include <DeviceArray.h>

//#define STORE_MATRIX

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

#ifdef STORE_MATRIX    
    Eigen::MatrixXd m_matrix;
    Eigen::VectorXd m_RHS;
#else
    int m_matrixRows = 0;
    int m_matrixCols = 0;
#endif

public:
    SparseSystem() = default;
    SparseSystem(std::vector<DofMapper>& mappers, 
                 const Eigen::VectorXi& dims);

#ifdef STORE_MATRIX    
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
#else
    void getDataVector(std::vector<int>& intDataOffsets, 
                       std::vector<int>& data_int) const;
    void getDataVector(std::vector<int>& intDataOffsets, 
                       DeviceArray<int>& data_int) const
    {
        std::vector<int> data_int_vec;
        std::vector<double> data_double_vec;
        getDataVector(intDataOffsets, data_int_vec);
        data_int = data_int_vec;
    }
#endif

#ifdef STORE_MATRIX    
    Eigen::MatrixXd& matrix() { return m_matrix; }
    const Eigen::MatrixXd& matrix() const { return m_matrix; }

    Eigen::VectorXd& rhs() { return m_RHS; }
    const Eigen::VectorXd& rhs() const { return m_RHS; }
#else
    int matrixRows() const { return m_matrixRows; }
    int matrixCols() const { return m_matrixCols; }
#endif

    int numColBlocks() const {return m_col.size();}
    int numRowBlocks() const {return m_row.size();}

    

    const DofMapper& colMapper(int c) const { return m_mappers[m_col[c]]; }
};