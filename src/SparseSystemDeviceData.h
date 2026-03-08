#pragma once

#include <DeviceArray.h>
#include <SparseSystemDeviceView.h>
#include <SparseSystem.h>
#include <vector>
#include <DeviceCSRMatrix.h>

class SparseSystemDeviceData
{
private:
    int m_matrixRows = 0;
    int m_matrixCols = 0;
    //int m_numMatrixEntries = 0;
    std::vector<int> m_intDataOffsets;
    DeviceArray<int> m_intData;
    //DeviceArray<double> m_doubleData;
    
    //DeviceArray<int> m_rows;
    //DeviceArray<int> m_cols;
    //DeviceArray<double> m_values;
    DeviceArray<double> m_RHS;

    DeviceCSRMatrix m_csrMatrix;

    DeviceArray<int> m_perm_old2new;
    DeviceArray<int> m_perm_new2old;

public:
	__host__
	SparseSystemDeviceData() = default;
	
    __host__
    SparseSystemDeviceData(int matrixRows, int matrixCols,
                           const std::vector<int>& intDataOffsets,
                           const std::vector<int>& intData,
                           const std::vector<double>& doubleData,
                           const std::vector<int>& permOld2New,
                           const std::vector<int>& permNew2Old)
                         : m_matrixRows(matrixRows),
                           m_matrixCols(matrixCols),
                           m_intDataOffsets(intDataOffsets),
                           m_intData(intData),
                           m_perm_old2new(permOld2New),
                           m_perm_new2old(permNew2Old)
                           //m_doubleData(doubleData)
    {
    }

#ifdef STORE_MATRIX
    __host__
    SparseSystemDeviceData(const SparseSystem& sparseSystem)
    : m_matrixRows(sparseSystem.matrix().rows()),
      m_matrixCols(sparseSystem.matrix().cols())
    {
        sparseSystem.getDataVector(m_intDataOffsets,
                                   m_intData,
                                   m_doubleData);
    }

    __host__
    SparseSystemDeviceData(const MultiPatch& multiPatch,
                           const MultiBasis& multiBasis,
						   const BoundaryConditions& bc)
    {
		int targetDim = multiPatch.getCPDim();
		std::vector<DofMapper> dofMappers_stdVec(targetDim);
		multiBasis.getMappers(true, bc, dofMappers_stdVec, true);
		SparseSystem sparseSystem(dofMappers_stdVec, 
                                  Eigen::VectorXi::Ones(targetDim));
		m_matrixRows = sparseSystem.matrix().rows();
		m_matrixCols = sparseSystem.matrix().cols();
		sparseSystem.getDataVector(m_intDataOffsets,
								   m_intData,
								   m_doubleData);
    }
#else
    __host__
    SparseSystemDeviceData(const SparseSystem& sparseSystem)
    : m_matrixRows(sparseSystem.matrixRows()),
      m_matrixCols(sparseSystem.matrixCols())
      //m_doubleData(sparseSystem.matrixRows()*
      //             sparseSystem.matrixCols()+
      //             sparseSystem.matrixRows())
    {
        sparseSystem.getDataVector(m_intDataOffsets,
                                   m_intData);
    }

    __host__
    SparseSystemDeviceData(const MultiPatch& multiPatch,
                           const MultiBasis& multiBasis,
						   const BoundaryConditions& bc)
    {
		int targetDim = multiPatch.getCPDim();
		std::vector<DofMapper> dofMappers_stdVec(targetDim);
		multiBasis.getMappers(true, bc, dofMappers_stdVec, true);
		SparseSystem sparseSystem(dofMappers_stdVec, 
                                  Eigen::VectorXi::Ones(targetDim));
		m_matrixRows = sparseSystem.matrixRows();
		m_matrixCols = sparseSystem.matrixCols();
		sparseSystem.getDataVector(m_intDataOffsets,
								   m_intData);
        //m_doubleData.resize(m_matrixRows*
        //                     m_matrixCols+
        //                     m_matrixRows);
    }
#endif

    

    __host__
    SparseSystemDeviceView deviceView() const
    {
        int mappersDataStart = m_intDataOffsets[0];
        int mappersDataEnd = m_intDataOffsets[1];
        DeviceVectorView<int> mappersData(m_intData.data() + mappersDataStart,
                                          mappersDataEnd - mappersDataStart);
        int rowStart = m_intDataOffsets[1];;
        int rowEnd = m_intDataOffsets[2];
        DeviceVectorView<int> row(m_intData.data() + rowStart,
                                  rowEnd - rowStart);
        int colStart = m_intDataOffsets[2];
        int colEnd = m_intDataOffsets[3];
        DeviceVectorView<int> col(m_intData.data() + colStart,
                                  colEnd - colStart);
        int rstrStart = m_intDataOffsets[3];
        int rstrEnd = m_intDataOffsets[4];
        DeviceVectorView<int> rstr(m_intData.data() + rstrStart,
                                   rstrEnd - rstrStart);
        int cstrStart = m_intDataOffsets[4];
        int cstrEnd = m_intDataOffsets[5];
        DeviceVectorView<int> cstr(m_intData.data() + cstrStart,
                                   cstrEnd - cstrStart);
        int cvarStart = m_intDataOffsets[5];
        int cvarEnd = m_intDataOffsets[6];
        DeviceVectorView<int> cvar(m_intData.data() + cvarStart,
                                   cvarEnd - cvarStart);
        int dimsStart = m_intDataOffsets[6];
        int dimsEnd = m_intDataOffsets[7];
        DeviceVectorView<int> dims(m_intData.data() + dimsStart,
                                   dimsEnd - dimsStart);
        
        //DeviceMatrixView<double> matrix(m_doubleData.data(),
        //                                m_matrixRows,
        //                                m_matrixCols);
        //DeviceVectorView<double> rhs(m_doubleData.data() + (m_matrixRows * m_matrixCols),
        //                            m_matrixRows);
        
        return SparseSystemDeviceView(mappersData, row, col, rstr,
                                      cstr, cvar, dims, //matrix, rhs, 
                                      //m_rows.vectorView(), 
                                      //m_cols.vectorView(), 
                                      //m_values.vectorView(), 
                                      m_RHS.vectorView(),
                                      m_csrMatrix.view(),
                                      m_perm_old2new.vectorView(),
                                      m_perm_new2old.vectorView());    
    }

	__host__
	const std::vector<int> & intDataOffsets() const 
	{ return m_intDataOffsets; }
	__host__
	const DeviceArray<int> & intData() const { return m_intData; }
	//__host__
	//const DeviceArray<double> & doubleData() const { return m_doubleData; }

	__host__
	void setMatrixRows(int rows) { m_matrixRows = rows; }
	__host__
	void setMatrixCols(int cols) { m_matrixCols = cols; }
	__host__
	void setIntDataOffsets(const std::vector<int> & offsets) 
	{ m_intDataOffsets = offsets; }
	__host__
	void setIntData(const std::vector<int> & intData) 
	{ m_intData = intData; }
	//__host__
	//void setDoubleData(const std::vector<double> & doubleData) 
	//{ m_doubleData = doubleData; }
    //__host__
    //void resizeDoubleData(int size) { m_doubleData.resize(size); }
    __host__
    void setPermVectors(const std::vector<int>& permOld2New, 
                        const std::vector<int>& permNew2Old)
    {
        m_perm_old2new = permOld2New;
        m_perm_new2old = permNew2Old;
    }

	__host__
    int numDofs() const { return m_matrixRows; }

    //__host__
    //int* numMatrixEntriesPtr() { return &m_numMatrixEntries; }

    //__host__
    //int setNumMatrixEntries(int numEntries) 
    //{ return m_numMatrixEntries = numEntries; }

    //__host__
    //int numMatrixEntries() const { return m_numMatrixEntries; }

    //__host__
    //void resizeMatrixData(int numEntries)
    //{
    //    m_rows.resize(numEntries);
    //    m_cols.resize(numEntries);
    //    m_values.resize(numEntries);
    //}

    __host__
    void resizeRHS(int size) { m_RHS.resize(size); }

    __host__
    void setCSRMatrixFromCOO(int numRows, int numCols,
                             DeviceVectorView<int> cooR, 
                             DeviceVectorView<int> cooC, 
                             DeviceVectorView<double> cooV)
    { m_csrMatrix.setFromCOO(numRows, numCols, cooR, cooC, cooV); }

    __host__
    void setCSRMatrixFromCOO(int numRows, int numCols,
                             DeviceVectorView<int> cooR, 
                             DeviceVectorView<int> cooC)
    { m_csrMatrix.setFromCOO(numRows, numCols, cooR, cooC); }

    __host__
    const DeviceCSRMatrix& csrMatrix() const { return m_csrMatrix; }
    __host__
    DeviceCSRMatrix& csrMatrix() { return m_csrMatrix; }

    __host__
    Eigen::VectorXd hostRHS() const
    {
        std::vector<double> h_rhs(m_RHS.size());
        m_RHS.copyToHost(h_rhs);
        return Eigen::Map<Eigen::VectorXd>(h_rhs.data(), h_rhs.size());
    }
};