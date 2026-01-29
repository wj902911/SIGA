#pragma once

#include <DeviceArray.h>
#include <SparseSystemDeviceView.h>
#include <SparseSystem.h>
#include <vector>

class SparseSystemDeviceData
{
private:
    int m_matrixRows = 0;
    int m_matrixCols = 0;
    std::vector<int> m_intDataOffsets;
    DeviceArray<int> m_intData;
    DeviceArray<double> m_doubleData;
public:
	__host__
	SparseSystemDeviceData() = default;
	
    __host__
    SparseSystemDeviceData(int matrixRows, int matrixCols,
                           const std::vector<int>& intDataOffsets,
                           const std::vector<int>& intData,
                           const std::vector<double>& doubleData)
                         : m_matrixRows(matrixRows),
                           m_matrixCols(matrixCols),
                           m_intDataOffsets(intDataOffsets),
                           m_intData(intData),
                           m_doubleData(doubleData)
    {
    }

    __host__
    SparseSystemDeviceData(const SparseSystem& sparseSystem)
    : m_matrixRows(sparseSystem.matrix().rows()),
      m_matrixCols(sparseSystem.matrix().cols())
    {
        std::vector<int> intData;
        std::vector<double> doubleData;
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
        
        DeviceMatrixView<double> matrix(m_doubleData.data(),
                                        m_matrixRows,
                                        m_matrixCols);
        DeviceVectorView<double> rhs(m_doubleData.data() + (m_matrixRows * m_matrixCols),
                                      m_matrixRows);
        
        return SparseSystemDeviceView(mappersData,
                                      row,
                                      col,
                                      rstr,
                                      cstr,
                                      cvar,
                                      dims,
                                      matrix,
                                      rhs);    
    }

	__host__
	const std::vector<int> & intDataOffsets() const 
	{ return m_intDataOffsets; }
	__host__
	const DeviceArray<int> & intData() const { return m_intData; }
	__host__
	const DeviceArray<double> & doubleData() const { return m_doubleData; }

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
	__host__
	void setDoubleData(const std::vector<double> & doubleData) 
	{ m_doubleData = doubleData; }

	__host__
    int numDofs() const { return m_matrixRows; }
};