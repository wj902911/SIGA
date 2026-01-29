#include "SparseSystem.h"

SparseSystem::SparseSystem(std::vector<DofMapper> &mappers, const Eigen::VectorXi &dims)
    : m_mappers(mappers), m_dims(dims), m_row(dims.sum()), m_col(dims.sum()), m_rstr(dims.sum()), m_cstr(dims.sum())
{
    const int d = dims.size();
    const int s = dims.sum();
    const int ms = mappers.size();

    m_mappers.swap(mappers);

    int k=0;
    for(int i=0;i<d;++i)
        for(int j=0; j<dims[i];++j)
        {
            m_row[k] = i;
            ++k;
        }
    m_col = m_row;
    
    if (ms == 1 )
    {
        m_row.setZero();
        m_col.setZero();
        m_cvar.setZero(1);
    }
    else if ( ms == 2*s )
        m_col.array() += s;

    m_cvar = m_row;
    m_rstr[0] = m_cstr[0] = 0;
    for (int r = 1; r < d; ++r)
        m_rstr[r] = int(m_rstr[r-1]) + mappers[m_row[r-1]].freeSize(); // Use the original mappers to get freeSize
    for (int c = 1; c < d; ++c)
        m_cstr[c] = int(m_cstr[c-1]) + mappers[m_col[c-1]].freeSize(); // Use the original mappers to get freeSize

    m_matrix.setZero(int(m_rstr[d-1]) + mappers[m_row[d-1]].freeSize(),
                    int(m_cstr[d-1]) + mappers[m_col[d-1]].freeSize());
                    
    m_RHS.setZero(m_matrix.rows());
}

void SparseSystem::getDataVector(std::vector<int>& intDataOffsets, 
                                 std::vector<int> &data_int, 
                                 std::vector<double> &data_double) const
{
    intDataOffsets.clear();
    data_int.clear();
    data_double.clear();

    intDataOffsets.reserve(8);
    intDataOffsets.push_back(0);
    int mappersSize = 0;
    for (const auto &mapper : m_mappers)
    {
        mappersSize += mapper.getDataSize();
    }
    intDataOffsets.push_back(intDataOffsets.back() + mappersSize);
    intDataOffsets.push_back(intDataOffsets.back() + m_row.size());
    intDataOffsets.push_back(intDataOffsets.back() + m_col.size());
    intDataOffsets.push_back(intDataOffsets.back() + m_rstr.size());
    intDataOffsets.push_back(intDataOffsets.back() + m_cstr.size());
    intDataOffsets.push_back(intDataOffsets.back() + m_cvar.size());
    intDataOffsets.push_back(intDataOffsets.back() + m_dims.size()); 

    data_int.reserve(intDataOffsets.back());
    for (const auto &mapper : m_mappers)
    {
        std::vector<int> mapper_data;
        mapper.getDofMapperDataVec(mapper_data);
        data_int.insert(data_int.end(), mapper_data.begin(), mapper_data.end());
    }
    data_int.insert(data_int.end(), m_row.data(), m_row.data() + m_row.size());
    data_int.insert(data_int.end(), m_col.data(), m_col.data() + m_col.size());
    data_int.insert(data_int.end(), m_rstr.data(), m_rstr.data() + m_rstr.size());
    data_int.insert(data_int.end(), m_cstr.data(), m_cstr.data() + m_cstr.size());
    data_int.insert(data_int.end(), m_cvar.data(), m_cvar.data() + m_cvar.size());
    data_int.insert(data_int.end(), m_dims.data(), m_dims.data() + m_dims.size());

    data_double.reserve(m_matrix.size() + m_RHS.size());
    data_double.insert(data_double.end(), m_matrix.data(), m_matrix.data() + m_matrix.size());
    data_double.insert(data_double.end(), m_RHS.data(), m_RHS.data() + m_RHS.size());
#if 0
    // calculate data sizes
    int size_int = 0;
    size_int += 1; // for m_mappers.size()
    for (const auto &mapper : m_mappers)
    {
        size_int += 1; // for mapper size
        size_int += mapper.getDataSize();
    }
    size_int += 1 + m_row.size(); // for m_row
    size_int += 1 + m_col.size(); // for m_col
    size_int += 1 + m_rstr.size(); // for m_rstr
    size_int += 1 + m_cstr.size(); // for m_cstr
    size_int += 1 + m_cvar.size(); // for m_cvar
    size_int += 1 + m_dims.size(); // for m_dims
    size_int += 2; // for m_matrix rows and cols
    size_int += 1; // for m_RHS size
    data_int.reserve(size_int);
    data_double.reserve(1 + 1 + m_matrix.size() + 1 + m_RHS.size());

    // Store m_mappers size
    data_int.push_back(m_mappers.size());
    
    // Store m_mappers
    for (const auto &mapper : m_mappers)
    {
        std::vector<int> mapper_data;
        mapper.getDofMapperDataVec(mapper_data);
        data_int.push_back(mapper_data.size());
        data_int.insert(data_int.end(), mapper_data.begin(), mapper_data.end());
    }

    // Store m_row
    data_int.push_back(m_row.size());
    data_int.insert(data_int.end(), m_row.data(), m_row.data() + m_row.size());

    // Store m_col
    data_int.push_back(m_col.size());
    data_int.insert(data_int.end(), m_col.data(), m_col.data() + m_col.size());

    // Store m_rstr
    data_int.push_back(m_rstr.size());
    data_int.insert(data_int.end(), m_rstr.data(), m_rstr.data() + m_rstr.size());

    // Store m_cstr
    data_int.push_back(m_cstr.size());
    data_int.insert(data_int.end(), m_cstr.data(), m_cstr.data() + m_cstr.size());

    // Store m_cvar
    data_int.push_back(m_cvar.size());
    data_int.insert(data_int.end(), m_cvar.data(), m_cvar.data() + m_cvar.size());

    // Store m_dims
    data_int.push_back(m_dims.size());
    data_int.insert(data_int.end(), m_dims.data(), m_dims.data() + m_dims.size());

    // Store m_matrix
    data_int.push_back(m_matrix.rows());
    data_int.push_back(m_matrix.cols());
    data_double.insert(data_double.end(), m_matrix.data(), m_matrix.data() + m_matrix.size());

    // Store m_RHS
    data_int.push_back(m_RHS.size());
    data_double.insert(data_double.end(), m_RHS.data(), m_RHS.data() + m_RHS.size());
#endif
}
