#pragma once

#include <Function.h>
#include <MultiPatch_d.h>

class PostProcessor
{
public:
    PostProcessor(const MultiPatch &geometry)
        : m_geometry(geometry) {}

    void addFunction(const std::string &name, Function *function)
    {
        m_functions[name] = function;
    }

    const MultiPatch &getGeometry() const { return m_geometry; }

    const std::map<std::string, Function*> &getFunctions() const
    {
        return m_functions;
    }

    void distributePoints(const Eigen::VectorXi &numPoints, Eigen::MatrixXi& numPointsPerDir) const;

    void evalGeometryAtPoints(const Eigen::MatrixXi &numPointsPerDir,
                              Eigen::MatrixXd& values) const;

private:
    MultiPatch m_geometry;
    std::map<std::string, Function*> m_functions;
};