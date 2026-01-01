#pragma once

#include <Function.h>
#include <MultiPatch_d.h>
#include <ParaviewCollection.h>

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

    void evalFunctionsAtPoints(std::map<std::string, Eigen::MatrixXd>& data,
                               const Eigen::MatrixXi &numPointsPerDir) const;

    void outputToParaview(const std::string &fn, 
                          const Eigen::VectorXi& numPoints, 
                          int step, 
                          ParaviewCollection& collection) const;

    void writeParaview(const std::string &fn, 
                       const Eigen::MatrixXi& numPoints, 
                       const Eigen::MatrixXd& points,
                       const std::map<std::string, Eigen::MatrixXd>& data,
                       int step,
                       ParaviewCollection& collection) const;
    
    void writeParaviewSinglePatch(const std::string &fn, 
                                  const Eigen::VectorXi& np, 
                                  const Eigen::MatrixXd& points,
                                  const std::map<std::string, Eigen::MatrixXd>& data) const;

private:
    MultiPatch m_geometry;
    std::map<std::string, Function*> m_functions;
};