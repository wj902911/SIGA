#pragma once

#include <MultiPatch.h>

class Function
{
public:
    virtual void eval_into(const Eigen::MatrixXi &numPointsPerDir,
                           Eigen::MatrixXd &values) const = 0;
};

class DisplacementFunction : public Function
{
public:
    DisplacementFunction(const MultiPatch &displacement)
        : m_displacement(&displacement) {}

    void eval_into(const Eigen::MatrixXi &numPointsPerDir,
                         Eigen::MatrixXd &values) const override;
private:
    const MultiPatch* m_displacement;
};