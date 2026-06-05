#pragma once

#include <GPUAssembler.h>

/**
 * GPU assembler for finite-strain strain-gradient elasticity.
 *
 * Usage:
 *   1. Construct it the same way as GPUAssembler, with the displacement basis.
 *   2. Set material and strain-gradient options through options().
 *   3. Pass the assembler to GPUSolver, or call assemble() with the solver vector.
 *
 * Example:
 *   GPUStrainGradientElasticityAssembler assembler(geometry, basis, bcInfo, bodyForce);
 *   assembler.options().setReal("youngs_modulus", YM);
 *   assembler.options().setReal("poissons_ratio", PR);
 *   assembler.options().setReal("length_scale", l);
 *   assembler.options().setInt("material_law", 1);
 *
 * Options:
 *   youngs_modulus    - Young's modulus.
 *   poissons_ratio    - Poisson's ratio.
 *   length_scale      - Strain-gradient length scale l.
 *   force_scaling     - Multiplier for volumetric body force.
 *   local_stiffening  - Exponent used in weightBody = w * det(J)^(-local_stiffening) * det(J).
 *   material_law      - 0 for St. Venant-Kirchhoff, 1 for neo-Hookean.
 *
 * Notes:
 *   - The constructor requests second derivatives from GPUAssembler.
 *   - The displacement basis should be at least C1 inside patches for the
 *     Hessian-dependent strain-gradient contribution to be meaningful.
 *   - Boundary conditions use the same DofMapper/SparseSystem path as GPUAssembler.
 */
class GPUStrainGradientElasticityAssembler : public GPUAssembler
{
private:
    DeviceArray<double> m_sgGPData;

    __host__
    void constructStrainGradientStressFunctions(MultiPatchDeviceView displacementView,
                                                GPUFunction* firstPiolaStressFunction,
                                                GPUFunction* cauchyStressFunction);

public:
    /**
     * Builds the displacement sparse system and precomputes geometry,
     * displacement basis values, first derivatives, and second derivatives at
     * Gauss points.
     */
    __host__
    GPUStrainGradientElasticityAssembler(const MultiPatch& multiPatch,
                                         const MultiBasis& multiBasis,
                                         const BoundaryConditions& bc,
                                         const Eigen::VectorXd& bodyForce);

    /**
     * Returns the default option set listed above. This is called by
     * setDefaultOptions() and by the constructor.
     */
    __host__
    static OptionList defaultStrainGradientOptions();

    /**
     * Resets options to the default strain-gradient material settings.
     * Call this only before custom option values are applied.
     */
    __host__
    void setDefaultOptions();

    /**
     * Assembles tangent matrix and residual/RHS for the current solution vector.
     *
     * numIter follows GPUAssembler convention:
     *   - 0 uses the prescribed Dirichlet values in elimination terms.
     *   - nonzero uses zero Dirichlet increments.
     */
    __host__
    void assemble(const DeviceVectorView<double>& solVector,
                  int numIter,
                  const DeviceNestedArrayView<double>& fixedDoFs) override;

    /**
     * Constructs a full first Piola-Kirchhoff stress function from the current
     * strain-gradient constitutive law. The output function must have target
     * dimension dim * dim, with components stored as P(a, A) in row-major order.
     */
    __host__
    void constructFirstPiolaStressFunction(const DeviceVectorView<double>& solVector,
                                           const DeviceNestedArrayView<double>& fixedDoFs,
                                           GPUFunction& firstPiolaStressFunction);

    /**
     * Same recovery, but reuses an already constructed displacement function.
     */
    __host__
    void constructFirstPiolaStressFunction(GPUFunction& displacementFunction,
                                           GPUFunction& firstPiolaStressFunction);

    /**
     * Constructs first Piola and Cauchy stress functions from one strain-gradient
     * Gauss-point evaluation. first Piola must have target dimension dim * dim;
     * Cauchy stress must have target dimension dim * (dim + 1) / 2.
     */
    __host__
    void constructStrainGradientStressFunctions(const DeviceVectorView<double>& solVector,
                                                const DeviceNestedArrayView<double>& fixedDoFs,
                                                GPUFunction& firstPiolaStressFunction,
                                                GPUFunction& cauchyStressFunction);

    /**
     * Same combined recovery, but reuses an already constructed displacement
     * function.
     */
    __host__
    void constructStrainGradientStressFunctions(GPUFunction& displacementFunction,
                                                GPUFunction& firstPiolaStressFunction,
                                                GPUFunction& cauchyStressFunction);
};
