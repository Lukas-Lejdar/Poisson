
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>

#include "poisson.h"

struct RadialCapacitor{
    const double r0;
    const double r1;
    const double r2;

    const double voltage0;
    const double voltage1;

    const double epsilon0_1;
    const double epsilon1_2;
};

template <int dim>
class PermittivityFunction : public dealii::Function<dim> {
public:
    const RadialCapacitor capacitor;
    PermittivityFunction(const RadialCapacitor capacitor) : capacitor(capacitor) {}

    double value(const dealii::Point<dim> &p, const unsigned int = 0) const override {
        return p.norm() < capacitor.r1 ? capacitor.epsilon0_1 : capacitor.epsilon1_2;
    }
};

class Exact2DPotentialSolution : public dealii::Function<2> {
private:
    // phi(r ∈ [r0, r1]) = solution[0] ln r + solution[2]
    // phi(r ∈ [r1, r2]) = solution[2] ln r + solution[3]
    dealii::Vector<double> consts; 

public:
    const RadialCapacitor capacitor;
    Exact2DPotentialSolution(const RadialCapacitor capacitor) 
        : capacitor(capacitor) 
    {
        const double rhs_vec[4] = { capacitor.voltage0, capacitor.voltage1, 0, 0 };
        const double system_mat[4][4] = {
            { std::log(capacitor.r0), 1.0, 0.0, 0.0 },
            { 0.0, 0.0, std::log(capacitor.r2), 1.0 },
            { std::log(capacitor.r1), 1.0, -std::log(capacitor.r1), -1.0 },
            { capacitor.epsilon0_1, 0.0, -capacitor.epsilon1_2, 0.0 }
        };

        dealii::FullMatrix<double> system(4, 4); // system matrix 
        dealii::Vector<double> rhs(4);
        consts.reinit(4);

        for (uint i=0; i < 4; i++) {
            rhs[i] = rhs_vec[i];
            for (uint j = 0; j < 4; j++) {
                system(i,j) = system_mat[i][j];
            }
        }

        system.gauss_jordan();             // in-place Gauss-Jordan invert
        system.vmult(consts, rhs);         // x = A^{-1} * b
    }

    double value(const dealii::Point<2> &p, const unsigned int = 0) const override {
        double r = p.norm();
        return (r < capacitor.r1)
            ? consts[0]*std::log(r) + consts[1] 
            : consts[2]*std::log(r) + consts[3];
    }
};

template<int dim>
dealii::Triangulation<2> get_capacitor_triangulation(const RadialCapacitor& capacitor) {
    dealii::Triangulation<2> triangulation;
    const dealii::Point<2> center(0, 0);
    dealii::GridGenerator::hyper_shell( triangulation, center, capacitor.r0, capacitor.r2, 10, true);
    return triangulation;
} 

int main() {

    // problem definition

    const   RadialCapacitor capacitor{0.5, 0.75, 1., 0., 1., 1., 10.}; // r0, r1, r2, U0, U2, eps0_1, eps1_2
    const Exact2DPotentialSolution ex_solution(capacitor); 
    const PermittivityFunction<2> permittivity{capacitor};

    // Create triangulation

    dealii::Triangulation<2> triangulation = get_capacitor_triangulation<2>(capacitor);
    triangulation.refine_global(2);

    // boundary ids assigned in dealii::GridGenerator::hyper_shell
    const dealii::types::boundary_id inner_id = 0, outer_id = 1; 

    // solve

    dealii::FE_Q<2> fe{1};
    dealii::DoFHandler<2> dof_handler{triangulation};
    dof_handler.distribute_dofs(fe);

    std::ofstream error_file("l2_errors.txt");

    for (int i = 0; i < 9; i++) {

        dealii::AffineConstraints<double> constraints;
        dealii::VectorTools::interpolate_boundary_values(dof_handler, inner_id, dealii::Functions::ConstantFunction<2>(capacitor.voltage0), constraints); 
        dealii::VectorTools::interpolate_boundary_values(dof_handler, outer_id, dealii::Functions::ConstantFunction<2>(capacitor.voltage1), constraints); 

        dealii::Vector<double> solution = solve_poisson_system(dof_handler, permittivity, constraints);
        write_out_solution(dof_handler, solution, "solutions/solution" + std::to_string(i) + ".vtu");

        // l2 error

        double l2_error = get_l2_error(dof_handler, solution, ex_solution);
        double cell_size = smallest_cell_size(triangulation);
        std::cout << "smallest cell size: " << cell_size << " l2: " << l2_error << "\n";
        error_file << cell_size << " " << l2_error << "\n";
        error_file.flush();

        // refinement

        dealii::Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
        dealii::KellyErrorEstimator<2>::estimate( dof_handler, dealii::QGauss<1>(fe.degree + 1), {}, solution, estimated_error_per_cell);
        dealii::GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.3, 0.03);

        //triangulation.refine_global(1);
        triangulation.execute_coarsening_and_refinement();
        dof_handler.distribute_dofs(fe);
    }

    error_file.close();
}
