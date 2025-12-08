
#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>

#include <vector>
#include <string>
#include <iostream>
#include <limits>
#include <cmath>
#include <fstream>


dealii::UpdateFlags POISSON_VOLUME_FLAGS = 
    dealii::update_gradients |
    dealii::update_JxW_values |
    dealii::update_quadrature_points;


template<int dim>
void assemble_local_poisson_volume_matrix(
    const dealii::FEValues<dim> &fe_values,
    const dealii::Function<dim> &permittivity,
    dealii::FullMatrix<double> &local_mat
) {
    const auto flags = fe_values.get_update_flags();
    Assert((flags & POISSON_VOLUME_FLAGS) == POISSON_VOLUME_FLAGS,
       dealii::ExcMessage("assemble_local_poisson_volume_matrix: FEValues missing required update flags."));

    for (const uint q : fe_values.quadrature_point_indices()) {
        const dealii::Point<dim> &x_q = fe_values.quadrature_point(q);
        for (uint i = 0; i < fe_values.get_fe().dofs_per_cell; i++) {
            for (uint j = 0; j < fe_values.get_fe().dofs_per_cell; j++) {
                local_mat(i, j) += fe_values.shape_grad(i, q) 
                    * permittivity.value(x_q) 
                    * fe_values.shape_grad(j, q) 
                    * fe_values.JxW(q);
            }
        }
    }
}


dealii::UpdateFlags POISSON_BOUNDARY_FLAGS = 
    dealii::update_values |
    dealii::update_normal_vectors |
    dealii::update_gradients |
    dealii::update_JxW_values  |
    dealii::update_quadrature_points;


template<int dim>
void assemble_local_poisson_boundary_matrix(
    const dealii::FEFaceValues<dim> &fe_face_values,
    const dealii::Function<dim> &permittivity,
    dealii::FullMatrix<double> &local_mat
) {
    const auto flags = fe_face_values.get_update_flags();
    Assert((flags & POISSON_BOUNDARY_FLAGS) == POISSON_BOUNDARY_FLAGS,
       dealii::ExcMessage("assemble_local_poisson_boundary_matrix: FEValues missing required update flags."));

    for (const uint q : fe_face_values.quadrature_point_indices()) {
        const dealii::Point<dim> &x_q = fe_face_values.quadrature_point(q);
        const double eps = permittivity.value(x_q);

        for (uint i = 0; i < fe_face_values.get_fe().dofs_per_cell; ++i) {
            for (uint j = 0; j < fe_face_values.get_fe().dofs_per_cell; ++j) {
                const dealii::Tensor<1,dim>& normal = fe_face_values.normal_vector(q);
                local_mat(i, j) -= fe_face_values.shape_value(i, q) 
                    * normal * eps * fe_face_values.shape_grad(j, q) 
                    * fe_face_values.JxW(q);
            }
        }
    }
}


template<int dim>
void assemble_poisson_system(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::AffineConstraints<double>& constraints,
    const dealii::Function<dim>& permittivity,
    dealii::SparseMatrix<double>& system_matrix,
    dealii::Vector<double>& system_rhs
) {
    auto& fe = dof_handler.get_fe();
    
    dealii::QGauss<dim> quadrature{fe.degree + 1};
    dealii::QGauss<dim-1> face_quadrature{fe.degree + 1};

    dealii::FEValues<dim> fe_values{fe, quadrature, POISSON_VOLUME_FLAGS };
    dealii::FEFaceValues<dim> fe_face_values{fe, face_quadrature, POISSON_BOUNDARY_FLAGS };

    dealii::FullMatrix<double> local_mat(fe.dofs_per_cell, fe.dofs_per_cell);
    dealii::Vector<double> cell_rhs(fe.dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof(fe.dofs_per_cell);

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        local_mat = 0;
        cell_rhs = 0;

        fe_values.reinit(cell);
        assemble_local_poisson_volume_matrix(fe_values, permittivity, local_mat);

        for (uint face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (!cell->face(face)->at_boundary()) continue;
            fe_face_values.reinit(cell, face);
            assemble_local_poisson_boundary_matrix(fe_face_values, permittivity, local_mat);
        }

        cell->get_dof_indices(local_dof);
        constraints.distribute_local_to_global(local_mat, cell_rhs, local_dof, system_matrix, system_rhs);
    }
}


template <typename MatrixType, typename VectorType>
void solve_cg(const MatrixType& system_matrix, VectorType& solution, const VectorType& rhs) {
    dealii::SolverControl solver_control(1000, 1e-6 * rhs.l2_norm());
    dealii::PreconditionSSOR<MatrixType> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    dealii::SolverCG<VectorType> solver(solver_control);
    solver.solve(system_matrix, solution, rhs, dealii::PreconditionIdentity());
    std::cout << solver_control.last_step() << " CG iterations needed to converge.\n";
}

template <int dim>
dealii::Vector<double> solve_poisson_system(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::Function<dim>& permittivity,
    const dealii::AffineConstraints<double>& user_constraints
) {

    dealii::AffineConstraints<double> constraints{user_constraints};
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    dealii::SparsityPattern sp;
    sp.copy_from(dsp);

    dealii::Vector<double> solution;
    dealii::Vector<double> system_rhs;
    dealii::SparseMatrix<double> system_matrix;

    system_matrix.reinit(sp);
    system_rhs.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());

    assemble_poisson_system(dof_handler, constraints, permittivity, system_matrix, system_rhs);
    solve_cg(system_matrix, solution, system_rhs);
    constraints.distribute(solution);

    return solution;
}


template<int dim>
double smallest_cell_size(const dealii::Triangulation<dim>& triangulation) {
    double h_min = std::numeric_limits<double>::max();
    for (const auto &cell : triangulation.active_cell_iterators())
        h_min = std::min(h_min, cell->minimum_vertex_distance());
    return h_min;
}

template<int dim>
double get_l2_error(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::Vector<double>& solution,
    const dealii::Function<dim>& ex_solution
) {
    dealii::Vector<double> difference_per_cell(dof_handler.get_triangulation().n_active_cells());
    dealii::VectorTools::integrate_difference(
        dof_handler,
        solution,
        ex_solution,
        difference_per_cell,
        dealii::QGauss<2>(dof_handler.get_fe().degree + 1),
        dealii::VectorTools::L2_norm
    );

    return difference_per_cell.l2_norm();
}


template<int dim>
void write_out_solution(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::Vector<double>& solution,
    std::string file) 
{
    dealii::DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
    std::vector<std::string> solution_names;

    solution_names.emplace_back("potential");
    data_out.add_data_vector(solution, solution_names);
    data_out.build_patches();

    std::ofstream output(file);
    data_out.write_vtu(output);
}
