
# pragma once

#include <boost/tuple/detail/tuple_basic.hpp>
#include <deal.II/base/function.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/vector_tools_boundary.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <format>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/cell_data.h>

#include "mesh.h"
#include "poisson.h"
#include "assembly_predicates.h"
#include "poisson_local_assembly.h"

using namespace dealii::Functions;

const double water_permitivity = 78.;
const double air_permitivity = 1.;
const double wedge_permitivity = 2.;

const double V2 = 5.;
const double V1= 0.;

//struct GradientDensity
//{
//    const dealii::Vector<double> &potential;
//    const unsigned int c;
//
//    GradientDensity(
//        const dealii::Vector<double> &sol,
//        unsigned int c
//    ) : potential(sol), c(c) {};
//
//    template <typename FEType>
//    double operator()(const FEType &fe_values, unsigned int q) {
//        dealii::Tensor<1, FEType::dimension, double> grad;
//        for (uint i = 0; i < fe.dofs_per_cell; i++) {
//            int dof_index = fe_values.get_cell()->vertex_dof_index(i, c);
//            grad += potential[dof_index] * fe_values.shape_grad_component(i, q, c);
//        }
//
//        std::cout << grad.norm_square() << " ";
//
//        return std::sqrt(1 + grad.norm_square());
//    }
//};


template <int dim>
class ComponentIdentityFunction : public dealii::Function<dim>
{
public:
    ComponentIdentityFunction(unsigned int component)
        : dealii::Function<dim>(1), component(component)
    {}

    virtual double value(const dealii::Point<dim> &p, const unsigned int c = 0) const override {
        return p[component];
    }

private:
    unsigned int component;
};

template<int dim=2>
void improve_mesh_winslow() {

    auto triangulation = build_triangulation(vertices, faces, material_ids, manifold_ids, circle_centers, boundary_ids, boundary_manifold_ids);

    for (int i = 0; i < 10; i++) {
        dealii::DoFHandler<dim> dof_handler{triangulation};
        dealii::FESystem<dim> fe{ dealii::FE_Q<dim>(1), dim+1 };
        dof_handler.distribute_dofs(fe);

        dealii::AffineConstraints<double> constraints;
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);

        add_face_dirichlet_conditions(dof_handler, constraints,
                ConstantFunction<dim>(V1, dim+1),
                BoundaryIDPredicate<dim>(ELECTRODE1_BOUNDARY_ID),
                {2});

        add_face_dirichlet_conditions(dof_handler, constraints,
                ConstantFunction<dim>(V2, dim+1),
                BoundaryIDPredicate<dim>(ELECTRODE2_BOUNDARY_ID),
                {2});

        add_vertex_dirichlet_conditions(dof_handler, constraints,
                ConstantFunction<dim>(0., dim+1),
                AllVerticesPredicate<dim>(),
                {0, 1});

        constraints.close();
        auto potential_system = LinearSystem(dof_handler, constraints);

        assemble_poisson_volume(dof_handler, constraints, potential_system.matrix, potential_system.rhs,
                ConstantQuadratureFunction(water_permitivity),
                MaterialIDPredicate<2>{.material_id=WATER_MAT_ID},
                {2});

        assemble_poisson_volume(dof_handler, constraints, potential_system.matrix, potential_system.rhs,
                ConstantQuadratureFunction(air_permitivity),
                MaterialIDPredicate<2>{.material_id=AIR_MAT_ID},
                {2});

        assemble_poisson_volume(dof_handler, constraints, potential_system.matrix, potential_system.rhs,
                ConstantQuadratureFunction(wedge_permitivity),
                MaterialIDPredicate<2>{.material_id=WEDGE_MAT_ID},
                {2});

        dealii::Vector<double> potential_solution = solve_cg(potential_system.matrix, potential_system.rhs);
        constraints.distribute(potential_solution);

        dealii::AffineConstraints<double> constraints2;
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints2);

        add_vertex_dirichlet_conditions(dof_handler, constraints2,
                potential_solution,
                AllVerticesPredicate<dim>(),
                {2});

        add_face_tangency_conditions(dof_handler, constraints2, AllBoundariesPredicate<dim>(), structure_verex_ids, {0, 1});
        add_face_tangency_conditions(dof_handler, constraints2, AllInterfacesPredicate<dim>(), structure_verex_ids, {0, 1});

        add_vertex_dirichlet_conditions(dof_handler, constraints2,
                IdentityFunction<dim>(),
                VertexListPredicate<dim>(structure_verex_ids),
                {0, 1});

        //add_face_dirichlet_conditions(dof_handler, constraints2,
        //        IdentityFunction<dim>(),
        //        AllInterfacesPredicate<dim>(),
        //        {0, 1});

        constraints2.close();
        potential_system.reinit(dof_handler, constraints2);

        assemble_winslow_system(dof_handler, constraints2, potential_system.matrix, potential_system.rhs, potential_solution, 0, 2);
        assemble_winslow_system(dof_handler, constraints2, potential_system.matrix, potential_system.rhs, potential_solution, 1, 2);

        auto solution = solve_cg(potential_system.matrix, potential_system.rhs);
        constraints2.distribute(solution);

        std::vector<std::vector<float>> new_vertices = vertices;
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            for (unsigned int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v) {
                for (unsigned int c : {0, 1}) {
                    unsigned int dof_index = cell->vertex_dof_index(v, c);
                    new_vertices[cell->vertex_index(v)][c] = solution[dof_index];
                }
            }
        }

        triangulation = build_triangulation(new_vertices, faces, material_ids, manifold_ids, circle_centers, boundary_ids, boundary_manifold_ids);

        dealii::GridOut grid_out;
        std::ofstream out(std::format("winslow_solutions/mesh_smoothed{:1}.msh", i));
        grid_out.write_msh(triangulation, out);
    }
}


//template<int dim=2>
//void winslow() {
//    auto reactor_triangulation = build_triangulation(vertices, faces, material_ids, manifold_ids, circle_centers, boundary_ids, boundary_manifold_ids);
//
//    dealii::DoFHandler<dim> dof_handler{triangulation};
//    dealii::FESystem<dim> fe{ dealii::FE_Q<dim>(1), dim };
//    dof_handler.distribute_dofs(fe);
//
//    dealii::AffineConstraints<double> constraints;
//    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
//
//    constraints.close();
//    auto potential_system = LinearSystem(dof_handler, constraints);
//
//    assemble_poisson_volume(dof_handler, constraints, potential_system.matrix, potential_system.rhs,
//            ConstantQuadratureFunction(1),
//            AllCellsPredicate<dim>(),
//            {0});
//
//    assemble_poisson_volume(dof_handler, constraints, potential_system.matrix, potential_system.rhs,
//            ConstantQuadratureFunction(1),
//            AllCellsPredicate<dim>(),
//            {0});
//
//    dealii::Vector<double> potential_solution = solve_cg(potential_system.matrix, potential_system.rhs);
//    constraints.distribute(potential_solution);
//
//
//
//}

