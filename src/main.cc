
#include <boost/math/special_functions/math_fwd.hpp>
#include <cstdlib>
#include <deal.II/base/function.h>
#include <deal.II/base/types.h>
#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <iterator>
#include <string>
#include <format>
#include <chrono>
#include <iostream>
#include <cstdlib>


#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/cell_data.h>

//#include "mesh_processor.h"
#include "assembly_predicates.h"
#include "mesh.h"
#include "poisson.h"
#include "poisson_local_assembly.h"

using namespace dealii::Functions;

struct RadialCapacitor{
    const double r0;
    const double r1;
    const double r2;

    const double voltage0;
    const double voltage2;

    const double epsilon0_1;
    const double epsilon1_2;

    const double surface_charge;
};

//template <int dim>
//class PermittivityFunction : public dealii::Function<dim> {
//public:
//    const RadialCapacitor capacitor;
//    PermittivityFunction(const RadialCapacitor capacitor) : capacitor(capacitor) {}
//
//    double value(const dealii::Point<dim> &p, const unsigned int = 0) const override {
//        return p.norm() < capacitor.r1 ? capacitor.epsilon0_1 : capacitor.epsilon1_2;
//    }
//};

class Exact2DPotentialSolution : public dealii::Function<2> {
private:
    dealii::Vector<double> consts; 
    const RadialCapacitor capacitor;

public:
    const dealii::Vector<double>& get_consts() const { return consts; }

    double value(const dealii::Point<2> &p, const unsigned int = 0) const override {
        double r = p.norm();
        return (r < capacitor.r1)
            ? consts[0]*std::log(r) + consts[1] 
            : consts[2]*std::log(r) + consts[3];
    }

    Exact2DPotentialSolution(const RadialCapacitor& capacitor) : capacitor(capacitor) {

        const double rhs_vec[4] = { capacitor.voltage0, capacitor.voltage2, 0, capacitor.surface_charge*capacitor.r1 };
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
};

template<int dim>
void change_boundary_id(
    dealii::Triangulation<dim>& triangulation,
    const dealii::types::boundary_id from,
    const dealii::types::boundary_id to
) {
    for (auto &cell : triangulation.active_cell_iterators()) {
        for (uint f = 0; f < dealii::GeometryInfo<2>::faces_per_cell; ++f) {
            if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == from) {
                cell->face(f)->set_boundary_id(to);
            }
        }
    }
}

const dealii::types::boundary_id INNER_ID = 1, OUTER_ID = 2, MIDDLE_ID = 3, SIDES_ID = 4; 
const dealii::types::material_id INNER_MAT_ID = 4, OUTER_MAT_ID = 5;


template<int dim>
dealii::Triangulation<dim> create_capacitor_triangulation(const RadialCapacitor& capacitor) {
    dealii::Triangulation<dim> triangulation;
    const dealii::Point<dim> center(0, 0);
    dealii::GridGenerator::hyper_shell( triangulation, center, capacitor.r0, capacitor.r2, 10, true);
    change_boundary_id(triangulation, 1, OUTER_ID);
    change_boundary_id(triangulation, 0, INNER_ID);
    return triangulation;
} 

dealii::Triangulation<2> import_gmsh_capacitor_triangulation(std::string&& file) {
    dealii::Triangulation<2> triangulation;
    dealii::GridIn<2> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream input_file(file);
    gridin.read_msh(input_file);

    return triangulation;
} 

template <int dim>
unsigned int count_faces_with_boundary_id(const dealii::Triangulation<dim> &triangulation, const unsigned int boundary_id) {
    unsigned int count = 0;
    for (const auto &cell : triangulation.active_cell_iterators()) {
        for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
            if (cell->face(f)->at_boundary() &&
                cell->face(f)->boundary_id() == boundary_id)
                ++count;
        }
    }
    return count;
}

const double MIN_CELL_SIZE = 0.001;
const double MAX_CELL_SIZE = 0.3;



//void test_on_radial_capacitor() {
//
//    // problem definition
//
//    const RadialCapacitor capacitor{0.5, 0.75, 1., 0., 0.5, 1., 2., 1.0}; // r0, r1, r2, U0, U2, eps0_1, eps1_2, surface_charge
//    const Exact2DPotentialSolution ex_solution(capacitor); 
//    //const PermittivityFunction<2> permittivity{capacitor};
//
//    // Create triangulation
//
//    dealii::Triangulation<2> triangulation = import_gmsh_capacitor_triangulation("../capacitor.msh");
//    dealii::Point<2> center(0.0, 0.0);
//    triangulation.set_manifold(1, dealii::SphericalManifold<2>(center));
//    triangulation.set_all_manifold_ids(1);
//
//    // solve
//
//    dealii::FE_Q<2> fe{1};
//    dealii::DoFHandler<2> dof_handler{triangulation};
//    dof_handler.distribute_dofs(fe);
//
//    std::ofstream error_file("l2_errors.txt");
//
//    double smalles_initial_cell_size = smallest_cell_size(triangulation);
//
//    for (int i = 0; i < 12; i++) {
//
//        dealii::AffineConstraints<double> constraints;
//        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
//        dealii::VectorTools::interpolate_boundary_values(dof_handler, INNER_ID, ConstantFunction<2>(capacitor.voltage0), constraints);
//        dealii::VectorTools::interpolate_boundary_values(dof_handler, OUTER_ID, ConstantFunction<2>(capacitor.voltage2), constraints);
//        constraints.close();
//
//        auto poisson_system = LinearSystem(dof_handler, constraints);
//
//        assemble_poisson_volume(dof_handler, constraints, poisson_system.matrix, poisson_system.rhs,
//                ConstantFunction<2>(capacitor.epsilon0_1),
//                MaterialIDPredicate<2>{.material_id=INNER_MAT_ID});
//
//        assemble_poisson_volume(dof_handler, constraints, poisson_system.matrix, poisson_system.rhs,
//                ConstantFunction<2>(capacitor.epsilon1_2),
//                MaterialIDPredicate<2>{.material_id=OUTER_MAT_ID});
//
//        // surface charge
//
//        assemble_poisson_boundary_source(dof_handler, poisson_system.rhs,
//                ConstantFunction<2>(capacitor.surface_charge),
//                InterfacePredicate<2>{.mat1=INNER_MAT_ID, .mat2=OUTER_MAT_ID});
//
//        // boundary conditions
//
//        assemble_poisson_neuman_condition(dof_handler, poisson_system.rhs,
//                ConstantFunction<2>(capacitor.epsilon1_2),
//                ConstantFunction<2>(ex_solution.get_consts()(2)/capacitor.r2),
//                BoundaryIDPredicate<2>{.boundary_id=OUTER_ID});
//
//        //assemble_poisson_neuman_condition(dof_handler, poisson_system.rhs,
//        //        ConstantFunction<2>(capacitor.epsilon1_2),
//        //        ZeroFunction<2>(),
//        //        BoundaryAndMaterialPredicate<2>{.boundary_id=OUTER_ID, .material_id=OUTER_MAT_ID});
//
//        //assemble_poisson_neuman_condition(dof_handler, poisson_system.rhs,
//        //        ConstantFunction<2>(capacitor.epsilon0_1),
//        //        ZeroFunction<2>(),
//        //        BoundaryAndMaterialPredicate<2>{.boundary_id=OUTER_ID, .material_id=INNER_MAT_ID});
//
//
//        auto solution = solve_cg(poisson_system.matrix, poisson_system.rhs);
//        constraints.distribute(solution);
//
//        // out
//
//        //dealii::VectorTools::interpolate(dof_handler, ex_solution, solution);
//        write_out_solution(dof_handler, solution, std::format("solutions/solution{:02}.vtu", i));
//
//        // l2 error
//
//        double l2_error = get_l2_error(dof_handler, solution, ex_solution);
//        double cell_size = smallest_cell_size(triangulation);
//        std::cout << "smallest cell size: " << cell_size << " l2: " << l2_error << "\n";
//        error_file << smalles_initial_cell_size * std::pow(2, -i) << " " << l2_error << "\n";
//        error_file << cell_size << " " << l2_error << "\n";
//        error_file.flush();
//
//        // refinement
//
//        //dealii::Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
//        //dealii::KellyErrorEstimator<2>::estimate( dof_handler, dealii::QGauss<1>(fe.degree + 1), {}, solution, estimated_error_per_cell);
//        //dealii::GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.3, 0.03);
//        //restrict_refinement_by_cell_size(triangulation, MIN_CELL_SIZE, MAX_CELL_SIZE);
//
//        triangulation.refine_global(1);
//        dof_handler.distribute_dofs(fe);
//
//        //triangulation.execute_coarsening_and_refinement();
//    }
//
//    error_file.close();
//}

dealii::Triangulation<2> build_triangulation(
    const std::vector<std::vector<float>> vertices,
    const std::vector<std::vector<int>> faces,
    const std::vector<int> material_ids,
    const std::vector<std::pair<int, int>> manifold_ids,
    const std::vector<std::pair<int, std::vector<float>>> circle_centers,
    const std::vector<std::pair<std::vector<int>, int>> boundary_ids,
    const std::vector<std::pair<std::vector<int>, int>> boundary_manifold_ids
) {
    dealii::Triangulation<2> triangulation;

    std::vector<dealii::Point<2>> vertices_dealii(vertices.size());
    for (unsigned int i = 0; i < vertices.size(); ++i) {
        vertices_dealii[i] = dealii::Point<2>(vertices[i][0], vertices[i][1]);
    }

    std::vector<dealii::CellData<2>> cells(faces.size());
    for (unsigned int i = 0; i < faces.size(); ++i) {
        cells[i].vertices[0] = faces[i][0];
        cells[i].vertices[1] = faces[i][1];
        cells[i].vertices[2] = faces[i][3];
        cells[i].vertices[3] = faces[i][2];
        cells[i].material_id = i;
    }

    triangulation.create_triangulation(vertices_dealii, cells, dealii::SubCellData());

    for (auto center : circle_centers) {
        dealii::Point<2> point(center.second[0], center.second[1]);
        triangulation.set_manifold(center.first, dealii::SphericalManifold<2>(point));
    }

    for (auto &cell : triangulation.active_cell_iterators()) {
        int idx = cell->material_id();
        cell->set_material_id(material_ids[idx]);
        cell->set_user_index(idx);
        
        dealii::Point<2> center;
        for (unsigned int v = 0; v < dealii::GeometryInfo<2>::vertices_per_cell; ++v) {
            center += cell->vertex(v);
        }
        center /= dealii::GeometryInfo<2>::vertices_per_cell;

        if (center[1] >= 3.) {
            cell->set_material_id(AIR_MAT_ID);
        }

        for (int i = 0; i < manifold_ids.size(); i++) {
            if (manifold_ids[i].first == idx) {
                //cell->set_manifold_id(manifold_ids[i].second);
            }
        }
    }

    for (auto &cell : triangulation.active_cell_iterators()) {
        for (unsigned int f = 0; f < dealii::GeometryInfo<2>::faces_per_cell; ++f) {
            int v0 = cell->face(f)->vertex_index(0);
            int v1 = cell->face(f)->vertex_index(1);

            
            if (!cell->face(f)->at_boundary() && cell->neighbor(f)->index() < cell->index())
                continue;

            if (v0 > v1)
                std::swap(v0, v1);


            for (auto be : boundary_ids) {
                if (v0 == be.first[0] && v1 == be.first[1]) {
                    cell->face(f)->set_boundary_id(be.second);
                }
            }

            for (auto me : boundary_manifold_ids) {
                if (v0 == me.first[0] && v1 == me.first[1]) {
                    cell->face(f)->set_manifold_id(me.second);
                }
            }
        }
    }

    return triangulation;
}

auto duration_ms(auto start, auto end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

const double water_permitivity = 78.;
const double air_permitivity = 1.;
const double wedge_permitivity = 2.;

const double V2 = 10.;
const double V1= 0.;

void solve_reactor_potential(
    dealii::DoFHandler<2>& dof_handler,
    dealii::Vector<double>& solution,
    dealii::Vector<double>& residual
) {

    auto t_start = std::chrono::high_resolution_clock::now();

    dealii::AffineConstraints<double> constraints;
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    dealii::VectorTools::interpolate_boundary_values(dof_handler, ELECTRODE2_BOUNDARY_ID, ConstantFunction<2>(V2), constraints);
    dealii::VectorTools::interpolate_boundary_values(dof_handler, ELECTRODE1_BOUNDARY_ID, ConstantFunction<2>(V1), constraints);
    constraints.close();

    auto t_constraints = std::chrono::high_resolution_clock::now();
    std::cout << "Constraint setup: " << duration_ms(t_start, t_constraints) << " ms\n";

    auto system = LinearSystem(dof_handler, constraints);

    assemble_poisson_volume(dof_handler, constraints, system.matrix, system.rhs,
            ConstantQuadratureFunction(water_permitivity),
            MaterialIDPredicate<2>{.material_id=WATER_MAT_ID});

    assemble_poisson_volume(dof_handler, constraints, system.matrix, system.rhs,
            ConstantQuadratureFunction(air_permitivity),
            MaterialIDPredicate<2>{.material_id=AIR_MAT_ID});

    assemble_poisson_volume(dof_handler, constraints, system.matrix, system.rhs,
            ConstantQuadratureFunction(wedge_permitivity),
            MaterialIDPredicate<2>{.material_id=WEDGE_MAT_ID});

    auto t_assembly = std::chrono::high_resolution_clock::now();
    std::cout << "Assembly: " << duration_ms(t_constraints, t_assembly) << " ms\n";

    solve_cg(system.matrix, system.rhs, solution);
    constraints.distribute(solution);

    residual = 0;
    residual += system.rhs;
    residual *= -1;
    system.matrix.vmult(residual, solution);

    auto t_solve = std::chrono::high_resolution_clock::now();
    std::cout << "Solver: " << duration_ms(t_assembly, t_solve) << " ms\n";

}

void compute_refined_potential() {
    auto triangulation = build_triangulation(vertices, faces, material_ids, manifold_ids, circle_centers, boundary_ids, boundary_manifold_ids);
    triangulation.refine_global(3);

    dealii::FE_Q<2> fe{1};
    dealii::DoFHandler<2> dof_handler{triangulation};
    dof_handler.distribute_dofs(fe);

    //solve_reactor_potential(dof_handler, solution);
}


void compute_reactor_potential() {
    auto triangulation = build_triangulation(vertices, faces, material_ids, manifold_ids, circle_centers, boundary_ids, boundary_manifold_ids);
    triangulation.set_mesh_smoothing( dealii::Triangulation<2>::limit_level_difference_at_vertices );

    std::cout << "built triangulation\n";

    dealii::FE_Q<2> fe{1};
    dealii::DoFHandler<2> dof_handler{triangulation};
    dof_handler.distribute_dofs(fe);

    dealii::Vector<double> residual;
    dealii::Vector<double> solution(dof_handler.n_dofs());
    dealii::Vector<double> prev_solution(dof_handler.n_dofs());
    //dealii::Triangulation<2> prev_triangulation;

    auto permittivity = MaterialIdQuadratureFunction(
                {WATER_MAT_ID, AIR_MAT_ID, WEDGE_MAT_ID},
                {water_permitivity, air_permitivity, wedge_permitivity});

    std::ofstream diff_out("max_diff.txt");
    diff_out << "# iter  n_cells  n_dofs  max_abs_diff  max_grad_diff_norm\n";

    for (int i = 0; i < 9; i++) {

        std::cout << "\n";

        solution = prev_solution;
        residual.reinit(dof_handler.n_dofs());
        solve_reactor_potential(dof_handler, solution, residual);

        dealii::SolutionTransfer<2> solution_transfer(dof_handler);
        solution_transfer.prepare_for_coarsening_and_refinement(solution);

        std::cout << "solved\n";

        // refinement

        auto t_start = std::chrono::high_resolution_clock::now();

        dealii::Vector<double> cell_residual(triangulation.n_active_cells());
        calculate_poisson_residual(dof_handler, solution, cell_residual,
                permittivity, AllCellsPredicate<2>());

        auto t_cell_residual = std::chrono::high_resolution_clock::now();
        std::cout << "Calculating cell residual: " << duration_ms(t_start, t_cell_residual) << " ms\n";

        dealii::Vector<float> face_residual(triangulation.n_active_cells());
        calculate_poisson_face_residual(dof_handler, solution, face_residual,
                permittivity, AllNonBoundaryFacesPredicate<2>());

        auto t_face_residual = std::chrono::high_resolution_clock::now();

        std::cout << "Calculating face residual: " << duration_ms(t_cell_residual, t_face_residual) << " ms\n";
        std::cout << "cell residual " << cell_residual.l2_norm() << "\n"; 
        std::cout << "face residual " << face_residual.l2_norm() << "\n"; 

        dealii::Vector<float> error_per_cell(triangulation.n_active_cells());
        for (unsigned int i = 0; i < triangulation.n_active_cells(); ++i) {
            error_per_cell[i] = std::sqrt( cell_residual[i] + face_residual[i] );
        }

        write_out_solution( dof_handler, solution, prev_solution, error_per_cell, i); 
        //write_out_solution( dof_handler, next_solution, solution, error_per_cell, i);

        auto t_output = std::chrono::high_resolution_clock::now();
        //std::cout << "Writing to files: " << duration_ms(t_face_residual, t_output) << " ms\n";

        //dealii::KellyErrorEstimator<2>::estimate( dof_handler, dealii::QGauss<1>(fe.degree + 1), {}, solution, error_per_cell);
        dealii::GridRefinement::refine_and_coarsen_fixed_number(triangulation, error_per_cell, 0.3, 0.0);
        //refine_hanging_nodes_on_material_interfaces(triangulation);
        triangulation.execute_coarsening_and_refinement();

        //triangulation.refine_global(1);
    
        dof_handler.distribute_dofs(fe);
        prev_solution.reinit(dof_handler.n_dofs());
        solution_transfer.interpolate(prev_solution);

        auto t_refinement = std::chrono::high_resolution_clock::now();
        std::cout << "Executing refinement: " << duration_ms(t_output, t_refinement) << " ms\n";

        //dealii::GridOut grid_out;
        //std::ofstream out("exported.msh");
        //grid_out.write_msh(triangulation, out);

    }
}




int main() {

    //auto triangulation = build_triangulation(vertices, faces, material_ids, manifold_ids, circle_centers, boundary_ids, boundary_manifold_ids);

    //std::cout << "start\n";

    //auto triangulation = build_triangulation();

    //std::cout << "triangulation built \n";
    //triangulation.refine_global(2);

    //dealii::GridOut grid_out;
    //std::ofstream out("exported.msh");
    //grid_out.write_msh(triangulation, out);

    compute_reactor_potential();

    //improve_mesh_winslow();
}

