import backend
from .assembly import assemble, assemble_system
from .solving import solve
from .projection import project
from .interpolation import interpolate
from .ufl_constraints import UFLEqualityConstraint, UFLInequalityConstraint
from .shapead_transformations import (transfer_from_boundary,
                                      transfer_to_boundary)
from .types import (Function, Constant, DirichletBC,
                    Mesh, UnitSquareMesh, UnitIntervalMesh, IntervalMesh,
                    UnitCubeMesh)
from .boundarymesh import BoundaryMesh
from .types import Function, Constant, DirichletBC
from .ufl_constraints import UFLEqualityConstraint, UFLInequalityConstraint
if backend.__name__ != "firedrake":
    from .types import Expression, UserExpression, CompiledExpression
    from .types import io
    from .newton_solver import NewtonSolver
    from .lu_solver import LUSolver
    from .krylov_solver import KrylovSolver
from .variational_solver import (NonlinearVariationalProblem, NonlinearVariationalSolver,
                                 LinearVariationalProblem, LinearVariationalSolver)
from pyadjoint import (Tape, set_working_tape, get_working_tape,
                       pause_annotation, continue_annotation,
                       ReducedFunctional,
                       taylor_test, taylor_to_dict,
                       compute_gradient, compute_hessian,
                       AdjFloat, Control, minimize, MinimizationProblem,
                       IPOPTSolver, ROLSolver, InequalityConstraint, EqualityConstraint,
                       MoolaOptimizationProblem, print_optimization_methods)

set_working_tape(Tape())
