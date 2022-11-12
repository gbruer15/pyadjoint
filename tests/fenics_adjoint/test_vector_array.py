import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *
from pyadjoint import create_overloaded_object

import numpy
import pprint


op_map = {
    "inner": ("vec", "vec"),
    "__add__": ("vec", "vec"),
    "__sub__": ("vec", "vec"),
    "__mul__": ("vec", "vec"),
    "__truediv__": ("vec", "float"),
    "__neg__": ("vec",),
}

@pytest.mark.parametrize("op_name", op_map.keys())
def test_simple_solve(op_name):
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    input_specs = op_map[op_name]

    args = []
    for ipt in input_specs:
        if ipt == "vec":
            f = Function(V)
            fvec = create_overloaded_object(f.vector())
            fvec[:] = numpy.arange(len(fvec))
            args.append(fvec)
        elif ipt == "float":
            args.append(AdjFloat(3))
        else:
            args.append(create_overloaded_object(ipt))


    op = getattr(type(args[0]), op_name)
    J = op(*args)
    if not isinstance(J, AdjFloat):
        J = J.inner(J)

    hs = []
    for ipt in input_specs:
        if ipt == "vec":
            f = Function(V)
            fvec = create_overloaded_object(f.vector())
            fvec[:] = numpy.arange(len(fvec)) * 2
            hs.append(fvec)
        elif ipt == "float":
            hs.append(AdjFloat(2))
        else:
            hs.append(create_overloaded_object(ipt))

    for arg, h in zip(args, hs):
        m = Control(arg)
        Jhat = ReducedFunctional(J, m)

        results = taylor_to_dict(Jhat, arg, h)

        for (i, Ri) in enumerate(["R0","R1","R2"]):
            correct_rate = min(results[Ri]["Rate"]) >= i + 0.95
            exact = max(results[Ri]["Residual"]) <= 1e-10
            assert correct_rate or exact, pprint.pformat(results)
