import backend
import numpy

from dolfin_adjoint_common import compat
from functools import wraps
from pyadjoint import Block
from pyadjoint.overloaded_type import OverloadedType, create_overloaded_object, register_overloaded_type
from pyadjoint.overloaded_function import overload_function

compat = compat.compat(backend)

__all__ = []


class VectorOperatorBlock(Block):
    """ Let z = op(x, y) and v_z = adj_inputs[0] and (h_x, h_y) = tlm_inputs.

    Or let's say one of them is a scalar.
        Let y = a*ones and dy/da = ones (an array of all ones).

    At least one of x and y is a vector. One may be a scalar.

    h_x, v_x, and w_x have the same shape as x.

    evaluate_adj_component():
        v_x = dz/dx * v_z for idx == 0
        v_y = dz/dy * v_z for idx == 1
        v_a = dy/da * v_y == sum(v_y) for scalar input a

    evaluate_tlm_component():
        h_z = dz/dx * h_x + dz/dy * h_y
            = dz/dx * h_x + dz/dy * dy/da * h_a for scalar input a

    evaluate_hessian_component():
        w_x = v_z d^2z/dx^2 * h_x + v_z d^2z/dxdy * h_y + dz/dx * w_z for idx == 0
        w_y = v_z d^2z/dy^2 * h_y + v_z d^2z/dxdy * h_x + dz/dy * w_z for idx == 1
        w_a = v_y d^2y/da^2 * h_a + dy/da * w_y for scalar input a
            = sum(w_y)
    """
    def __init__(self, *args):
        super().__init__()
        self._scalar_idx = -1
        for idx, dep in enumerate(args):
            self.add_dependency(dep)
            if not isinstance(dep, backend.GenericVector):
                self._scalar_idx = idx

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return self._operator(*inputs)


class AddBlock(VectorOperatorBlock):
    def evaluate_adj_component(
        self, inputs, adj_inputs, block_variable, idx, prepared=None
    ):
        if idx == self._scalar_idx:
            return adj_inputs[0].sum()
        return adj_inputs[0]

    def evaluate_tlm_component(
        self, inputs, tlm_inputs, block_variable, idx, prepared=None
    ):
        return sum(x if x is not None else 0 for x in tlm_inputs)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        if idx == self._scalar_idx:
            return hessian_inputs[0].sum()
        return hessian_inputs[0]


class SubBlock(VectorOperatorBlock):
    def evaluate_adj_component(
        self, inputs, adj_inputs, block_variable, idx, prepared=None
    ):
        adj = adj_inputs[0]
        if idx == self._scalar_idx:
            adj = adj.sum()
        if idx == 0:
            return adj
        return -adj

    def evaluate_tlm_component(
        self, inputs, tlm_inputs, block_variable, idx, prepared=None
    ):
        x_dot, y_dot = tlm_inputs
        output = 0
        if x_dot is not None:
            output = output + x_dot
        if y_dot is not None:
            output = output - y_dot

        return output

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        hessian = hessian_inputs[0]
        if idx == self._scalar_idx:
            hessian = hessian.sum()
        if idx == 0:
            return hessian
        return -hessian


class MulBlock(VectorOperatorBlock):

    def evaluate_adj_component(
        self, inputs, adj_inputs, block_variable, idx, prepared=None
    ):
        other_idx = 0 if idx == 1 else 1
        adj = self._operator(adj_inputs[0], inputs[other_idx])
        if idx == self._scalar_idx:
            adj = adj.sum()
        return adj

    def evaluate_tlm_component(
        self, inputs, tlm_inputs, block_variable, idx, prepared=None
    ):
        x_dot, y_dot = tlm_inputs
        output = 0
        if x_dot is not None:
            output = output + self._operator(inputs[1], x_dot)
        if y_dot is not None:
            output = output + self._operator(inputs[0], y_dot)
        return output

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        other_idx = 0 if idx == 1 else 1
        hessian = self._operator(hessian_inputs[0], inputs[other_idx])
        if idx == self._scalar_idx:
            hessian = hessian.sum()
        return hessian


class DivBlock(VectorOperatorBlock):
    def evaluate_adj_component(
        self, inputs, adj_inputs, block_variable, idx, prepared=None
    ):
        adj = adj_inputs[0]
        if idx == 0:
            adj = adj / inputs[1]
        else:
            adj = -adj * inputs[0] / (inputs[1] ** 2)

        if idx == self._scalar_idx:
            adj = adj.sum()
        return adj

    def evaluate_tlm_component(
        self, inputs, tlm_inputs, block_variable, idx, prepared=None
    ):
        x, y = inputs
        x_dot, y_dot = tlm_inputs

        tlm = []
        if x_dot is not None:
            tlm.append(x_dot / y)
        if y_dot is not None:
            tlm.append(-y_dot * x / (y ** 2))
        if len(tlm) == 0:
            return None
        return sum(tlm)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        # z = x / y
        # dz/dx = 1 / y
        # dz/dy = -x / y**2
        # d2z/dxdy = -1 / y**2
        # d2z/dx2 = 0
        # d2z/dy2 = 2 x / y**3

        x, y = inputs
        y2 = y ** 2

        hessian = hessian_inputs[0]
        adj = adj_inputs[0]
        if idx == 0:
            hessian = hessian / y
        else:
            hessian = -hessian * x / y2
            tlm = block_variable.tlm_value
            if tlm is not None:
                hessian += adj * 2 * (x / (y * y2)) * tlm

        other_idx = 0 if idx == 1 else 1
        other_tlm = self.get_dependencies()[other_idx].tlm_value
        if other_tlm is not None:
            hessian += - other_tlm * (adj / y2)

        if idx == self._scalar_idx:
            hessian = hessian.sum()
        return hessian


class NegBlock(VectorOperatorBlock):
    def evaluate_adj_component(
        self, inputs, adj_inputs, block_variable, idx, prepared=None
    ):
        return -adj_inputs[0]

    def evaluate_tlm_component(
        self, inputs, tlm_inputs, block_variable, idx, prepared=None
    ):
        return -tlm_inputs[0]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return -hessian_inputs[0]


class PowBlock(VectorOperatorBlock):
    """
        z = x ** y = exp(ln(x)*y)
        dz/dx = y * x ** (y-1)
        dz/dy = x ** y * ln(x)
        d2z/dxdy = d/dx(x ** y) * ln(x) + x ** y * d/dx(ln(x))
                 = y * x ** (y-1) * ln(x) + x ** (y-1)
                 = y * x ** (y-1) * ln(x) + x ** (y-1)
        d2z/dx2 = y * (y-1) * x ** (y-2)
        d2z/dy2 = x ** y * (ln(x) ** 2)
    """
    def evaluate_adj_component(
        self, inputs, adj_inputs, block_variable, idx, prepared=None
    ):
        x, y = inputs
        adj = adj_inputs[0]

        if idx == 0:
            adj = adj * y * (x ** (y - 1))
        else:
            output = self.get_outputs()[0].saved_output
            adj = adj * output
            adj.set_local(adj.array() * numpy.log(x.array()))

        if idx == self._scalar_idx:
            adj = adj.sum()
        return adj

    def evaluate_tlm_component(
        self, inputs, tlm_inputs, block_variable, idx, prepared=None
    ):
        x, y = inputs
        x_dot, y_dot = tlm_inputs

        tlm = []
        if x_dot is not None:
            tlm.append(x_dot * y * (x ** (y - 1)))
        if y_dot is not None:
            output = self.get_outputs()[0].saved_output
            tmp = y_dot * output
            tmp.set_local(tmp.array() * numpy.log(x.array()))
            tlm.append(tmp)
        if len(tlm) == 0:
            return None
        return sum(tlm)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):

        """
            z = x ** y = exp(ln(x)*y)
            dz/dx = y * x ** (y-1)
            dz/dy = x ** y * ln(x)
            d2z/dxdy = d/dx(x ** y) * ln(x) + x ** y * d/dx(ln(x))
                     = y * x ** (y-1) * ln(x) + x ** (y-1)
                     = x ** (y-1) * (y * ln(x) + 1)
            d2z/dx2 = y * (y-1) * x ** (y-2)
            d2z/dy2 = x ** y * (ln(x) ** 2)
        """
        x, y = inputs

        hessian = hessian_inputs[0]
        adj = adj_inputs[0]
        tlm = block_variable.tlm_value

        other_idx = 0 if idx == 1 else 1
        other_tlm = self.get_dependencies()[other_idx].tlm_value

        if idx == 0:
            hessian = hessian * y * (x ** (y - 1))
            if tlm is not None:
                hessian += adj * y * (y - 1) * x ** (y - 2) * tlm
        else:
            output = self.get_outputs()[0].saved_output
            hessian = hessian * output
            logx = numpy.log(x.array())
            hessian.set_local(hessian.array() * logx)
            if tlm is not None:
                hessian += adj * output * (logx ** 2) * tlm

        # Cross-derivatives.
        if other_tlm is not None:
            logx = x.copy()
            logx = logx.set_local(numpy.log(x.array()))

            d2zdxdy = x ** (y - 1) * (y * logx + 1)
            hessian += adj * d2zdxdy * other_tlm

        if idx == self._scalar_idx:
            hessian = hessian.sum()
        return hessian


class InnerBlock(VectorOperatorBlock):
    def evaluate_adj_component(
        self, inputs, adj_inputs, block_variable, idx, prepared=None
    ):
        other_idx = 0 if idx == 1 else 1
        return adj_inputs[0] * inputs[other_idx]

    def evaluate_tlm_component(
        self, inputs, tlm_inputs, block_variable, idx, prepared=None
    ):
        tlm = 0
        if tlm_inputs[0] is not None:
            tlm += tlm_inputs[0].inner(inputs[1])
        if tlm_inputs[1] is not None:
            tlm += tlm_inputs[1].inner(inputs[0])
        return tlm

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):

        adj = adj_inputs[0]
        other_idx = 0 if idx == 1 else 1
        other_tlm = self.get_dependencies()[other_idx].tlm_value
        hessian = hessian_inputs[0] * inputs[other_idx]
        if other_tlm is not None:
            hessian += adj * other_tlm
        return hessian


class GenericVectorMixin(OverloadedType):

    def _ad_convert_type(self, value, options={}):
        return value

    def _ad_create_checkpoint(self):
        return self.copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def _ad_mul(self, other):
        return self * other

    def _ad_add(self, other):
        return self + other

    def _ad_dot(self, other):
        return self.inner(other)

    @staticmethod
    def _ad_to_list(self):
        return compat.gather(self).tolist()

    def _ad_copy(self):
        return self.copy()

    def _ad_dim(self):
        return self.size()
        # return self.local_size()


def _reverse_block(block):
    class ReversedBlock(block):
        def __init__(self, *args):
            super().__init__(*reversed(args))

    return ReversedBlock


def make_basic_operator(k):
    backend_func = getattr(backend.GenericVector, k)

    @wraps(backend_func)
    def operator(*args, backend_func=backend_func):
        args = [backend.as_backend_type(x) for x in args]
        return backend_func(*args)
    return operator


_basic_op_map = {
    "inner": (None, InnerBlock),
    "__add__": (None, AddBlock),
    "__sub__": (None, SubBlock),
    "__mul__": (None, MulBlock),
    "__truediv__": (None, DivBlock),
    "__neg__": (None, NegBlock),
    "__rmul__": (None, _reverse_block(MulBlock)),
}

for k, (operator, block) in _basic_op_map.items():
    if operator is None:
        operator = make_basic_operator(k)
    block._operator = staticmethod(operator)
    overloaded_op = overload_function(operator, block)
    setattr(GenericVectorMixin, k, overloaded_op)


class PETScVector(GenericVectorMixin, backend.PETScVector):
    def __init__(self, vec):
        backend_vec = backend.as_backend_type(vec)
        backend.PETScVector.__init__(self, backend_vec.vec())
        GenericVectorMixin.__init__(self)


class GenericVector(GenericVectorMixin, backend.GenericVector):
    def __new__(cls, vec):
        # untested
        backend_vec = backend.as_backend_type(vec)
        return create_overloaded_object(backend_vec)


register_overloaded_type(GenericVector, backend.GenericVector)
register_overloaded_type(PETScVector, backend.PETScVector)
