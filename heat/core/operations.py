from copy import copy as _copy
import torch

from . import stride_tricks
from . import types
from . import tensor
from .halo import halorize

__all__ = [
    'abs',
    'absolute',
    'clip',
    'copy',
    'exp',
    'floor',
    'log',
    'sin',
    'sqrt'
]


def abs(x, out=None, dtype=None):
    """
    Calculate the absolute value element-wise.

    Parameters
    ----------
    x : ht.tensor
        The values for which the compute the absolute value.
    out : ht.tensor, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or None, a freshly-allocated array is returned.
    dtype : ht.type, optional
        Determines the data type of the output array. The values are cast to this type with potential loss of
        precision.

    Returns
    -------
    absolute_values : ht.tensor
        A tensor containing the absolute value of each element in x.
    """
    if dtype is not None and not issubclass(dtype, types.generic):
        raise TypeError('dtype must be a heat data type')

    absolute_values = __local_operation(torch.abs, x, out)
    if dtype is not None:
        absolute_values._tensor__array = absolute_values._tensor__array.type(dtype.torch_type())
        absolute_values._tensor__dtype = dtype

    return absolute_values


def absolute(x, out=None, dtype=None):
    """
    Calculate the absolute value element-wise.

    np.abs is a shorthand for this function.

    Parameters
    ----------
    x : ht.tensor
        The values for which the compute the absolute value.
    out : ht.tensor, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or None, a freshly-allocated array is returned.
    dtype : ht.type, optional
        Determines the data type of the output array. The values are cast to this type with potential loss of
        precision.

    Returns
    -------
    absolute_values : ht.tensor
        A tensor containing the absolute value of each element in x.
    """
    return abs(x, out, dtype)


def clip(a, a_min, a_max, out=None):
    """
    Parameters
    ----------
    a : ht.tensor
        Array containing elements to clip.
    a_min : scalar or None
        Minimum value. If None, clipping is not performed on lower interval edge. Not more than one of a_min and
        a_max may be None.
    a_max : scalar or None
        Maximum value. If None, clipping is not performed on upper interval edge. Not more than one of a_min and
        a_max may be None.
    out : ht.tensor, optional
        The results will be placed in this array. It may be the input array for in-place clipping. out must be of
        the right shape to hold the output. Its type is preserved.

    Returns
    -------
    clipped_values : ht.tensor
        A tensor with the elements of this tensor, but where values < a_min are replaced with a_min, and those >
        a_max with a_max.
    """
    if not isinstance(a, tensor.tensor):
        raise TypeError('a must be a tensor')
    if a_min is None and a_max is None:
        raise ValueError('either a_min or a_max must be set')

    if out is None:
        return tensor.tensor(a._tensor__array.clamp(a_min, a_max), a.shape, a.dtype, a.split, _copy(a._tensor__comm))
    if not isinstance(out, tensor.tensor):
        raise TypeError('out must be a tensor')

    return a._tensor__array.clamp(a_min, a_max, out=out._tensor__array) and out


def copy(a):
    """
    Return an array copy of the given object.

    Parameters
    ----------
    a : ht.tensor
        Input data to be copied.

    Returns
    -------
    copied : ht.tensor
        A copy of the original
    """
    if not isinstance(a, tensor.tensor):
        raise TypeError('input needs to be a tensor')
    return tensor.tensor(a._tensor__array.clone(), a.shape, a.dtype, a.split, _copy(a._tensor__comm))


def exp(x, out=None):
    """
    Calculate the exponential of all elements in the input array.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the exponential.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    exponentials : ht.tensor
        A tensor of the same shape as x, containing the positive exponentials of each element in this tensor. If out
        was provided, logarithms is a reference to it.

    Examples
    --------
    >>> ht.exp(ht.arange(5))
    tensor([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981])
    """
    return __local_operation(torch.exp, x, out)


def floor(x, out=None):
    """
    Return the floor of the input, element-wise.

    The floor of the scalar x is the largest integer i, such that i <= x. It is often denoted as \lfloor x \rfloor.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the floored values.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    floored : ht.tensor
        A tensor of the same shape as x, containing the floored valued of each element in this tensor. If out was
        provided, logarithms is a reference to it.

    Examples
    --------
    >>> ht.floor(ht.arange(-2.0, 2.0, 0.4))
    tensor([-2., -2., -2., -1., -1.,  0.,  0.,  0.,  1.,  1.])
    """
    return __local_operation(torch.floor, x, out)


def log(x, out=None):
    """
    Natural logarithm, element-wise.

    The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x. The natural
    logarithm is logarithm in base e.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the logarithm.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    logarithms : ht.tensor
        A tensor of the same shape as x, containing the positive logarithms of each element in this tensor.
        Negative input elements are returned as nan. If out was provided, logarithms is a reference to it.

    Examples
    --------
    >>> ht.log(ht.arange(5))
    tensor([  -inf, 0.0000, 0.6931, 1.0986, 1.3863])
    """
    return __local_operation(torch.log, x, out)


def sin(x, out=None):
    """
    Return the trigonometric sine, element-wise.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the trigonometric sine.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    sine : ht.tensor
        A tensor of the same shape as x, containing the trigonometric sine of each element in this tensor.
        Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.

    Examples
    --------
    >>> ht.sin(ht.arange(-6, 7, 2))
    tensor([ 0.2794,  0.7568, -0.9093,  0.0000,  0.9093, -0.7568, -0.2794])
    """
    return __local_operation(torch.sin, x, out)


def sqrt(x, out=None):
    """
    Return the non-negative square-root of a tensor element-wise.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the square-roots.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided or
        set to None, a fresh tensor is allocated.

    Returns
    -------
    square_roots : ht.tensor
        A tensor of the same shape as x, containing the positive square-root of each element in x. Negative input
        elements are returned as nan. If out was provided, square_roots is a reference to it.

    Examples
    --------
    >>> ht.sqrt(ht.arange(5))
    tensor([0.0000, 1.0000, 1.4142, 1.7321, 2.0000])
    >>> ht.sqrt(ht.arange(-5, 0))
    tensor([nan, nan, nan, nan, nan])
    """
    return __local_operation(torch.sqrt, x, out)

@halorize
def __local_operation(operation, x, out):
    """
    Generic wrapper for local operations, which do not require communication. Accepts the actual operation function as
    argument and takes only care of buffer allocation/writing.

    Parameters
    ----------
    operation : function
        A function implementing the element-wise local operation, e.g. torch.sqrt
    x : ht.tensor
        The value for which to compute 'operation'.
    out : ht.tensor or None
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided or
        set to None, a fresh tensor is allocated.

    Returns
    -------
    result : ht.tensor
        A tensor of the same shape as x, containing the result of 'operation' for each element in x. If out was
        provided, result is a reference to it.

    Raises
    -------
    TypeError
        If the input is not a tensor or the output is not a tensor or None.
    """
    # perform sanitation
    if not isinstance(x, tensor.tensor):
        raise TypeError('expected x to be a ht.tensor, but was {}'.format(type(x)))
    if out is not None and not isinstance(out, tensor.tensor):
        raise TypeError('expected out to be None or an ht.tensor, but was {}'.format(type(out)))

    # infer the output type of the tensor
    # we need floating point numbers here, due to PyTorch only providing sqrt() implementation for float32/64
    promoted_type = types.promote_types(x.dtype, types.float32)
    torch_type = promoted_type.torch_type()

    # no defined output tensor, return a freshly created one
    if out is None:
        return tensor.tensor(
            operation(x._tensor__array.type(torch_type)),
            x.gshape,
            promoted_type,
            x.split,
            _copy(x._tensor__comm)
        )

    # output buffer writing requires a bit more work
    # we need to determine whether the operands are broadcastable and the multiple of the broadcasting
    # reason: manually repetition for each dimension as PyTorch does not conform to numpy's broadcast semantic
    # PyTorch always recreates the input shape and ignores broadcasting/too large buffers
    broadcast_shape = stride_tricks.broadcast_shape(x.lshape, out.lshape)
    padded_shape = (1,) * (len(broadcast_shape) - len(x.lshape)) + x.lshape
    multiples = [int(a / b) for a, b in zip(broadcast_shape, padded_shape)]
    needs_repetition = any(multiple > 1 for multiple in multiples)

    # do an inplace operation into a provided buffer
    casted = x._tensor__array.type(torch_type)
    operation(casted.repeat(multiples) if needs_repetition else casted, out=out._tensor__array)
    return out
