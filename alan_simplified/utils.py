import inspect
import math

import torch as t
import torch.distributions as td
import functorch
from functorch.dim import Dim
from torch.utils.checkpoint import checkpoint
import collections

Tensor = (functorch.dim.Tensor, t.Tensor)
OptionalTensor = (type(None), functorch.dim.Tensor, t.Tensor)
Number = (int, float)
TensorNumber = (*Tensor, *Number)

#New and definitely used.
def function_arguments(f):
    """
    Extracts the arguments of f as a list of strings.

    Does lots of error checking to ensure the function signature is very simple
    (e.g. no *args, no **kwargs, no default args, no kw-only args, no type annotations)
    """
    argspec = inspect.getfullargspec(f)

    #no *args
    if argspec.varargs is not None:
        raise Exception("In Alan, functions may not have *args")
    #no **kwargs
    if argspec.varkw is not None:
        raise Exception("In Alan, functions may not have **kwargs")
    #no defaults (positional or keyword only)
    if (argspec.defaults is not None) or (argspec.kwonlydefaults is not None):
        raise Exception("In Alan, functions may not have defaults")
    #no keyword only arguments
    if argspec.kwonlyargs:
        #kwonlyargs is a list, and lists evaluate to False if empty
        raise Exception("In Alan, functions may not have keyword only arguments")
    #no type annotations
    if argspec.annotations:
        #Annotations is a dict, and dicts evaluate to False if empty
        raise Exception("In Alan, functions may not have type annotations")

    return argspec.args


def list_duplicates(xs:list):
    dups = set()
    xs_so_far = set()
    for x in xs:
        if x in xs_so_far:
            dups.add(x)
        else:
            xs_so_far.add(x)
    return list(dups)

reserved_names = [
    "plate", 
    "prog", 
    "sample", 
    "groupvarname2Kdim", 
    "inputs", 
    "params", 
    "inputs_params_named",
]
reserved_prefixes = [
    "K_",
]


def check_name(name:str):
    if name in reserved_names:
        raise Exception(f"{name} is reserved in Alan")
    for prefix in reserved_prefixes:
        if (len(prefix) <= len(name)) and (prefix == name[:len(prefix)]):
            raise Exception(f"You can't use the prefix {prefix} in {name} in Alan")




def named2dim_dict(tensors: dict[str, t.Tensor], all_platedims: dict[str, Dim], setting=""):
    result = {}
    for varname, tensor in tensors.items():
        if not isinstance(tensor, t.Tensor):
            raise Exception(f"{varname} in {setting} must be a (named) torch Tensor")

        for dimname in tensor.names:
            if (dimname is not None): 
                if (dimname not in all_platedims):
                    raise Exception(f"{dimname} appears as a named dimension in {varname} in {setting}, but we don't have a Dim for that plate.")
                else:
                    dim = all_platedims[dimname]
                    if dim.size != tensor.size(dimname):
                        raise Exception(f"Dimension size mismatch along {dimname} in tensor {varname} in {setting}.  Specifically, the size provided in all_platesizes is {dim.size}, while the size of the tensor along this dimension is {tensor.size(dimname)}.")

        torchdims = [(slice(None) if (dimname is None) else all_platedims[dimname]) for dimname in tensor.names]
        result[varname] = generic_getitem(tensor.rename(None), torchdims)

    return result


#### Utilities for working with torchdims
def sum_non_dim(x):
    """
    Sums over all non-torchdim dimensions.
    Returns x for anything that isn't a tensor.
    """
    return x.sum() if (isinstance(x, Tensor) and x.ndim > 0) else x

"""
Defines a series of reduction functions that are called e.g. as
sum_dims(x, (i, j)), where i, j are torchdims.
"""
def assert_iter(dims, varname='dims'):
    if not isinstance(dims, (list, tuple)):
        raise Exception(varname + ' must be a list or tuple')

def assert_unique_iter(dims, varname='dims'):
    assert_iter(dims, varname=varname)
    if len(set(dims)) != len(dims):
        raise Exception(f'Non-unique elements in {varname}')

def assert_unique_dim_iter(dims, varname='dims'):
    assert_unique_iter(dims, varname=varname)
    for dim in dims:
        if not isinstance(dim, Dim):
            raise Exception(f'dim in {varname} is not torchdim dimension')

def assert_no_ellipsis(dims):
    if 0<len(dims):
        assert dims[-1] is not Ellipsis

def reduce_dims(func):
    """
    Reduces over specified torchdim dimensions.
    Returns itself if no dims given.
    """
    def inner(x, dims, ignore_extra_dims=False):
        assert_unique_dim_iter(dims)

        set_x_dims = set(generic_dims(x)) 
        if ignore_extra_dims:
            dims = tuple(dim for dim in dims if dim in set_x_dims)

        if not all(dim in set_x_dims for dim in dims):
            raise Exception("dims provided that aren't in x; can ignore them by providing ignore_extra_dims=True kwarg")

        if 0<len(dims):
            x = func(x.order(dims), 0)
        return x
    return inner

sum_dims        = reduce_dims(t.sum)
prod_dims       = reduce_dims(t.prod)
mean_dims       = reduce_dims(t.mean)
min_dims        = reduce_dims(lambda x, dim: t.min(x, dim).values)
max_dims        = reduce_dims(lambda x, dim: t.max(x, dim).values)
logsumexp_dims  = reduce_dims(t.logsumexp)

def logmeanexp_dims(x, dims):
    return logsumexp_dims(x, dims) - sum([math.log(dim.size) for dim in dims])


def is_dimtensor(tensor):
    return isinstance(tensor, functorch.dim.Tensor)

def unify_dims(tensors):
    """
    Returns unique ordered list of dims for tensors in args
    """
    return ordered_unique([dim for tensor in tensors for dim in generic_dims(tensor)])

def generic_ndim(x):
    """
    Generalises x.ndim, which is only defined for tensors
    """
    assert isinstance(x, TensorNumber)
    return x.ndim if isinstance(x, Tensor) else 0

def generic_dims(x):
    """
    Generalises x.dims, which is only defined for torchdim tensors
    """
    return x.dims if is_dimtensor(x) else ()

def generic_order(x, dims):
    """
    Generalises x.order(dims), which is only defined for torchdim tensors
    """
    assert_unique_dim_iter(dims)
    assert_no_ellipsis(dims)

    #If x is not a dimtensor, then we can't have any dims.
    if not is_dimtensor(x):
        assert 0 == len(dims)

    return x.order(*dims) if 0<len(dims) else x

def generic_getitem(x, dims):
    assert_iter(dims) #dims doesn't have to be unique, e.g. [2,2]
    assert_no_ellipsis(dims)

    if len(dims)==0:
        return x
    else:
        return x[dims]

def generic_setitem(x, dims, value):
    assert_iter(dims) #dims doesn't have to be unique, e.g. [2,2]
    assert_no_ellipsis(dims)

    if len(dims)==0:
        dims = (Ellipsis,)

    x[dims] = value

def ordered_unique(ls):
    """
    Exploits the fact that in Python 3.7<, dict keys retain ordering

    Arguments:
        ls: list with duplicate elements
    Returns:
        list of unique elements, in the order they first appeared in ls
    """
    assert_iter(ls, 'ls')
    d = {l:None for l in ls}
    return list(d.keys())

def partition_tensors(lps, dim):
    """
    Partitions a list of tensors into two sets, one list with all tensors
    that have dim, and another list with all tensors that don't have that
    dim
    """
    has_dim = [lp for lp in lps if dim     in set(generic_dims(lp))]
    no_dim  = [lp for lp in lps if dim not in set(generic_dims(lp))]

    return has_dim, no_dim


def singleton_order(x, dims):
    """
    Takes a torchdim tensor and returns a standard tensor,
    in a manner that mirrors `x.order(dims)` (if `dims` had all
    dimensions in `x`). However, with `x.order(dims)`,
    all dimensions in `dims` must be in `x`.  Here, we allow new
    dimensions in `dims` (i.e. dimensions in `dims` that aren't
    in `x`), and add singleton dimensions to the result for those
    dimensions.
    """
    #This will be applied in dist.py to distribution arguments, 
    #which may be non-tensors.  These non-tensors should broadcast 
    #properly whatever happens, so we can return immediately.
    if not isinstance(x, Tensor):
        assert isinstance(x, Number)
        return x

    assert_no_ellipsis(dims)

    x_dims = set(generic_dims(x))

    dims_in_x = [dim for dim in dims if dim in x_dims]
    x = generic_order(x, dims_in_x)

    idxs = [(slice(None) if (dim in x_dims) else None) for dim in dims]
    x = generic_getitem(x, idxs)

    return x

def dim2named_tensor(x):
    """
    Converts a torchdim to a named tensor.
    """
    dims = generic_dims(x)
    names = [repr(dim) for dim in dims]
    return generic_order(x, dims).rename(*names, ...)

#### Utilities for working with dictionaries of plates

def named2dim_tensor(d, x):
    """
    Args:
        d (dict): dictionary mapping plate name to torchdim Dim.
        x (t.Tensor): named tensor.
    Returns:
        A torchdim tensor.

    Assumes that all named dimensions appear in the dict.
    """
    #can't already be a dimtensor
    assert not is_dimtensor(x)

    #if a number then just return
    if isinstance(x, Number):
        return x

    assert isinstance(x, t.Tensor)

    for name in x.names:
        if (name is not None) and (name not in d):
            raise Exception(f"No torchdim dimension for named dimension {name} in named2dim_tensor")

    torchdims = [(slice(None) if (name is None) else d[name]) for name in x.names]

    return generic_getitem(x.rename(None), torchdims)

def named2dim_tensordict(d, tensordict):
    """Maps named2dim_tensor over a dict of tensors
    Args:
        d (dict): dictionary mapping plate name to torchdim Dim.
        tensordict (dict): dictionary mapping variable name to named tensor.
    Returns:
        dictionary mapping variable name to torchdim tensor.
    """
    return {k: named2dim_tensor(d, tensor) for (k, tensor) in tensordict.items()}


def extend_plates_with_sizes(plates, size_dict):
    """Extends a plate dict using a size dict.
    Args:
        d (plate dict): dictionary mapping plate name to torchdim Dim.
        size_dict: dictionary mapping plate name to integer size.
    Returns:
        a plate dict extended with the sizes in size_dict.
    """
    new_dict = {}
    for (name, size) in size_dict.items():
        if (name not in plates):
            new_dict[name] = Dim(name, size)
        elif size != plates[name].size:
            raise Exception(
                f"Mismatch in sizes for plate '{name}', "
                f"tensor has size {size} but we already have the plate-size as {plates[name].size}"
            )
    return {**plates, **new_dict}

def extend_plates_with_named_tensor(plates, tensor):
    """Extends a plate dict using any named dimensions in `tensor`.
    Args:
        d (plate dict): dictionary mapping plate name to torchdim Dim.
        tensor: named tensor.
    Returns:
        a plate dict extended with the sizes of the named dimensions in `tensor`.
    """
    size_dict = {name: tensor.size(name) for name in tensor.names if name is not None}
    return extend_plates_with_sizes(plates, size_dict)

def extend_plates_with_named_tensors(plates, tensors):
    """Extends a plate dict using any named dimensions in `tensors`.
    Args:
        d (plate dict): dictionary mapping plate name to torchdim Dim.
        tensors: an iterable of named tensor.
    Returns:
        a plate dict extended with the sizes of the named dimensions in `tensors`.
    """
    for tensor in tensors:
        plates = extend_plates_with_named_tensor(plates, tensor)
    return plates

def platenames2platedims(plates, platenames):
    if isinstance(platenames, str):
        platenames = (platenames,)
    return [plates[pn] for pn in platenames]


def corresponding_plates(platedims1 : dict[str, Tensor], platedims2 : dict[str, Tensor],
                         sample1 : Tensor, sample2 : Tensor):
    '''For any platedims which share a name (but not necessarily size) and appear in both samples,
    returns two lists of those platedims in the same order.'''
    assert isinstance(platedims1, dict)
    assert isinstance(platedims2, dict)
    assert isinstance(sample1, Tensor)
    assert isinstance(sample2, Tensor)

    dimnames1 = [name for (name, dim) in platedims1.items() if dim in set(sample1.dims)]
    dimnames2 = [name for (name, dim) in platedims2.items() if dim in set(sample2.dims)]
    assert set(dimnames1) == set(dimnames2)

    dimnames = dimnames1

    dims1 = [platedims1[name] for name in dimnames]
    dims2 = [platedims2[name] for name in dimnames]

    return dims1, dims2




def merge_dict_with_subdicts(dict1: dict, dict2: dict) -> dict:
    """
    similar behaviour to builtin dict.update - but knows how to handle nested dicts
    """
    q = collections.deque([(dict1, dict2)])
    while len(q) > 0:
        d1, d2 = q.pop()
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                q.append((d1[k], v))
            else:
                d1[k] = v

    return dict1