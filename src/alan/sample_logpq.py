import math
from typing import Optional, Union

from .Plate import Plate, tree_values, update_scope
from .BoundPlate import BoundPlate
from .Group import Group
from .utils import *
from .reduce_Ks import reduce_Ks, sample_Ks
from .Split import Split
from .SamplingType import SamplingType
from .dist import Dist
from .logpq import logPQ_dist, logPQ_group, logPQ_plate, lp_getter
from .Data import Data

PBP = Union[Plate, BoundPlate]

def logPQ_sample(
    name:Optional[str],
    P: Plate, 
    Q: Plate, 
    sample: dict, 
    inputs_params: dict,
    data: dict,
    extra_log_factors: dict, 
    scope: dict[str, Tensor], 
    active_platedims:list[Dim],
    all_platedims:dict[str: Dim],
    groupvarname2Kdim:dict[str, Tensor],
    sampling_type:SamplingType,
    split:Optional[Split],
    indices:dict[str, Tensor],
    N_dim:Dim,
    num_samples:int):

    assert isinstance(P, Plate)
    assert isinstance(Q, Plate)
    assert isinstance(sample, dict)
    assert isinstance(inputs_params, dict)
    assert isinstance(data, dict)
    assert isinstance(extra_log_factors, dict)
    assert isinstance(indices, dict)

    #Push an extra plate, if not the top-layer plate (top-layer plate is signalled
    #by name=None.
    if name is not None:
        active_platedims = [*active_platedims, all_platedims[name]]

    scope = update_scope(scope, Q, sample, inputs_params)
    
    lps, all_Ks = lp_getter(
        name=name,
        P=P, 
        Q=Q, 
        sample=sample, 
        inputs_params=inputs_params,
        data=data,
        extra_log_factors=extra_log_factors, 
        scope=scope, 
        active_platedims=active_platedims,
        all_platedims=all_platedims,
        groupvarname2Kdim=groupvarname2Kdim,
        sampling_type=sampling_type,
        split=split)

    # Index into each lp with the indices we've collected so far
    for i in range(len(lps)):
        for dim in list(set(generic_dims(lps[i])).intersection(set(indices.keys()))):
            lps[i] = lps[i].order(dim)[indices[dim]]


    if len(all_Ks) > 0:
        indices = {**indices, **sample_Ks(lps, all_Ks,N_dim, num_samples)}
        
    for childname, childP in P.prog.items():
        childQ = Q.prog.get(childname)
        
        if isinstance(childP, Plate):
            assert isinstance(childQ, Plate)
            indices = logPQ_sample(name=childname,
            P=childP, 
            Q=childQ, 
            sample=sample.get(childname),
            data=data.get(childname),
            inputs_params=inputs_params.get(childname),
            extra_log_factors=extra_log_factors.get(childname),
            scope=scope,
            active_platedims=active_platedims,
            all_platedims=all_platedims,
            groupvarname2Kdim=groupvarname2Kdim,
            sampling_type=sampling_type,
            split=split,
            indices=indices,
            num_samples=num_samples,
            N_dim = N_dim)

    return indices

