import torch as t
import torch.distributions as td
from functorch.dim import Dim

from alan_simplified import Normal, Bernoulli, Plate, BoundPlate, Group, Problem, IndependentSample, Data, Mean, Mean2, Var, Split

#t.manual_seed(0)


P = Plate(
    ab = Group(
        a = Normal(0, 1),
        b = Normal("a", 1),),
    c = Normal(0, lambda a: a.exp()),
    p1 = Plate(
        d = Normal("a", 1),
        p2 = Plate(
            e = Normal("d", 1.),
        ),
    ),
)

Q = Plate(
    ab = Group(
        a = Normal("a_mean", 1),
        b = Normal("a", 1),
    ),
    c = Normal(0, lambda a: a.exp()),
    p1 = Plate(
        d = Normal("d_mean", 1),
        p2 = Plate(
            e = Data()
        ),
    ),
)

P = BoundPlate(P)
Q = BoundPlate(Q, params={'a_mean': t.zeros(()), 'd_mean':t.zeros(3, names=('p1',))})

all_platesizes = {'p1': 3, 'p2': 4}
data = {'e': t.randn(3, 4, names=('p1', 'p2'))}

prob = Problem(P, Q, all_platesizes, data)
prob.to('mps')

sampling_type = IndependentSample
sample = prob.sample(3, True, sampling_type)

marginals = sample.marginals(("ab", "c"))
importance_sample = sample.importance_sample(100000)

a_mean = ("a", Mean)
print(sample.moments(*a_mean))
print(marginals.moments(*a_mean))
print(importance_sample.moments(*a_mean))

a_mean_mean2 = ("a", (Mean, Mean2))
print(sample.moments(*a_mean_mean2))
print(marginals.moments(*a_mean_mean2))
print(importance_sample.moments(*a_mean_mean2))

ad_mean = {"a": Mean, "b": (Mean, Mean2)}
print(sample.moments(ad_mean))
print(marginals.moments(ad_mean))
print(importance_sample.moments(ad_mean))


print(sample.elbo_vi())
print(sample.elbo_vi(split=Split('p1', 2)))