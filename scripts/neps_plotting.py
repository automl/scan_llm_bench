import neps
from neps.optimizers.algorithms import chinchilla_guided_scaling, random_search
from neps.space.neps_spaces.neps_space import convert_classic_to_neps_search_space

from space import SpacePipeline


def _optimizer(space):
    if not hasattr(space, "get_attrs"):
        space = convert_classic_to_neps_search_space(space)
    return chinchilla_guided_scaling(
        space,
        base_optimizer=random_search(space, use_priors=False),
    )


space = SpacePipeline()
neps.extrapolate(
    "../neps/neps_configs/Neps_Scale_Search_Space",
    max_target_flops=1_000_000_000_000_000_000,
    optimizer=_optimizer,
)
