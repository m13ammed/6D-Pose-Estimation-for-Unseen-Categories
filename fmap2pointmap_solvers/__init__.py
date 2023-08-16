import gin
from .spacial_filtering import spacial_filtering_fmap2pointmap 
from .naive import naive_fmap2pointmap


@gin.configurable()
def choose_fmap2pointmap_solver(solver):
    return solver