import numpy as np
from halotools.empirical_models.ia_models.ia_model_components import alignment_strength
from vonmises_distribution import VonMisesHalf
import matplotlib.pyplot as plt

def test_vonmises_perfect_rvs():
    N = 100000
    mu = -1

    vm = VonMisesHalf()
    angles = vm.rvs(alignment_strength(mu), size=N)

    assert( (angles == np.pi/2).all() )

    mu = 1

    vm = VonMisesHalf()
    angles = vm.rvs(alignment_strength(mu), size=N)

    assert( ( (angles == 0) | (angles == np.pi) ).all() )

if __name__ == "__main__":
    print("TESTING")

    test_vonmises_perfect_rvs()

    print("DONE TESTING")