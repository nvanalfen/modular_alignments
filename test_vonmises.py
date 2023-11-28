import numpy as np
from halotools.empirical_models.ia_models.ia_model_components import alignment_strength
from vonmises_distribution import VonMisesHalf
import matplotlib.pyplot as plt

def test_vonmises_perfect_rvs():
    N = 100000
    mu = 1

    vm = VonMisesHalf()
    angles = vm.rvs(alignment_strength(mu), size=N)

    # plt.hist(angles, bins=100)
    # plt.show()

if __name__ == "__main__":
    print("TESTING")

    test_vonmises_perfect_rvs()

    print("DONE TESTING")