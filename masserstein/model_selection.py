import numpy as np


def _weighted_var(x, weights):
    return np.average((x - np.average(x, weights=weights)) ** 2, weights=weights)


def goodness_of_fit(transport_plan, metric="L1"):
    # todo: docs
    if not isinstance(transport_plan, np.ndarray):
        transport_plan = np.array(list(transport_plan))

    assert (
        transport_plan[:, 2] >= 0
    ).all(), "Negative values passed in transport plan!"
    transport_plan = transport_plan[transport_plan[:, 2] > 0]
    transport_distance = transport_plan[:, 1] - transport_plan[:, 0]
    transport_mass = transport_plan[:, 2]

    if metric == "L1":
        return np.average(np.abs(transport_distance), weights=transport_mass)
    elif metric == "L2":
        return np.average(transport_distance ** 2, weights=transport_mass) ** 0.5
    elif metric == "std":
        return _weighted_var(transport_distance, transport_mass) ** 0.5
    else:
        raise KeyError("Unknown metric, has to be one of L1, L2, std.")


# tmp tests
from masserstein.spectrum import Spectrum

s1 = Spectrum("H")
s2 = Spectrum("C")
transport_plan = s1.WSDistanceMoves(s2)
goodness_of_fit(transport_plan)
