import random
import numpy as np
import re


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


def generate_random_spectrum(target_mass, elements):
    elements = list(set(elements) - set("H"))
    random.shuffle(elements)
    composition = dict()
    while elements:
        e = elements.pop()
        m = Spectrum(e).confs[0][0]
        if m > target_mass:
            continue
        u = random.randint(0, target_mass // m)
        composition[e] = u
        target_mass -= m * u
    H_mass = Spectrum("H").confs[0][0]
    composition["H"] = np.round(target_mass / H_mass)
    spectrum = Spectrum(
        "".join(f"{k}{int(v)}" for k, v in sorted(composition.items()) if v != 0)
    )
    return spectrum


def get_composition(formula):
    composition = dict()
    for element_and_potential_number in re.findall("[A-Z][a-z]*\d*", formula):
        element = re.findall("[A-Z][a-z]*", element_and_potential_number)[0]
        potential_number = re.findall("\d+$", element_and_potential_number)
        number = int(potential_number[0]) if potential_number else 1
        composition[element] = number
    return composition


def calculate_metrics_for_randomizations(spectrum, elements=None, n_replications=100, metric="L1"):
    if not elements:
        elements = list(get_composition(spectrum.formula).keys())
    mass = spectrum.confs[0][0]
    metrics = []
    for _ in range(n_replications):
        random_spectrum = generate_random_spectrum(mass, elements)
        transport_plan = spectrum.WSDistanceMoves(random_spectrum)
        metric_value = goodness_of_fit(transport_plan, metric=metric)
        metrics.append(metric_value)

    return metrics


# tmp tests
from masserstein.spectrum import Spectrum

s1 = Spectrum("H")
s2 = Spectrum("C")
transport_plan = s1.WSDistanceMoves(s2)
goodness_of_fit(transport_plan)

for i in range(3):
    s = generate_random_spectrum(100, list("HCO"))
    print(s.formula, s.confs[0][0], get_composition(s.formula))

metrics = calculate_metrics_for_randomizations(
    Spectrum("C8H10N4O2", label="Caffeine"), n_replications=10000, metric="std"
)
import matplotlib.pyplot as plt

plt.hist(metrics, bins=20)
plt.savefig("fig.png")
