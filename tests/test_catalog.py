import math
import numpy as np

from lensgraph.data.catalog_simulator import CatalogConfig, generate_catalog
from lensgraph.data.lens_models import ALLOWED_MORSE_PHASES, PM_MAX_MASS_MSUN


def _cfg(n=600):
    return CatalogConfig(
        n_total=n,
        lens_prevalence=0.3,
        multiplicity_distribution={"doublet": 0.8, "triplet": 0.15, "quadruplet": 0.05},
        seed=7,
        strain_length=4096,
    )


def test_schema_complete():
    events = generate_catalog(_cfg(100))
    required = {
        "event_id", "source_id", "system_type", "image_index", "magnification", "time_delay",
        "morse_phase", "intrinsic_params", "strain", "lens_family", "lens_mass_msun"
    }
    assert required.issubset(events[0].keys())
    assert events[0]["strain"].shape == (4096,)


def test_source_invariants():
    events = generate_catalog(_cfg(200))
    by_source = {}
    for e in events:
        by_source.setdefault(e["source_id"], []).append(e)
    for _, group in by_source.items():
        if len(group) <= 1:
            continue
        ref = group[0]["intrinsic_params"]
        for g in group[1:]:
            assert g["intrinsic_params"] == ref


def test_prevalence_convergence():
    events = generate_catalog(_cfg(2000))
    prevalence = sum(e["system_type"] != "isolated" for e in events) / len(events)
    assert math.isclose(prevalence, 0.3, rel_tol=0.1)


def test_morse_phase_and_pm_mass_limits():
    events = generate_catalog(_cfg(800))
    for e in events:
        assert e["morse_phase"] in ALLOWED_MORSE_PHASES
        if e["lens_family"] == "PM":
            assert e["lens_mass_msun"] is not None
            assert e["lens_mass_msun"] <= PM_MAX_MASS_MSUN


def test_no_nan_inf():
    events = generate_catalog(_cfg(120))
    arr = np.stack([e["strain"] for e in events])
    assert np.isfinite(arr).all()
