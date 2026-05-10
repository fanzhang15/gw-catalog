from __future__ import annotations

from dataclasses import dataclass
import uuid
import numpy as np

from .lens_models import sample_pm_doublet, sample_sis_doublet, sample_sie_multiplet


@dataclass
class CatalogConfig:
    n_total: int
    lens_prevalence: float
    multiplicity_distribution: dict[str, float]
    seed: int = 42
    strain_length: int = 4096


def _sample_intrinsic(rng: np.random.Generator) -> dict:
    return {
        "m1": float(rng.uniform(10, 60)),
        "m2": float(rng.uniform(8, 45)),
        "chi1": float(rng.uniform(-0.99, 0.99)),
        "chi2": float(rng.uniform(-0.99, 0.99)),
        "z": float(rng.uniform(0.05, 2.5)),
        "ra": float(rng.uniform(0, 2 * np.pi)),
        "dec": float(rng.uniform(-np.pi / 2, np.pi / 2)),
        "iota": float(rng.uniform(0, np.pi)),
        "psi": float(rng.uniform(0, np.pi)),
        "phi_c": float(rng.uniform(0, 2 * np.pi)),
    }


def _base_strain(rng: np.random.Generator, length: int) -> np.ndarray:
    x = np.linspace(0, 1, length)
    f = rng.uniform(20, 120)
    phase = rng.uniform(0, 2 * np.pi)
    env = np.exp(-6 * x)
    wave = np.sin(2 * np.pi * f * x + phase) * env
    noise = rng.normal(0.0, 0.03, size=length)
    out = (wave + noise).astype(np.float32)
    out /= np.std(out) + 1e-8
    return out


def _lensed_strain(base: np.ndarray, magnification: float, shift: int) -> np.ndarray:
    return (np.roll(base, shift) * np.sqrt(magnification)).astype(np.float32)


def generate_catalog(cfg: CatalogConfig) -> list[dict]:
    rng = np.random.default_rng(cfg.seed)
    events: list[dict] = []
    n_lensed = int(cfg.n_total * cfg.lens_prevalence)
    n_isolated = cfg.n_total - n_lensed

    lensed_budget = 0
    while lensed_budget < n_lensed:
        source_id = f"SRC-{uuid.uuid4().hex[:12]}"
        intrinsic = _sample_intrinsic(rng)
        base = _base_strain(rng, cfg.strain_length)
        mult = rng.choice(list(cfg.multiplicity_distribution.keys()), p=list(cfg.multiplicity_distribution.values()))
        if mult == "doublet":
            images = sample_sis_doublet() if rng.random() > 0.5 else sample_pm_doublet()
        elif mult == "triplet":
            images = sample_sie_multiplet(3)
        else:
            images = sample_sie_multiplet(4)
        for img in images:
            if lensed_budget >= n_lensed:
                break
            shift = int(min(cfg.strain_length - 1, img.time_delay * 8))
            strain = _lensed_strain(base, img.magnification, shift)
            events.append({
                "event_id": f"EVT-{uuid.uuid4().hex[:12]}",
                "source_id": source_id,
                "system_type": mult,
                "image_index": img.image_index,
                "magnification": float(img.magnification),
                "time_delay": float(img.time_delay),
                "morse_phase": float(img.morse_phase),
                "intrinsic_params": intrinsic,
                "strain": strain,
            })
            lensed_budget += 1

    for _ in range(n_isolated):
        intrinsic = _sample_intrinsic(rng)
        strain = _base_strain(rng, cfg.strain_length)
        events.append({
            "event_id": f"EVT-{uuid.uuid4().hex[:12]}",
            "source_id": f"SRC-{uuid.uuid4().hex[:12]}",
            "system_type": "isolated",
            "image_index": 0,
            "magnification": 1.0,
            "time_delay": 0.0,
            "morse_phase": 0.0,
            "intrinsic_params": intrinsic,
            "strain": strain,
        })

    rng.shuffle(events)
    return events
