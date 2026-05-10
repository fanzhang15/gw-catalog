from __future__ import annotations

from dataclasses import dataclass
import math
import random

ALLOWED_MORSE_PHASES = (0.0, math.pi / 2, math.pi)
PM_MAX_MASS_MSUN = 1e7


@dataclass
class LensImage:
    image_index: int
    magnification: float
    time_delay: float
    morse_phase: float


@dataclass
class LensSystemSample:
    lens_family: str
    lens_mass_msun: float | None
    images: list[LensImage]


def sample_pm_doublet() -> LensSystemSample:
    lens_mass_msun = random.uniform(10.0, PM_MAX_MASS_MSUN)
    mag1 = random.uniform(1.1, 3.0)
    mag2 = random.uniform(0.7, 2.0)
    dt = random.uniform(0.01, 7.5)
    return LensSystemSample(
        lens_family="PM",
        lens_mass_msun=lens_mass_msun,
        images=[
            LensImage(0, mag1, 0.0, 0.0),
            LensImage(1, mag2, dt, math.pi / 2),
        ],
    )


def sample_sis_doublet() -> LensSystemSample:
    mag1 = random.uniform(1.2, 4.0)
    mag2 = random.uniform(0.8, 2.5)
    dt = random.uniform(0.1, 15.0)
    return LensSystemSample(
        lens_family="SIS",
        lens_mass_msun=None,
        images=[
            LensImage(0, mag1, 0.0, 0.0),
            LensImage(1, mag2, dt, math.pi / 2),
        ],
    )


def sample_sie_multiplet(multiplicity: int) -> LensSystemSample:
    assert multiplicity in (3, 4)
    images: list[LensImage] = [LensImage(0, random.uniform(1.2, 3.0), 0.0, 0.0)]
    for idx in range(1, multiplicity):
        images.append(
            LensImage(
                idx,
                random.uniform(0.6, 2.2),
                random.uniform(0.2, 20.0) * idx,
                ALLOWED_MORSE_PHASES[idx % len(ALLOWED_MORSE_PHASES)],
            )
        )
    return LensSystemSample(lens_family="SIE", lens_mass_msun=None, images=images)
