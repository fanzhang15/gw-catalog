from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass
class LensImage:
    image_index: int
    magnification: float
    time_delay: float
    morse_phase: float


def _morse_phase_from_parity(parity: int) -> float:
    if parity > 0:
        return 0.0
    return math.pi / 2


def sample_pm_doublet() -> list[LensImage]:
    mag1 = random.uniform(1.1, 3.0)
    mag2 = random.uniform(0.7, 2.0)
    dt = random.uniform(0.01, 7.5)
    return [
        LensImage(0, mag1, 0.0, 0.0),
        LensImage(1, mag2, dt, _morse_phase_from_parity(-1)),
    ]


def sample_sis_doublet() -> list[LensImage]:
    mag1 = random.uniform(1.2, 4.0)
    mag2 = random.uniform(0.8, 2.5)
    dt = random.uniform(0.1, 15.0)
    return [
        LensImage(0, mag1, 0.0, 0.0),
        LensImage(1, mag2, dt, math.pi / 2),
    ]


def sample_sie_multiplet(multiplicity: int) -> list[LensImage]:
    assert multiplicity in (3, 4)
    images: list[LensImage] = [LensImage(0, random.uniform(1.2, 3.0), 0.0, 0.0)]
    for idx in range(1, multiplicity):
        images.append(
            LensImage(
                idx,
                random.uniform(0.6, 2.2),
                random.uniform(0.2, 20.0) * idx,
                [0.0, math.pi / 2, math.pi][idx % 3],
            )
        )
    return images
