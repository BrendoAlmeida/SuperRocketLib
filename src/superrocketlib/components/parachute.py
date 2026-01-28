from __future__ import annotations

from typing import Optional
import random

from rocketpy import Parachute

from ..core.structures import ParachuteRanges


class SuperParachute(Parachute):
    """Parachute com geração aleatória de parâmetros."""

    @classmethod
    def generate_random(
        cls,
        ranges: ParachuteRanges,
        name: str = "Main",
        rng: Optional[random.Random] = None,
    ) -> "SuperParachute":
        """Gera um paraquedas aleatório.

        Args:
            ranges: Ranges do paraquedas.
            name: Nome do paraquedas.
            rng: Gerador aleatório.

        Returns:
            Instância de SuperParachute.
        """

        random_source = rng or random

        cd_s = ranges.cd_s.random(random_source)
        trigger_altitude = ranges.trigger_altitude.random(random_source)
        sampling_rate = ranges.sampling_rate.random(random_source)
        lag = ranges.lag.random(random_source)
        noise = (
            ranges.noise_mean.random(random_source),
            ranges.noise_std.random(random_source),
            ranges.noise_time_correlation.random(random_source),
        )

        return cls(
            name=name,
            cd_s=cd_s,
            trigger=trigger_altitude,
            sampling_rate=sampling_rate,
            lag=lag,
            noise=noise,
        )

    def export_to_dict(self) -> dict:
        """Exporta parâmetros do paraquedas para dicionário."""

        return {
            "name": self.name,
            "cd_s": self.cd_s,
            "trigger": self.trigger,
            "sampling_rate": self.sampling_rate,
            "lag": self.lag,
            "noise": self.noise,
        }
