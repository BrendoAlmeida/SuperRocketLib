from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math


@dataclass(frozen=True)
class ThrustCurveParameters:
    """Parâmetros para geração de curva de empuxo.

    Args:
        burn_time: Tempo total de queima (s).
        average_thrust: Empuxo médio (N).
        peak_thrust_ratio: Razão entre empuxo de pico e empuxo médio.
        ignition_duration_fraction: Fração do tempo de queima dedicada à ignição.
        tail_off_fraction: Fração do tempo de queima dedicada ao tail-off.
        main_burn_end_ratio: Razão do empuxo no fim da fase principal.
        thrust_profile_type: Tipo de perfil ("progressive", "neutral", "regressive").
        points: Número de pontos na curva.
    """

    burn_time: float
    average_thrust: float
    peak_thrust_ratio: float
    ignition_duration_fraction: float
    tail_off_fraction: float
    main_burn_end_ratio: float
    thrust_profile_type: str
    points: int


@dataclass(frozen=True)
class DragCurveParameters:
    """Parâmetros para geração de curva de arrasto.

    Args:
        rocket_radius: Raio do foguete (m).
        rocket_length: Comprimento total do foguete (m).
        fin_area: Área total das aletas (m^2).
        fin_count: Número de aletas.
        tail_bottom_radius: Raio da base do tail (m).
    """

    rocket_radius: float
    rocket_length: float
    fin_area: float
    fin_count: int
    tail_bottom_radius: float


class ThrustCurveGenerator:
    """Gera uma curva de empuxo sintética a partir de parâmetros físicos."""

    @staticmethod
    def generate(parameters: ThrustCurveParameters) -> List[Tuple[float, float]]:
        """Gera a curva de empuxo.

        Args:
            parameters: Parâmetros da curva de empuxo.

        Returns:
            Lista de pares (tempo, empuxo).
        """

        burn_time = max(parameters.burn_time, 0.01)
        points = max(parameters.points, 20)
        ignition_fraction = max(min(parameters.ignition_duration_fraction, 0.3), 0.01)
        tail_off_fraction = max(min(parameters.tail_off_fraction, 0.3), 0.01)
        ignition_time = burn_time * ignition_fraction
        tail_off_time = burn_time * tail_off_fraction
        main_burn_time = max(burn_time - ignition_time - tail_off_time, burn_time * 0.4)

        peak_ratio = max(parameters.peak_thrust_ratio, 1.05)
        main_end_ratio = max(parameters.main_burn_end_ratio, 0.2)

        times: List[float] = [i * burn_time / (points - 1) for i in range(points)]
        ratios: List[float] = []

        for t in times:
            if t <= ignition_time:
                ratio = peak_ratio * (t / ignition_time) if ignition_time > 0 else peak_ratio
            elif t <= ignition_time + main_burn_time:
                progress = (t - ignition_time) / main_burn_time
                if parameters.thrust_profile_type == "progressive":
                    ratio = 1.0 + (main_end_ratio - 1.0) * progress
                elif parameters.thrust_profile_type == "regressive":
                    ratio = 1.0 + (main_end_ratio - 1.0) * progress
                else:
                    ratio = 1.0
            else:
                progress = (t - ignition_time - main_burn_time) / tail_off_time
                ratio = max(main_end_ratio * (1.0 - progress), 0.0)
            ratios.append(ratio)

        average_ratio = sum(ratios) / len(ratios)
        if average_ratio <= 0:
            average_ratio = 1.0

        scale = parameters.average_thrust / average_ratio
        thrust_curve = [(t, max(r * scale, 0.0)) for t, r in zip(times, ratios)]
        thrust_curve[0] = (0.0, 0.0)
        thrust_curve[-1] = (burn_time, 0.0)
        return thrust_curve


class DragCurveGenerator:
    """Gera curvas de arrasto (power-off e power-on) baseadas na geometria."""

    @staticmethod
    def generate(parameters: DragCurveParameters) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Gera as curvas de arrasto em função do Mach.

        Args:
            parameters: Parâmetros para curva de arrasto.

        Returns:
            Tupla (power_off_drag, power_on_drag) com listas (mach, cd).
        """

        rocket_radius = max(parameters.rocket_radius, 1e-4)
        rocket_length = max(parameters.rocket_length, rocket_radius * 5.0)
        fin_area = max(parameters.fin_area, 0.0)
        fin_count = max(parameters.fin_count, 0)
        tail_bottom_radius = max(parameters.tail_bottom_radius, rocket_radius)

        reference_area = math.pi * rocket_radius ** 2
        fin_area_ratio = fin_area / reference_area if reference_area > 0 else 0.0

        slenderness = rocket_length / (2.0 * rocket_radius)
        base_cd = 0.25 + 0.015 * slenderness + 0.02 * fin_area_ratio
        base_cd *= 1.0 + 0.02 * max(fin_count - 3, 0)
        base_cd *= 1.0 + 0.1 * max((tail_bottom_radius / rocket_radius) - 1.0, 0.0)

        mach_values = [i * 5.0 / 49 for i in range(50)]
        power_off: List[Tuple[float, float]] = []
        power_on: List[Tuple[float, float]] = []

        for mach in mach_values:
            wave_drag = 0.0
            if mach >= 0.8:
                wave_drag = 0.3 * (mach - 0.8) ** 2

            cd_off = base_cd + wave_drag
            cd_on = cd_off * 0.95

            power_off.append((mach, cd_off))
            power_on.append((mach, cd_on))

        return power_off, power_on
