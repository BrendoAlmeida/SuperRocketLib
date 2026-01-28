from __future__ import annotations

from typing import Optional
import random

from rocketpy import SolidMotor

from ..core.structures import MotorRanges
from ..generators.curve_generators import ThrustCurveGenerator, ThrustCurveParameters


class SuperMotor(SolidMotor):
    """Motor sólido com geração aleatória de parâmetros e curva de empuxo.

    Args:
        motor_diameter: Diâmetro externo do motor (m).
        motor_length: Comprimento do motor (m).
        kwargs: Parâmetros do SolidMotor do RocketPy.
    """

    def __init__(self, motor_diameter: float, motor_length: float, **kwargs):
        super().__init__(**kwargs)
        self.motor_diameter = motor_diameter
        self.motor_length = motor_length

    @classmethod
    def generate_random(
        cls,
        ranges: MotorRanges,
        rocket_inner_radius: float,
        rng: Optional[random.Random] = None,
    ) -> "SuperMotor":
        """Gera um motor sólido aleatório dentro dos ranges fornecidos.

        Args:
            ranges: Ranges de parâmetros do motor.
            rocket_inner_radius: Raio interno do foguete (m).
            rng: Gerador de números aleatórios para reprodutibilidade.

        Returns:
            Instância de SuperMotor.
        """

        random_source = rng or random

        motor_diameter = ranges.motor_diameter.random(random_source)
        motor_length = ranges.motor_length.random(random_source)

        if motor_diameter / 2.0 > rocket_inner_radius:
            raise ValueError("Diâmetro do motor maior que o raio interno do foguete.")

        nozzle_radius = ranges.nozzle_radius.random(random_source)
        throat_radius = ranges.throat_radius.random(random_source)

        dry_mass = ranges.dry_mass.random(random_source)

        dry_inertia = (
            ranges.dry_I_11.random(random_source),
            ranges.dry_I_22.random(random_source),
            ranges.dry_I_33.random(random_source),
            ranges.dry_I_12.random(random_source),
            ranges.dry_I_13.random(random_source),
            ranges.dry_I_23.random(random_source),
        )

        grain_number = ranges.grain_number.random(random_source)
        grain_density = ranges.grain_density.random(random_source)
        grain_outer_radius = min(
            ranges.grain_outer_radius.random(random_source),
            motor_diameter / 2.0,
        )
        grain_initial_inner_radius = min(
            ranges.grain_initial_inner_radius.random(random_source),
            grain_outer_radius * 0.9,
        )
        grain_initial_height = ranges.grain_initial_height.random(random_source)
        grain_separation = ranges.grain_separation.random(random_source)

        grains_center_of_mass_position = ranges.grains_center_of_mass_position.random(random_source)
        center_of_dry_mass_position = ranges.center_of_dry_mass_position.random(random_source)

        burn_time = ranges.burn_time.random(random_source)
        average_thrust = ranges.average_thrust.random(random_source)
        peak_thrust_ratio = ranges.peak_thrust_ratio.random(random_source)
        ignition_duration_fraction = ranges.ignition_duration_fraction.random(random_source)
        tail_off_fraction = ranges.tail_off_fraction.random(random_source)
        main_burn_end_ratio = ranges.main_burn_end_ratio.random(random_source)
        thrust_profile_type = ranges.thrust_profile_type.random(random_source)
        curve_points = ranges.thrust_curve_points.random(random_source)

        curve_params = ThrustCurveParameters(
            burn_time=burn_time,
            average_thrust=average_thrust,
            peak_thrust_ratio=peak_thrust_ratio,
            ignition_duration_fraction=ignition_duration_fraction,
            tail_off_fraction=tail_off_fraction,
            main_burn_end_ratio=main_burn_end_ratio,
            thrust_profile_type=thrust_profile_type,
            points=curve_points,
        )
        thrust_source = ThrustCurveGenerator.generate(curve_params)

        motor = cls(
            motor_diameter=motor_diameter,
            motor_length=motor_length,
            thrust_source=thrust_source,
            dry_mass=dry_mass,
            dry_inertia=dry_inertia,
            nozzle_radius=nozzle_radius,
            throat_radius=throat_radius,
            grain_number=grain_number,
            grain_density=grain_density,
            grain_outer_radius=grain_outer_radius,
            grain_initial_inner_radius=grain_initial_inner_radius,
            grain_initial_height=grain_initial_height,
            grain_separation=grain_separation,
            grains_center_of_mass_position=grains_center_of_mass_position,
            center_of_dry_mass_position=center_of_dry_mass_position,
            burn_time=(0, burn_time),
            coordinate_system_orientation=ranges.coordinate_system_orientation,
        )

        motor.dry_inertia = dry_inertia
        motor.burn_time = burn_time
        motor.average_thrust = average_thrust
        motor.peak_thrust_ratio = peak_thrust_ratio
        motor.ignition_duration_fraction = ignition_duration_fraction
        motor.tail_off_fraction = tail_off_fraction
        motor.main_burn_end_ratio = main_burn_end_ratio
        motor.thrust_profile_type = thrust_profile_type
        motor.thrust_curve_points = curve_points

        return motor

    def export_to_dict(self) -> dict:
        """Exporta parâmetros do motor para dicionário."""

        return {
            "motor_diameter": self.motor_diameter,
            "motor_length": self.motor_length,
            "dry_mass": self.dry_mass,
            "dry_inertia_11": getattr(self, "dry_inertia", (None,) * 6)[0],
            "dry_inertia_22": getattr(self, "dry_inertia", (None,) * 6)[1],
            "dry_inertia_33": getattr(self, "dry_inertia", (None,) * 6)[2],
            "dry_inertia_12": getattr(self, "dry_inertia", (None,) * 6)[3],
            "dry_inertia_13": getattr(self, "dry_inertia", (None,) * 6)[4],
            "dry_inertia_23": getattr(self, "dry_inertia", (None,) * 6)[5],
            "nozzle_radius": self.nozzle_radius,
            "throat_radius": self.throat_radius,
            "grain_number": self.grain_number,
            "grain_density": self.grain_density,
            "grain_outer_radius": self.grain_outer_radius,
            "grain_initial_inner_radius": self.grain_initial_inner_radius,
            "grain_initial_height": self.grain_initial_height,
            "grain_separation": self.grain_separation,
            "grains_center_of_mass_position": self.grains_center_of_mass_position,
            "center_of_dry_mass_position": self.center_of_dry_mass_position,
            "burn_time": getattr(self, "burn_time", None),
            "average_thrust": getattr(self, "average_thrust", None),
            "peak_thrust_ratio": getattr(self, "peak_thrust_ratio", None),
            "ignition_duration_fraction": getattr(self, "ignition_duration_fraction", None),
            "tail_off_fraction": getattr(self, "tail_off_fraction", None),
            "main_burn_end_ratio": getattr(self, "main_burn_end_ratio", None),
            "thrust_profile_type": getattr(self, "thrust_profile_type", None),
            "thrust_curve_points": getattr(self, "thrust_curve_points", None),
            "thrust_curve": getattr(self, "thrust_source", None),
        }
