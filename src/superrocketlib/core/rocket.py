from __future__ import annotations

from typing import Any, Dict, List, Optional
import random

from rocketpy import Rocket

from .structures import RocketConfigRanges
from .validators import RocketValidator
from ..components.motor import SuperMotor
from ..components.aerodynamics import SuperNoseCone, SuperTrapezoidalFins, SuperTail
from ..components.parachute import SuperParachute
from ..components.rail_buttons import SuperRailButtons
from ..generators.curve_generators import DragCurveGenerator, DragCurveParameters


class SuperRocket(Rocket):
    """RocketPy Rocket com geração aleatória de parâmetros e componentes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.motor_component: Optional[SuperMotor] = None
        self.nose_component: Optional[SuperNoseCone] = None
        self.fins_component: Optional[SuperTrapezoidalFins] = None
        self.tail_component: Optional[SuperTail] = None
        self.parachute_component: Optional[SuperParachute] = None
        self.rail_buttons_component: Optional[SuperRailButtons] = None
        self.simulation_results: Dict[str, Any] = {}
        self.simulation_objects: Dict[str, Any] = {}

    @classmethod
    def generate_random(
        cls,
        config: Optional[RocketConfigRanges] = None,
        seed: Optional[int] = None,
    ) -> "SuperRocket":
        """Gera um foguete completo com componentes aleatórios.

        Args:
            config: Configuração de ranges para foguete e componentes.
            seed: Seed para reprodutibilidade.

        Returns:
            Instância de SuperRocket.
        """

        if config is None:
            from ..defaults import DEFAULT_CONFIG

            config = DEFAULT_CONFIG

        rng = random.Random(seed)

        rocket_radius = config.rocket.radius.random(rng)
        rocket_length = config.rocket.length.random(rng)
        wall_thickness = config.rocket.wall_thickness.random(rng)
        rocket_inner_radius = max(rocket_radius - wall_thickness, rocket_radius * 0.8)

        mass = config.rocket.mass.random(rng)
        center_of_mass_without_motor = config.rocket.center_of_mass_without_motor.random(rng)
        center_of_mass_without_motor = max(0.0, min(center_of_mass_without_motor, rocket_length))

        inertia = (
            config.rocket.I_11_without_motor.random(rng),
            config.rocket.I_22_without_motor.random(rng),
            config.rocket.I_33_without_motor.random(rng),
            config.rocket.I_12_without_motor.random(rng),
            config.rocket.I_13_without_motor.random(rng),
            config.rocket.I_23_without_motor.random(rng),
        )

        motor = SuperMotor.generate_random(
            ranges=config.motor,
            rocket_inner_radius=rocket_inner_radius,
            rng=rng,
        )

        nose = SuperNoseCone.generate_random(
            ranges=config.nosecone,
            rocket_radius=rocket_radius,
            rng=rng,
        )

        fins = SuperTrapezoidalFins.generate_random(
            ranges=config.fins,
            rocket_radius=rocket_radius,
            rocket_length=rocket_length,
            rng=rng,
        )

        tail = SuperTail.generate_random(
            ranges=config.tail,
            rocket_radius=rocket_radius,
            rng=rng,
        )

        parachute = SuperParachute.generate_random(
            ranges=config.parachute,
            name="Main",
            rng=rng,
        )

        rail_buttons = SuperRailButtons.generate_random(
            ranges=config.rail_buttons,
            rng=rng,
        )

        geometry_validation = RocketValidator.validate_geometry(
            rocket_length=rocket_length,
            nose_length=nose.length,
            tail_length=tail.length,
            fin_span=fins.span,
            rocket_radius=rocket_radius,
            fin_root_chord=fins.root_chord,
        )
        RocketValidator.log_warnings(geometry_validation)
        geometry_validation.raise_if_errors()

        motor_fit_validation = RocketValidator.validate_motor_fit(
            rocket_inner_radius=rocket_inner_radius,
            motor_diameter=motor.motor_diameter,
        )
        RocketValidator.log_warnings(motor_fit_validation)
        motor_fit_validation.raise_if_errors()

        motor_position = config.rocket.motor_position.random(rng)
        motor_position_validation = RocketValidator.validate_motor_position(
            motor_position=motor_position,
            motor_length=motor.motor_length,
            rocket_length=rocket_length,
        )
        RocketValidator.log_warnings(motor_position_validation)
        motor_position_validation.raise_if_errors()

        fin_area = fins.planform_area() * fins.n
        drag_params = DragCurveParameters(
            rocket_radius=rocket_radius,
            rocket_length=rocket_length,
            fin_area=fin_area,
            fin_count=fins.n,
            tail_bottom_radius=tail.bottom_radius,
        )
        power_off_drag, power_on_drag = DragCurveGenerator.generate(drag_params)

        rocket = cls(
            radius=rocket_radius,
            mass=mass,
            inertia=inertia,
            power_off_drag=power_off_drag,
            power_on_drag=power_on_drag,
            center_of_mass_without_motor=center_of_mass_without_motor,
            coordinate_system_orientation=config.rocket.coordinate_system_orientation,
        )

        rocket.add_motor(motor, position=motor_position)

        nose_position = max(rocket_length - nose.length, 0.0)
        tail_position = 0.0
        fins_position = rocket_length * 0.1

        nose.add_to_rocket(rocket, position=nose_position)
        tail.add_to_rocket(rocket, position=tail_position)
        fins.add_to_rocket(rocket, position=fins_position)

        if hasattr(rocket, "add_parachute"):
            rocket.add_parachute(
                parachute.name,
                parachute.cd_s,
                parachute.trigger,
                parachute.sampling_rate,
                parachute.lag,
                parachute.noise,
            )
        else:
            rocket.parachutes.append(parachute)

        if hasattr(rocket, "set_rail_buttons"):
            rocket.set_rail_buttons(
                rail_buttons.upper_button_position,
                rail_buttons.lower_button_position,
                rail_buttons.angular_position,
            )
        else:
            rocket.rail_buttons = rail_buttons

        rocket.motor_component = motor
        rocket.nose_component = nose
        rocket.fins_component = fins
        rocket.tail_component = tail
        rocket.parachute_component = parachute
        rocket.rail_buttons_component = rail_buttons

        rocket._rocket_length = rocket_length
        rocket._rocket_inner_radius = rocket_inner_radius
        rocket._wall_thickness = wall_thickness
        rocket._center_of_mass_without_motor = center_of_mass_without_motor
        rocket._inertia = inertia
        rocket._motor_position = motor_position
        rocket._nose_position = nose_position
        rocket._fins_position = fins_position
        rocket._tail_position = tail_position
        rocket._coordinate_system_orientation = config.rocket.coordinate_system_orientation

        return rocket

    def export_to_dict(self) -> dict:
        """Exporta parâmetros do foguete e componentes para dicionário."""

        return {
            "radius": self.radius,
            "mass": self.mass,
            "rocket_length": getattr(self, "_rocket_length", None),
            "rocket_inner_radius": getattr(self, "_rocket_inner_radius", None),
            "wall_thickness": getattr(self, "_wall_thickness", None),
            "center_of_mass_without_motor": getattr(self, "_center_of_mass_without_motor", None),
            "inertia_11": getattr(self, "_inertia", (None,) * 6)[0],
            "inertia_22": getattr(self, "_inertia", (None,) * 6)[1],
            "inertia_33": getattr(self, "_inertia", (None,) * 6)[2],
            "inertia_12": getattr(self, "_inertia", (None,) * 6)[3],
            "inertia_13": getattr(self, "_inertia", (None,) * 6)[4],
            "inertia_23": getattr(self, "_inertia", (None,) * 6)[5],
            "motor_position": getattr(self, "_motor_position", None),
            "nose_position": getattr(self, "_nose_position", None),
            "fins_position": getattr(self, "_fins_position", None),
            "tail_position": getattr(self, "_tail_position", None),
            "coordinate_system_orientation": getattr(self, "_coordinate_system_orientation", None),
            "motor": self.motor_component.export_to_dict() if self.motor_component else None,
            "nosecone": self.nose_component.export_to_dict() if self.nose_component else None,
            "fins": self.fins_component.export_to_dict() if self.fins_component else None,
            "tail": self.tail_component.export_to_dict() if self.tail_component else None,
            "parachute": self.parachute_component.export_to_dict() if self.parachute_component else None,
            "rail_buttons": self.rail_buttons_component.export_to_dict() if self.rail_buttons_component else None,
            "simulations": self.simulation_results or None,
        }

    def simulate(
        self,
        env: Any,
        use_monte_carlo: bool = False,
        monte_carlo_runs: int = 30,
        **flight_kwargs: Any,
    ) -> Dict[str, Any]:
        """Simula o foguete com os parâmetros atuais.

        Args:
            env: Ambiente do RocketPy.
            use_monte_carlo: Se True, executa múltiplas simulações.
            monte_carlo_runs: Número de execuções no Monte Carlo.
            **flight_kwargs: Parâmetros adicionais para Flight.

        Returns:
            Dicionário com resultados da simulação.
        """

        if use_monte_carlo:
            result = self.simulate_monte_carlo(env, runs=monte_carlo_runs, **flight_kwargs)
            self.simulation_results["monte_carlo"] = result
            return result

        result = self.simulate_flight(env, **flight_kwargs)
        self.simulation_results["flight"] = result
        return result

    def simulate_flight(self, env: Any, **flight_kwargs: Any) -> Dict[str, Any]:
        """Simula o voo completo do foguete.

        Args:
            env: Ambiente do RocketPy.
            **flight_kwargs: Parâmetros adicionais para Flight.

        Returns:
            Resumo da simulação de voo.
        """

        from rocketpy import Flight

        flight = Flight(rocket=self, environment=env, **flight_kwargs)
        self.simulation_objects["flight"] = flight
        summary = self._summarize_flight(flight)
        return summary

    def simulate_monte_carlo(
        self,
        env: Any,
        runs: int = 30,
        **flight_kwargs: Any,
    ) -> Dict[str, Any]:
        """Executa simulações Monte Carlo (múltiplos voos).

        Args:
            env: Ambiente do RocketPy.
            runs: Número de execuções.
            **flight_kwargs: Parâmetros adicionais para Flight.

        Returns:
            Dicionário com os resultados agregados.
        """

        summaries: List[Dict[str, Any]] = []
        for _ in range(max(runs, 1)):
            summary = self.simulate_flight(env, **flight_kwargs)
            summaries.append(summary)

        self.simulation_results["monte_carlo_runs"] = summaries
        return {
            "runs": len(summaries),
            "summaries": summaries,
        }

    def simulate_motor(self) -> Dict[str, Any]:
        """Simula apenas o motor (curva de empuxo)."""

        motor = self.motor_component or getattr(self, "motor", None)
        thrust_curve = getattr(motor, "thrust_source", None)
        result = {"thrust_curve": thrust_curve}
        self.simulation_results["motor"] = result
        return result

    def simulate_aerodynamics(self) -> Dict[str, Any]:
        """Simula apenas a parte aerodinâmica (curvas de arrasto)."""

        result = {
            "power_off_drag": getattr(self, "power_off_drag", None),
            "power_on_drag": getattr(self, "power_on_drag", None),
        }
        self.simulation_results["aerodynamics"] = result
        return result

    @staticmethod
    def _summarize_flight(flight: Any) -> Dict[str, Any]:
        """Extrai um resumo do voo do RocketPy.

        Args:
            flight: Instância de Flight.

        Returns:
            Dicionário com métricas principais.
        """

        return {
            "apogee": getattr(flight, "apogee", None),
            "max_speed": getattr(flight, "max_speed", None),
            "max_acceleration": getattr(flight, "max_acceleration", None),
            "max_mach": getattr(
                flight,
                "max_mach",
                getattr(flight, "max_mach_number", None),
            ),
            "flight_time": getattr(flight, "t_final", None),
        }
