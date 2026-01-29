from __future__ import annotations

from typing import Any, Dict, List, Optional
import random

from rocketpy import Rocket
try:
    from rocketpy import Function as RocketPyFunction
except Exception:  # pragma: no cover - compatibilidade com versões sem Function
    RocketPyFunction = None

from .structures import RocketConfigRanges
from .validators import RocketValidator
from ..components.motor import SuperMotor
from ..components.aerodynamics import SuperNoseCone, SuperTrapezoidalFins, SuperTail
from ..components.parachute import SuperParachute
from ..components.rail_buttons import SuperRailButtons
from ..generators.curve_generators import (
    DragCurveGenerator,
    DragCurveParameters,
    ThrustCurveGenerator,
    ThrustCurveParameters,
)


def _ensure_rocketpy_function(curve):
    if RocketPyFunction is None:
        return curve
    try:
        return RocketPyFunction(curve)
    except Exception:
        return curve


def _update_component_position(component, position: float) -> None:
    if component is None:
        return
    if hasattr(component, "position"):
        try:
            component.position = position
            return
        except Exception:
            pass
    if hasattr(component, "set_position"):
        try:
            component.set_position(position)
            return
        except Exception:
            pass
    if hasattr(component, "_position"):
        try:
            component._position = position
        except Exception:
            pass


def _extract_static_margin_value(static_margin) -> Optional[float]:
    if static_margin is None:
        return None
    if isinstance(static_margin, (int, float)):
        return float(static_margin)
    if callable(static_margin):
        try:
            return float(static_margin(0.0, 0.0))
        except Exception:
            return None
    return None


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

        # Criar rocket com drag temporário (será recalculado após ajustes)
        rocket = cls(
            radius=rocket_radius,
            mass=mass,
            inertia=inertia,
            power_off_drag=[(0, 0), (1, 0)],  # Placeholder
            power_on_drag=[(0, 0), (1, 0)],   # Placeholder
            center_of_mass_without_motor=center_of_mass_without_motor,
            coordinate_system_orientation=config.rocket.coordinate_system_orientation,
        )

        rocket.add_motor(motor, position=motor_position)

        original_nose_length = nose.length
        nose_position = max(rocket_length - nose.length, 0.0)
        tail_position = 0.0
        fins_position = rocket_length * 0.1

        nose.add_to_rocket(rocket, position=nose_position)
        tail.add_to_rocket(rocket, position=tail_position)
        fins.add_to_rocket(rocket, position=fins_position)
        
        # 1. Primeiro atualizamos a referência Mestra (Comprimento Total)
        rocket_length = max(rocket_length + (nose.length - original_nose_length), 0.0)

        # 2. Agora recalculamos a posição baseada no NOVO comprimento
        # (Isso deve manter a posição da base do nariz inalterada, ex: 8.0)
        nose_position = max(rocket_length - nose.length, 0.0)
        _update_component_position(getattr(rocket, "nose", None), nose_position)
        
        # Recalcular drag com valores reais após ajustes do RocketPy
        fin_area = fins.planform_area() * fins.n
        drag_params = DragCurveParameters(
            rocket_radius=rocket_radius,
            rocket_length=rocket_length,
            fin_area=fin_area,
            fin_count=fins.n,
            tail_bottom_radius=tail.bottom_radius,
        )
        power_off_drag, power_on_drag = DragCurveGenerator.generate(drag_params)
        rocket.power_off_drag = _ensure_rocketpy_function(power_off_drag)
        rocket.power_on_drag = _ensure_rocketpy_function(power_on_drag)
        if hasattr(rocket, "evaluate_static_margin"):
            try:
                static_margin = rocket.evaluate_static_margin()
                static_margin_value = _extract_static_margin_value(static_margin)
            except Exception:
                static_margin_value = None
            if static_margin_value is not None and static_margin_value < 1.0:
                raise ValueError(
                    f"Static margin after adjustments is unstable: {static_margin_value:.3f}"
                )

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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuperRocket":
        def _get(key: str, default: Any = None) -> Any:
            value = data.get(key, default)
            return default if value is None else value

        rocket_radius = float(_get("radius", 0.05))
        rocket_length = float(_get("rocket_length", rocket_radius * 8.0))
        wall_thickness = float(_get("wall_thickness", rocket_radius * 0.05))
        rocket_inner_radius = float(
            _get("rocket_inner_radius", max(rocket_radius - wall_thickness, rocket_radius * 0.8))
        )
        mass = float(_get("mass", 1.0))
        center_of_mass_without_motor = float(
            _get("center_of_mass_without_motor", rocket_length * 0.5)
        )
        inertia = (
            float(_get("inertia_11", 0.0)),
            float(_get("inertia_22", 0.0)),
            float(_get("inertia_33", 0.0)),
            float(_get("inertia_12", 0.0)),
            float(_get("inertia_13", 0.0)),
            float(_get("inertia_23", 0.0)),
        )

        motor_diameter = float(_get("motor.motor_diameter", rocket_inner_radius * 1.8))
        motor_length = float(_get("motor.motor_length", rocket_length * 0.4))
        nozzle_radius = float(_get("motor.nozzle_radius", motor_diameter * 0.2))
        throat_radius = float(_get("motor.throat_radius", nozzle_radius * 0.6))
        dry_mass = float(_get("motor.dry_mass", mass * 0.2))
        dry_inertia = (
            float(_get("motor.dry_inertia_11", 0.0)),
            float(_get("motor.dry_inertia_22", 0.0)),
            float(_get("motor.dry_inertia_33", 0.0)),
            float(_get("motor.dry_inertia_12", 0.0)),
            float(_get("motor.dry_inertia_13", 0.0)),
            float(_get("motor.dry_inertia_23", 0.0)),
        )
        grain_number = int(round(_get("motor.grain_number", 1)))
        grain_density = float(_get("motor.grain_density", 1500.0))
        grain_outer_radius = float(_get("motor.grain_outer_radius", motor_diameter * 0.4))
        grain_initial_inner_radius = float(
            _get("motor.grain_initial_inner_radius", grain_outer_radius * 0.5)
        )
        grain_initial_height = float(_get("motor.grain_initial_height", motor_length * 0.2))
        grain_separation = float(_get("motor.grain_separation", motor_length * 0.02))
        grains_center_of_mass_position = float(
            _get("motor.grains_center_of_mass_position", motor_length * 0.5)
        )
        center_of_dry_mass_position = float(
            _get("motor.center_of_dry_mass_position", motor_length * 0.5)
        )
        burn_time = float(_get("motor.burn_time", _get("motor.thrust_curve.burn_time", 1.0)))
        average_thrust = float(
            _get("motor.average_thrust", _get("motor.thrust_curve.avg_thrust", 100.0))
        )
        peak_thrust_ratio = float(_get("motor.peak_thrust_ratio", 2.0))
        ignition_duration_fraction = float(_get("motor.ignition_duration_fraction", 0.05))
        tail_off_fraction = float(_get("motor.tail_off_fraction", 0.08))
        main_burn_end_ratio = float(_get("motor.main_burn_end_ratio", 1.0))
        thrust_profile_type = _get("motor.thrust_profile_type", "neutral")
        curve_points = int(round(_get("motor.thrust_curve_points", 150)))

        curve_params = ThrustCurveParameters(
            burn_time=burn_time,
            average_thrust=average_thrust,
            peak_thrust_ratio=peak_thrust_ratio,
            ignition_duration_fraction=ignition_duration_fraction,
            tail_off_fraction=tail_off_fraction,
            main_burn_end_ratio=main_burn_end_ratio,
            thrust_profile_type=str(thrust_profile_type),
            points=curve_points,
        )
        thrust_source = ThrustCurveGenerator.generate(curve_params)
        motor = SuperMotor(
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
            coordinate_system_orientation=_get(
                "motor.coordinate_system_orientation", "nozzle_to_combustion_chamber"
            ),
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

        nose = SuperNoseCone(
            length=float(_get("nosecone.length", rocket_length * 0.15)),
            kind=_get("nosecone.kind", "von karman"),
            base_radius=rocket_radius,
            bluffness=_get("nosecone.bluffness", None),
            rocket_radius=rocket_radius,
            power=_get("nosecone.power", None),
        )

        fins = SuperTrapezoidalFins(
            n=int(round(_get("fins.n", 4))),
            root_chord=float(_get("fins.root_chord", rocket_length * 0.15)),
            tip_chord=float(_get("fins.tip_chord", rocket_length * 0.08)),
            span=float(_get("fins.span", rocket_radius * 1.5)),
            rocket_radius=rocket_radius,
            sweep_length=float(_get("fins.sweep_length", rocket_length * 0.05)),
            cant_angle=float(_get("fins.cant_angle", 0.0)),
        )

        tail = SuperTail(
            length=float(_get("tail.length", rocket_length * 0.1)),
            top_radius=float(_get("tail.top_radius", rocket_radius)),
            bottom_radius=float(_get("tail.bottom_radius", rocket_radius)),
            rocket_radius=rocket_radius,
        )

        parachute_noise = (
            float(_get("parachute.noise[0]", 0.0)),
            float(_get("parachute.noise[1]", 0.0)),
            float(_get("parachute.noise[2]", 0.0)),
        )
        parachute = SuperParachute(
            name=str(_get("parachute.name", "Main")),
            cd_s=float(_get("parachute.cd_s", 1.0)),
            trigger=float(_get("parachute.trigger", 200.0)),
            sampling_rate=float(_get("parachute.sampling_rate", 100.0)),
            lag=float(_get("parachute.lag", 0.5)),
            noise=parachute_noise,
        )

        rail_buttons = SuperRailButtons(
            upper_button_position=float(_get("rail_buttons.upper_button_position", rocket_length * 0.6)),
            lower_button_position=float(_get("rail_buttons.lower_button_position", rocket_length * 0.2)),
            angular_position=float(_get("rail_buttons.angular_position", 45.0)),
            rocket_radius=rocket_radius,
        )

        # Criar rocket com drag placeholder (será recalculado após ajustes)
        rocket = cls(
            radius=rocket_radius,
            mass=mass,
            inertia=inertia,
            power_off_drag=[(0, 0), (1, 0)],  # Placeholder
            power_on_drag=[(0, 0), (1, 0)],   # Placeholder
            center_of_mass_without_motor=center_of_mass_without_motor,
            coordinate_system_orientation=_get(
                "coordinate_system_orientation", "tail_to_nose"
            ),
        )

        motor_position = float(_get("motor_position", rocket_length * 0.1))
        rocket.add_motor(motor, position=motor_position)

        original_nose_length = nose.length
        nose_position = float(_get("nose_position", max(rocket_length - nose.length, 0.0)))
        tail_position = float(_get("tail_position", 0.0))
        fins_position = float(_get("fins_position", rocket_length * 0.1))

        nose.add_to_rocket(rocket, position=nose_position)
        tail.add_to_rocket(rocket, position=tail_position)
        fins.add_to_rocket(rocket, position=fins_position)
        
        # 1. Primeiro atualizamos a referência Mestra (Comprimento Total)
        rocket_length = max(rocket_length + (nose.length - original_nose_length), 0.0)

        # 2. Agora recalculamos a posição baseada no NOVO comprimento
        # (Isso deve manter a posição da base do nariz inalterada, ex: 8.0)
        nose_position = max(rocket_length - nose.length, 0.0)
        _update_component_position(getattr(rocket, "nose", None), nose_position)
        
        # Recalcular drag com valores reais após ajustes do RocketPy
        fin_area = fins.planform_area() * fins.n
        drag_params = DragCurveParameters(
            rocket_radius=rocket_radius,
            rocket_length=rocket_length,
            fin_area=fin_area,
            fin_count=fins.n,
            tail_bottom_radius=tail.bottom_radius,
        )
        power_off_drag, power_on_drag = DragCurveGenerator.generate(drag_params)
        rocket.power_off_drag = _ensure_rocketpy_function(power_off_drag)
        rocket.power_on_drag = _ensure_rocketpy_function(power_on_drag)
        if hasattr(rocket, "evaluate_static_margin"):
            try:
                rocket.evaluate_static_margin()
            except Exception:
                pass

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
        rocket._coordinate_system_orientation = _get(
            "coordinate_system_orientation", "tail_to_nose"
        )

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
