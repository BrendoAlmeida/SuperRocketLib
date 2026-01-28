from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, Sequence, TypeVar
import random

T = TypeVar("T")


@dataclass(frozen=True)
class Range:
    """Define um range contínuo ou discreto para geração de valores aleatórios.

    Args:
        min: Valor mínimo do range.
        max: Valor máximo do range.
        step: Passo discreto. Se None, gera valor contínuo.
    """

    min: float
    max: float
    step: Optional[float] = None

    def random(self, rng: Optional[random.Random] = None) -> float:
        """Gera um valor aleatório dentro do range.

        Args:
            rng: Gerador de números aleatórios. Se None, usa random padrão.

        Returns:
            Valor aleatório dentro do range.
        """

        random_source = rng or random
        if self.step is None:
            return random_source.uniform(self.min, self.max)

        if self.step <= 0:
            raise ValueError("step deve ser maior que 0 para ranges discretos.")

        num_steps = int(round((self.max - self.min) / self.step))
        index = random_source.randint(0, num_steps)
        return self.min + index * self.step

    @classmethod
    def continuous(cls, min_val: float, max_val: float) -> "Range":
        """Cria um range contínuo.

        Args:
            min_val: Valor mínimo.
            max_val: Valor máximo.

        Returns:
            Range contínuo.
        """

        return cls(min=min_val, max=max_val, step=None)

    @classmethod
    def discrete(cls, min_val: float, max_val: float, step: float) -> "Range":
        """Cria um range discreto.

        Args:
            min_val: Valor mínimo.
            max_val: Valor máximo.
            step: Passo discreto.

        Returns:
            Range discreto.
        """

        return cls(min=min_val, max=max_val, step=step)


@dataclass(frozen=True)
class IntRange:
    """Define um range inteiro para geração de valores aleatórios.

    Args:
        min: Valor mínimo do range.
        max: Valor máximo do range.
        step: Passo discreto.
    """

    min: int
    max: int
    step: int = 1

    def random(self, rng: Optional[random.Random] = None) -> int:
        """Gera um valor inteiro aleatório dentro do range.

        Args:
            rng: Gerador de números aleatórios. Se None, usa random padrão.

        Returns:
            Valor inteiro aleatório dentro do range.
        """

        if self.step <= 0:
            raise ValueError("step deve ser maior que 0 para ranges inteiros.")

        random_source = rng or random
        num_steps = (self.max - self.min) // self.step
        index = random_source.randint(0, num_steps)
        return self.min + index * self.step


@dataclass(frozen=True)
class Choice(Generic[T]):
    """Define um conjunto de opções discretas.

    Args:
        options: Lista de opções possíveis.
    """

    options: Sequence[T]

    def random(self, rng: Optional[random.Random] = None) -> T:
        """Seleciona uma opção aleatória.

        Args:
            rng: Gerador de números aleatórios. Se None, usa random padrão.

        Returns:
            Opção selecionada.
        """

        if not self.options:
            raise ValueError("options não pode ser vazio.")

        random_source = rng or random
        return random_source.choice(list(self.options))


@dataclass
class RocketRanges:
    """Ranges necessários para inicializar um Rocket do RocketPy."""

    radius: Range
    length: Range
    mass: Range
    center_of_mass_without_motor: Range

    I_11_without_motor: Range
    I_22_without_motor: Range
    I_33_without_motor: Range
    I_12_without_motor: Range
    I_13_without_motor: Range
    I_23_without_motor: Range

    wall_thickness: Range
    motor_position: Range

    coordinate_system_orientation: str = "tail_to_nose"


@dataclass
class MotorRanges:
    """Ranges necessários para inicializar um SolidMotor do RocketPy."""

    motor_diameter: Range
    motor_length: Range

    nozzle_radius: Range
    throat_radius: Range

    dry_mass: Range

    dry_I_11: Range
    dry_I_22: Range
    dry_I_33: Range
    dry_I_12: Range
    dry_I_13: Range
    dry_I_23: Range

    grain_number: IntRange
    grain_density: Range
    grain_outer_radius: Range
    grain_initial_inner_radius: Range
    grain_initial_height: Range
    grain_separation: Range

    grains_center_of_mass_position: Range
    center_of_dry_mass_position: Range

    burn_time: Range
    average_thrust: Range
    peak_thrust_ratio: Range
    ignition_duration_fraction: Range
    tail_off_fraction: Range
    main_burn_end_ratio: Range

    thrust_profile_type: Choice[str]
    thrust_curve_points: IntRange

    coordinate_system_orientation: str = "nozzle_to_combustion_chamber"


@dataclass
class NoseConeRanges:
    """Ranges necessários para inicializar um NoseCone do RocketPy."""

    length: Range
    kind: Choice[str]
    bluffness: Optional[Range] = None
    power: Optional[Range] = None


@dataclass
class TrapezoidalFinsRanges:
    """Ranges necessários para inicializar TrapezoidalFins do RocketPy."""

    n: IntRange
    root_chord: Range
    tip_chord: Range
    span: Range
    sweep_length: Range
    cant_angle: Range


@dataclass
class TailRanges:
    """Ranges necessários para inicializar Tail do RocketPy."""

    length: Range
    top_radius_ratio: Range
    bottom_radius_ratio: Range


@dataclass
class ParachuteRanges:
    """Ranges necessários para inicializar Parachute do RocketPy."""

    cd_s: Range
    trigger_altitude: Range
    sampling_rate: Range
    lag: Range
    noise_mean: Range
    noise_std: Range
    noise_time_correlation: Range


@dataclass
class EnvironmentRanges:
    """Ranges necessários para inicializar Environment do RocketPy."""

    gravity: Range
    latitude: Range
    longitude: Range
    elevation: Range
    max_expected_height: Range


@dataclass
class FlightRanges:
    """Ranges necessários para inicializar Flight do RocketPy."""

    rail_length: Range
    inclination: Range
    heading: Range
    max_time: Range


@dataclass
class RailButtonsRanges:
    """Ranges necessários para inicializar RailButtons do RocketPy."""

    upper_button_position: Range
    lower_button_position: Range
    angular_position: Range


@dataclass
class RocketConfigRanges:
    """Configuração completa de ranges para gerar um foguete completo."""

    rocket: RocketRanges
    motor: MotorRanges
    nosecone: NoseConeRanges
    fins: TrapezoidalFinsRanges
    tail: TailRanges
    parachute: ParachuteRanges
    rail_buttons: RailButtonsRanges
    environment: EnvironmentRanges
    flight: FlightRanges
