from __future__ import annotations

from typing import Optional
import random

from rocketpy import NoseCone, TrapezoidalFins, Tail

from ..core.structures import NoseConeRanges, TrapezoidalFinsRanges, TailRanges


class SuperNoseCone(NoseCone):
    """NoseCone com geração aleatória de parâmetros."""

    @classmethod
    def generate_random(
        cls,
        ranges: NoseConeRanges,
        rocket_radius: float,
        rng: Optional[random.Random] = None,
    ) -> "SuperNoseCone":
        """Gera um nose cone aleatório.

        Args:
            ranges: Ranges do nose cone.
            rocket_radius: Raio do foguete (m).
            rng: Gerador aleatório.

        Returns:
            Instância de SuperNoseCone.
        """

        random_source = rng or random
        length = ranges.length.random(random_source)
        kind = ranges.kind.random(random_source)
        bluffness = ranges.bluffness.random(random_source) if ranges.bluffness else None
        power = ranges.power.random(random_source) if ranges.power else None

        return cls(
            length=length,
            kind=kind,
            base_radius=rocket_radius,
            bluffness=bluffness,
            rocket_radius=rocket_radius,
            power=power,
        )

    def add_to_rocket(self, rocket, position: float) -> None:
        """Adiciona o nose cone ao foguete.

        Args:
            rocket: Instância de RocketPy.
            position: Posição do nose cone no foguete (m).
        """

        if hasattr(rocket, "add_nose"):
            rocket.add_nose(
                length=self.length,
                kind=self.kind,
                position=position,
                base_radius=self.base_radius,
                bluffness=getattr(self, "bluffness", None),
                power=getattr(self, "power", None),
            )
        else:
            rocket.add_surfaces(self, position)

    def export_to_dict(self) -> dict:
        """Exporta parâmetros do nose cone para dicionário."""

        return {
            "length": self.length,
            "kind": self.kind,
            "base_radius": self.base_radius,
            "bluffness": getattr(self, "bluffness", None),
            "power": getattr(self, "power", None),
        }


class SuperTrapezoidalFins(TrapezoidalFins):
    """Conjunto de aletas com geração aleatória de parâmetros."""

    @classmethod
    def generate_random(
        cls,
        ranges: TrapezoidalFinsRanges,
        rocket_radius: float,
        rocket_length: float,
        rng: Optional[random.Random] = None,
    ) -> "SuperTrapezoidalFins":
        """Gera um conjunto de aletas trapezoidais aleatórias.

        Args:
            ranges: Ranges das aletas.
            rocket_radius: Raio do foguete (m).
            rocket_length: Comprimento do foguete (m).
            rng: Gerador aleatório.

        Returns:
            Instância de SuperTrapezoidalFins.
        """

        random_source = rng or random

        n = ranges.n.random(random_source)
        root_chord = ranges.root_chord.random(random_source)
        tip_chord = ranges.tip_chord.random(random_source)
        span = ranges.span.random(random_source)
        sweep_length = ranges.sweep_length.random(random_source)
        cant_angle = ranges.cant_angle.random(random_source)

        if root_chord > rocket_length * 0.4:
            root_chord = rocket_length * 0.4

        if span > rocket_radius * 2.0:
            span = rocket_radius * 2.0

        return cls(
            n=n,
            root_chord=root_chord,
            tip_chord=tip_chord,
            span=span,
            rocket_radius=rocket_radius,
            sweep_length=sweep_length,
            cant_angle=cant_angle,
        )

    def add_to_rocket(self, rocket, position: float) -> None:
        """Adiciona as aletas ao foguete.

        Args:
            rocket: Instância de RocketPy.
            position: Posição das aletas no foguete (m).
        """

        if hasattr(rocket, "add_trapezoidal_fins"):
            rocket.add_trapezoidal_fins(
                n=self.n,
                root_chord=self.root_chord,
                tip_chord=self.tip_chord,
                span=self.span,
                position=position,
                cant_angle=self.cant_angle,
                sweep_length=self.sweep_length,
            )
        else:
            rocket.add_surfaces(self, position)

    def planform_area(self) -> float:
        """Calcula área de cada aleta (aproximação trapezoidal)."""

        return 0.5 * (self.root_chord + self.tip_chord) * self.span

    def export_to_dict(self) -> dict:
        """Exporta parâmetros das aletas para dicionário."""

        return {
            "n": self.n,
            "root_chord": self.root_chord,
            "tip_chord": self.tip_chord,
            "span": self.span,
            "sweep_length": self.sweep_length,
            "cant_angle": self.cant_angle,
        }


class SuperTail(Tail):
    """Tail com geração aleatória de parâmetros."""

    @classmethod
    def generate_random(
        cls,
        ranges: TailRanges,
        rocket_radius: float,
        rng: Optional[random.Random] = None,
    ) -> "SuperTail":
        """Gera um tail aleatório.

        Args:
            ranges: Ranges do tail.
            rocket_radius: Raio do foguete (m).
            rng: Gerador aleatório.

        Returns:
            Instância de SuperTail.
        """

        random_source = rng or random
        length = ranges.length.random(random_source)
        top_radius = rocket_radius * ranges.top_radius_ratio.random(random_source)
        bottom_radius = rocket_radius * ranges.bottom_radius_ratio.random(random_source)

        return cls(
            length=length,
            top_radius=top_radius,
            bottom_radius=bottom_radius,
            rocket_radius=rocket_radius,
        )

    def add_to_rocket(self, rocket, position: float) -> None:
        """Adiciona o tail ao foguete.

        Args:
            rocket: Instância de RocketPy.
            position: Posição do tail no foguete (m).
        """

        if hasattr(rocket, "add_tail"):
            rocket.add_tail(
                top_radius=self.top_radius,
                bottom_radius=self.bottom_radius,
                length=self.length,
                position=position,
            )
        else:
            rocket.add_surfaces(self, position)

    def export_to_dict(self) -> dict:
        """Exporta parâmetros do tail para dicionário."""

        return {
            "length": self.length,
            "top_radius": self.top_radius,
            "bottom_radius": self.bottom_radius,
        }
