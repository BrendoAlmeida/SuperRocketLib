from __future__ import annotations

from typing import Optional
import random

from rocketpy import RailButtons

from ..core.structures import RailButtonsRanges


class SuperRailButtons(RailButtons):
    """Rail buttons com geração aleatória de parâmetros."""

    def __init__(
        self,
        upper_button_position: float,
        lower_button_position: float,
        angular_position: float = 45,
        name: str = "Rail Buttons",
        rocket_radius: Optional[float] = None,
    ) -> None:
        upper = float(upper_button_position)
        lower = float(lower_button_position)
        if lower > upper:
            lower, upper = upper, lower

        buttons_distance = abs(upper - lower)
        super().__init__(
            buttons_distance=buttons_distance,
            angular_position=angular_position,
            name=name,
            rocket_radius=rocket_radius,
        )
        self.upper_button_position = upper
        self.lower_button_position = lower

    @classmethod
    def generate_random(
        cls,
        ranges: RailButtonsRanges,
        rng: Optional[random.Random] = None,
    ) -> "SuperRailButtons":
        """Gera um conjunto de rail buttons aleatórios.

        Args:
            ranges: Ranges dos rail buttons.
            rng: Gerador aleatório.

        Returns:
            Instância de SuperRailButtons.
        """

        random_source = rng or random

        upper = ranges.upper_button_position.random(random_source)
        lower = ranges.lower_button_position.random(random_source)
        if lower > upper:
            lower, upper = upper, lower

        angular_position = ranges.angular_position.random(random_source)

        return cls(
            upper_button_position=upper,
            lower_button_position=lower,
            angular_position=angular_position,
        )

    def export_to_dict(self) -> dict:
        """Exporta parâmetros dos rail buttons para dicionário."""

        return {
            "upper_button_position": self.upper_button_position,
            "lower_button_position": self.lower_button_position,
            "angular_position": self.angular_position,
        }
