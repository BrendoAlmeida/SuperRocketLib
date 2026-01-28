from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import logging


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado de validação de constraints físicos."""

    errors: List[str]
    warnings: List[str]

    def raise_if_errors(self) -> None:
        """Lança ValueError se houver erros."""

        if self.errors:
            raise ValueError("; ".join(self.errors))


class RocketValidator:
    """Valida constraints físicos entre componentes do foguete."""

    @staticmethod
    def validate_motor_fit(
        rocket_inner_radius: float,
        motor_diameter: float,
    ) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        if motor_diameter / 2.0 > rocket_inner_radius:
            errors.append("Motor não cabe no diâmetro interno do foguete.")

        if motor_diameter / 2.0 > rocket_inner_radius * 0.98:
            warnings.append("Motor muito justo no tubo do foguete.")

        return ValidationResult(errors=errors, warnings=warnings)

    @staticmethod
    def validate_geometry(
        rocket_length: float,
        nose_length: float,
        tail_length: float,
        fin_span: float,
        rocket_radius: float,
        fin_root_chord: float,
    ) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        if nose_length + tail_length >= rocket_length:
            errors.append("Nose e tail excedem o comprimento total do foguete.")

        if fin_span > rocket_radius * 2.0:
            errors.append("Span das aletas maior que o recomendado para o raio do foguete.")

        if fin_root_chord > rocket_length * 0.4:
            warnings.append("Root chord das aletas alto para o comprimento do foguete.")

        return ValidationResult(errors=errors, warnings=warnings)

    @staticmethod
    def validate_motor_position(
        motor_position: float,
        motor_length: float,
        rocket_length: float,
    ) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        if motor_position + motor_length > rocket_length:
            errors.append("Motor ultrapassa o comprimento total do foguete.")

        if motor_position < 0:
            errors.append("Posição do motor não pode ser negativa.")

        return ValidationResult(errors=errors, warnings=warnings)

    @staticmethod
    def log_warnings(result: ValidationResult) -> None:
        for warning in result.warnings:
            logger.warning(warning)
