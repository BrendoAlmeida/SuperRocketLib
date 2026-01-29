from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..core import IntRange, Range, RocketConfigRanges
from ..defaults import DEFAULT_CONFIG

class full_rocket_model:
    def __init__(
        self,
        stage1_model,
        stage2_model,
        full_dataset_columns,
        full_scaler,
        config: RocketConfigRanges = DEFAULT_CONFIG,
        manufacturing_constraints: Optional[
            Mapping[str, Union[Range, IntRange, Tuple[float, float], Sequence[float]]]
        ] = None,
    ):
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        self.columns = full_dataset_columns
        self.full_scaler = full_scaler
        self.config = config

        constraints = self._constraints_from_config(config)
        if manufacturing_constraints:
            constraints.update(manufacturing_constraints)
        self.manufacturing_constraints = constraints

    def run(self, target_apogee, user_ranges=None):
        print(f"🚀 Iniciando design para Apogeu: {target_apogee}m")

        print("- Rodando Estágio 1 (Física Macro)...")
        effective_ranges = user_ranges or self.config
        macro_design_dict, estimated_apogee = self.stage1.find_optimal_design(target_apogee, effective_ranges)
        
        print(f"  -> Macro Definido: Massa={macro_design_dict['mass']:.2f}kg, "
              f"Impulso={macro_design_dict['motor.thrust_curve.total_impulse']:.0f}Ns")

        macro_vector = [macro_design_dict[feat] for feat in self.stage1.macro_features]
        
        macro_vector_reshaped = np.array(macro_vector).reshape(1, -1)
        macro_vector_normalized = self.stage1.scaler_x.transform(macro_vector_reshaped)

        print("- Rodando Estágio 2 (Geração de Detalhes)...")
        full_rocket_norm = self.stage2.generate_design(macro_vector_normalized)

        if self.full_scaler is None:
            raise ValueError("full_scaler é obrigatório para desnormalizar o vetor completo.")

        full_rocket_real = self.full_scaler.inverse_transform(full_rocket_norm)

        rocket_candidate = dict(zip(self.columns, full_rocket_real[0]))

        print("- Aplicando restrições de manufatura...")
        final_rocket = self._apply_manufacturing_constraints(rocket_candidate, macro_design_dict)

        return final_rocket

    @staticmethod
    def _constraints_from_config(config: RocketConfigRanges) -> dict:
        return {
            "fins.n": config.fins.n,
            "motor.grain_number": config.motor.grain_number,
            "wall_thickness": config.rocket.wall_thickness,
        }

    @staticmethod
    def _snap_to_range(value: float, range_value: Union[Range, IntRange, Tuple[float, float], Sequence[float]]):
        if isinstance(range_value, IntRange):
            clamped = max(range_value.min, min(value, range_value.max))
            index = round((clamped - range_value.min) / range_value.step)
            return int(range_value.min + index * range_value.step)

        if isinstance(range_value, Range):
            clamped = max(range_value.min, min(value, range_value.max))
            if range_value.step is None:
                return clamped
            index = round((clamped - range_value.min) / range_value.step)
            return range_value.min + index * range_value.step

        if isinstance(range_value, tuple) and len(range_value) == 2:
            return max(range_value[0], min(value, range_value[1]))

        if isinstance(range_value, Sequence):
            return min(range_value, key=lambda x: abs(x - value))

        return value

    def _apply_manufacturing_constraints(self, rocket_dict, macro_locks):
        for key, value in macro_locks.items():
            rocket_dict[key] = value

        for key, rule in self.manufacturing_constraints.items():
            if key in rocket_dict:
                rocket_dict[key] = self._snap_to_range(rocket_dict[key], rule)

        for key in rocket_dict:
            if "number" in key or "n_" in key:
                rocket_dict[key] = int(round(rocket_dict[key]))

        return rocket_dict