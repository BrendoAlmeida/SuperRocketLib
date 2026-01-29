from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class LogMinMaxScaler:
	r"""MinMaxScaler com suporte a log1p por coluna.

	Aplica $\log(1 + x)$ em colunas selecionadas antes de normalizar
	e faz $\exp(x) - 1$ na desnormalização.
	"""

	def __init__(
		self,
		columns: Sequence[str],
		log_columns: Iterable[str],
		*,
		base_scaler: MinMaxScaler | None = None,
	):
		self.columns = list(columns)
		self.log_columns = set(log_columns)
		self.log_indices = [
			index for index, name in enumerate(self.columns) if name in self.log_columns
		]
		self.base_scaler = base_scaler or MinMaxScaler()

	def _apply_log(self, values: np.ndarray) -> np.ndarray:
		if not self.log_indices:
			return values

		logged = np.asarray(values, dtype=float).copy()
		for index in self.log_indices:
			column = logged[:, index]
			column = np.maximum(column, 0.0)
			logged[:, index] = np.log1p(column)
		return logged

	def _apply_exp(self, values: np.ndarray) -> np.ndarray:
		if not self.log_indices:
			return values

		recovered = np.asarray(values, dtype=float).copy()
		for index in self.log_indices:
			recovered[:, index] = np.expm1(recovered[:, index])
		return recovered

	def fit(self, values: np.ndarray) -> "LogMinMaxScaler":
		logged = self._apply_log(values)
		self.base_scaler.fit(logged)
		return self

	def fit_transform(self, values: np.ndarray) -> np.ndarray:
		self.fit(values)
		return self.transform(values)

	def transform(self, values: np.ndarray) -> np.ndarray:
		logged = self._apply_log(values)
		return self.base_scaler.transform(logged)

	def inverse_transform(self, values: np.ndarray) -> np.ndarray:
		recovered = self.base_scaler.inverse_transform(values)
		return self._apply_exp(recovered)

	@property
	def data_min_(self) -> np.ndarray:
		return self.base_scaler.data_min_

	@property
	def data_max_(self) -> np.ndarray:
		return self.base_scaler.data_max_

	@property
	def scale_(self) -> np.ndarray:
		return self.base_scaler.scale_

	@property
	def min_(self) -> np.ndarray:
		return self.base_scaler.min_
