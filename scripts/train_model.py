from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from superrocketlib.models.scalers import LogMinMaxScaler

from superrocketlib import DEFAULT_CONFIG, SMALL_CONFIG, STANDARD_COMPETITION_CONFIG
from superrocketlib.core import IntRange, Range, RocketConfigRanges
from superrocketlib.models.model import first_stage_model, second_stage_model


LOG_CANDIDATES = {
	"simulations.flight.apogee",
	"environment.max_expected_height",
	"motor.thrust_curve.total_impulse",
	"motor.thrust_curve.avg_thrust",
	"motor.thrust_curve.max_thrust",
	"motor.average_thrust",
	"inertia_11",
	"inertia_22",
	"inertia_33",
	"motor.dry_inertia_11",
	"motor.dry_inertia_22",
	"motor.dry_inertia_33",
	"parachute.cd_s",
}


def _coerce_ok_column(values: pd.Series) -> pd.Series:
	if values.dtype == bool:
		return values
	return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _filter_columns(columns: Iterable[str]) -> list[str]:
	excluded_prefixes = ("simulations.", "environment.", "flight_config.")
	excluded_columns = {"ok", "error", "simulations.flight.apogee"}
	filtered = []
	for column in columns:
		if column in excluded_columns:
			continue
		if any(column.startswith(prefix) for prefix in excluded_prefixes):
			continue
		filtered.append(column)
	return filtered


def _numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
	numeric = df.copy()
	for column in columns:
		numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
	numeric = numeric[columns]
	numeric = numeric.dropna(axis=1, how="all")
	numeric = numeric.fillna(numeric.median(numeric_only=True))
	return numeric


def _range_bounds(range_obj: object) -> Optional[Tuple[float, float]]:
	if isinstance(range_obj, Range):
		return float(range_obj.min), float(range_obj.max)
	if isinstance(range_obj, IntRange):
		return float(range_obj.min), float(range_obj.max)
	return None


def _range_product(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
	values = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
	return float(min(values)), float(max(values))


def _build_range_mapping(config: RocketConfigRanges) -> dict:
	rocket = config.rocket
	motor = config.motor
	nosecone = config.nosecone
	fins = config.fins
	tail = config.tail
	parachute = config.parachute
	rail = config.rail_buttons

	rocket_radius = _range_bounds(rocket.radius) or (0.0, 1.0)
	wall_thickness = _range_bounds(rocket.wall_thickness) or (0.0, 0.0)
	inner_min = max(0.0, rocket_radius[0] - wall_thickness[1])
	inner_max = max(0.0, rocket_radius[1] - wall_thickness[0])

	avg_thrust = _range_bounds(motor.average_thrust) or (0.0, 1.0)
	burn_time = _range_bounds(motor.burn_time) or (0.0, 1.0)
	peak_thrust_ratio = _range_bounds(motor.peak_thrust_ratio) or (1.0, 1.0)

	impulse_min, impulse_max = _range_product(avg_thrust, burn_time)
	max_thrust_min, max_thrust_max = _range_product(avg_thrust, peak_thrust_ratio)

	top_ratio = _range_bounds(tail.top_radius_ratio) or (0.0, 1.0)
	bottom_ratio = _range_bounds(tail.bottom_radius_ratio) or (0.0, 1.0)
	top_radius = _range_product(rocket_radius, top_ratio)
	bottom_radius = _range_product(rocket_radius, bottom_ratio)

	mapping = {
		"radius": rocket_radius,
		"mass": _range_bounds(rocket.mass),
		"rocket_length": _range_bounds(rocket.length),
		"rocket_inner_radius": (inner_min, inner_max),
		"wall_thickness": wall_thickness,
		"center_of_mass_without_motor": _range_bounds(rocket.center_of_mass_without_motor),
		"inertia_11": _range_bounds(rocket.I_11_without_motor),
		"inertia_22": _range_bounds(rocket.I_22_without_motor),
		"inertia_33": _range_bounds(rocket.I_33_without_motor),
		"inertia_12": _range_bounds(rocket.I_12_without_motor),
		"inertia_13": _range_bounds(rocket.I_13_without_motor),
		"inertia_23": _range_bounds(rocket.I_23_without_motor),
		"motor_position": _range_bounds(rocket.motor_position),
		"motor.motor_diameter": _range_bounds(motor.motor_diameter),
		"motor.motor_length": _range_bounds(motor.motor_length),
		"motor.nozzle_radius": _range_bounds(motor.nozzle_radius),
		"motor.throat_radius": _range_bounds(motor.throat_radius),
		"motor.dry_mass": _range_bounds(motor.dry_mass),
		"motor.dry_inertia_11": _range_bounds(motor.dry_I_11),
		"motor.dry_inertia_22": _range_bounds(motor.dry_I_22),
		"motor.dry_inertia_33": _range_bounds(motor.dry_I_33),
		"motor.dry_inertia_12": _range_bounds(motor.dry_I_12),
		"motor.dry_inertia_13": _range_bounds(motor.dry_I_13),
		"motor.dry_inertia_23": _range_bounds(motor.dry_I_23),
		"motor.grain_number": _range_bounds(motor.grain_number),
		"motor.grain_density": _range_bounds(motor.grain_density),
		"motor.grain_outer_radius": _range_bounds(motor.grain_outer_radius),
		"motor.grain_initial_inner_radius": _range_bounds(motor.grain_initial_inner_radius),
		"motor.grain_initial_height": _range_bounds(motor.grain_initial_height),
		"motor.grain_separation": _range_bounds(motor.grain_separation),
		"motor.grains_center_of_mass_position": _range_bounds(motor.grains_center_of_mass_position),
		"motor.center_of_dry_mass_position": _range_bounds(motor.center_of_dry_mass_position),
		"motor.burn_time": burn_time,
		"motor.average_thrust": avg_thrust,
		"motor.peak_thrust_ratio": peak_thrust_ratio,
		"motor.ignition_duration_fraction": _range_bounds(motor.ignition_duration_fraction),
		"motor.tail_off_fraction": _range_bounds(motor.tail_off_fraction),
		"motor.main_burn_end_ratio": _range_bounds(motor.main_burn_end_ratio),
		"motor.thrust_curve_points": _range_bounds(motor.thrust_curve_points),
		"motor.thrust_curve.total_impulse": (impulse_min, impulse_max),
		"motor.thrust_curve.burn_time": burn_time,
		"motor.thrust_curve.avg_thrust": avg_thrust,
		"motor.thrust_curve.max_thrust": (max_thrust_min, max_thrust_max),
		"motor.thrust_curve.peak_time": (0.0, burn_time[1]),
		"nosecone.length": _range_bounds(nosecone.length),
		"nosecone.bluffness": _range_bounds(nosecone.bluffness),
		"nosecone.power": _range_bounds(nosecone.power),
		"fins.n": _range_bounds(fins.n),
		"fins.root_chord": _range_bounds(fins.root_chord),
		"fins.tip_chord": _range_bounds(fins.tip_chord),
		"fins.span": _range_bounds(fins.span),
		"fins.sweep_length": _range_bounds(fins.sweep_length),
		"fins.cant_angle": _range_bounds(fins.cant_angle),
		"tail.length": _range_bounds(tail.length),
		"tail.top_radius": top_radius,
		"tail.bottom_radius": bottom_radius,
		"parachute.cd_s": _range_bounds(parachute.cd_s),
		"parachute.trigger": _range_bounds(parachute.trigger_altitude),
		"parachute.sampling_rate": _range_bounds(parachute.sampling_rate),
		"parachute.lag": _range_bounds(parachute.lag),
		"parachute.noise[0]": _range_bounds(parachute.noise_mean),
		"parachute.noise[1]": _range_bounds(parachute.noise_std),
		"parachute.noise[2]": _range_bounds(parachute.noise_time_correlation),
		"rail_buttons.upper_button_position": _range_bounds(rail.upper_button_position),
		"rail_buttons.lower_button_position": _range_bounds(rail.lower_button_position),
		"rail_buttons.angular_position": _range_bounds(rail.angular_position),
	}

	return {key: value for key, value in mapping.items() if value is not None}


def _build_global_scaler(
	config: RocketConfigRanges,
	columns: Sequence[str],
	fallback_df: pd.DataFrame,
) -> LogMinMaxScaler:
	range_mapping = _build_range_mapping(config)
	min_values: list[float] = []
	max_values: list[float] = []
	for column in columns:
		bounds = range_mapping.get(column)
		if bounds is None:
			series = pd.to_numeric(fallback_df[column], errors="coerce")
			col_min = float(np.nanmin(series.values)) if series.size else 0.0
			col_max = float(np.nanmax(series.values)) if series.size else 1.0
			if np.isnan(col_min) or np.isnan(col_max):
				col_min, col_max = 0.0, 1.0
			if col_min == col_max:
				col_max = col_min + 1e-6
			bounds = (col_min, col_max)
		span = float(bounds[1]) - float(bounds[0])
		if span == 0.0:
			span = max(1e-6, abs(float(bounds[0])) * 0.1)
		margin = 0.1 * span
		min_values.append(float(bounds[0]) - margin)
		max_values.append(float(bounds[1]) + margin)

	scaler = LogMinMaxScaler(columns, LOG_CANDIDATES)
	scaler.fit(np.vstack([min_values, max_values]))
	return scaler


def _select_scaler_config(name: str) -> RocketConfigRanges:
	if name == "small":
		return SMALL_CONFIG
	if name == "standard":
		return STANDARD_COMPETITION_CONFIG
	return DEFAULT_CONFIG


def train_stage1(
	df: pd.DataFrame,
	validation_split: float,
	early_stopping_rounds: Optional[int],
	xgb_params: Optional[dict],
) -> first_stage_model:
	stage1 = first_stage_model(dataset=df)
	stage1.train(
		validation_split=validation_split,
		random_state=42,
		early_stopping_rounds=early_stopping_rounds,
		xgb_params=xgb_params,
	)
	return stage1


def train_stage2(
	df: pd.DataFrame,
	stage1: first_stage_model,
	epochs: int,
	batch_size: int,
	learning_rate: float,
	latent_dim: int,
	hidden_dim: int,
	device: torch.device,
	test_df: pd.DataFrame,
	early_stop_patience: Optional[int],
	early_stop_min_delta: float,
	config: RocketConfigRanges,
):
	macro_features = stage1.macro_features
	macro_df = _numeric_frame(df, macro_features)

	full_columns = _filter_columns(df.columns)
	full_df = _numeric_frame(df, full_columns)
	full_test_df = _numeric_frame(test_df, full_columns)

	full_scaler = _build_global_scaler(config, full_df.columns, full_df)
	full_scaled = full_scaler.transform(full_df.values)
	full_test_scaled = full_scaler.transform(full_test_df.values)

	condition_scaled = stage1.scaler_x.transform(macro_df.values)
	macro_test_df = _numeric_frame(test_df, macro_features)
	condition_test_scaled = stage1.scaler_x.transform(macro_test_df.values)

	x_tensor = torch.tensor(full_scaled, dtype=torch.float32)
	c_tensor = torch.tensor(condition_scaled, dtype=torch.float32)

	x_train, x_val, c_train, c_val = train_test_split(
		x_tensor,
		c_tensor,
		test_size=0.1,
		random_state=42,
	)

	train_loader = torch.utils.data.DataLoader(
		torch.utils.data.TensorDataset(x_train, c_train),
		batch_size=batch_size,
		shuffle=True,
	)
	val_loader = torch.utils.data.DataLoader(
		torch.utils.data.TensorDataset(x_val, c_val),
		batch_size=batch_size,
		shuffle=False,
	)

	model = second_stage_model(
		input_dim=x_tensor.shape[1],
		condition_dim=len(macro_features),
		latent_dim=latent_dim,
		hidden_dim=hidden_dim,
	).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	history = {"train": [], "val": []}
	best_state = None
	best_val = None
	patience_counter = 0

	for epoch in range(1, epochs + 1):
		model.train()
		train_loss = 0.0
		for batch_x, batch_c in train_loader:
			batch_x = batch_x.to(device)
			batch_c = batch_c.to(device)
			optimizer.zero_grad()
			recon_x, mu, logvar = model(batch_x, batch_c)
			loss = model.loss_function(recon_x, batch_x, mu, logvar)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()

		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for batch_x, batch_c in val_loader:
				batch_x = batch_x.to(device)
				batch_c = batch_c.to(device)
				recon_x, mu, logvar = model(batch_x, batch_c)
				loss = model.loss_function(recon_x, batch_x, mu, logvar)
				val_loss += loss.item()

		train_loss /= max(1, len(train_loader.dataset))
		val_loss /= max(1, len(val_loader.dataset))
		history["train"].append(train_loss)
		history["val"].append(val_loss)
		print(f"Epoch {epoch}/{epochs} | loss={train_loss:.6f} | val={val_loss:.6f}")

		if early_stop_patience is not None:
			if best_val is None or val_loss < best_val - early_stop_min_delta:
				best_val = val_loss
				best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
				patience_counter = 0
			else:
				patience_counter += 1
				if patience_counter >= early_stop_patience:
					print(f"Early stop ativado na época {epoch}")
					break

	if best_state is not None:
		model.load_state_dict(best_state)

	model.eval()
	with torch.no_grad():
		x_test = torch.tensor(full_test_scaled, dtype=torch.float32, device=device)
		c_test = torch.tensor(condition_test_scaled, dtype=torch.float32, device=device)
		recon_x, mu, logvar = model(x_test, c_test)
		test_loss = model.loss_function(recon_x, x_test, mu, logvar).item()
		test_loss /= max(1, len(x_test))
		recon_error = torch.mean((recon_x - x_test) ** 2, dim=1).cpu().numpy()

	return model, full_scaler, full_df.columns.tolist(), history, test_loss, recon_error


def evaluate_stage1(
	stage1: first_stage_model,
	test_df: pd.DataFrame,
) -> dict:
	macro_df = _numeric_frame(test_df, stage1.macro_features)
	target = pd.to_numeric(test_df[stage1.target_col], errors="coerce").values.reshape(-1, 1)
	valid_mask = ~np.isnan(target).reshape(-1)
	macro_df = macro_df.iloc[valid_mask]
	target = target[valid_mask]

	x_scaled = stage1.scaler_x.transform(macro_df.values)
	y_scaled_pred = stage1.model.predict(x_scaled).reshape(-1, 1)
	y_pred = stage1.scaler_y.inverse_transform(y_scaled_pred).reshape(-1)
	y_true = target.reshape(-1)

	residuals = y_pred - y_true
	mae = float(np.mean(np.abs(residuals)))
	rmse = float(np.sqrt(np.mean(residuals**2)))
	denom = np.clip(np.abs(y_true), 1e-9, None)
	ape = np.abs(residuals) / denom
	mape = float(np.mean(ape))
	precision_10 = float(np.mean(ape <= 0.10))
	ss_res = float(np.sum((y_true - y_pred) ** 2))
	ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
	r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

	return {
		"y_true": y_true,
		"y_pred": y_pred,
		"residuals": residuals,
		"mae": mae,
		"rmse": rmse,
		"mape": mape,
		"precision_10": precision_10,
		"r2": r2,
	}


def _save_stage1_plots(stage1_metrics: dict, output_dir: Path) -> None:
	y_true = stage1_metrics["y_true"]
	y_pred = stage1_metrics["y_pred"]
	residuals = stage1_metrics["residuals"]

	plt.figure(figsize=(6, 6))
	plt.scatter(y_true, y_pred, alpha=0.4, s=10)
	min_val = float(min(y_true.min(), y_pred.min()))
	max_val = float(max(y_true.max(), y_pred.max()))
	plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
	plt.xlabel("Apogeu real")
	plt.ylabel("Apogeu previsto")
	plt.title("Stage 1: Real vs Previsto")
	plt.tight_layout()
	plt.savefig(output_dir / "stage1_pred_vs_true.pdf")
	plt.close()

	plt.figure(figsize=(6, 4))
	plt.hist(residuals, bins=40, alpha=0.8)
	plt.xlabel("Residual (previsto - real)")
	plt.ylabel("Contagem")
	plt.title("Stage 1: Distribuição de Resíduos")
	plt.tight_layout()
	plt.savefig(output_dir / "stage1_residuals.pdf")
	plt.close()


def _save_stage2_plots(history: dict, recon_error: np.ndarray, output_dir: Path) -> None:
	plt.figure(figsize=(6, 4))
	plt.plot(history["train"], label="train")
	plt.plot(history["val"], label="val")
	plt.xlabel("Época")
	plt.ylabel("Loss")
	plt.title("Stage 2: Curva de Loss")
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_dir / "stage2_loss_curve.pdf")
	plt.close()

	plt.figure(figsize=(6, 4))
	plt.hist(recon_error, bins=40, alpha=0.8)
	plt.xlabel("MSE por amostra")
	plt.ylabel("Contagem")
	plt.title("Stage 2: Erro de Reconstrução")
	plt.tight_layout()
	plt.savefig(output_dir / "stage2_reconstruction_error.pdf")
	plt.close()


def main() -> None:
	parser = argparse.ArgumentParser(description="Treina os modelos do SuperRocketLib")
	parser.add_argument("--input", type=str, default="data.csv", help="CSV de treino")
	parser.add_argument("--output-dir", type=str, default="models", help="Pasta de saída")
	parser.add_argument("--stage", type=str, default="all", choices=["1", "2", "all"], help="Stage a treinar")
	parser.add_argument("--stage1-model", type=str, default=None, help="Modelo Stage 1 para treinar só o Stage 2")
	parser.add_argument("--test-split", type=float, default=0.1, help="Proporção do conjunto de teste")
	parser.add_argument("--stage1-early-stop", type=int, default=None, help="Early stop rounds do Stage 1")
	parser.add_argument("--stage1-val-split", type=float, default=0.1, help="Split de validação do Stage 1")
	parser.add_argument("--stage1-n-estimators", type=int, default=None, help="n_estimators do XGBoost")
	parser.add_argument("--stage1-learning-rate", type=float, default=None, help="learning_rate do XGBoost")
	parser.add_argument("--stage1-max-depth", type=int, default=None, help="max_depth do XGBoost")
	parser.add_argument("--stage1-subsample", type=float, default=None, help="subsample do XGBoost")
	parser.add_argument("--stage1-colsample-bytree", type=float, default=None, help="colsample_bytree do XGBoost")
	parser.add_argument("--epochs", type=int, default=150, help="Épocas do CVAE")
	parser.add_argument("--batch-size", type=int, default=256, help="Batch size do CVAE")
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate do CVAE")
	parser.add_argument("--latent-dim", type=int, default=16, help="Latent dim do CVAE")
	parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dim do CVAE")
	parser.add_argument("--stage2-early-stop", type=int, default=20, help="Paciencia do early stop do Stage 2")
	parser.add_argument("--stage2-min-delta", type=float, default=1e-5, help="Delta mínimo para melhora")
	parser.add_argument(
		"--scaler-config",
		type=str,
		default="default",
		choices=["default", "small", "standard"],
		help="Config de ranges para o scaler global",
	)
	parser.add_argument("--device", type=str, default=None, help="cpu/cuda")
	args = parser.parse_args()

	input_path = Path(args.input)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(input_path)
	if "ok" in df.columns:
		df["ok"] = _coerce_ok_column(df["ok"])
		df = df[df["ok"] == True].copy()
	if "simulations.stability_margin.avg_value" in df.columns:
		stability = pd.to_numeric(
			df["simulations.stability_margin.avg_value"], errors="coerce"
		)
		df = df[stability > 1.5].copy()

	macro_features = first_stage_model().macro_features
	macro_cols = [col for col in macro_features if col in df.columns]
	df[macro_cols] = df[macro_cols].apply(pd.to_numeric, errors="coerce")
	df = df.dropna(subset=macro_cols)

	train_df, test_df = train_test_split(df, test_size=args.test_split, random_state=42)

	stage = args.stage
	stage1 = None
	stage1_metrics = None
	history = None
	stage2_test_loss = None
	recon_error = None

	xgb_params = {}
	if args.stage1_n_estimators is not None:
		xgb_params["n_estimators"] = args.stage1_n_estimators
	if args.stage1_learning_rate is not None:
		xgb_params["learning_rate"] = args.stage1_learning_rate
	if args.stage1_max_depth is not None:
		xgb_params["max_depth"] = args.stage1_max_depth
	if args.stage1_subsample is not None:
		xgb_params["subsample"] = args.stage1_subsample
	if args.stage1_colsample_bytree is not None:
		xgb_params["colsample_bytree"] = args.stage1_colsample_bytree
	xgb_params = xgb_params or None

	if stage in {"1", "all"}:
		stage1 = train_stage1(
			train_df,
			validation_split=args.stage1_val_split,
			early_stopping_rounds=args.stage1_early_stop,
			xgb_params=xgb_params,
		)
		stage1_metrics = evaluate_stage1(stage1, test_df)

	if stage == "2":
		if not args.stage1_model:
			raise ValueError("--stage1-model é obrigatório para treinar apenas o Stage 2")
		stage1 = joblib.load(args.stage1_model)

	device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
	if stage in {"2", "all"}:
		scaler_config = _select_scaler_config(args.scaler_config)
		stage2, full_scaler, full_columns, history, stage2_test_loss, recon_error = train_stage2(
			df=train_df,
			stage1=stage1,
			epochs=args.epochs,
			batch_size=args.batch_size,
			learning_rate=args.lr,
			latent_dim=args.latent_dim,
			hidden_dim=args.hidden_dim,
			device=device,
			test_df=test_df,
			early_stop_patience=args.stage2_early_stop,
			early_stop_min_delta=args.stage2_min_delta,
			config=scaler_config,
		)

	stage1_path = output_dir / "stage1_model.joblib"
	stage2_path = output_dir / "stage2_model.pt"
	scaler_path = output_dir / "full_scaler.joblib"
	columns_path = output_dir / "full_columns.joblib"

	if stage in {"1", "all"}:
		joblib.dump(stage1, stage1_path)

	if stage in {"2", "all"}:
		joblib.dump(full_scaler, scaler_path)
		joblib.dump(full_columns, columns_path)
		torch.save(
			{
				"state_dict": stage2.state_dict(),
				"input_dim": stage2.input_dim,
				"condition_dim": stage2.condition_dim,
				"latent_dim": stage2.latent_dim,
				"hidden_dim": stage2.encoder_fc1.out_features,
			},
			stage2_path,
		)

	log_path = output_dir / "test_log.txt"
	with log_path.open("w", encoding="utf-8") as handle:
		if stage in {"1", "all"} and stage1_metrics is not None:
			handle.write("Stage 1 (teste)\n")
			handle.write(f"R2: {stage1_metrics['r2']:.6f}\n")
			handle.write(f"MAE: {stage1_metrics['mae']:.6f}\n")
			handle.write(f"RMSE: {stage1_metrics['rmse']:.6f}\n")
			handle.write(f"MAPE: {stage1_metrics['mape']:.6f}\n")
			handle.write(f"Precision@10%: {stage1_metrics['precision_10']:.6f}\n")
			handle.write("\n")
		if stage in {"2", "all"} and stage2_test_loss is not None:
			handle.write("Stage 2 (teste)\n")
			handle.write(f"Loss: {stage2_test_loss:.6f}\n")
			handle.write(f"Recon MSE (mean): {float(np.mean(recon_error)):.6f}\n")
			handle.write(f"Recon MSE (median): {float(np.median(recon_error)):.6f}\n")

	if stage in {"1", "all"} and stage1_metrics is not None:
		_save_stage1_plots(stage1_metrics, output_dir)
	if stage in {"2", "all"} and history is not None and recon_error is not None:
		_save_stage2_plots(history, recon_error, output_dir)

	print("Modelos salvos em:")
	if stage in {"1", "all"}:
		print(stage1_path)
	if stage in {"2", "all"}:
		print(stage2_path)
		print(scaler_path)
		print(columns_path)
	print("Relatórios de teste e gráficos salvos em:")
	print(log_path)
	if stage in {"1", "all"}:
		print(output_dir / "stage1_pred_vs_true.pdf")
		print(output_dir / "stage1_residuals.pdf")
	if stage in {"2", "all"}:
		print(output_dir / "stage2_loss_curve.pdf")
		print(output_dir / "stage2_reconstruction_error.pdf")


if __name__ == "__main__":
	main()
