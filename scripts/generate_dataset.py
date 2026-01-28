import argparse
import csv
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Union

from rocketpy import Environment

from superrocketlib import SuperRocket, DEFAULT_CONFIG


logger = logging.getLogger(__name__)

FLIGHT_KWARGS: dict = {}


def _aggregate_curve_stats(curve: list[tuple[float, float]], prefix: str) -> dict:
	"""Converte curva de empuxo em features estatísticas para ML."""
	if not curve:
		return {}

	times = [t for t, _ in curve]
	thrusts = [f for _, f in curve]

	return {
		f"{prefix}.total_impulse": sum(
			(thrusts[i] + thrusts[i + 1]) / 2 * (times[i + 1] - times[i])
			for i in range(len(times) - 1)
		),
		f"{prefix}.burn_time": times[-1] if times else 0,
		f"{prefix}.max_thrust": max(thrusts) if thrusts else 0,
		f"{prefix}.avg_thrust": sum(thrusts) / len(thrusts) if thrusts else 0,
		f"{prefix}.peak_time": times[thrusts.index(max(thrusts))] if thrusts else 0,
	}


def _aggregate_function_stats(function_obj: object, prefix: str) -> dict:
	"""Converte Function do RocketPy em features estatísticas."""
	x_array = getattr(function_obj, "x_array", None)
	y_array = getattr(function_obj, "y_array", None)
	if x_array is None or y_array is None:
		return {}

	curve = list(zip(x_array, y_array))
	if not curve:
		return {}

	values = [value for _, value in curve]
	return {
		f"{prefix}.min_value": min(values),
		f"{prefix}.max_value": max(values),
		f"{prefix}.avg_value": sum(values) / len(values),
		f"{prefix}.start_value": values[0],
		f"{prefix}.end_value": values[-1],
	}


def _aggregate_stability_margin(function_obj: object, prefix: str) -> dict:
	"""Gera estatísticas da margem de estabilidade em pontos amostrados."""
	if not callable(function_obj):
		return {}

	sample_points = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]
	values: list[float] = []
	for mach, time in sample_points:
		try:
			values.append(float(function_obj(mach, time)))
		except Exception:
			continue

	if not values:
		return {}

	return {
		f"{prefix}.min_value": min(values),
		f"{prefix}.max_value": max(values),
		f"{prefix}.avg_value": sum(values) / len(values),
		f"{prefix}.value_mach0_time0": values[0],
	}


def _flatten_params(data: dict, prefix: str = "") -> dict:
	flattened: dict[str, object] = {}
	for key, value in data.items():
		full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
		if key in {"power_on_drag", "power_off_drag", "static_margin"}:
			function_stats = _aggregate_function_stats(value, full_key)
			if function_stats:
				flattened.update(function_stats)
				continue
		if key == "stability_margin":
			stability_stats = _aggregate_stability_margin(value, full_key)
			if stability_stats:
				flattened.update(stability_stats)
				continue
		if (
			key in {"thrust_curve", "power_on_drag", "power_off_drag"}
			and isinstance(value, list)
			and value
			and isinstance(value[0], tuple)
		):
			flattened.update(_aggregate_curve_stats(value, full_key))
		elif isinstance(value, dict):
			flattened.update(_flatten_params(value, full_key))
		elif isinstance(value, (list, tuple)):
			for index, item in enumerate(value):
				item_key = f"{full_key}[{index}]"
				if isinstance(item, dict):
					flattened.update(_flatten_params(item, item_key))
				else:
					flattened[item_key] = item
		elif hasattr(value, "x_array") and hasattr(value, "y_array"):
			function_stats = _aggregate_function_stats(value, full_key)
			if function_stats:
				flattened.update(function_stats)
			else:
				flattened[full_key] = None
		else:
			flattened[full_key] = value
	return flattened


def write_params_to_csv(rows: list[dict], csv_path: Union[str, Path]) -> None:
	path = Path(csv_path)
	path.parent.mkdir(parents=True, exist_ok=True)

	if not rows:
		return

	fieldnames = sorted({key for row in rows for key in row.keys()})

	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _append_rows_to_csv(rows: list[dict], csv_path: Path, fieldnames: list[str]) -> None:
	for row in rows:
		for field in fieldnames:
			row.setdefault(field, None)

	with csv_path.open("a", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writerows(rows)


def build_params(seed: int = 33) -> dict:
	params: dict = {"ok": True, "error": ""}
	rocket = None
	try:
		config = DEFAULT_CONFIG
		rocket = SuperRocket.generate_random(config, seed=seed)

		env_ranges = getattr(config, "environment", None)
		flight_ranges = getattr(config, "flight", None)

		gravity = env_ranges.gravity.random() if env_ranges else None
		latitude = env_ranges.latitude.random() if env_ranges else 0.0
		longitude = env_ranges.longitude.random() if env_ranges else 0.0
		elevation = env_ranges.elevation.random() if env_ranges else 0.0
		max_expected_height = (
			env_ranges.max_expected_height.random() if env_ranges else 80000.0
		)

		rail_length = (
			flight_ranges.rail_length.random()
			if flight_ranges
			else max(3.0, getattr(rocket, "_rocket_length", 0.0) * 1.2)
		)
		inclination = flight_ranges.inclination.random() if flight_ranges else 85.0
		heading = flight_ranges.heading.random() if flight_ranges else 90.0
		max_time = flight_ranges.max_time.random() if flight_ranges else 600.0

		params["environment"] = {
			"gravity": gravity,
			"latitude": latitude,
			"longitude": longitude,
			"elevation": elevation,
			"max_expected_height": max_expected_height,
		}
		params["flight_config"] = {
			"rail_length": rail_length,
			"inclination": inclination,
			"heading": heading,
			"max_time": max_time,
		}

		rocket.simulate_motor()
		rocket.simulate_aerodynamics()

		static_margin = rocket.evaluate_static_margin()
		rocket.simulation_results["static_margin"] = static_margin

		stability_margin = rocket.evaluate_stability_margin()
		rocket.simulation_results["stability_margin"] = stability_margin

		env = Environment(
			gravity=gravity,
			latitude=latitude,
			longitude=longitude,
			elevation=elevation,
			max_expected_height=max_expected_height,
		)
		try:
			flight_summary = rocket.simulate_flight(
				env,
				rail_length=rail_length,
				inclination=inclination,
				heading=heading,
				max_time=max_time,
				**FLIGHT_KWARGS,
			)
			rocket.simulation_results["flight"] = flight_summary
		except Exception as exc:
			params["ok"] = False
			params["error"] = f"flight_error: {exc}"

		params.update(rocket.export_to_dict())
		return params
	except Exception as exc:
		params["ok"] = False
		params["error"] = str(exc)
		if rocket is not None:
			try:
				params.update(rocket.export_to_dict())
			except Exception:
				pass
		return params

def _build_params_worker(seed: int) -> dict:
	return _flatten_params(build_params(seed=seed))


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate rocket dataset CSV")
	parser.add_argument("-n", "--num", type=int, default=1, help="Number of rockets to generate")
	parser.add_argument(
		"-o",
		"--output",
		type=str,
		default=str(Path(__file__).with_suffix(".csv")),
		help="Output CSV path",
	)
	parser.add_argument("--seed", type=int, default=33, help="Base random seed")
	parser.add_argument("--log-every", type=int, default=10, help="Log progress every N rockets")
	parser.add_argument("--log-interval", type=float, default=10.0, help="Log heartbeat every N seconds")
	parser.add_argument(
		"--no-terminate-on-apogee",
		action="store_false",
		dest="terminate_on_apogee",
		default=True,
		help="Do not stop simulation at apogee",
	)
	parser.add_argument(
		"--max-time-step",
		type=float,
		default=0.5,
		help="Maximum integration time step",
	)
	parser.add_argument(
		"--min-time-step",
		type=float,
		default=1e-3,
		help="Minimum integration time step",
	)
	parser.add_argument(
		"--rtol",
		type=float,
		default=1e-4,
		help="Relative tolerance for ODE solver",
	)
	parser.add_argument(
		"--atol",
		type=float,
		default=1e-7,
		help="Absolute tolerance for ODE solver",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=mp.cpu_count(),
		help="Number of worker processes",
	)
	args = parser.parse_args()

	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(message)s",
	)

	global FLIGHT_KWARGS
	FLIGHT_KWARGS = {
		"terminate_on_apogee": args.terminate_on_apogee,
		"max_time_step": args.max_time_step,
		"min_time_step": args.min_time_step,
		"rtol": args.rtol,
		"atol": args.atol,
	}

	num = max(1, args.num)
	seeds = [args.seed + i for i in range(num)]

	output_path = Path(args.output)
	temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

	fieldnames: list[str] | None = None
	batch: list[dict] = []

	processed = 0
	logger.info("Iniciando geração de %s foguetes", num)
	try:
		with mp.Pool(processes=max(1, args.workers)) as pool:
			iterator = pool.imap_unordered(_build_params_worker, seeds, chunksize=1)
			while processed < num:
				try:
					row = iterator.next(timeout=args.log_interval)
				except mp.TimeoutError:
					logger.info("Processando... %s/%s concluídos", processed, num)
					continue
				batch.append(row)
				processed += 1
				if args.log_every > 0 and processed % args.log_every == 0:
					logger.info("Gerados %s/%s foguetes", processed, num)
				if len(batch) >= 10:
					if fieldnames is None:
						fieldnames = sorted({key for item in batch for key in item.keys()})
						with temp_path.open("w", newline="", encoding="utf-8") as handle:
							writer = csv.DictWriter(handle, fieldnames=fieldnames)
							writer.writeheader()
							writer.writerows(batch)
					else:
						_append_rows_to_csv(batch, temp_path, fieldnames)
					batch.clear()
	except KeyboardInterrupt:
		logger.info("Interrompido pelo usuário. Salvando progresso...")

	if batch:
		if fieldnames is None:
			write_params_to_csv(batch, temp_path)
		else:
			_append_rows_to_csv(batch, temp_path, fieldnames)

	if temp_path.exists():
		if output_path.exists():
			output_path.unlink()
		temp_path.rename(output_path)

	logger.info("Finalizado: %s linhas em %s", processed, output_path)


if __name__ == "__main__":
	main()
