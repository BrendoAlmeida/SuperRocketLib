from __future__ import annotations

import argparse
import io
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
from rocketpy import Environment
from scipy.optimize import differential_evolution

from superrocketlib import DEFAULT_CONFIG, SuperRocket
from superrocketlib.models.model import first_stage_model, second_stage_model


def _range_midpoint(range_obj, default: float) -> float:
	if range_obj is None:
		return default
	min_val = getattr(range_obj, "min", None)
	max_val = getattr(range_obj, "max", None)
	if min_val is None or max_val is None:
		return default
	return float(min_val + (max_val - min_val) * 0.5)


def load_system(model_dir: Path, device: torch.device):
	try:
		stage1 = joblib.load(model_dir / "stage1_model.joblib")
		full_scaler = joblib.load(model_dir / "full_scaler.joblib")
		full_columns = joblib.load(model_dir / "full_columns.joblib")
		checkpoint = torch.load(model_dir / "stage2_model.pt", map_location=device)
		stage2 = second_stage_model(
			input_dim=checkpoint["input_dim"],
			condition_dim=checkpoint["condition_dim"],
			latent_dim=checkpoint["latent_dim"],
			hidden_dim=checkpoint["hidden_dim"],
		).to(device)
		stage2.load_state_dict(checkpoint["state_dict"])
		stage2.eval()
		return stage1, stage2, full_scaler, full_columns
	except FileNotFoundError as exc:
		print(f"Erro ao carregar modelos: {exc}")
		sys.exit(1)


def build_environment(config=DEFAULT_CONFIG) -> Environment:
	env = getattr(config, "environment", None)
	return Environment(
		gravity=_range_midpoint(getattr(env, "gravity", None), 9.81),
		latitude=_range_midpoint(getattr(env, "latitude", None), 0.0),
		longitude=_range_midpoint(getattr(env, "longitude", None), 0.0),
		elevation=_range_midpoint(getattr(env, "elevation", None), 0.0),
		max_expected_height=_range_midpoint(
			getattr(env, "max_expected_height", None), 80000.0
		),
	)


def build_flight_kwargs(config=DEFAULT_CONFIG) -> dict:
	flight = getattr(config, "flight", None)
	return {
		"rail_length": _range_midpoint(getattr(flight, "rail_length", None), 3.0),
		"inclination": _range_midpoint(getattr(flight, "inclination", None), 85.0),
		"heading": _range_midpoint(getattr(flight, "heading", None), 90.0),
		"max_time": _range_midpoint(getattr(flight, "max_time", None), 600.0),
	}


def optimize_macro(stage1: first_stage_model, target_apogee: float):
	target_scaled = stage1.scaler_y.transform([[target_apogee]])[0][0]

	def objective_function(x_norm):
		pred_scaled = stage1.model.predict([x_norm])[0]
		return (pred_scaled - target_scaled) ** 2

	bounds = [(0, 1)] * len(stage1.macro_features)
	result = differential_evolution(
		objective_function,
		bounds,
		strategy="best1bin",
		maxiter=100,
		popsize=20,
		tol=1e-4,
		seed=42,
	)
	best_macro_norm = result.x
	pred_scaled = stage1.model.predict([best_macro_norm])[0]
	est_apogee = stage1.scaler_y.inverse_transform([[pred_scaled]])[0][0]
	return best_macro_norm, est_apogee


def generate_candidates(
	stage1: first_stage_model,
	stage2: second_stage_model,
	full_scaler,
	full_columns,
	macro_norm,
	target_apogee: float,
	device: torch.device,
	n_samples: int,
):
	macro_norm_arr = np.array(macro_norm, dtype=np.float32)
	c_tensor = torch.tensor(macro_norm_arr).unsqueeze(0).repeat(n_samples, 1).to(device)
	z = torch.randn(n_samples, stage2.latent_dim).to(device)
	with torch.no_grad():
		batch_norm = stage2.decode(z, c_tensor).cpu().numpy()
	batch_real = full_scaler.inverse_transform(batch_norm)
	candidates = []
	for row in batch_real:
		rocket_dict = {col: row[idx] for idx, col in enumerate(full_columns)}
		macro_vals = [rocket_dict[feat] for feat in stage1.macro_features]
		macro_input_scaled = stage1.scaler_x.transform([macro_vals])
		pred_scaled = stage1.model.predict(macro_input_scaled)[0]
		validation_apogee = stage1.scaler_y.inverse_transform([[pred_scaled]])[0][0]
		error = abs(validation_apogee - target_apogee)
		rocket_dict["validation_apogee"] = validation_apogee
		rocket_dict["error_score"] = error
		candidates.append(rocket_dict)
	return candidates


def select_top_candidates(candidates: list[dict]) -> list[dict]:
	sorted_candidates = sorted(candidates, key=lambda x: x["error_score"])
	top_picks: list[dict] = []
	if not sorted_candidates:
		return top_picks
	top_picks.append(sorted_candidates[0])
	best_fins = int(round(sorted_candidates[0].get("fins.n", 0)))
	for cand in sorted_candidates[1:]:
		if int(round(cand.get("fins.n", 0))) != best_fins:
			top_picks.append(cand)
			break
	for cand in sorted_candidates[1:]:
		if cand not in top_picks:
			top_picks.append(cand)
			if len(top_picks) >= 3:
				break
	while len(top_picks) < 3 and len(sorted_candidates) > len(top_picks):
		for cand in sorted_candidates:
			if cand not in top_picks:
				top_picks.append(cand)
				break
	return top_picks


def simulate_candidate(rocket_dict: dict, env: Environment, flight_kwargs: dict) -> tuple[float | None, dict]:
	try:
		rocket = SuperRocket.from_dict(rocket_dict)
		# Captura valores reais após ajustes do RocketPy
		if hasattr(rocket, 'nose_component') and rocket.nose_component:
			rocket_dict['nosecone.length'] = rocket.nose_component.length
		if hasattr(rocket, 'fins_component') and rocket.fins_component:
			rocket_dict['fins.span'] = rocket.fins_component.span
			rocket_dict['fins.root_chord'] = rocket.fins_component.root_chord
			rocket_dict['fins.tip_chord'] = rocket.fins_component.tip_chord
		if hasattr(rocket, 'tail_component') and rocket.tail_component:
			rocket_dict['tail.length'] = rocket.tail_component.length
			rocket_dict['tail.top_radius'] = rocket.tail_component.top_radius
			rocket_dict['tail.bottom_radius'] = rocket.tail_component.bottom_radius
		
		result = rocket.simulate(env, **flight_kwargs)
		return (result.get("apogee") if isinstance(result, dict) else None), rocket_dict
	except Exception:
		return None, rocket_dict


def smart_sweep_refinement(
    target: float,
    stage1: first_stage_model,
    stage2: second_stage_model,
    full_scaler,
    full_columns,
    device: torch.device,
    env: Environment,
    flight_kwargs: dict,
    n_samples: int,
):
    """
    Em vez de ajustar iterativamente, dispara em vários alvos prováveis
    e seleciona o melhor resultado físico.
    """
    sweep_targets = [
        target * 1.0
    ]
    
    print(f"📡 Iniciando Varredura (Sweep) nos alvos: {[int(t) for t in sweep_targets]}")
    
    candidates_pool = []
    
    for t in sweep_targets:
        macro_norm, _ = optimize_macro(stage1, t)
        
        batch_samples = max(50, n_samples // 2) 
        batch = generate_candidates(
            stage1, stage2, full_scaler, full_columns,
            macro_norm, t, device, batch_samples
        )
        
        top_batch = select_top_candidates(batch)
        
        for cand in top_batch:
            cand['_aimed_target'] = t
            candidates_pool.append(cand)

    print(f"🧪 Simulando os {len(candidates_pool)} melhores candidatos da varredura...")
    
    validated_candidates = []
    
    for cand in candidates_pool:
        real_apogee = simulate_candidate(cand, env, flight_kwargs)
        
        if real_apogee is not None:
            cand['simulated_apogee'] = real_apogee
            cand['final_error'] = abs(real_apogee - target)
            validated_candidates.append(cand)
            
            symbol = "✅" if cand['final_error'] < 50 else "❌"
            print(f"   Alvo Mira: {cand['_aimed_target']:.0f}m -> Real: {real_apogee:.1f}m (Erro: {cand['final_error']:.1f}m) {symbol}")

    if not validated_candidates:
        print("❌ Nenhuma simulação válida.")
        return None, []

    validated_candidates.sort(key=lambda x: x['final_error'])
    
    best_candidate = validated_candidates[0]
    
    return best_candidate, validated_candidates[:3]


def print_rocket_card(label: str, rocket: dict, target: float) -> None:
	error = rocket["error_score"]
	error_pct = (error / target) * 100 if target else 0.0
	sim_apogee = rocket.get("simulated_apogee")
	print(f"\nOPÇÃO {label}")
	print("-" * 30)
	print(f"Apogeu previsto: {rocket['validation_apogee']:.1f} m")
	print(f"Diferença alvo:  {error:.1f} m ({error_pct:.2f}%)")
	if sim_apogee is not None:
		sim_error = abs(sim_apogee - target)
		sim_error_pct = (sim_error / target) * 100 if target else 0.0
		print(f"Apogeu simulado: {sim_apogee:.1f} m")
		print(f"Erro simulado:  {sim_error:.1f} m ({sim_error_pct:.2f}%)")
	print(f"Massa total:     {rocket['mass']:.3f} kg")
	print(f"Comprimento:     {rocket['rocket_length']:.3f} m")
	print(f"Diâmetro (int):  {rocket['radius'] * 2 * 1000:.1f} mm")
	print(f"Impulso total:   {rocket['motor.thrust_curve.total_impulse']:.1f} Ns")
	print(f"Queima:          {rocket['motor.thrust_curve.burn_time']:.2f} s")
	print(f"Aletas:          {int(round(rocket['fins.n']))}")
	print(f"Envergadura:     {rocket['fins.span'] * 1000:.1f} mm")
	print("-" * 30)


def main() -> None:
	parser = argparse.ArgumentParser(description="Gera foguetes usando a SuperRocketLib AI")
	parser.add_argument("--target", type=float, default=3000.0, help="Apogeu alvo em metros")
	parser.add_argument("--samples", type=int, default=100, help="Número de candidatos")
	parser.add_argument("--models-dir", type=str, default="models", help="Pasta dos modelos")
	parser.add_argument("--device", type=str, default=None, help="cpu/cuda")
	parser.add_argument(
		"--debug",
		action="store_true",
		help="Enable debug mode (shows all warnings)",
	)
	args = parser.parse_args()
	
	# Suprimir warnings do RocketPy quando não estiver em modo debug
	if not args.debug:
		warnings.filterwarnings("ignore", message=".*nose cone length was reduced.*")
		# RocketPy usa print() para warnings, então redirecionamos stdout/stderr
		class SuppressRocketPyWarnings:
			def __init__(self, original_stream):
				self.original_stream = original_stream
				self._pending = ""
			
			def write(self, text):
				self._pending += text
				lines = self._pending.splitlines(keepends=True)
				if lines and not lines[-1].endswith(("\n", "\r")):
					self._pending = lines.pop()
				else:
					self._pending = ""
				for line in lines:
					if "nose cone length was reduced" not in line:
						self.original_stream.write(line)
			
			def flush(self):
				if self._pending:
					if "nose cone length was reduced" not in self._pending:
						self.original_stream.write(self._pending)
					self._pending = ""
				self.original_stream.flush()
		
		sys.stderr = SuppressRocketPyWarnings(sys.stderr)
		sys.stdout = SuppressRocketPyWarnings(sys.stdout)

	device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
	model_dir = Path(args.models_dir)
	stage1, stage2, full_scaler, full_columns = load_system(model_dir, device)

	macro_norm, est_macro = optimize_macro(stage1, args.target)
	print(f"Macro estimado: {est_macro:.1f} m")
	env = build_environment()
	flight_kwargs = build_flight_kwargs()
	best_candidate, best_options = smart_sweep_refinement(
        args.target,
        stage1,
        stage2,
        full_scaler,
        full_columns,
        device,
        env,
        flight_kwargs,
        args.samples,
    )
	if not best_options and best_candidate is not None:
		best_options = [best_candidate]
	for rocket in best_options:
		if "simulated_apogee" not in rocket:
			sim_apogee = simulate_candidate(rocket, env, flight_kwargs)
			if sim_apogee is not None:
				rocket["simulated_apogee"] = sim_apogee

	print(f"\nTOP 3 RESULTADOS PARA {args.target:.0f} METROS")
	labels = ["A", "B", "C"]
	for label, rocket in zip(labels, best_options):
		print_rocket_card(label, rocket, args.target)


if __name__ == "__main__":
	main()