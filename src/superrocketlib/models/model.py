from typing import Dict, Mapping, Optional, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy.optimize import differential_evolution
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import Range, RocketConfigRanges, SuperRocket
from .scalers import LogMinMaxScaler


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

class first_stage_model:
    def __init__(self, dataset=None, model=None):
        """
        Inicializa a classe com o dataset bruto.
        """
        self.dataset = dataset
        
        self.macro_features = [
            'mass', 
            'radius', 
            'rocket_length', 
            'motor.thrust_curve.total_impulse',
            'motor.thrust_curve.avg_thrust',
            'motor.thrust_curve.burn_time',
            'fins.span',
            'nosecone.length'
        ]
        self.target_col = 'simulations.flight.apogee'
        
        self.model = model
        self.scaler_x = LogMinMaxScaler(self.macro_features, LOG_CANDIDATES)
        self.scaler_y = LogMinMaxScaler([self.target_col], LOG_CANDIDATES)

    @staticmethod
    def macro_features_from_rocket(rocket: SuperRocket) -> Dict[str, Optional[float]]:
        """Extrai as features macro a partir de um `SuperRocket`."""

        data = rocket.export_to_dict()
        motor = data.get("motor") or {}
        fins = data.get("fins") or {}
        nose = data.get("nosecone") or {}

        burn_time = motor.get("burn_time")
        avg_thrust = motor.get("average_thrust")
        total_impulse = None
        if burn_time is not None and avg_thrust is not None:
            total_impulse = burn_time * avg_thrust

        return {
            "mass": data.get("mass"),
            "radius": data.get("radius"),
            "rocket_length": data.get("rocket_length"),
            "motor.thrust_curve.total_impulse": total_impulse,
            "motor.thrust_curve.avg_thrust": avg_thrust,
            "motor.thrust_curve.burn_time": burn_time,
            "fins.span": fins.get("span"),
            "nosecone.length": nose.get("length"),
        }

    @staticmethod
    def _macro_ranges_from_config(config: RocketConfigRanges) -> Dict[str, Range]:
        """Converte `RocketConfigRanges` para constraints macro do modelo."""

        min_impulse = config.motor.average_thrust.min * config.motor.burn_time.min
        max_impulse = config.motor.average_thrust.max * config.motor.burn_time.max

        return {
            "mass": config.rocket.mass,
            "radius": config.rocket.radius,
            "rocket_length": config.rocket.length,
            "motor.thrust_curve.total_impulse": Range.continuous(min_impulse, max_impulse),
            "motor.thrust_curve.avg_thrust": config.motor.average_thrust,
            "motor.thrust_curve.burn_time": config.motor.burn_time,
            "fins.span": config.fins.span,
            "nosecone.length": config.nosecone.length,
        }

    @staticmethod
    def _range_bounds(range_value: Union[Range, Tuple[float, float]]) -> Tuple[float, float]:
        """Extrai bounds contínuos (min, max) de um Range ou tupla.
        
        Nota: Ignora o atributo 'step' de ranges discretos para facilitar
        a convergência do differential_evolution. Ranges discretos são
        tratados como contínuos durante a otimização.
        """
        if isinstance(range_value, Range):
            return range_value.min, range_value.max
        return range_value

    def train(
        self,
        validation_split: float = 0.1,
        random_state: int = 42,
        early_stopping_rounds: Optional[int] = None,
        xgb_params: Optional[Dict[str, object]] = None,
    ):
        """
        Treina o 'Simulador Rápido' (Surrogate Model).
        Aprende a física: Dado (Massa, Motor, Geo) -> Qual o Apogeu?
        """
        print("--- Iniciando Pré-processamento ---")

        df_clean = self.dataset[self.dataset['ok'] == True].copy()
        
        X = df_clean[self.macro_features]
        y = df_clean[[self.target_col]]

        X_scaled = self.scaler_x.fit_transform(X.values)
        y_scaled = self.scaler_y.fit_transform(y.values)

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled,
            y_scaled,
            test_size=validation_split,
            random_state=random_state,
        )

        print("--- Iniciando Treinamento do XGBoost ---")
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "n_jobs": -1,
        }
        if xgb_params:
            params.update(xgb_params)

        self.model = XGBRegressor(**params)
        
        fit_kwargs: Dict[str, object] = {"eval_set": [(X_val, y_val)], "verbose": False}
        if early_stopping_rounds:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        self.model.fit(X_train, y_train, **fit_kwargs)
        
        score = self.model.score(X_val, y_val)
        print(f"Treinamento Concluído! Acurácia (R²) no set de teste: {score:.4f}")

    def find_optimal_design(
        self,
        target_apogee: float,
        user_constraints: Optional[
            Union[Mapping[str, Union[Range, Tuple[float, float]]], RocketConfigRanges]
        ] = None,
    ):
        """
        Aqui acontece a mágica do 'Design Inverso'.
        Usa o modelo treinado para achar os parâmetros ideais.
        
        Args:
            target_apogee (float): Apogeu desejado em metros (ex: 3000).
            user_constraints: Constraints usando `Range`, tuplas (min, max)
                ou `RocketConfigRanges` da biblioteca.
        """
        if not self.model:
            raise Exception("O modelo ainda não foi treinado! Rode .train() primeiro.")

        target_scaled = self.scaler_y.transform([[target_apogee]])[0][0]

        def objective_function(x_normalized):
            predicted_apogee_scaled = self.model.predict([x_normalized])[0]
            
            error = (predicted_apogee_scaled - target_scaled) ** 2
            return error

        if isinstance(user_constraints, RocketConfigRanges):
            constraints = self._macro_ranges_from_config(user_constraints)
        else:
            constraints = dict(user_constraints or {})

        bounds = []
        for i, feature in enumerate(self.macro_features):
            data_min = float(self.scaler_x.data_min_[i])
            data_max = float(self.scaler_x.data_max_[i])
            
            if feature in constraints:
                user_min, user_max = self._range_bounds(constraints[feature])
                if feature in self.scaler_x.log_columns:
                    user_min = float(np.log1p(max(0.0, user_min)))
                    user_max = float(np.log1p(max(0.0, user_max)))
                norm_min = (user_min - data_min) / (data_max - data_min)
                norm_max = (user_max - data_min) / (data_max - data_min)
                bounds.append((max(0, norm_min), min(1, norm_max)))
            else:
                bounds.append((0, 1))

        print(f"Buscando design ideal para {target_apogee}m...")
        result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=100, popsize=15)

        best_x_normalized = [result.x]
        best_x_real = self.scaler_x.inverse_transform(best_x_normalized)[0]
        
        optimal_design = dict(zip(self.macro_features, best_x_real))
        
        final_apogee_scaled = self.model.predict(best_x_normalized)[0]
        final_apogee_real = self.scaler_y.inverse_transform([[final_apogee_scaled]])[0][0]
        
        return optimal_design, final_apogee_real
    
class second_stage_model(nn.Module):
    def __init__(self, input_dim=150, condition_dim=8, latent_dim=16, hidden_dim=64):
        """
        CVAE: Conditional Variational Autoencoder.
        
        Args:
            input_dim: Número total de parâmetros do foguete (ex: 150).
            condition_dim: Número de parâmetros macro definidos no Estágio 1 (ex: 8).
            latent_dim: Tamanho do vetor de ruído (quanto maior, mais 'criativo').
        """
        super(second_stage_model, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        self.encoder_fc1 = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.encoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_fc1 = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        h = F.relu(self.encoder_fc1(inputs))
        h = F.relu(self.encoder_fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([z, c], dim=1)
        h = F.relu(self.decoder_fc1(inputs))
        h = F.relu(self.decoder_fc2(h))
        
        return torch.sigmoid(self.decoder_out(h)) 

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def generate_design(self, condition_vector):
        """
        Gera um foguete completo dado APENAS a condição macro.
        (Usado na inferência/produção).
        """
        with torch.no_grad():
            c = torch.FloatTensor(condition_vector)
            if c.dim() == 1:
                c = c.unsqueeze(0)

            z = torch.randn(c.size(0), self.latent_dim)
            generated_normalized = self.decode(z, c)

            return generated_normalized.numpy()

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD