# Models

Este módulo implementa a pipeline de ML em dois estágios para o SuperRocketLib.

## Stage 1 — first_stage_model
Modelo XGBoost que aprende a relação entre parâmetros macro do foguete e o apogeu simulado.

Responsabilidades:
- Normalização com `MinMaxScaler`.
- Treinamento supervisionado do surrogate model.
- Otimização inversa com `differential_evolution` usando ranges do `RocketConfigRanges`.

## Stage 2 — second_stage_model
CVAE (Conditional Variational Autoencoder) que gera o vetor completo de parâmetros do foguete a partir do vetor macro.

Responsabilidades:
- Reconstrução condicionada por features macro.
- Geração de amostras coerentes com o condicionamento.

## full_rocket_model
Orquestra a execução completa:
1. Otimiza o design macro pelo Stage 1.
2. Gera o foguete detalhado com o Stage 2.
3. Aplica restrições de manufatura (snap em ranges discretos).

## LogMinMaxScaler
O scaler global está em `models/scalers.py` e aplica `log1p` em colunas com crescimento exponencial antes de normalizar.
Isso melhora a estabilidade numérica e a distribuição das features.

Observações:
- A lista de colunas com log é configurada no script de treino.
- `inverse_transform()` desfaz automaticamente o `log1p` usando `expm1`.
