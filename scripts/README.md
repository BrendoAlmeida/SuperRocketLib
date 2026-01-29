# Scripts

Este diretório contém utilitários de linha de comando para geração de dados e treinamento dos modelos.

## generate_dataset.py
Gera um dataset sintético executando simulações do RocketPy a partir do `DEFAULT_CONFIG`.

Saída:
- CSV com parâmetros do foguete, ambiente, configuração de voo e métricas de simulação.

## train_model.py
Treina a pipeline em dois estágios:

- Stage 1: modelo XGBoost para prever o apogeu a partir de features macro.
- Stage 2: CVAE condicionado pelas features macro para gerar um foguete completo.

Artefatos gerados:
- `stage1_model.joblib`
- `stage2_model.pt`
- `full_scaler.joblib`
- `full_columns.joblib`
- `test_log.txt`
- PDFs com gráficos de avaliação

Observações importantes:
- O scaler global aplica `log1p` para colunas com comportamento exponencial antes de normalizar.
- O conjunto de colunas com log está definido no `LOG_CANDIDATES` dentro do script.
- Use `--scaler-config` para selecionar os ranges base (`default`, `small`, `standard`).

## _juntar_csv.py
Utilitário para unir CSVs gerados separadamente em um único arquivo.
