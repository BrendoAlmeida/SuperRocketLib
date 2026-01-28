# SuperRocketLib

Biblioteca para criação de foguetes com parâmetros gerados aleatoriamente a partir de ranges físicos, baseada no RocketPy.

## Objetivo

- Gerar foguetes completos (rocket + motor + aerodinâmica + recuperação)
- Garantir coerência física entre componentes
- Gerar curvas de empuxo e arrasto a partir de parâmetros
- Manter uma arquitetura extensível para futuras simulações

## Instalação

```bash
pip install superrocketlib
```

## Uso Rápido

```python
from superrocketlib import SuperRocket, DEFAULT_CONFIG

rocket = SuperRocket.generate_random(DEFAULT_CONFIG, seed=33)
params = rocket.export_to_dict()

# Simulação
# env deve ser uma instância de Environment do RocketPy
results = rocket.simulate(env, use_monte_carlo=False)
```

## Estrutura do Projeto

- [src/superrocketlib/core/README.md](src/superrocketlib/core/README.md)
- [src/superrocketlib/components/README.md](src/superrocketlib/components/README.md)
- [src/superrocketlib/generators/README.md](src/superrocketlib/generators/README.md)

## Configurações de Ranges

Os ranges padrão estão em [src/superrocketlib/defaults.py](src/superrocketlib/defaults.py).
Use `SMALL_CONFIG`, `STANDARD_COMPETITION_CONFIG` ou crie sua própria configuração com `RocketConfigRanges`.
