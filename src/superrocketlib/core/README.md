# Core

Esta pasta contém as estruturas fundamentais da biblioteca.

## Conteúdo

- `structures.py`
  - `Range`, `IntRange`, `Choice`
  - `RocketRanges`, `MotorRanges`, `NoseConeRanges`, `TrapezoidalFinsRanges`, `TailRanges`, `ParachuteRanges`, `RailButtonsRanges`
  - `RocketConfigRanges`

- `rocket.py`
  - `SuperRocket`: classe principal, gera foguete completo com componentes
  - `simulate()`, `simulate_flight()`, `simulate_monte_carlo()`: simulações

- `validators.py`
  - `RocketValidator`: valida coerência física entre componentes

## Exemplo

```python
from superrocketlib import SuperRocket, DEFAULT_CONFIG

rocket = SuperRocket.generate_random(DEFAULT_CONFIG, seed=33)
results = rocket.simulate(env)
```
