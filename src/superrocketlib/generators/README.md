# Generators

Geradores de curvas sintéticas.

## ThrustCurveGenerator

Gera uma curva de empuxo baseada em parâmetros do motor (tempo de queima, empuxo médio, pico, tail-off).

## DragCurveGenerator

Gera curvas de arrasto (power-off e power-on) com base em geometria do foguete e das aletas.

## Exemplo

```python
from superrocketlib.generators import ThrustCurveGenerator, ThrustCurveParameters

params = ThrustCurveParameters(
    burn_time=2.0,
    average_thrust=500,
    peak_thrust_ratio=2.0,
    ignition_duration_fraction=0.05,
    tail_off_fraction=0.08,
    main_burn_end_ratio=1.0,
    thrust_profile_type="neutral",
    points=150,
)
curve = ThrustCurveGenerator.generate(params)
```
