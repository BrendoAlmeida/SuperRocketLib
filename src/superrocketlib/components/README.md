# Components

Classes de componentes usados no foguete.

## Classes

- `SuperMotor`: motor sólido com curva de empuxo sintética
- `SuperNoseCone`: nose cone com parâmetros aleatórios
- `SuperTrapezoidalFins`: aletas trapezoidais
- `SuperTail`: tail cone
- `SuperParachute`: paraquedas
- `SuperRailButtons`: rail buttons

## Uso

```python
from superrocketlib import SuperMotor, DEFAULT_CONFIG

motor = SuperMotor.generate_random(DEFAULT_CONFIG.motor, rocket_inner_radius=0.05)
```
