# ¿Dónde cortan los árboles los montos en pesos? (generado por scripts/09_donde_corta.py)

Modelo final de F0 (semilla 100081, entrena 201903-201908). Cada corte pesa según su ganancia en el árbol.

- Los cortes sobre montos (nivel y lags de las variables en pesos) son el 37,3% de la ganancia de todos los cortes; los deltas de esas variables, otro 13,0%.
- Dónde caen los cortes sobre montos (% de su ganancia), según el lugar del umbral entre los clientes que tienen ese monto:

| dónde corta | % de la ganancia |
|---|---|
| separa el cero | 4,4% |
| montos bajos (el 25% más bajo) | 60,7% |
| montos medios | 24,7% |
| montos altos (el 25% más alto) | 10,3% |

- Si todos los montos subieran un 18% (del mes promedio del entrenamiento a octubre), cada corte cambiaría de rama, en promedio, al 1,4% de los clientes; con un 27% (de marzo a octubre), al 2,0%.
- Prueba directa en 201910: se le da al mismo modelo octubre llevado a pesos del entrenamiento (cada monto con el IPC de su mes) y se compara con octubre nominal, en los primeros 4.000 (la ganancia mira el test: solo descriptivo).

| octubre | mismos clientes en los primeros 4.000 | correlación de rangos | ganancia |
|---|---|---|---|
| nominal (como lo usa F0) | 100% | 1,000 | $293,7M |
| montos deflactados, deltas nominales | 96,0% | 0,995 | $287,1M |
| montos y deltas deflactados | 93,5% | 0,991 | $267,3M |
