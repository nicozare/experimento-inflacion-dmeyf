# Prueba de estrés de inflación (generado por scripts/11_estres_inflacion.py)

Seguimiento S7 (README). Montos en pesos de 201910 multiplicados por k (nivel, lags y deltas). Entre paréntesis en cada columna, la inflación total entre el mes promedio del entrenamiento y el test (hoy +18%). Ganancia en el corte de la bayesiana: la pérdida medida en 5 semillas, sumada a la media de las 20; entre paréntesis en cada celda, qué parte de los clientes elegidos son los mismos que sin inflar.

| familia | x1,00 (+18%) | x1,10 (+30%) | x1,25 (+47%) | x1,50 (+77%) | x2,00 (+136%) | x3,00 (+254%) |
|---|---|---|---|---|---|---|
| F0 nominal | $289,0M (100%) | $287,0M (97%) | $285,5M (94%) | $283,5M (91%) | $281,8M (87%) | $276,3M (83%) |
| F4 proporciones | $288,6M (100%) | $287,3M (97%) | $286,6M (94%) | $286,4M (91%) | $287,3M (88%) | $279,6M (83%) |

## Desde cuánta inflación conviene corregir

Con una inflación pareja, F1 (si el IPC la mide), F2 y F3 no cambian: su ganancia es la de las 20 semillas. "Desde" es el primer cruce, interpolando entre los factores probados.

| familia | queda por debajo de | su ganancia (no cambia) | desde una inflación total de |
|---|---|---|---|
| F0 nominal | F1 IPC | $279,6M | +183% |
| F0 nominal | F2 mediana | $287,6M | +26% |
| F0 nominal | F3 ranking | $282,9M | +99% |
| F4 proporciones | F1 IPC | $279,6M | no en este rango |
| F4 proporciones | F2 mediana | $287,6M | +26% |
| F4 proporciones | F3 ranking | $282,9M | +203% |
