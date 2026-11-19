# Resumen del experimento (generado por scripts/06_resumen.py)

Ganancia: +1.072.500 por cada BAJA+2 estimulado, -27.500 por cada estímulo a un cliente que no es BAJA+2.

## Optimización bayesiana (validación 201907)

| familia | AUC 201907 | árboles | num_leaves | learning_rate | min_data_in_leaf | feature_fraction | bagging_fraction | corte (201907) | ganancia 201907 | intentos |
|---|---|---|---|---|---|---|---|---|---|---|
| F0 nominal | 0,9334 | 889 | 327 | 0,0058 | 1.447 | 0,69 | 0,84 | 4.400 | $304,7M | 50 |
| F4 proporciones | 0,9332 | 1.097 | 301 | 0,0052 | 72 | 0,70 | 0,96 | 5.500 | $305,2M | 50 |

## Test 201909 (reentreno 201903-201907, mismas semillas en todas las familias)

| familia | semillas | ganancia (corte de validación) | ganancia máxima (optimista) | AUC 201909 | dif. vs F0 | gana en | p Wilcoxon | ¿supera a F0? |
|---|---|---|---|---|---|---|---|---|
| F0 nominal | 20 | $253,5M ± $2,8M | $256,9M ± $2,1M | 0,9338 ± 0,0003 |  |  |  |  |
| F4 proporciones | 20 | $252,7M ± $2,4M | $259,0M ± $2,3M | 0,9331 ± 0,0004 | -$0,8M | 8/20 | 0,268 | no |

## Sensibilidad al corte (exploratorio, no es parte de la regla de decisión)

Ganancia media en 201909 si todas las familias estimularan la misma cantidad de clientes.

| familia | 3.000 | 3.500 | 4.000 | 4.500 | 5.000 | 6.000 |
|---|---|---|---|---|---|---|
| F0 nominal | $250,7M | $253,1M | $252,9M | $253,5M | $252,9M | $252,3M |
| F4 proporciones | $251,1M | $253,1M | $255,3M | $255,1M | $254,9M | $250,9M |
