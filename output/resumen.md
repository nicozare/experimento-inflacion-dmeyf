# Resumen del experimento (generado por scripts/06_resumen.py)

Ganancia: +1.072.500 por cada BAJA+2 estimulado, -27.500 por cada estímulo a un cliente que no es BAJA+2.

## Optimización bayesiana (validación 201908)

| familia | AUC 201908 | árboles | num_leaves | learning_rate | min_data_in_leaf | feature_fraction | bagging_fraction | corte (201908) | ganancia 201908 | intentos |
|---|---|---|---|---|---|---|---|---|---|---|
| F0 nominal | 0,9334 | 715 | 432 | 0,0083 | 1.917 | 0,78 | 0,87 | 4.000 | $216,7M | 50 |
| F1 IPC | 0,9331 | 612 | 366 | 0,0099 | 1.457 | 0,50 | 0,79 | 5.000 | $209,0M | 50 |
| F2 mediana | 0,9323 | 1.125 | 322 | 0,0051 | 2.642 | 0,32 | 0,96 | 3.900 | $208,4M | 50 |
| F3 ranking | 0,9316 | 1.223 | 567 | 0,0053 | 2.717 | 0,64 | 0,91 | 3.500 | $212,8M | 50 |
| F4 proporciones | 0,9332 | 603 | 251 | 0,0116 | 2.492 | 0,65 | 0,90 | 4.500 | $212,8M | 50 |
| F5 diferencias | 0,9326 | 484 | 284 | 0,0119 | 1.018 | 0,95 | 0,57 | 4.800 | $215,6M | 50 |

## Test 201910 (reentreno 201903-201908, mismas semillas en todas las familias)

| familia | semillas | ganancia (corte de validación) | ganancia máxima (optimista) | AUC 201910 | dif. vs F0 | gana en | p Wilcoxon | ¿supera a F0? |
|---|---|---|---|---|---|---|---|---|
| F0 nominal | 20 | $289,0M ± $2,2M | $290,5M ± $1,9M | 0,9364 ± 0,0004 |  |  |  |  |
| F1 IPC | 20 | $279,6M ± $3,4M | $283,2M ± $2,4M | 0,9362 ± 0,0004 | -$9,5M | 0/20 | < 0,001 | no |
| F2 mediana | 20 | $287,6M ± $2,1M | $289,7M ± $1,5M | 0,9376 ± 0,0002 | -$1,4M | 7/20 | 0,034 | no |
| F3 ranking | 20 | $282,9M ± $3,0M | $291,2M ± $1,6M | 0,9366 ± 0,0003 | -$6,2M | 2/20 | < 0,001 | no |
| F4 proporciones | 20 | $288,6M ± $2,1M | $293,0M ± $2,1M | 0,9360 ± 0,0003 | -$0,4M | 11/20 | 0,626 | no |
| F5 diferencias | 20 | $288,0M ± $3,2M | $292,2M ± $2,3M | 0,9340 ± 0,0005 | -$1,0M | 7/20 | 0,567 | no |

## Corrección interna contra IPC

| familia | dif. vs F1 IPC | gana en | p Wilcoxon | ¿empata o le gana al IPC? |
|---|---|---|---|---|
| F2 mediana | $8,1M | 20/20 | < 0,001 | sí |
| F3 ranking | $3,3M | 15/20 | 0,003 | sí |

## Proporciones contra diferencias (S8, corte de la bayesiana)

| comparación | dif. media | desvío de F5 | gana en | p Wilcoxon | ¿F4 supera a F5? |
|---|---|---|---|---|---|
| F4 proporciones vs F5 diferencias | $0,6M | $3,2M | 13/20 | 0,561 | no |

## Sensibilidad al corte (exploratorio, no es parte de la regla de decisión)

Ganancia media en 201910 si todas las familias estimularan la misma cantidad de clientes.

| familia | 3.000 | 3.500 | 4.000 | 4.500 | 5.000 | 6.000 |
|---|---|---|---|---|---|---|
| F0 nominal | $278,7M | $284,5M | $289,0M | $286,0M | $282,9M | $278,6M |
| F1 IPC | $274,7M | $276,8M | $279,2M | $280,4M | $279,6M | $276,2M |
| F2 mediana | $279,8M | $286,0M | $287,2M | $285,3M | $281,3M | $279,2M |
| F3 ranking | $275,6M | $282,9M | $288,9M | $288,5M | $286,6M | $282,9M |
| F4 proporciones | $281,5M | $289,1M | $291,3M | $288,6M | $286,5M | $282,4M |
| F5 diferencias | $282,7M | $286,6M | $289,0M | $287,9M | $287,7M | $281,7M |

## Drift de las variables en pesos (201903-201908 contra 201910)

| familia | PSI mediano (todas) | PSI mediano, variables de mercado | PSI mediano, límites y comisión | variables con PSI > 0,1 | AUC adversarial (continuas) | AUC adversarial (todas) |
|---|---|---|---|---|---|---|
| F0 nominal | 0,008 | 0,006 | 0,031 | 6% | 0,810 | 0,892 |
| F1 IPC | 0,004 | 0,003 | 1,014 | 13% | 0,874 | 1,000 |
| F2 mediana | 0,002 | 0,002 | 0,039 | 5% | 0,854 | 0,921 |
| F3 ranking | 0,001 | 0,001 | 0,330 | 11% | 0,867 | 1,000 |
