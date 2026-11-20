# Barrida con ventana móvil (generado por scripts/15_barrida_movil.py)

Seguimiento S10 (regla en el README). Para cada mes de test t: entrena t-7 a t-4, elige el corte estable en t-2, reentrena t-7 a t-2 y mide la ganancia en t. Hiperparámetros de la bayesiana de 2019 de cada familia, 5 semillas. Una familia supera a otra si la diferencia media por mes es positiva y el Wilcoxon pareado por mes da p < 0,05; pierde si es negativa con p < 0,05; si no, empata.

## Resultado de la regla

| comparación | diferencia media por mes | meses en que gana | Wilcoxon p (por mes) | resultado |
|---|---|---|---|---|
| F1 IPC vs F0 nominal | -$6,8M | 2 de 7 | 0,109 | empata |
| F2 mediana vs F0 nominal | -$3,0M | 4 de 7 | 1,000 | empata |
| F3 ranking vs F0 nominal | -$6,9M | 2 de 7 | 0,156 | empata |
| F6 sin las de mucho drift vs F0 nominal | $1,1M | 5 de 7 | 0,312 | empata |
| F7 mediana sin las de mucho drift vs F0 nominal | -$2,4M | 3 de 7 | 0,938 | empata |
| F8 ranking de todo lo que drifta vs F0 nominal | -$6,4M | 2 de 7 | 0,297 | empata |
| F7 mediana sin las de mucho drift vs F2 mediana | $0,6M | 5 de 7 | 0,438 | empata |
| F8 ranking de todo lo que drifta vs F3 ranking | $0,5M | 3 de 7 | 1,000 | empata |

## Mes a mes: diferencia contra F0 (en cuántas semillas gana)

| mes de test | entrena / valida | BAJA+2 del mes | F0 (corte) | F1 IPC | F2 mediana | F3 ranking | F6 sin las de mucho drift | F7 mediana sin las de mucho drift | F8 ranking de todo lo que drifta |
|---|---|---|---|---|---|---|---|---|---|
| 201910 | 201903-201908 / 201908 | 632 | $289,7M (4.000) | -$9,2M (0/5) | -$3,4M (0/5) | -$8,1M (0/5) | $0,1M (3/5) | -$0,8M (3/5) | -$3,3M (0/5) |
| 202001 | 201906-201911 / 201911 | 521 | $238,7M (6.400) | -$10,7M (0/5) | -$19,8M (0/5) | -$4,8M (0/5) | $0,9M (2/5) | -$7,0M (0/5) | -$12,1M (0/5) |
| 202004 | 201909-202002 / 202002 | 524 | $158,8M (2.000) | -$25,6M (0/5) | -$25,7M (0/5) | -$25,9M (0/5) | -$6,9M (0/5) | -$24,1M (0/5) | -$14,7M (0/5) |
| 202007 | 201912-202005 / 202005 | 541 | $187,9M (5.600) | $4,0M (5/5) | $6,4M (5/5) | $4,6M (4/5) | -$0,1M (2/5) | $7,3M (5/5) | $4,3M (5/5) |
| 202010 | 202003-202008 / 202008 | 519 | $94,8M (2.800) | $0,4M (2/5) | $5,2M (5/5) | $4,7M (4/5) | $10,0M (5/5) | $6,9M (5/5) | $4,3M (5/5) |
| 202101 | 202006-202011 / 202011 | 850 | $356,6M (7.600) | -$1,0M (2/5) | $9,9M (5/5) | -$14,7M (0/5) | $1,0M (2/5) | -$5,8M (1/5) | -$19,9M (0/5) |
| 202104 | 202009-202102 / 202102 | 1.139 | $552,0M (9.200) | -$5,2M (0/5) | $6,8M (5/5) | -$4,0M (0/5) | $3,1M (4/5) | $6,8M (4/5) | -$3,2M (1/5) |

## Sin depender del corte (exploratorio)

La regla compara cada familia con su propio corte estable, y el corte cambia mucho de un mes a otro. Para separar el orden de los clientes del corte se miran también el AUC, la ganancia si las dos estimulan la misma cantidad de clientes y la ganancia con el mejor corte de cada semilla (que mira el test, igual para todas). Media de las 5 semillas en cada mes.

| comparación | medida | diferencia media por mes | meses en que gana |
|---|---|---|---|
| F1 IPC vs F0 nominal | AUC (milésimas) | 0,3 | 5 de 7 |
| F1 IPC vs F0 nominal | ganancia con el corte estable de F0 | -$3,5M | 1 de 7 |
| F1 IPC vs F0 nominal | ganancia con el mejor corte de cada semilla (mira el test) | $0,0M | 3 de 7 |
| F1 IPC vs F0 nominal | ganancia con el corte estable de cada una (la regla) | -$6,8M | 2 de 7 |
| F2 mediana vs F0 nominal | AUC (milésimas) | 0,9 | 5 de 7 |
| F2 mediana vs F0 nominal | ganancia con el corte estable de F0 | -$0,3M | 3 de 7 |
| F2 mediana vs F0 nominal | ganancia con el mejor corte de cada semilla (mira el test) | $0,5M | 2 de 7 |
| F2 mediana vs F0 nominal | ganancia con el corte estable de cada una (la regla) | -$3,0M | 4 de 7 |
| F3 ranking vs F0 nominal | AUC (milésimas) | 0,7 | 7 de 7 |
| F3 ranking vs F0 nominal | ganancia con el corte estable de F0 | -$2,4M | 1 de 7 |
| F3 ranking vs F0 nominal | ganancia con el mejor corte de cada semilla (mira el test) | $0,8M | 3 de 7 |
| F3 ranking vs F0 nominal | ganancia con el corte estable de cada una (la regla) | -$6,9M | 2 de 7 |
| F6 sin las de mucho drift vs F0 nominal | AUC (milésimas) | -0,7 | 1 de 7 |
| F6 sin las de mucho drift vs F0 nominal | ganancia con el corte estable de F0 | $0,4M | 3 de 7 |
| F6 sin las de mucho drift vs F0 nominal | ganancia con el mejor corte de cada semilla (mira el test) | $0,4M | 3 de 7 |
| F6 sin las de mucho drift vs F0 nominal | ganancia con el corte estable de cada una (la regla) | $1,1M | 5 de 7 |
| F7 mediana sin las de mucho drift vs F0 nominal | AUC (milésimas) | 0,4 | 4 de 7 |
| F7 mediana sin las de mucho drift vs F0 nominal | ganancia con el corte estable de F0 | $0,5M | 4 de 7 |
| F7 mediana sin las de mucho drift vs F0 nominal | ganancia con el mejor corte de cada semilla (mira el test) | $0,3M | 3 de 7 |
| F7 mediana sin las de mucho drift vs F0 nominal | ganancia con el corte estable de cada una (la regla) | -$2,4M | 3 de 7 |
| F8 ranking de todo lo que drifta vs F0 nominal | AUC (milésimas) | 0,8 | 7 de 7 |
| F8 ranking de todo lo que drifta vs F0 nominal | ganancia con el corte estable de F0 | -$2,7M | 1 de 7 |
| F8 ranking de todo lo que drifta vs F0 nominal | ganancia con el mejor corte de cada semilla (mira el test) | -$0,6M | 2 de 7 |
| F8 ranking de todo lo que drifta vs F0 nominal | ganancia con el corte estable de cada una (la regla) | -$6,4M | 2 de 7 |
| F7 mediana sin las de mucho drift vs F2 mediana | AUC (milésimas) | -0,5 | 2 de 7 |
| F7 mediana sin las de mucho drift vs F2 mediana | ganancia con el corte estable de F2 | $0,3M | 5 de 7 |
| F7 mediana sin las de mucho drift vs F2 mediana | ganancia con el mejor corte de cada semilla (mira el test) | -$0,2M | 5 de 7 |
| F7 mediana sin las de mucho drift vs F2 mediana | ganancia con el corte estable de cada una (la regla) | $0,6M | 5 de 7 |
| F8 ranking de todo lo que drifta vs F3 ranking | AUC (milésimas) | 0,1 | 6 de 7 |
| F8 ranking de todo lo que drifta vs F3 ranking | ganancia con el corte estable de F3 | -$0,8M | 2 de 7 |
| F8 ranking de todo lo que drifta vs F3 ranking | ganancia con el mejor corte de cada semilla (mira el test) | -$1,4M | 3 de 7 |
| F8 ranking de todo lo que drifta vs F3 ranking | ganancia con el corte estable de cada una (la regla) | $0,5M | 3 de 7 |

## Corte estable de cada familia (clientes a estimular)

| mes de test | valida en | BAJA+2 en validación | BAJA+2 en el test | F0 | F1 | F2 | F3 | F6 | F7 | F8 | mejor corte de F0 (mira el test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 201910 | 201908 | 543 | 632 | 4.000 | 5.000 | 3.900 | 3.400 | 4.100 | 3.700 | 4.800 | 4.000 |
| 202001 | 201911 | 791 | 521 | 6.400 | 7.100 | 7.200 | 6.400 | 6.200 | 6.200 | 7.000 | 2.800 |
| 202004 | 202002 | 213 | 524 | 2.000 | 1.300 | 1.200 | 1.300 | 1.700 | 1.300 | 1.600 | 6.300 |
| 202007 | 202005 | 566 | 541 | 5.600 | 4.400 | 4.800 | 5.000 | 5.500 | 5.000 | 4.900 | 4.200 |
| 202010 | 202008 | 434 | 519 | 2.800 | 2.600 | 2.500 | 2.700 | 2.900 | 2.500 | 2.700 | 4.500 |
| 202101 | 202011 | 675 | 850 | 7.600 | 8.100 | 8.800 | 7.000 | 7.900 | 6.700 | 6.700 | 8.700 |
| 202104 | 202102 | 1.018 | 1.139 | 9.200 | 9.100 | 10.400 | 9.000 | 10.000 | 10.600 | 9.300 | 10.500 |

## Variables con drift en cada ventana (PSI > 0,25 se sacan; no monetarias con PSI > 0,1 se rankean)

| mes de test | sacadas en F6 | sacadas en F7 | rankeadas de más en F8 | las de más drift (PSI sin corregir) |
|---|---|---|---|---|
| 201910 | 20 | 20 | 13 | mrentabilidad_annual, mrentabilidad, mpasivos_margen, ccomisiones_otras |
| 202001 | 14 | 15 | 10 | Visa_fultimo_cierre, Master_fultimo_cierre, cmobile_app_trx, tmobile_app |
| 202004 | 12 | 12 | 13 | Visa_fultimo_cierre, Master_fultimo_cierre, mrentabilidad_annual, mrentabilidad |
| 202007 | 5 | 5 | 7 | Master_fultimo_cierre, Visa_fultimo_cierre, mforex_sell, cforex_sell |
| 202010 | 20 | 19 | 14 | Visa_fultimo_cierre, Master_fultimo_cierre, cmobile_app_trx, internet |
| 202101 | 17 | 17 | 11 | Visa_fultimo_cierre, Master_fultimo_cierre, cmobile_app_trx, internet |
| 202104 | 4 | 3 | 8 | Visa_fultimo_cierre, Master_fultimo_cierre, cmobile_app_trx, mcomisiones_mantenimiento |
