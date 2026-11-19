# El modelo de 2019 que envejece (generado por scripts/14_modelo_que_envejece.py)

Seguimiento S9 (regla en el README). Modelos finales de cada familia entrenados con 201903-201908 (los de 04), aplicados sin reentrenar a cada mes. 5 semillas; cada familia con el corte de su bayesiana. "Inflación acumulada": IPC del mes contra el IPC promedio de los meses del entrenamiento. En cada celda, la diferencia media contra F0 y en cuántas de las 5 semillas gana.

## La predicción de la prueba de estrés (regla de S9)

Se cumple si la corrección le gana a F0 en al menos 2 de cada 3 meses con más inflación acumulada que su umbral.

| corrección | predicción | meses por encima | meses en que le gana | diferencia media en esos meses | ¿se cumple? |
|---|---|---|---|---|---|
| F2 mediana | le gana a F0 desde +26% | 19 | 11 | $4,9M | no |
| F3 ranking | le gana a F0 desde +99% | 3 | 1 | -$4,8M | no |

El IPC, que según la prueba de estrés recién convendría desde +183%, no debería ganarle en este rango: le gana en 13 de 21 meses, y en todos desde 202009 (+56%). Tampoco se cumple.

## ¿La ventaja de corregir crece con la inflación? (descriptivo)

| corrección | meses en que le gana a F0 | diferencia media (todos los meses) | correlación con la inflación acumulada (Spearman) | p |
|---|---|---|---|---|
| F1 IPC | 13 de 21 | $13,4M | 0,86 | < 0,001 |
| F2 mediana | 11 de 21 | $3,4M | 0,70 | < 0,001 |
| F3 ranking | 8 de 21 | -$2,6M | 0,08 | 0,741 |

## Mes a mes

| mes | inflación acumulada | BAJA+2 del mes | F0 (sin corregir) | F1 IPC - F0 | F2 mediana - F0 | F3 ranking - F0 |
|---|---|---|---|---|---|---|
| 201910 | +18% | 632 | $289,7M | -$9,2M (0/5) | -$3,4M (0/5) | -$5,8M (1/5) |
| 201911 | +23% | 791 | $377,7M | -$6,2M (0/5) | -$18,4M (0/5) | -$22,8M (0/5) |
| 201912 | +27% | 601 | $306,5M | -$19,1M (0/5) | -$5,6M (0/5) | -$11,8M (0/5) |
| 202001 | +30% | 521 | $265,5M | -$11,4M (0/5) | -$3,6M (0/5) | -$2,8M (1/5) |
| 202002 | +33% | 213 | $34,5M | -$17,8M (0/5) | -$7,2M (0/5) | $0,3M (3/5) |
| 202003 | +37% | 398 | $160,4M | -$7,9M (0/5) | $2,8M (5/5) | $7,8M (5/5) |
| 202004 | +39% | 524 | $178,2M | $13,6M (5/5) | -$2,5M (1/5) | -$2,1M (1/5) |
| 202005 | +42% | 566 | $222,9M | $22,0M (5/5) | $1,4M (2/5) | $1,6M (3/5) |
| 202006 | +45% | 610 | $201,5M | -$2,9M (0/5) | $35,1M (5/5) | $14,6M (5/5) |
| 202007 | +48% | 541 | $166,3M | $5,5M (5/5) | -$2,3M (0/5) | -$7,6M (0/5) |
| 202008 | +52% | 434 | $68,6M | -$6,4M (0/5) | $3,0M (5/5) | $4,1M (5/5) |
| 202009 | +56% | 531 | $84,9M | $7,5M (5/5) | $0,6M (2/5) | $2,8M (4/5) |
| 202010 | +62% | 519 | $79,2M | $7,3M (4/5) | -$0,1M (3/5) | -$0,6M (3/5) |
| 202011 | +67% | 675 | $131,3M | $12,3M (5/5) | -$1,0M (2/5) | -$6,0M (0/5) |
| 202012 | +74% | 661 | $136,2M | $18,0M (5/5) | -$2,1M (2/5) | -$4,3M (1/5) |
| 202101 | +81% | 850 | $251,7M | $45,3M (5/5) | $9,6M (5/5) | -$19,0M (0/5) |
| 202102 | +87% | 1.018 | $238,0M | $44,2M (5/5) | $19,5M (5/5) | -$8,2M (0/5) |
| 202103 | +96% | 960 | $228,6M | $42,2M (5/5) | $17,9M (5/5) | $18,8M (5/5) |
| 202104 | +104% | 1.139 | $343,9M | $55,0M (5/5) | $9,3M (4/5) | -$1,4M (3/5) |
| 202105 | +111% | 870 | $141,2M | $40,5M (5/5) | $16,2M (5/5) | $10,4M (5/5) |
| 202106 | +117% | 1.098 | $334,6M | $48,0M (5/5) | $1,4M (3/5) | -$23,4M (0/5) |

## Exploratorio: con otros cortes (no es parte de la regla)

| corte | corrección | meses por encima de su umbral en que le gana a F0 | diferencia media en esos meses |
|---|---|---|---|
| todas con 4.000 clientes | F2 mediana | 13 de 19 | $6,2M |
| todas con 4.000 clientes | F3 ranking | 3 de 3 | $21,6M |
| mejor corte de cada semilla (mira el test) | F2 mediana | 13 de 19 | $5,6M |
| mejor corte de cada semilla (mira el test) | F3 ranking | 3 de 3 | $20,3M |

## Por qué el corte de la bayesiana confunde (exploratorio, no es parte de la regla)

Cada familia estimula siempre la misma cantidad de clientes: F0 4.000, IPC 5.000, mediana 3.900 y ranking 3.500. Pero el mejor corte de F0 (media de las 5 semillas) pasa de 4.000 clientes en 201910 a 9.800 en promedio en 2021, porque hay casi el doble de BAJA+2. Con un corte fijo, la familia que estimula más clientes gana sola en 2021 (el IPC) y la que estimula menos pierde sola (el ranking), corrijan o no. Por eso se miran también el AUC, que no depende del corte, y la ganancia con el mejor corte de cada semilla (que mira el test, igual para todas).

## Sin depender del corte (exploratorio)

Diferencia media contra F0 en cada tramo de inflación acumulada y en cuántos meses del tramo le gana.

| corrección | medida | meses en que le gana a F0 | hasta +30% (201910-202001) | +33% a +74% (202002-202012) | +81% a +117% (202101-202106) | Spearman con la inflación |
|---|---|---|---|---|---|---|
| F1 IPC | AUC (milésimas) | 18 de 21 | 0,0 (2 de 4) | 4,0 (10 de 11) | 10,0 (6 de 6) | 0,87 (p < 0,001) |
| F2 mediana | AUC (milésimas) | 20 de 21 | 0,8 (4 de 4) | 5,0 (10 de 11) | 10,4 (6 de 6) | 0,80 (p < 0,001) |
| F3 ranking | AUC (milésimas) | 19 de 21 | 0,2 (3 de 4) | 3,5 (10 de 11) | 11,0 (6 de 6) | 0,86 (p < 0,001) |
| F1 IPC | ganancia con el mejor corte | 16 de 21 | -$5,5M (0 de 4) | $5,8M (10 de 11) | $14,4M (6 de 6) | 0,73 (p < 0,001) |
| F2 mediana | ganancia con el mejor corte | 13 de 21 | -$3,2M (0 de 4) | $3,8M (7 de 11) | $12,2M (6 de 6) | 0,73 (p < 0,001) |
| F3 ranking | ganancia con el mejor corte | 16 de 21 | -$3,6M (1 de 4) | $4,7M (9 de 11) | $19,0M (6 de 6) | 0,80 (p < 0,001) |
| F1 IPC | ganancia con 4.000 clientes | 15 de 21 | -$9,3M (0 de 4) | $4,7M (9 de 11) | $14,0M (6 de 6) | 0,65 (p = 0,002) |
| F2 mediana | ganancia con 4.000 clientes | 13 de 21 | -$7,7M (0 de 4) | $3,0M (7 de 11) | $16,0M (6 de 6) | 0,75 (p < 0,001) |
| F3 ranking | ganancia con 4.000 clientes | 13 de 21 | -$6,4M (0 de 4) | $1,9M (8 de 11) | $18,8M (5 de 6) | 0,72 (p < 0,001) |
| F1 IPC | ganancia con el corte de la bayesiana (la regla) | 13 de 21 | -$11,5M (0 de 4) | $4,7M (7 de 11) | $45,9M (6 de 6) | 0,86 (p < 0,001) |
| F2 mediana | ganancia con el corte de la bayesiana (la regla) | 11 de 21 | -$7,8M (0 de 4) | $2,5M (5 de 11) | $12,3M (6 de 6) | 0,70 (p < 0,001) |
| F3 ranking | ganancia con el corte de la bayesiana (la regla) | 8 de 21 | -$10,8M (0 de 4) | $1,0M (6 de 11) | -$3,8M (2 de 6) | 0,08 (p = 0,741) |

## AUC mes a mes (milésimas de AUC de la corrección menos las de F0)

| mes | inflación acumulada | mejor corte de F0 | F1 IPC - F0 | F2 mediana - F0 | F3 ranking - F0 |
|---|---|---|---|---|---|
| 201910 | +18% | 4.000 | -0,4 (2/5) | 1,2 (5/5) | 0,1 (3/5) |
| 201911 | +23% | 6.400 | -0,2 (1/5) | 0,1 (4/5) | -0,8 (0/5) |
| 201912 | +27% | 4.100 | 0,5 (3/5) | 1,1 (5/5) | 1,2 (5/5) |
| 202001 | +30% | 3.500 | 0,2 (3/5) | 0,6 (5/5) | 0,4 (4/5) |
| 202002 | +33% | 1.900 | -0,7 (0/5) | 0,9 (5/5) | -0,1 (3/5) |
| 202003 | +37% | 3.500 | 2,4 (5/5) | 1,7 (5/5) | 1,7 (5/5) |
| 202004 | +39% | 4.200 | 3,0 (5/5) | 0,1 (2/5) | 1,8 (5/5) |
| 202005 | +42% | 6.000 | 1,4 (5/5) | -0,2 (1/5) | 0,1 (3/5) |
| 202006 | +45% | 3.200 | 11,5 (5/5) | 22,3 (5/5) | 10,5 (5/5) |
| 202007 | +48% | 4.400 | 2,0 (5/5) | 2,4 (5/5) | 1,6 (5/5) |
| 202008 | +52% | 3.100 | 6,2 (5/5) | 6,9 (5/5) | 5,4 (5/5) |
| 202009 | +56% | 5.000 | 3,8 (5/5) | 3,3 (5/5) | 1,6 (5/5) |
| 202010 | +62% | 5.900 | 4,2 (5/5) | 4,8 (5/5) | 3,9 (5/5) |
| 202011 | +67% | 9.900 | 3,6 (5/5) | 3,6 (5/5) | 3,1 (5/5) |
| 202012 | +74% | 8.600 | 7,3 (5/5) | 9,3 (5/5) | 9,2 (5/5) |
| 202101 | +81% | 9.400 | 6,5 (5/5) | 8,2 (5/5) | 8,6 (5/5) |
| 202102 | +87% | 9.100 | 9,3 (5/5) | 11,4 (5/5) | 10,9 (5/5) |
| 202103 | +96% | 9.900 | 10,8 (5/5) | 9,1 (5/5) | 10,1 (5/5) |
| 202104 | +104% | 10.400 | 9,8 (5/5) | 9,6 (5/5) | 9,8 (5/5) |
| 202105 | +111% | 9.800 | 9,5 (5/5) | 10,4 (5/5) | 11,0 (5/5) |
| 202106 | +117% | 10.000 | 13,8 (5/5) | 13,9 (5/5) | 15,5 (5/5) |

## Control de hiperparámetros (criterio escrito antes de correr)

Las familias difieren en la corrección y también en los hiperparámetros de su bayesiana. Cada variante se compara con el modelo sin corregir (F0), mes y semilla pareados. Criterio, con el AUC medio de 2021: ganar al menos la mitad de lo que gana el IPC con sus propios hiperparámetros.

| variante | medida | hasta +30% (201910-202001) | +33% a +74% (202002-202012) | +81% a +117% (202101-202106) |
|---|---|---|---|---|
| IPC con sus propios hiperparámetros (F1, el de arriba) | AUC (milésimas) | 0,0 (2 de 4) | 4,0 (10 de 11) | 10,0 (6 de 6) |
| IPC con los hiperparámetros de F0 | AUC (milésimas) | 0,2 (2 de 4) | 3,3 (10 de 11) | 10,1 (6 de 6) |
| F0 con cada mes a precios del entrenamiento (un solo factor) | AUC (milésimas) | 0,4 (4 de 4) | 3,5 (11 de 11) | 10,6 (6 de 6) |
| F0 con cada monto deflactado con el IPC de su mes | AUC (milésimas) | -0,5 (0 de 4) | 3,4 (10 de 11) | 10,6 (6 de 6) |
| IPC con sus propios hiperparámetros (F1, el de arriba) | ganancia con el mejor corte | -$5,5M (0 de 4) | $5,8M (10 de 11) | $14,4M (6 de 6) |
| IPC con los hiperparámetros de F0 | ganancia con el mejor corte | -$5,8M (0 de 4) | $5,5M (11 de 11) | $16,2M (6 de 6) |
| F0 con cada mes a precios del entrenamiento (un solo factor) | ganancia con el mejor corte | -$4,3M (0 de 4) | $5,5M (10 de 11) | $18,9M (6 de 6) |
| F0 con cada monto deflactado con el IPC de su mes | ganancia con el mejor corte | -$14,9M (0 de 4) | $5,0M (9 de 11) | $19,8M (6 de 6) |
| IPC con sus propios hiperparámetros (F1, el de arriba) | ganancia con 4.000 clientes (el corte de F0) | -$9,3M (0 de 4) | $4,7M (9 de 11) | $14,0M (6 de 6) |
| IPC con los hiperparámetros de F0 | ganancia con 4.000 clientes (el corte de F0) | -$9,9M (0 de 4) | $4,3M (9 de 11) | $13,2M (6 de 6) |
| F0 con cada mes a precios del entrenamiento (un solo factor) | ganancia con 4.000 clientes (el corte de F0) | -$5,3M (0 de 4) | $3,5M (9 de 11) | $13,6M (6 de 6) |
| F0 con cada monto deflactado con el IPC de su mes | ganancia con 4.000 clientes (el corte de F0) | -$17,4M (0 de 4) | $3,3M (8 de 11) | $17,1M (5 de 6) |

- **IPC con los hiperparámetros de F0**: 10,1 milésimas de AUC en 2021, contra 10,0 del IPC con sus propios hiperparámetros. Gana al menos la mitad: la ventaja viene de la corrección, no de los hiperparámetros.
- **F0 con cada mes a precios del entrenamiento (un solo factor)**: 10,6 milésimas. Gana al menos la mitad: alcanza con deflactar al predecir, sin reentrenar.
- **F0 con cada monto deflactado con el IPC de su mes** (como la prueba directa de S4): 10,6 milésimas.
