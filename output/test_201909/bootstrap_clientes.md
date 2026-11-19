# Bootstrap de clientes (generado por scripts/10_bootstrap_clientes.py)

Seguimiento S6 (regla en el README). 1.000 remuestreos con reposición de los clientes de 201909, los mismos para todas las familias y semillas; en cada uno, ganancia media de las 20 semillas en el corte de cada familia. El intervalo no incluye el azar de las semillas: es el que mide "error estándar: semillas" (la diferencia media con los clientes fijos, lo que ya medía el Wilcoxon).

| comparación | corte | diferencia observada | intervalo del 95% | remuestreos con diferencia > 0 | error estándar: clientes | error estándar: semillas | ¿robusta al azar de los clientes? |
|---|---|---|---|---|---|---|---|
| F4 proporciones vs F0 nominal | corte de la bayesiana | -$0,8M | -$10,6M a $10,5M | 43% | $5,2M | $0,7M | no |
| F4 proporciones vs F0 nominal | corte estable (S1) | $2,7M | -$7,0M a $11,4M | 72% | $4,7M | $1,0M | no |
