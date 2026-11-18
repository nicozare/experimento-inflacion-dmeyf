# Bootstrap de clientes (generado por scripts/10_bootstrap_clientes.py)

Seguimiento S6 (regla en el README). 1.000 remuestreos con reposición de los clientes de 201910, los mismos para todas las familias y semillas; en cada uno, ganancia media de las 20 semillas en el corte de cada familia. El intervalo no incluye el azar de las semillas: es el que mide "error estándar: semillas" (la diferencia media con los clientes fijos, lo que ya medía el Wilcoxon).

| comparación | corte | diferencia observada | intervalo del 95% | remuestreos con diferencia > 0 | error estándar: clientes | error estándar: semillas | ¿robusta al azar de los clientes? |
|---|---|---|---|---|---|---|---|
| F1 IPC vs F0 nominal | corte de la bayesiana | -$9,5M | -$17,8M a -$0,4M | 2% | $4,4M | $0,8M | sí |
| F1 IPC vs F0 nominal | corte estable (S1) | -$8,1M | -$15,4M a $0,2M | 3% | $3,9M | $0,8M | no |
| F2 mediana vs F0 nominal | corte de la bayesiana | -$1,4M | -$8,9M a $6,3M | 42% | $3,8M | $0,6M | no |
| F2 mediana vs F0 nominal | corte estable (S1) | -$0,2M | -$8,7M a $7,7M | 49% | $4,2M | $0,6M | no |
| F3 ranking vs F0 nominal | corte de la bayesiana | -$6,2M | -$14,6M a $2,8M | 10% | $4,5M | $0,9M | no |
| F3 ranking vs F0 nominal | corte estable (S1) | -$5,0M | -$15,4M a $4,9M | 17% | $5,2M | $0,9M | no |
| F4 proporciones vs F0 nominal | corte de la bayesiana | -$0,4M | -$5,1M a $5,9M | 47% | $2,7M | $0,8M | no |
| F4 proporciones vs F0 nominal | corte estable (S1) | $2,8M | -$0,3M a $6,1M | 96% | $1,6M | $0,8M | no |
| F2 mediana vs F1 IPC | corte de la bayesiana | $8,1M | -$0,7M a $16,7M | 96% | $4,5M | $0,8M | no |
| F2 mediana vs F1 IPC | corte estable (S1) | $7,9M | -$0,2M a $15,8M | 97% | $4,2M | $0,8M | no |
| F3 ranking vs F1 IPC | corte de la bayesiana | $3,3M | -$9,1M a $13,9M | 70% | $5,8M | $1,0M | no |
| F3 ranking vs F1 IPC | corte estable (S1) | $3,1M | -$8,5M a $12,9M | 70% | $5,5M | $0,9M | no |
