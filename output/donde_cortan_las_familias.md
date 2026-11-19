# ¿Dónde cortan los árboles en cada familia? (generado por scripts/13_donde_cortan_las_familias.py)

Extensión de S4. Modelo final de cada familia (semilla 100081, entrena 201903-201908). Cortes sobre el nivel y los lags; para cada uno, dónde cae el umbral entre los clientes del entrenamiento que tienen ese valor con el mismo signo. Cada corte pesa según su ganancia; las categorías suman 100% dentro de cada fila.

| familia | columnas | parte de la ganancia de los cortes del modelo | separa el cero | montos bajos (el 25% más bajo) | montos medios | montos altos (el 25% más alto) |
|---|---|---|---|---|---|---|
| F0 nominal | montos en pesos | 37,3% | 4,4% | 60,7% | 24,7% | 10,3% |
| F1 IPC | montos en pesos | 40,0% | 5,7% | 60,7% | 24,6% | 8,9% |
| F2 mediana | montos en pesos | 38,3% | 4,8% | 65,4% | 20,5% | 9,3% |
| F3 ranking | montos en pesos | 38,4% | 3,6% | 62,4% | 23,2% | 10,8% |
| F4 proporciones | montos en pesos | 32,6% | 1,7% | 63,1% | 25,5% | 9,7% |
| F4 proporciones | cocientes | 7,6% | 2,9% | 28,7% | 21,0% | 47,4% |
| F5 diferencias | montos en pesos | 29,2% | 3,2% | 56,0% | 30,2% | 10,5% |
| F5 diferencias | diferencias | 10,1% | 10,9% | 39,2% | 27,8% | 22,2% |
