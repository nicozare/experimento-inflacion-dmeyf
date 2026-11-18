# CLAUDE.md: experimento_inflacion/

## Qué es

Experimento colaborativo de DMEyF 2026 sobre **inflación** (tema 02 del temario,
data drifting), grupo **Inflacion_2**: Ana Dominguez, Nicolás Zarewsky (el
usuario) y Valentín Turri; tutor Joaquín I. Aramendía. Entrega: video grupal de
10 minutos como máximo antes del mediodía del lunes 23/11/2026; discusión en
clase el 24/11. Estructura del video que pide la cátedra: hipótesis, diseño
experimental, limitaciones, resultados, conclusiones y qué seguirían probando;
la plantilla además pide sesgos cognitivos de cada integrante y un anexo de
GitHub (scripts, cómo correrlos desde el dataset y salidas originales). Este
README es ese anexo.

El otro grupo del tema (Inflacion_1) deflacta con índices externos por tipo de
variable (IPC, RIPTE, CER, dólar). Este grupo va por el lado opuesto: corregir
con los propios datos del mes (mediana o ranking). F1 (IPC) está como punto de
comparación común con Inflacion_1.

`README.md` tiene la hipótesis, el diseño, las decisiones y los resultados.
Este archivo guarda el contexto y el historial que no van en el README.

## Autocontenido

Como `entrega_competencia_01/`, no importa nada de `scripts/` ni del resto de
`dmeyf/`: tiene su `pyproject.toml`, su `config.py` y busca el dataset en
`data/` (en esta máquina, `data/competencia_02_crudo.csv.gz` es un symlink a
`../../competencia_02_crudo.csv.gz`). Así los compañeros lo pueden clonar y
correr solos. No es (todavía) un repo git.

## Historial de diseño

1. Primera versión (17/11): el usuario dijo que "no hay un baseline per se" y
   que el punto de partida era su mejor modelo (`candidatos/competencia_01`).
   Se armó el pipeline sobre esa receta (aguinaldo, exclusión de 4 columnas con
   drift, submuestreo 50%, hiperparámetros fijos del candidato, ventanas corta y
   larga hasta 202106). Validación de esa etapa que sigue valiendo: las features
   sin corrección de 202106 coinciden celda por celda con el
   `dataset_features.csv` del candidato (salvo `ccajas_depositos`, celda rota, y
   29 clientes con huecos cuya historia anterior a 202103 el candidato no tenía).
2. El mismo día el usuario pasó la consigna de Alejandro Bolaños y el plan que
   armó Valentín: el baseline es el de Ale (solo 2019, entrena
   201903-201906, valida AUC en 201908, test con ganancia en 201910, reentreno
   201903-201908, bayesiana por familia, defaults de LightGBM, bagging_freq >= 1)
   y el plan del grupo es F0 nominal, F1 IPC, F2 mediana, F3 ranking, 20
   semillas, Wilcoxon pareado contra F0, PSI y AUC adversarial. Se rehízo todo
   sobre eso: sin aguinaldo, sin excluir columnas, sin submuestreo, sin celdas
   rotas corregidas. El "octubre" del mensaje de Valentín es 201910.
3. División de tareas propuesta en el plan de Valentín (a confirmar por el
   grupo): Valentín baseline y pipeline común; Ana F1 y F2; Nicolás F3 y
   la medición del drift. Con este repo el pipeline común ya existe.

## Cosas no obvias encontradas

- **El allocator de macOS retenía memoria**: armar las features pidiendo
  arreglos nuevos de ~150 MB por mes hacía crecer el footprint ~1,3 GB por mes
  (16 GB para un método) aunque Python los liberaba. Se resolvió reservando los
  buffers una vez con el tamaño del mes más grande (`02_corregir_y_fe.py`). No
  volver a escribir ese loop con arreglos nuevos por mes.
- **AUC adversarial engañoso**: con IPC o ranking da 1,000 porque las variables
  con pocos valores distintos (límites de tarjeta, que el banco fija por
  escalones) quedan con un valor exacto distinto en cada mes ("huella"). Por eso
  `05_drift.py` reporta también la versión con solo variables continuas. Aun así
  octubre sigue siendo distinguible (0,81 a 0,87) por comportamiento en dólares
  post-PASO, que no es inflación.
- **Variables administradas**: límites y comisión de mantenimiento se ajustan por
  escalones. IPC y ranking les suben mucho el PSI; la mediana no.
- **Celdas rotas en el test**: 201910 tiene 14 variables en 0 para todos; 201905
  tiene 9. El baseline no las corrige (`CORREGIR_CELDAS_ROTAS = False`).
- **IPC**: serie oficial bajada de la API de datos.gob.ar
  (`148.3_INIVELNAL_DICI_M_26`), cargada a mano en `config.IPC`.

## Estado

Ver la sección de resultados del README y `output/resumen.md`. Las corridas
largas (03 y 04) dejan su log en `output/logs/`.
