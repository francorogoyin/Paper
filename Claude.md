---

# Instrucciones generales.

Usar siempre los skills `estilo-python`, `estilo-archivos`
y `estilo-conversacion`.

---

## Contexto del proyecto.

Proyecto de investigacion cientifica sobre cambio de opinion
politica en el contexto electoral argentino 2023.

**Objetivo principal**: Medir el cambio de opinion sobre items
politizados segun su asociacion a candidatos presidenciales
y segun la ideologia politica del participante.

**Diseño experimental**: Experimentos ecologicos realizados
durante las Elecciones Generales (octubre 2023) y el
Ballotage (noviembre 2023).

**Participantes**:
- Elecciones Generales: 2786 participantes.
- Ballotage: 1254 participantes.

---

## Estructura del proyecto.

```
Paper/
├── Codigo/                 # Scripts de analisis en Python.
│   ├── Procesamiento_Bases.py
│   ├── Analisis_Descriptivo.py
│   ├── Analisis_Estadisticos.py
│   ├── Analisis_Correlaciones.py
│   ├── Modelos.py
│   └── Comparar_Bases.py
├── Datos crudos/           # Datos originales sin procesar.
│   ├── Generales 1-5.csv
│   ├── Ballotage 1-2.csv
│   └── Clusters.xlsx
├── Datos definitivos/      # Bases procesadas finales.
│   ├── Generales.xlsx
│   └── Ballotage.xlsx
├── Resultados/
│   ├── Reportes/           # Reportes de analisis (Reportes_*.txt).
│   ├── Graficos/           # Visualizaciones y script Graficos.py.
│   └── Tablas/             # Tablas exportadas (xlsx).
├── Controles/              # Comparacion con resultados del paper.
├── Paper.md                # Documento principal del paper.
├── Ejecutar_Todo.py        # Script maestro del pipeline.
└── Requirements.txt        # Dependencias del proyecto.
```

---

## Pipeline de ejecucion.

El archivo `Ejecutar_Todo.py` define el orden de ejecucion:

1. `Procesamiento_Bases.py`: Integra csv crudos y genera
   bases finales en Excel.
2. `Analisis_Descriptivo.py`: Estadisticas descriptivas
   por categoria electoral.
3. `Analisis_Estadisticos.py`: Tests Wilcoxon, Kruskal-Wallis,
   Mann-Whitney y Dunn.
4. `Analisis_Correlaciones.py`: Matrices de correlacion
   entre variables.
5. `Modelos.py`: Regresiones GLM, robustas y logisticas.
6. `Comparar_Bases.py`: Validacion de integridad de datos.

---

## Variables y estructuras clave.

### Variables de identificacion.

- `ID`: Identificador unico del participante.
- `Fecha`: Fecha de participacion.
- `Categoria_PASO_2023`: Categoria electoral segun voto en PASO.

### Categorias electorales (segun voto en PASO 2023).

- `Left_Wing`: Izquierda (Bregman, Del Caño).
- `Progressivism`: Progresismo (Grabois, Massa).
- `Centre`: Centro (Schiaretti).
- `Moderate_Right_A`: Derecha moderada - Larreta.
- `Moderate_Right_B`: Derecha moderada - Bullrich.
- `Right_Wing_Libertarian`: Derecha libertaria (Milei).

### Items de ideologia politica (IP_Items).

- `Items_Progresistas`: 10 items (IP_Item_5, 6, 9, 11, 16,
  20, 24, 25, 27, 28).
- `Items_Conservadores`: 10 items (IP_Item_3, 4, 7, 8, 10,
  19, 22, 23, 29, 30).

### Variables de cambio de opinion.

- `CO_*`: Cambio de Opinion (diferencia entre respuesta
  asociada y no asociada).
- `CT_*`: Cambio de Tiempo (diferencia entre tiempos de
  respuesta).

---

## Dependencias.

El proyecto requiere Python 3.10+ con las siguientes librerias
(ver `Requirements.txt`):

- `pandas`: Manipulacion y analisis de datos.
- `numpy`: Computacion numerica.
- `scipy`: Funciones cientificas y estadisticas.
- `statsmodels`: Modelos estadisticos avanzados.
- `scikit-posthocs`: Tests post-hoc para comparaciones multiples.
- `openpyxl`: Lectura y escritura de archivos Excel.

---

## Convenciones de codigo.

### Nomenclatura.

- Usar Pascal_Snake_Case en todas las variables, funciones,
  clases y estructuras.
- Ejemplo: `Variable_Descriptiva`, `Funcion_De_Calculo`.

### Nombres descriptivos.

- Prohibido usar abreviaciones o nombres genericos.
- Incorrecto: `Var`, `Tmp`, `Dec`.
- Correcto: `Variable_Configuracion`, `Diccionario_Usuarios`.

### Idioma.

- Todo el codigo debe estar en español, incluyendo variables,
  funciones, comentarios y docstrings.

### Comentarios.

- Deben ser descriptivos y terminar con punto.
- Correcto: `# El IMC se calcula como peso dividido altura².`
- Incorrecto: `# Calculo.`

### Formato.

- Espacios alrededor del signo `=` (variable = valor).
- Maximo 70 caracteres por linea.
- Usar parentesis o backslash para continuacion de lineas.

### Funciones.

- Deben hacer una sola cosa.
- Nombres deben comenzar con verbo: `Calcular_Total`,
  `Obtener_Datos`, `Generar_Reporte`.

### Docstrings.

Cada funcion debe tener docstring con:
- Descripcion detallada del proposito.
- Ejemplos de uso si es relevante.
- Type hinting en parametros y retorno.

Formato requerido:
```python
def Nombre_Funcion(Parametro: tipo) -> tipo:

    """
    Descripcion de la funcion.

    Parametros y ejemplos.

    """

    # Cuerpo de la funcion.
```

---

## Notas importantes.

- Los reportes se guardan en `Resultados/Reportes/`.
- Los graficos se guardan en `Resultados/Graficos/`.
- El archivo `Agents.md` contiene contexto detallado
  adicional sobre variables y verificaciones.
- Al final de cambios importantes, commitear y pushear.

