---

# Instrucciones generales.

Usar siempre los skills `estilo-python`, `estilo-archivos`
y `estilo-conversacion`.

Migrar el proyecto de forma ordenada a la carpeta `Paper`.

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

### Carpeta principal: `Paper`.

```
Paper/
├── Datos definitivos/      # Bases procesadas finales.
│   ├── Generales.xlsx
│   └── Ballotage.xlsx
├── Resultados/             # Reportes de analisis.
│   ├── Analisis_Descriptivo.txt
│   ├── Resultados_Analisis_Estadisticos.txt
│   ├── Proceso_Modelos.txt
│   └── Resultados_Modelos.txt
├── Paper.md                # Documento principal del paper.
└── Material suplementario/ # Documentos docx adicionales.
```

### Scripts principales.

| Script | Funcion |
|--------|---------|
| `Procesamiento_Bases.py` | Procesa datos crudos y genera bases finales. |
| `Analisis_Descriptivo.py` | Estadisticas descriptivas por categoria. |
| `Analisis_Correlaciones.py` | Correlaciones entre variables. |
| `Analisis_Estadisticos.py` | Tests estadisticos (Wilcoxon, Kruskal-Wallis, etc.). |
| `Modelos.py` | Modelos de regresion (GLM, Robusto, Logistico). |

---

## Variables y estructuras clave.

### Variables de identificacion.

- `ID`: Identificador unico del participante.
- `Fecha`: Fecha de participacion.
- `Timestamp`: Marca temporal de cada respuesta.
- `Categoria_PASO_2023`: Categoria electoral segun voto en PASO.

### Categorias electorales (segun voto en PASO 2023).

- `Left_Wing`: Izquierda (Bregman, De Caño).
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

### Variables para modelos.

**Modelos Robustos (OLS con errores HC3)**:
- Lista: `Variables_Dependientes_Robusto`.
- Variables de cambio de opinion agregadas.

**Modelos GLM**:
- Lista: `Variables_Dependientes_GLM`.
- Indices de Progresismo y Conservadurismo.
- Autopercepciones ideologicas.

**Modelos de Regresion Logistica**:
- Lista: `Candidatos_Cercania`.
- Cercania a cada candidato (Massa, Bullrich, Bregman,
  Milei, Schiaretti).

**Variables Independientes**:
- Lista: `Variables_Independientes`.
- Predictores comunes para todos los modelos.

### Variables a excluir en correlaciones.

- `Orden_IP_Items`: Orden de presentacion de items.
- `Orden_IP_Items_Asociados`: Orden de items asociados.

---

## Flujo de revision del paper.

### Paso 1: Lectura inicial.

- Revisar `Paper` y `Material suplementario` en formato docx.
- Preferir leer `Paper/Paper.md` cuando exista.

### Paso 2: Revision parrafo por parrafo.

Trabajar parrafo por parrafo en las secciones de resultados,
discusion y graficos.

Para cada parrafo:
1. Pegar el texto en el archivo txt de comparacion.
2. Agregar un titulo descriptivo del tipo:
   "Parrafo X - Modelo Y de Variable M".
3. Agregar la comparacion punto por punto usando:
   - Tilde para datos correctos.
   - Cruz para datos incorrectos o discrepantes.

### Paso 3: Verificaciones obligatorias.

**Afirmaciones metodologicas**:
- Verificar que los tests mencionados (Wilcoxon, Kruskal-
  Wallis, Mann-Whitney, Dunn, etc.) coincidan con los
  reportados en los txt de resultados.

**Tipo de modelo**:
- Indicar explicitamente el tipo de modelo cuando aparezca:
  - Modelos GLM (para indices y autopercepciones).
  - Modelos Robustos (para variables de cambio de opinion).
  - Regresion Logistica (para cercania a candidatos).

**Descripcion de variables**:
- En comparaciones, aclarar en una frase que tipo de
  analisis involucra cada variable o conjunto de variables
  para que se entienda el dato.

**Significancia estadistica**:
- Verificar si las variables mencionadas en el texto
  resultan significativas en los modelos reportados.
- Contrastar valores p, coeficientes y otros estadisticos.

---