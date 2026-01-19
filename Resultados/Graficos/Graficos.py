"""
Modulo de configuraciones y funciones generales para graficos.

Este modulo contiene constantes, estilos y funciones utilitarias
que son compartidas por todos los notebooks de graficos.

"""

from pathlib import Path

import matplotlib.pyplot as Matplotlib_Pyplot
import pandas as Pandas
import re as Expresiones_Regulares
import seaborn as Seaborn


# ========================================================================
# Constantes de colores y categorias.
# ========================================================================

Colores_Categorias_Paso = {
    "Progressivism": "#0078bf",
    "Moderate_Right_A": "#f7d117",
    "Moderate_Right_B": "#f7d117",
    "Centre": "#009cdd",
    "Left_Wing": "#f65058",
    "Right_Wing_Libertarian": "#753bbd",
}

Categorias_Paso = [
    "Left_Wing",
    "Progressivism",
    "Centre",
    "Moderate_Right_A",
    "Moderate_Right_B",
    "Right_Wing_Libertarian",
]

Items_Progresistas = [5, 6, 9, 11, 16, 20, 24, 25, 27, 28]

Items_Conservadores = [3, 4, 7, 8, 10, 19, 22, 23, 29, 30]

# Diccionario de mapeo para nombres de categorias formateados.
Mapeo_Nombres_Categorias = {
    "Left_Wing": "Left-wing",
    "Progressivism": "Progressivism",
    "Centre": "Centre",
    "Moderate_Right_A": "Moderate Right (a)",
    "Moderate_Right_B": "Moderate Right (b)",
    "Right_Wing_Libertarian": "Libertarian Right-wing",
}


# ========================================================================
# Funciones de configuracion y rutas.
# ========================================================================

def Configurar_Estilo_Grafico():

    """
    Configurar estilo general para graficos.

    Aplica el estilo seaborn y configura la paleta de colores.

    Retorna:
    - Ninguno.

    Ejemplos:
    - Configurar_Estilo_Grafico()

    """

    Matplotlib_Pyplot.style.use("seaborn-v0_8")
    Seaborn.set_palette("husl")
    Seaborn.set_context("notebook")


def Obtener_Ruta_Proyecto():

    """
    Obtener la ruta base del proyecto Paper.

    El archivo esta en Resultados/Graficos, sube 2 niveles.

    Retorna:
    - Ruta_Proyecto: Ruta base del proyecto.

    Ejemplos:
    - Ruta_Proyecto = Obtener_Ruta_Proyecto()

    """

    return Path(__file__).resolve().parents[2]


def Obtener_Ruta_Base_Datos(Nombre_Base: str) -> Path:

    """
    Obtener la ruta del archivo de datos por base.

    Parametros:
    - Nombre_Base (str): "Generales" o "Ballotage".

    Retorna:
    - Ruta_Base (Path): Ruta del archivo de datos.

    Ejemplos:
    - Ruta_Base = Obtener_Ruta_Base_Datos("Generales")

    """

    Ruta_Proyecto = Obtener_Ruta_Proyecto()
    Nombre_Base_Limpio = Nombre_Base.strip().lower()

    Rutas_Bases = {
        "generales": (
            Ruta_Proyecto / "Datos definitivos" / "Generales.xlsx"
        ),
        "ballotage": (
            Ruta_Proyecto / "Datos definitivos" / "Ballotage.xlsx"
        ),
    }

    Ruta_Base = Rutas_Bases.get(Nombre_Base_Limpio)
    if Ruta_Base is None:
        raise ValueError(
            "Nombre_Base invalido. Usar Generales o Ballotage."
        )

    return Ruta_Base


def Obtener_Ruta_Carpeta_Graficos(Subcarpeta: str = "") -> Path:

    """
    Crear y devolver la carpeta de salida de graficos.

    Parametros:
    - Subcarpeta (str): Nombre opcional de subcarpeta.

    Retorna:
    - Ruta_Carpeta (Path): Ruta de salida para imagenes.

    Ejemplos:
    - Ruta = Obtener_Ruta_Carpeta_Graficos()
    - Ruta = Obtener_Ruta_Carpeta_Graficos("Cleveland")

    """

    Ruta_Proyecto = Obtener_Ruta_Proyecto()
    if Subcarpeta:
        Ruta_Carpeta = (
            Ruta_Proyecto / "Resultados" / "Graficos" / Subcarpeta
        )
    else:
        Ruta_Carpeta = Ruta_Proyecto / "Resultados" / "Graficos"
    Ruta_Carpeta.mkdir(parents=True, exist_ok=True)
    return Ruta_Carpeta


def Cargar_Base_Datos(Nombre_Base: str) -> Pandas.DataFrame:

    """
    Cargar base de datos por nombre.

    Parametros:
    - Nombre_Base (str): "Generales" o "Ballotage".

    Retorna:
    - Datos (DataFrame): DataFrame con los datos.

    Ejemplos:
    - Datos = Cargar_Base_Datos("Generales")

    """

    Ruta_Base = Obtener_Ruta_Base_Datos(Nombre_Base)
    return Pandas.read_excel(Ruta_Base)


# ========================================================================
# Funciones de formato y utilidades.
# ========================================================================

def Determinar_Asteriscos(P_Valor: float) -> str:

    """
    Convertir un valor p a asteriscos de significancia.

    Parametros:
    - P_Valor (float): Valor p numerico.

    Retorna:
    - Asteriscos (str): Cadena de asteriscos segun nivel.

    Ejemplos:
    - Determinar_Asteriscos(0.03)  # Retorna "*"
    - Determinar_Asteriscos(0.001) # Retorna "***"

    """

    if P_Valor < 0.001:
        return "***"
    if P_Valor < 0.01:
        return "**"
    if P_Valor < 0.05:
        return "*"
    return ""


def Formatear_Nombre_Categoria(Nombre_Categoria: str) -> str:

    """
    Formatear nombre de categoria con espacios legibles.

    Parametros:
    - Nombre_Categoria (str): Nombre original de la categoria.

    Retorna:
    - Nombre_Formateado (str): Nombre con formato especifico.

    Ejemplos:
    - Formatear_Nombre_Categoria("Left_Wing")
      # Retorna "Left-wing"
    - Formatear_Nombre_Categoria("Moderate_Right_A")
      # Retorna "Moderate Right (a)"

    """

    # Usar mapeo especifico si existe.
    if Nombre_Categoria in Mapeo_Nombres_Categorias:
        return Mapeo_Nombres_Categorias[Nombre_Categoria]

    # Fallback: reemplazar guiones bajos por espacios.
    Nombre_Formateado = Nombre_Categoria.replace("_", " ")
    Nombre_Formateado = Expresiones_Regulares.sub(
        r"(?<!^)([A-Z])",
        r" \1",
        Nombre_Formateado,
    )
    return Nombre_Formateado.strip()


def Formatear_Nombre_Categoria_Mathtext(Nombre_Categoria: str) -> str:

    """
    Formatear nombre de categoria para mathtext de matplotlib.

    Parametros:
    - Nombre_Categoria (str): Nombre original de la categoria.

    Retorna:
    - Nombre_Formateado (str): Nombre con espacios escapados.

    Ejemplos:
    - Formatear_Nombre_Categoria_Mathtext("Left_Wing")
      # Retorna "Left-wing" (sin espacios que escapar)

    """

    Nombre_Formateado = Formatear_Nombre_Categoria(Nombre_Categoria)
    return Nombre_Formateado.replace(" ", r"\ ")
