# -*- coding: utf-8 -*-
"""
Segmented population analysis.

"""

import os
import numpy as Numpy
import pandas as Pandas
from scipy import stats as Estadisticas


# ------------------------------------------------------------
# CONSTANTS.
# ------------------------------------------------------------

Categorias_Validas = [
    "Left_Wing",
    "Progressivism",
    "Centre",
    "Moderate_Right_A",
    "Moderate_Right_B",
    "Right_Wing_Libertarian",
]

Poblacion_Izquierda = ["Left_Wing", "Progressivism"]
Poblacion_Centro = ["Centre"]
Poblacion_Derecha = [
    "Moderate_Right_A",
    "Moderate_Right_B",
    "Right_Wing_Libertarian",
]

Items_Progresistas = [5, 6, 9, 11, 16, 20, 24, 25, 27, 28]
Items_Conservadores = [3, 4, 7, 8, 10, 19, 22, 23, 29, 30]
Todos_Los_Items = Items_Conservadores + Items_Progresistas


# ------------------------------------------------------------
# GENERAL UTILITIES.
# ------------------------------------------------------------

def Cargar_Bases_Datos(Ruta_Datos: str) -> dict:

    """
    Loads General and Ballotage databases.

    Parameters:
    - Ruta_Datos: Path to the folder with the files.

    Returns:
    - Dictionary with dataframes.

    Examples:
    - Cargar_Bases_Datos("Processed_Data")
    """

    Datos = {}

    Ruta_Generales = os.path.join(Ruta_Datos, "Generales.xlsx")
    Ruta_Ballotage = os.path.join(Ruta_Datos, "Ballotage.xlsx")

    if os.path.exists(Ruta_Generales):
        Datos["Generales"] = Pandas.read_excel(Ruta_Generales)

    if os.path.exists(Ruta_Ballotage):
        Datos["Ballotage"] = Pandas.read_excel(Ruta_Ballotage)

    return Datos


def Filtrar_Por_Poblacion(
    Datos_Entrada: Pandas.DataFrame,
    Poblacion: list
) -> Pandas.DataFrame:

    """
    Filters a dataframe by electoral population.

    Parameters:
    - Datos_Entrada: DataFrame with the data.
    - Poblacion: List of valid categories.

    Returns:
    - Filtered DataFrame.

    Examples:
    - Filtrar_Por_Poblacion(Datos, Poblacion_Izquierda)
    """

    return Datos_Entrada[
        Datos_Entrada["Categoria_PASO_2023"].isin(Poblacion)
    ].copy()


def Calcular_Media_Y_Error(Valores: Numpy.ndarray) -> tuple:

    """
    Calculates mean, standard error, and count of valid values.

    Parameters:
    - Valores: Array with data.

    Returns:
    - Tuple with mean, standard error, and valid count.

    Examples:
    - Calcular_Media_Y_Error(Numpy.array([1, 2, 3]))
    """

    Valores_Array = Numpy.array(Valores, dtype = float)
    Valores_Limpios = Valores_Array[
        ~Numpy.isnan(Valores_Array)
    ]
    Cantidad = int(len(Valores_Limpios))

    if Cantidad == 0:
        return Numpy.nan, Numpy.nan, 0, Numpy.array([])

    Media = float(Numpy.mean(Valores_Limpios))
    if Cantidad > 1:
        Error_Estandar = float(
            Estadisticas.sem(Valores_Limpios)
        )
    else:
        Error_Estandar = Numpy.nan

    return Media, Error_Estandar, Cantidad, Valores_Limpios


def Formatear_Numero(
    Valor: float,
    Decimales: int = 3,
    Mostrar_Signo: bool = False
) -> str:

    """
    Formats a number for the report.

    Parameters:
    - Valor: Number to format.
    - Decimales: Number of decimals.
    - Mostrar_Signo: Whether to add a sign for positives.

    Returns:
    - Number formatted as text.

    Examples:
    - Formatear_Numero(0.1234, Decimales = 2)
    """

    if Numpy.isnan(Valor):
        return "NaN"

    Formato = f"{{:.{Decimales}f}}"
    Texto = Formato.format(Valor)

    if Mostrar_Signo and Valor > 0:
        Texto = f"+{Texto}"

    return Texto


def Formatear_P_Valor(Valor_P: float) -> str:

    """
    Formats a p-value for the report.

    Parameters:
    - Valor_P: P-value to format.

    Returns:
    - P-value as text.

    Examples:
    - Formatear_P_Valor(0.004)
    """

    if Numpy.isnan(Valor_P):
        return "NaN"

    if Valor_P < 0.001:
        return f"{Valor_P:.2E}"

    return f"{Valor_P:.4f}"


def Obtener_Significancia(Valor_P: float) -> str:

    """
    Converts a p-value to a significance mark.

    Parameters:
    - Valor_P: P-value.

    Returns:
    - Significance mark.

    Examples:
    - Obtener_Significancia(0.03)
    """

    if Numpy.isnan(Valor_P):
        return ""

    if Valor_P < 0.001:
        return "***"
    if Valor_P < 0.01:
        return "**"
    if Valor_P < 0.05:
        return "*"
    return ""


def Test_Mann_Whitney(
    Valores_Uno: Numpy.ndarray,
    Valores_Dos: Numpy.ndarray,
    Minimo: int = 3
) -> float:

    """
    Applies Mann-Whitney U test with minimum count control.

    Parameters:
    - Valores_Uno: Array with values from group one.
    - Valores_Dos: Array with values from group two.
    - Minimo: Minimum count required per group.

    Returns:
    - Test p-value.

    Examples:
    - Test_Mann_Whitney(Numpy.array([1]), Numpy.array([2]))
    """

    if len(Valores_Uno) < Minimo or len(Valores_Dos) < Minimo:
        return 1.0

    try:
        _, Valor_P = Estadisticas.mannwhitneyu(
            Valores_Uno,
            Valores_Dos,
            alternative = "two-sided"
        )
        return float(Valor_P)
    except Exception:
        return 1.0


def Test_Wilcoxon_Pareado(
    Valores_Uno: Pandas.Series,
    Valores_Dos: Pandas.Series,
    Minimo: int = 10
) -> float:

    """
    Applies paired Wilcoxon with minimum count control.

    Parameters:
    - Valores_Uno: Series with values for condition one.
    - Valores_Dos: Series with values for condition two.
    - Minimo: Minimum count required.

    Returns:
    - Test p-value.

    Examples:
    - Test_Wilcoxon_Pareado(Serie_Uno, Serie_Dos)
    """

    Pares = Pandas.concat([Valores_Uno, Valores_Dos], axis = 1)
    Pares = Pares.dropna()

    if len(Pares) < Minimo:
        return 1.0

    try:
        _, Valor_P = Estadisticas.wilcoxon(
            Pares.iloc[:, 0],
            Pares.iloc[:, 1]
        )
        return float(Valor_P)
    except Exception:
        return 1.0


def Test_Wilcoxon_Contra_Cero(
    Valores: Numpy.ndarray,
    Minimo: int = 3
) -> float:

    """
    Applies Wilcoxon against zero with minimum count control.

    Parameters:
    - Valores: Array with differences.
    - Minimo: Minimum count required.

    Returns:
    - Test p-value.

    Examples:
    - Test_Wilcoxon_Contra_Cero(Numpy.array([0.1, 0.2]))
    """

    if len(Valores) < Minimo:
        return 1.0

    try:
        _, Valor_P = Estadisticas.wilcoxon(
            Valores,
            alternative = "two-sided"
        )
        return float(Valor_P)
    except Exception:
        return 1.0


class Generador_Reporte:

    """
    Generates a txt report with formatted lines.

    Parameters:
    - No parameters.

    Returns:
    - Instance to add lines.

    Examples:
    - Reporte = Generador_Reporte()
    """

    def __init__(self) -> None:
        self.Lineas = []

    def Agregar(self, Texto: str = "") -> None:
        self.Lineas.append(Texto)

    def Agregar_Separador(
        self,
        Caracter: str = "=",
        Largo: int = 70
    ) -> None:
        self.Lineas.append(Caracter * Largo)

    def Guardar(self, Ruta_Archivo: str) -> None:
        with open(Ruta_Archivo, "w", encoding = "utf-8") as Archivo:
            Archivo.write("\n".join(self.Lineas))


# ------------------------------------------------------------
# SECTION 01. OPINION CHANGE BREAKDOWN.
# ------------------------------------------------------------

def Calcular_Media_Cambio(
    Datos_Entrada: Pandas.DataFrame,
    Items: list,
    Direccion: str,
    Tipo_Variable: str
) -> tuple:

    """
    Calculates mean and standard error for CO or CT by items.

    Parameters:
    - Datos_Entrada: Filtered DataFrame.
    - Items: List of item numbers.
    - Direccion: Izq or Der.
    - Tipo_Variable: CO or CT.

    Returns:
    - Tuple with mean, error, count, and values.

    Examples:
    - Calcular_Media_Cambio(Datos, Items, "Izq", "CO")
    """

    Columnas = [
        f"{Tipo_Variable}_Item_{Item}_{Direccion}"
        for Item in Items
    ]
    Columnas_Existentes = [
        Columna for Columna in Columnas
        if Columna in Datos_Entrada.columns
    ]

    if len(Columnas_Existentes) == 0:
        return Numpy.nan, Numpy.nan, 0, Numpy.array([])

    Valores = Datos_Entrada[Columnas_Existentes].values.flatten()
    return Calcular_Media_Y_Error(Valores)


def Calcular_Datos_Desglosados(
    Datos_Entrada: Pandas.DataFrame
) -> dict:

    """
    Calculates CO and CT broken down by population and congruence.

    Parameters:
    - Datos_Entrada: DataFrame with data.

    Returns:
    - Dictionary with results and counts.

    Examples:
    - Calcular_Datos_Desglosados(Datos)
    """

    Datos_Izq = Filtrar_Por_Poblacion(
        Datos_Entrada, Poblacion_Izquierda
    )
    Datos_Der = Filtrar_Por_Poblacion(
        Datos_Entrada, Poblacion_Derecha
    )

    Resultados = {
        "Cantidad_Izquierda": len(Datos_Izq),
        "Cantidad_Derecha": len(Datos_Der)
    }

    Resultados["CO_Cong_Izq"] = Calcular_Media_Cambio(
        Datos_Izq, Items_Progresistas, "Izq", "CO"
    )
    Resultados["CT_Cong_Izq"] = Calcular_Media_Cambio(
        Datos_Izq, Items_Progresistas, "Izq", "CT"
    )
    Resultados["CO_Cong_Der"] = Calcular_Media_Cambio(
        Datos_Der, Items_Conservadores, "Der", "CO"
    )
    Resultados["CT_Cong_Der"] = Calcular_Media_Cambio(
        Datos_Der, Items_Conservadores, "Der", "CT"
    )

    Resultados["CO_Incong_Izq"] = Calcular_Media_Cambio(
        Datos_Izq, Items_Progresistas, "Der", "CO"
    )
    Resultados["CT_Incong_Izq"] = Calcular_Media_Cambio(
        Datos_Izq, Items_Progresistas, "Der", "CT"
    )
    Resultados["CO_Incong_Der"] = Calcular_Media_Cambio(
        Datos_Der, Items_Conservadores, "Izq", "CO"
    )
    Resultados["CT_Incong_Der"] = Calcular_Media_Cambio(
        Datos_Der, Items_Conservadores, "Izq", "CT"
    )

    return Resultados


def Agregar_Seccion_01(
    Reporte: Generador_Reporte,
    Datos: dict
) -> None:

    """
    Adds section 01 to the report.

    Parameters:
    - Reporte: Report generator instance.
    - Datos: Dictionary with dataframes.

    Returns:
    - None.

    Examples:
    - Agregar_Seccion_01(Reporte, Datos)
    """

    Reporte.Agregar_Separador("=")
    Reporte.Agregar(
        "Seccion 01. Cambio de opinion desglosado por poblacion."
    )
    Reporte.Agregar_Separador("=")
    Reporte.Agregar("")

    Datos_Generales = Calcular_Datos_Desglosados(
        Datos["Generales"]
    )
    Datos_Ballotage = Calcular_Datos_Desglosados(
        Datos["Ballotage"]
    )

    Reporte.Agregar(
        "Cantidad de casos por poblacion y eleccion."
    )
    Reporte.Agregar(
        "Generales izquierda: "
        f"{Datos_Generales['Cantidad_Izquierda']}."
    )
    Reporte.Agregar(
        f"Generales derecha: {Datos_Generales['Cantidad_Derecha']}."
    )
    Reporte.Agregar(
        "Ballotage izquierda: "
        f"{Datos_Ballotage['Cantidad_Izquierda']}."
    )
    Reporte.Agregar(
        f"Ballotage derecha: {Datos_Ballotage['Cantidad_Derecha']}."
    )
    Reporte.Agregar("")

    Condiciones = [
        ("Congruente", "Izquierda", "Cong_Izq"),
        ("Congruente", "Derecha", "Cong_Der"),
        ("Incongruente", "Izquierda", "Incong_Izq"),
        ("Incongruente", "Derecha", "Incong_Der"),
    ]

    for Tipo_Variable in ["CO", "CT"]:
        Nombre = "Cambio de opinion" if Tipo_Variable == "CO" else (
            "Cambio de tiempo"
        )
        Reporte.Agregar_Separador("-")
        Reporte.Agregar(f"Analisis de {Nombre}.")
        Reporte.Agregar_Separador("-")
        Reporte.Agregar("")

        Reporte.Agregar("Medias y errores estandar.")
        Encabezado = [
            "Condicion",
            "Poblacion",
            "Media_GE",
            "Error_GE",
            "Media_BA",
            "Error_BA",
        ]
        Reporte.Agregar(" | ".join(Encabezado) + ".")

        for Condicion, Poblacion, Clave in Condiciones:
            Media_Ge, Error_Ge, _, _ = Datos_Generales[
                f"{Tipo_Variable}_{Clave}"
            ]
            Media_Ba, Error_Ba, _, _ = Datos_Ballotage[
                f"{Tipo_Variable}_{Clave}"
            ]
            Fila = [
                Condicion,
                Poblacion,
                Formatear_Numero(Media_Ge),
                Formatear_Numero(Error_Ge),
                Formatear_Numero(Media_Ba),
                Formatear_Numero(Error_Ba),
            ]
            Reporte.Agregar(" | ".join(Fila) + ".")

        Reporte.Agregar("")
        Reporte.Agregar("Tests Mann Whitney.")

        Comparaciones = [
            ("Izq_vs_Der_GE", "Cong_Izq", "Cong_Der", "GE"),
            ("Izq_vs_Der_BA", "Cong_Izq", "Cong_Der", "BA"),
            ("Izq_vs_Der_GE", "Incong_Izq", "Incong_Der", "GE"),
            ("Izq_vs_Der_BA", "Incong_Izq", "Incong_Der", "BA"),
        ]

        for Nombre, Clave_Uno, Clave_Dos, Eleccion in Comparaciones:
            if Eleccion == "GE":
                Valores_Uno = Datos_Generales[
                    f"{Tipo_Variable}_{Clave_Uno}"
                ][3]
                Valores_Dos = Datos_Generales[
                    f"{Tipo_Variable}_{Clave_Dos}"
                ][3]
            else:
                Valores_Uno = Datos_Ballotage[
                    f"{Tipo_Variable}_{Clave_Uno}"
                ][3]
                Valores_Dos = Datos_Ballotage[
                    f"{Tipo_Variable}_{Clave_Dos}"
                ][3]

            Valor_P = Test_Mann_Whitney(Valores_Uno, Valores_Dos)
            Marca = Obtener_Significancia(Valor_P)
            Linea = [
                Nombre,
                Eleccion,
                f"{Clave_Uno} vs {Clave_Dos}",
                Formatear_P_Valor(Valor_P),
                Marca,
            ]
            Reporte.Agregar(" | ".join(Linea) + ".")

        Reporte.Agregar("")
        Reporte.Agregar("Tests GE vs BA.")

        for Condicion, Poblacion, Clave in Condiciones:
            Valores_Ge = Datos_Generales[
                f"{Tipo_Variable}_{Clave}"
            ][3]
            Valores_Ba = Datos_Ballotage[
                f"{Tipo_Variable}_{Clave}"
            ][3]
            Valor_P = Test_Mann_Whitney(Valores_Ge, Valores_Ba)
            Marca = Obtener_Significancia(Valor_P)
            Linea = [
                Condicion,
                Poblacion,
                "GE vs BA",
                Formatear_P_Valor(Valor_P),
                Marca,
            ]
            Reporte.Agregar(" | ".join(Linea) + ".")

        Reporte.Agregar("")


# ------------------------------------------------------------
# SECTION 02. MEAN DIFFERENCE VS TOTAL.
# ------------------------------------------------------------

def Calcular_Area_Sombreada(
    Datos_Entrada: Pandas.DataFrame,
    Poblacion: list,
    Items: list,
    Direccion: str
) -> dict:

    """
    Calculates shaded area versus the unassociated total mean.

    Parameters:
    - Datos_Entrada: DataFrame with data.
    - Poblacion: List of population categories.
    - Items: List of items.
    - Direccion: Izq or Der.

    Returns:
    - Dictionary with differences, mean, error, and detail.

    Examples:
    - Calcular_Area_Sombreada(
        Datos, Poblacion_Izquierda, Items, "Izq"
      )
    """

    Datos_Total = Datos_Entrada[
        Datos_Entrada["Categoria_PASO_2023"].isin(Categorias_Validas)
    ].copy()
    Datos_Poblacion = Filtrar_Por_Poblacion(Datos_Entrada, Poblacion)

    Diferencias = []
    Detalle = []

    for Item in Items:
        Columna_Base = f"IP_Item_{Item}_Respuesta"
        Columna_Asoc = f"IP_Item_{Item}_{Direccion}_Respuesta"
        if Columna_Base in Datos_Total.columns and (
            Columna_Asoc in Datos_Poblacion.columns
        ):
            Media_Total = Datos_Total[Columna_Base].mean()
            Media_Asoc = Datos_Poblacion[Columna_Asoc].mean()
            Diferencia = Media_Asoc - Media_Total
            Diferencias.append(Diferencia)
            Detalle.append({
                "Item": Item,
                "Media_Total": Media_Total,
                "Media_Asoc": Media_Asoc,
                "Diferencia": Diferencia,
            })

    Media, Error, _, Valores = Calcular_Media_Y_Error(Diferencias)

    return {
        "Diferencias": Valores,
        "Media": Media,
        "Error": Error,
        "Detalle": Detalle,
    }


def Agregar_Seccion_02(
    Reporte: Generador_Reporte,
    Datos: dict
) -> None:

    """
    Adds section 02 to the report.

    Parameters:
    - Reporte: Report generator instance.
    - Datos: Dictionary with dataframes.

    Returns:
    - None.

    Examples:
    - Agregar_Seccion_02(Reporte, Datos)
    """

    Reporte.Agregar_Separador("=")
    Reporte.Agregar(
        "Seccion 02. Diferencia de medias vs total sin asociar."
    )
    Reporte.Agregar_Separador("=")
    Reporte.Agregar("")

    Resultados = {}
    for Nombre, Datos_Entrada in Datos.items():
        Resultados[Nombre] = {
            "Verde_Con": Calcular_Area_Sombreada(
                Datos_Entrada, Poblacion_Izquierda,
                Items_Conservadores, "Izq"
            ),
            "Verde_Pro": Calcular_Area_Sombreada(
                Datos_Entrada, Poblacion_Izquierda,
                Items_Progresistas, "Izq"
            ),
            "Naranja_Con": Calcular_Area_Sombreada(
                Datos_Entrada, Poblacion_Derecha,
                Items_Conservadores, "Der"
            ),
            "Naranja_Pro": Calcular_Area_Sombreada(
                Datos_Entrada, Poblacion_Derecha,
                Items_Progresistas, "Der"
            ),
        }

    Reporte.Agregar("Resumen de areas por eleccion.")
    Encabezado = [
        "Area",
        "Media_GE",
        "Error_GE",
        "Media_BA",
        "Error_BA",
    ]
    Reporte.Agregar(" | ".join(Encabezado) + ".")

    for Clave, Nombre in [
        ("Verde_Con", "Verde en conservadores"),
        ("Verde_Pro", "Verde en progresistas"),
        ("Naranja_Con", "Naranja en conservadores"),
        ("Naranja_Pro", "Naranja en progresistas"),
    ]:
        Media_Ge = Resultados["Generales"][Clave]["Media"]
        Error_Ge = Resultados["Generales"][Clave]["Error"]
        Media_Ba = Resultados["Ballotage"][Clave]["Media"]
        Error_Ba = Resultados["Ballotage"][Clave]["Error"]
        Fila = [
            Nombre,
            Formatear_Numero(Media_Ge, Mostrar_Signo = True),
            Formatear_Numero(Error_Ge),
            Formatear_Numero(Media_Ba, Mostrar_Signo = True),
            Formatear_Numero(Error_Ba),
        ]
        Reporte.Agregar(" | ".join(Fila) + ".")

    Reporte.Agregar("")
    Reporte.Agregar("Detalle por item y area.")

    for Clave, Titulo in [
        ("Verde_Con", "Verde en conservadores"),
        ("Verde_Pro", "Verde en progresistas"),
        ("Naranja_Con", "Naranja en conservadores"),
        ("Naranja_Pro", "Naranja en progresistas"),
    ]:
        Reporte.Agregar_Separador("-")
        Reporte.Agregar(f"Area {Titulo}.")
        Reporte.Agregar_Separador("-")
        for Eleccion in ["Generales", "Ballotage"]:
            Reporte.Agregar(f"Eleccion {Eleccion}.")
            Encabezado = [
                "Item",
                "Media_Total",
                "Media_Asoc",
                "Diferencia",
            ]
            Reporte.Agregar(" | ".join(Encabezado) + ".")
            for Detalle in Resultados[Eleccion][Clave]["Detalle"]:
                Fila = [
                    str(Detalle["Item"]),
                    Formatear_Numero(Detalle["Media_Total"]),
                    Formatear_Numero(Detalle["Media_Asoc"]),
                    Formatear_Numero(
                        Detalle["Diferencia"], Mostrar_Signo = True
                    ),
                ]
                Reporte.Agregar(" | ".join(Fila) + ".")
            Reporte.Agregar("")

    Reporte.Agregar("Comparaciones Mann Whitney.")
    Comparaciones = [
        ("Congruente", "Verde_Pro", "Naranja_Con"),
        ("Incongruente", "Verde_Con", "Naranja_Pro"),
    ]

    for Condicion, Clave_Izq, Clave_Der in Comparaciones:
        for Eleccion in ["Generales", "Ballotage"]:
            Valores_Izq = Resultados[Eleccion][
                Clave_Izq
            ]["Diferencias"]
            Valores_Der = Resultados[Eleccion][
                Clave_Der
            ]["Diferencias"]
            Valor_P = Test_Mann_Whitney(Valores_Izq, Valores_Der)
            Marca = Obtener_Significancia(Valor_P)
            Linea = [
                Condicion,
                Eleccion,
                "Izq vs Der",
                Formatear_P_Valor(Valor_P),
                Marca,
            ]
            Reporte.Agregar(" | ".join(Linea) + ".")

    for Condicion, Clave in [
        ("Congruente Izq", "Verde_Pro"),
        ("Congruente Der", "Naranja_Con"),
        ("Incongruente Izq", "Verde_Con"),
        ("Incongruente Der", "Naranja_Pro"),
    ]:
        Valores_Ge = Resultados["Generales"][Clave]["Diferencias"]
        Valores_Ba = Resultados["Ballotage"][Clave]["Diferencias"]
        Valor_P = Test_Mann_Whitney(Valores_Ge, Valores_Ba)
        Marca = Obtener_Significancia(Valor_P)
        Linea = [
            Condicion,
            "GE vs BA",
            Formatear_P_Valor(Valor_P),
            Marca,
        ]
        Reporte.Agregar(" | ".join(Linea) + ".")

    Reporte.Agregar("")


# ------------------------------------------------------------
# SECTION 03. OWN BASELINE CURVES.
# ------------------------------------------------------------

def Calcular_Medias_Por_Poblacion(
    Datos_Entrada: Pandas.DataFrame
) -> dict:

    """
    Calculates means by population with and without association.

    Parameters:
    - Datos_Entrada: DataFrame with data.

    Returns:
    - Dictionary with means and item order.

    Examples:
    - Calcular_Medias_Por_Poblacion(Datos)
    """

    Datos_Izq = Filtrar_Por_Poblacion(
        Datos_Entrada,
        Poblacion_Izquierda
    )
    Datos_Der = Filtrar_Por_Poblacion(
        Datos_Entrada,
        Poblacion_Derecha
    )

    Medias_Izq_Sin = {}
    Medias_Izq_Asoc = {}
    Medias_Der_Sin = {}
    Medias_Der_Asoc = {}

    for Item in Todos_Los_Items:
        Col_Base = f"IP_Item_{Item}_Respuesta"
        Col_Izq = f"IP_Item_{Item}_Izq_Respuesta"
        Col_Der = f"IP_Item_{Item}_Der_Respuesta"

        if Col_Base in Datos_Izq.columns:
            Medias_Izq_Sin[Item] = Datos_Izq[Col_Base].mean()
        if Col_Izq in Datos_Izq.columns:
            Medias_Izq_Asoc[Item] = Datos_Izq[Col_Izq].mean()
        if Col_Base in Datos_Der.columns:
            Medias_Der_Sin[Item] = Datos_Der[Col_Base].mean()
        if Col_Der in Datos_Der.columns:
            Medias_Der_Asoc[Item] = Datos_Der[Col_Der].mean()

    Items_Con_Ordenados = sorted(
        Items_Conservadores,
        key = lambda Item: Medias_Izq_Sin.get(Item, 0)
    )
    Items_Pro_Ordenados = sorted(
        Items_Progresistas,
        key = lambda Item: Medias_Izq_Sin.get(Item, 0),
        reverse = True
    )
    Orden_Items = Items_Con_Ordenados + Items_Pro_Ordenados

    return {
        "Orden_Items": Orden_Items,
        "Medias_Izq_Sin": Medias_Izq_Sin,
        "Medias_Izq_Asoc": Medias_Izq_Asoc,
        "Medias_Der_Sin": Medias_Der_Sin,
        "Medias_Der_Asoc": Medias_Der_Asoc,
        "Datos_Izq": Datos_Izq,
        "Datos_Der": Datos_Der,
    }


def Calcular_Significancia_Wilcoxon_Poblacion(
    Datos_Poblacion: dict
) -> dict:

    """
    Calculates Wilcoxon significance per item for populations.

    Parameters:
    - Datos_Poblacion: Dictionary with population data.

    Returns:
    - Dictionary with p-values per item.

    Examples:
    - Calcular_Significancia_Wilcoxon_Poblacion(Datos_Poblacion)
    """

    Orden_Items = Datos_Poblacion["Orden_Items"]
    Datos_Izq = Datos_Poblacion["Datos_Izq"]
    Datos_Der = Datos_Poblacion["Datos_Der"]

    P_Izq = {}
    P_Der = {}

    for Item in Orden_Items:
        Col_Base = f"IP_Item_{Item}_Respuesta"
        Col_Izq = f"IP_Item_{Item}_Izq_Respuesta"
        Col_Der = f"IP_Item_{Item}_Der_Respuesta"

        if (
            Col_Base in Datos_Izq.columns
            and Col_Izq in Datos_Izq.columns
        ):
            Valor_P = Test_Wilcoxon_Pareado(
                Datos_Izq[Col_Base], Datos_Izq[Col_Izq]
            )
        else:
            Valor_P = 1.0
        P_Izq[Item] = Valor_P

        if (
            Col_Base in Datos_Der.columns
            and Col_Der in Datos_Der.columns
        ):
            Valor_P = Test_Wilcoxon_Pareado(
                Datos_Der[Col_Base], Datos_Der[Col_Der]
            )
        else:
            Valor_P = 1.0
        P_Der[Item] = Valor_P

    return {"P_Izq": P_Izq, "P_Der": P_Der}


def Agregar_Seccion_03(
    Reporte: Generador_Reporte,
    Datos: dict
) -> None:

    """
    Adds section 03 to the report.

    Parameters:
    - Reporte: Report generator instance.
    - Datos: Dictionary with dataframes.

    Returns:
    - None.

    Examples:
    - Agregar_Seccion_03(Reporte, Datos)
    """

    Reporte.Agregar_Separador("=")
    Reporte.Agregar("Seccion 03. Curvas baseline propia.")
    Reporte.Agregar_Separador("=")
    Reporte.Agregar("")

    for Eleccion, Datos_Entrada in Datos.items():
        Reporte.Agregar_Separador("-")
        Reporte.Agregar(f"Eleccion {Eleccion}.")
        Reporte.Agregar_Separador("-")

        Datos_Poblacion = Calcular_Medias_Por_Poblacion(
            Datos_Entrada
        )
        Significancia = Calcular_Significancia_Wilcoxon_Poblacion(
            Datos_Poblacion
        )

        Orden_Items = Datos_Poblacion["Orden_Items"]
        Medias_Izq_Sin = Datos_Poblacion["Medias_Izq_Sin"]
        Medias_Izq_Asoc = Datos_Poblacion["Medias_Izq_Asoc"]
        Medias_Der_Sin = Datos_Poblacion["Medias_Der_Sin"]
        Medias_Der_Asoc = Datos_Poblacion["Medias_Der_Asoc"]

        Reporte.Agregar("Tabla de diferencias por item.")
        Encabezado = [
            "Item",
            "Tipo",
            "Izq_Base",
            "Izq_Asoc",
            "Dif_Izq",
            "Sig_Izq",
            "Der_Base",
            "Der_Asoc",
            "Dif_Der",
            "Sig_Der",
        ]
        Reporte.Agregar(" | ".join(Encabezado) + ".")

        Suma_Izq = 0.0
        Suma_Der = 0.0
        Cantidad_Items = len(Orden_Items)

        for Item in Orden_Items:
            Tipo = "Con" if Item in Items_Conservadores else "Pro"
            Izq_Base = Medias_Izq_Sin.get(Item, Numpy.nan)
            Izq_Asoc = Medias_Izq_Asoc.get(Item, Numpy.nan)
            Der_Base = Medias_Der_Sin.get(Item, Numpy.nan)
            Der_Asoc = Medias_Der_Asoc.get(Item, Numpy.nan)

            Dif_Izq = Izq_Asoc - Izq_Base
            Dif_Der = Der_Asoc - Der_Base

            Suma_Izq += Dif_Izq
            Suma_Der += Dif_Der

            Sig_Izq = "*" if Significancia["P_Izq"].get(
                Item, 1.0
            ) < 0.05 else ""
            Sig_Der = "*" if Significancia["P_Der"].get(
                Item, 1.0
            ) < 0.05 else ""

            Fila = [
                str(Item),
                Tipo,
                Formatear_Numero(Izq_Base),
                Formatear_Numero(Izq_Asoc),
                Formatear_Numero(Dif_Izq, Mostrar_Signo = True),
                Sig_Izq,
                Formatear_Numero(Der_Base),
                Formatear_Numero(Der_Asoc),
                Formatear_Numero(Dif_Der, Mostrar_Signo = True),
                Sig_Der,
            ]
            Reporte.Agregar(" | ".join(Fila) + ".")

        Reporte.Agregar(
            "Suma diferencias izquierda: "
            f"{Formatear_Numero(Suma_Izq)}."
        )
        Reporte.Agregar(
            "Suma diferencias derecha: "
            f"{Formatear_Numero(Suma_Der)}."
        )
        if Cantidad_Items > 0:
            Media_Izq = Suma_Izq / Cantidad_Items
            Media_Der = Suma_Der / Cantidad_Items
        else:
            Media_Izq = Numpy.nan
            Media_Der = Numpy.nan
        Reporte.Agregar(
            "Media diferencias izquierda: "
            f"{Formatear_Numero(Media_Izq)}."
        )
        Reporte.Agregar(
            "Media diferencias derecha: "
            f"{Formatear_Numero(Media_Der)}."
        )

        Cant_Sig_Izq = sum(
            1 for Valor in Significancia["P_Izq"].values()
            if Valor < 0.05
        )
        Cant_Sig_Der = sum(
            1 for Valor in Significancia["P_Der"].values()
            if Valor < 0.05
        )
        Reporte.Agregar(
            f"Diferencias significativas izquierda: {Cant_Sig_Izq}."
        )
        Reporte.Agregar(
            f"Diferencias significativas derecha: {Cant_Sig_Der}."
        )
        Reporte.Agregar("")


# ------------------------------------------------------------
# SECTION 04. CURVES WITH BALANCED SAMPLE.
# ------------------------------------------------------------

def Crear_Muestra_Balanceada(
    Datos_Entrada: Pandas.DataFrame
) -> dict:

    """
    Creates a balanced left and right sample with a fixed seed.

    Parameters:
    - Datos_Entrada: DataFrame with data.

    Returns:
    - Dictionary with subsamples and counts.

    Examples:
    - Crear_Muestra_Balanceada(Datos)
    """

    Datos_Izq = Filtrar_Por_Poblacion(
        Datos_Entrada,
        Poblacion_Izquierda
    )
    Datos_Der = Filtrar_Por_Poblacion(
        Datos_Entrada,
        Poblacion_Derecha
    )

    Cantidad_Der = len(Datos_Der)
    Datos_Izq_Bal = Datos_Izq.sample(
        n = Cantidad_Der, random_state = 42
    )

    Datos_Bal = Pandas.concat(
        [Datos_Izq_Bal, Datos_Der],
        ignore_index = True
    )

    return {
        "Datos_Bal": Datos_Bal,
        "Datos_Izq_Bal": Datos_Izq_Bal,
        "Datos_Der": Datos_Der,
        "Cantidad_Izq": len(Datos_Izq_Bal),
        "Cantidad_Der": Cantidad_Der,
    }


def Calcular_Medias_Baseline_Balanceada(
    Muestra: dict
) -> dict:

    """
    Calculates means with a balanced baseline.

    Parameters:
    - Muestra: Dictionary with balanced data.

    Returns:
    - Dictionary with means and data.

    Examples:
    - Calcular_Medias_Baseline_Balanceada(Muestra)
    """

    Datos_Bal = Muestra["Datos_Bal"]
    Datos_Izq = Muestra["Datos_Izq_Bal"]
    Datos_Der = Muestra["Datos_Der"]

    Medias_Base = {}
    Medias_Izq = {}
    Medias_Der = {}

    for Item in Todos_Los_Items:
        Col_Base = f"IP_Item_{Item}_Respuesta"
        Col_Izq = f"IP_Item_{Item}_Izq_Respuesta"
        Col_Der = f"IP_Item_{Item}_Der_Respuesta"

        if Col_Base in Datos_Bal.columns:
            Medias_Base[Item] = Datos_Bal[Col_Base].mean()
        if Col_Izq in Datos_Izq.columns:
            Medias_Izq[Item] = Datos_Izq[Col_Izq].mean()
        if Col_Der in Datos_Der.columns:
            Medias_Der[Item] = Datos_Der[Col_Der].mean()

    Items_Con_Ordenados = sorted(
        Items_Conservadores,
        key = lambda Item: Medias_Base.get(Item, 0)
    )
    Items_Pro_Ordenados = sorted(
        Items_Progresistas,
        key = lambda Item: Medias_Base.get(Item, 0),
        reverse = True
    )
    Orden_Items = Items_Con_Ordenados + Items_Pro_Ordenados

    return {
        "Orden_Items": Orden_Items,
        "Medias_Base": Medias_Base,
        "Medias_Izq": Medias_Izq,
        "Medias_Der": Medias_Der,
        "Datos_Izq": Datos_Izq,
        "Datos_Der": Datos_Der,
        "Datos_Bal": Datos_Bal,
    }


def Agregar_Seccion_04(
    Reporte: Generador_Reporte,
    Datos: dict
) -> None:

    """
    Adds section 04 to the report.

    Parameters:
    - Reporte: Report generator instance.
    - Datos: Dictionary with dataframes.

    Returns:
    - None.

    Examples:
    - Agregar_Seccion_04(Reporte, Datos)
    """

    Reporte.Agregar_Separador("=")
    Reporte.Agregar("Seccion 04. Curvas con muestra balanceada.")
    Reporte.Agregar_Separador("=")
    Reporte.Agregar("")

    for Eleccion, Datos_Entrada in Datos.items():
        Reporte.Agregar_Separador("-")
        Reporte.Agregar(f"Eleccion {Eleccion}.")
        Reporte.Agregar_Separador("-")

        Muestra = Crear_Muestra_Balanceada(Datos_Entrada)
        Datos_Bal = Calcular_Medias_Baseline_Balanceada(Muestra)
        Orden_Items = Datos_Bal["Orden_Items"]
        Medias_Base = Datos_Bal["Medias_Base"]
        Medias_Izq = Datos_Bal["Medias_Izq"]
        Medias_Der = Datos_Bal["Medias_Der"]
        Datos_Izq = Datos_Bal["Datos_Izq"]
        Datos_Der = Datos_Bal["Datos_Der"]
        Datos_Bal_Ref = Datos_Bal["Datos_Bal"]

        Reporte.Agregar(
            "Cantidad balanceada izquierda: "
            f"{Muestra['Cantidad_Izq']}."
        )
        Reporte.Agregar(
            "Cantidad balanceada derecha: "
            f"{Muestra['Cantidad_Der']}."
        )

        Reporte.Agregar(
            "Tabla de diferencias vs baseline balanceada."
        )
        Encabezado = [
            "Item",
            "Tipo",
            "Baseline",
            "Izq_Asoc",
            "Dif_Izq",
            "Sig_Izq",
            "Der_Asoc",
            "Dif_Der",
            "Sig_Der",
        ]
        Reporte.Agregar(" | ".join(Encabezado) + ".")

        Suma_Izq = 0.0
        Suma_Der = 0.0

        for Item in Orden_Items:
            Tipo = "Con" if Item in Items_Conservadores else "Pro"
            Base = Medias_Base.get(Item, Numpy.nan)
            Izq = Medias_Izq.get(Item, Numpy.nan)
            Der = Medias_Der.get(Item, Numpy.nan)

            Dif_Izq = Izq - Base
            Dif_Der = Der - Base

            Suma_Izq += Dif_Izq
            Suma_Der += Dif_Der

            Col_Base = f"IP_Item_{Item}_Respuesta"
            Col_Izq = f"IP_Item_{Item}_Izq_Respuesta"
            Col_Der = f"IP_Item_{Item}_Der_Respuesta"

            if Col_Base in Datos_Izq.columns and Col_Izq in (
                Datos_Izq.columns
            ):
                Valor_P_Izq = Test_Wilcoxon_Pareado(
                    Datos_Izq[Col_Base],
                    Datos_Izq[Col_Izq]
                )
            else:
                Valor_P_Izq = 1.0

            if Col_Base in Datos_Der.columns and Col_Der in (
                Datos_Der.columns
            ):
                Valor_P_Der = Test_Wilcoxon_Pareado(
                    Datos_Der[Col_Base],
                    Datos_Der[Col_Der]
                )
            else:
                Valor_P_Der = 1.0

            Sig_Izq = "*" if Valor_P_Izq < 0.05 else ""
            Sig_Der = "*" if Valor_P_Der < 0.05 else ""

            Fila = [
                str(Item),
                Tipo,
                Formatear_Numero(Base),
                Formatear_Numero(Izq),
                Formatear_Numero(Dif_Izq, Mostrar_Signo = True),
                Sig_Izq,
                Formatear_Numero(Der),
                Formatear_Numero(Dif_Der, Mostrar_Signo = True),
                Sig_Der,
            ]
            Reporte.Agregar(" | ".join(Fila) + ".")

        Cantidad_Items = len(Orden_Items)
        Media_Izq = Suma_Izq / Cantidad_Items if (
            Cantidad_Items > 0
        ) else Numpy.nan
        Media_Der = Suma_Der / Cantidad_Items if (
            Cantidad_Items > 0
        ) else Numpy.nan

        Reporte.Agregar(
            "Media diferencias izquierda: "
            f"{Formatear_Numero(Media_Izq)}."
        )
        Reporte.Agregar(
            "Media diferencias derecha: "
            f"{Formatear_Numero(Media_Der)}."
        )
        Reporte.Agregar("")

        Reporte.Agregar(
            "Comparacion baseline original vs balanceada."
        )
        Encabezado = [
            "Item",
            "Tipo",
            "Base_Original",
            "Base_Balanceada",
            "Diferencia",
        ]
        Reporte.Agregar(" | ".join(Encabezado) + ".")

        for Item in Todos_Los_Items:
            Tipo = "Con" if Item in Items_Conservadores else "Pro"
            Col_Base = f"IP_Item_{Item}_Respuesta"
            if Col_Base in Datos_Entrada.columns:
                Media_Original = Datos_Entrada[Col_Base].mean()
                Media_Bal = Datos_Bal_Ref[Col_Base].mean()
                Dif = Media_Bal - Media_Original
                Fila = [
                    str(Item),
                    Tipo,
                    Formatear_Numero(Media_Original),
                    Formatear_Numero(Media_Bal),
                    Formatear_Numero(Dif, Mostrar_Signo = True),
                ]
                Reporte.Agregar(" | ".join(Fila) + ".")

        Reporte.Agregar("")


# ------------------------------------------------------------
# SECTION 05. OWN BASELINE AREAS.
# ------------------------------------------------------------

def Calcular_Area_Baseline_Propia(
    Datos_Entrada: Pandas.DataFrame,
    Poblacion: list,
    Items: list,
    Direccion: str
) -> dict:

    """
    Calculates areas comparing a population with itself.

    Parameters:
    - Datos_Entrada: DataFrame with data.
    - Poblacion: List of categories.
    - Items: List of items.
    - Direccion: Izq or Der.

    Returns:
    - Dictionary with differences, mean, error, and detail.

    Examples:
    - Calcular_Area_Baseline_Propia(
        Datos, Poblacion_Izquierda, Items, "Izq"
      )
    """

    Datos_Poblacion = Filtrar_Por_Poblacion(Datos_Entrada, Poblacion)

    Diferencias = []
    Detalle = []

    for Item in Items:
        Col_Base = f"IP_Item_{Item}_Respuesta"
        Col_Asoc = f"IP_Item_{Item}_{Direccion}_Respuesta"
        if Col_Base in Datos_Poblacion.columns and (
            Col_Asoc in Datos_Poblacion.columns
        ):
            Media_Base = Datos_Poblacion[Col_Base].mean()
            Media_Asoc = Datos_Poblacion[Col_Asoc].mean()
            Diferencia = Media_Asoc - Media_Base
            Diferencias.append(Diferencia)
            Detalle.append({
                "Item": Item,
                "Media_Base": Media_Base,
                "Media_Asoc": Media_Asoc,
                "Diferencia": Diferencia,
            })

    Media, Error, _, Valores = Calcular_Media_Y_Error(Diferencias)

    return {
        "Diferencias": Valores,
        "Media": Media,
        "Error": Error,
        "Detalle": Detalle,
    }


def Agregar_Seccion_05(
    Reporte: Generador_Reporte,
    Datos: dict
) -> None:

    """
    Adds section 05 to the report.

    Parameters:
    - Reporte: Report generator instance.
    - Datos: Dictionary with dataframes.

    Returns:
    - None.

    Examples:
    - Agregar_Seccion_05(Reporte, Datos)
    """

    Reporte.Agregar_Separador("=")
    Reporte.Agregar("Seccion 05. Areas baseline propia.")
    Reporte.Agregar_Separador("=")
    Reporte.Agregar("")

    Resultados = {}
    for Nombre, Datos_Entrada in Datos.items():
        Resultados[Nombre] = {
            "Verde_Con": Calcular_Area_Baseline_Propia(
                Datos_Entrada, Poblacion_Izquierda,
                Items_Conservadores, "Izq"
            ),
            "Verde_Pro": Calcular_Area_Baseline_Propia(
                Datos_Entrada, Poblacion_Izquierda,
                Items_Progresistas, "Izq"
            ),
            "Naranja_Con": Calcular_Area_Baseline_Propia(
                Datos_Entrada, Poblacion_Derecha,
                Items_Conservadores, "Der"
            ),
            "Naranja_Pro": Calcular_Area_Baseline_Propia(
                Datos_Entrada, Poblacion_Derecha,
                Items_Progresistas, "Der"
            ),
        }

    Reporte.Agregar("Resumen de areas por eleccion.")
    Encabezado = [
        "Area",
        "Media_GE",
        "Error_GE",
        "Media_BA",
        "Error_BA",
    ]
    Reporte.Agregar(" | ".join(Encabezado) + ".")

    for Clave, Nombre in [
        ("Verde_Con", "Verde en conservadores"),
        ("Verde_Pro", "Verde en progresistas"),
        ("Naranja_Con", "Naranja en conservadores"),
        ("Naranja_Pro", "Naranja en progresistas"),
    ]:
        Media_Ge = Resultados["Generales"][Clave]["Media"]
        Error_Ge = Resultados["Generales"][Clave]["Error"]
        Media_Ba = Resultados["Ballotage"][Clave]["Media"]
        Error_Ba = Resultados["Ballotage"][Clave]["Error"]
        Fila = [
            Nombre,
            Formatear_Numero(Media_Ge, Mostrar_Signo = True),
            Formatear_Numero(Error_Ge),
            Formatear_Numero(Media_Ba, Mostrar_Signo = True),
            Formatear_Numero(Error_Ba),
        ]
        Reporte.Agregar(" | ".join(Fila) + ".")

    Reporte.Agregar("")
    Reporte.Agregar("Detalle por item y area.")

    for Clave, Titulo in [
        ("Verde_Con", "Verde en conservadores"),
        ("Verde_Pro", "Verde en progresistas"),
        ("Naranja_Con", "Naranja en conservadores"),
        ("Naranja_Pro", "Naranja en progresistas"),
    ]:
        Reporte.Agregar_Separador("-")
        Reporte.Agregar(f"Area {Titulo}.")
        Reporte.Agregar_Separador("-")
        for Eleccion in ["Generales", "Ballotage"]:
            Reporte.Agregar(f"Eleccion {Eleccion}.")
            Encabezado = [
                "Item",
                "Media_Base",
                "Media_Asoc",
                "Diferencia",
            ]
            Reporte.Agregar(" | ".join(Encabezado) + ".")
            for Detalle in Resultados[Eleccion][Clave]["Detalle"]:
                Fila = [
                    str(Detalle["Item"]),
                    Formatear_Numero(Detalle["Media_Base"]),
                    Formatear_Numero(Detalle["Media_Asoc"]),
                    Formatear_Numero(
                        Detalle["Diferencia"], Mostrar_Signo = True
                    ),
                ]
                Reporte.Agregar(" | ".join(Fila) + ".")
            Reporte.Agregar("")

    Reporte.Agregar("Tests Wilcoxon contra cero.")
    for Clave, Nombre in [
        ("Verde_Pro", "Congruente Izq"),
        ("Naranja_Con", "Congruente Der"),
        ("Verde_Con", "Incongruente Izq"),
        ("Naranja_Pro", "Incongruente Der"),
    ]:
        for Eleccion in ["Generales", "Ballotage"]:
            Valores = Resultados[Eleccion][Clave]["Diferencias"]
            Valor_P = Test_Wilcoxon_Contra_Cero(Valores)
            Marca = Obtener_Significancia(Valor_P)
            Linea = [
                Nombre,
                Eleccion,
                Formatear_P_Valor(Valor_P),
                Marca,
            ]
            Reporte.Agregar(" | ".join(Linea) + ".")

    Reporte.Agregar("")
    Reporte.Agregar("Comparaciones Mann Whitney.")

    Comparaciones = [
        ("Congruente", "Verde_Pro", "Naranja_Con"),
        ("Incongruente", "Verde_Con", "Naranja_Pro"),
    ]

    for Condicion, Clave_Izq, Clave_Der in Comparaciones:
        for Eleccion in ["Generales", "Ballotage"]:
            Valores_Izq = Resultados[Eleccion][
                Clave_Izq
            ]["Diferencias"]
            Valores_Der = Resultados[Eleccion][
                Clave_Der
            ]["Diferencias"]
            Valor_P = Test_Mann_Whitney(Valores_Izq, Valores_Der)
            Marca = Obtener_Significancia(Valor_P)
            Linea = [
                Condicion,
                Eleccion,
                "Izq vs Der",
                Formatear_P_Valor(Valor_P),
                Marca,
            ]
            Reporte.Agregar(" | ".join(Linea) + ".")

    for Condicion, Clave in [
        ("Congruente Izq", "Verde_Pro"),
        ("Congruente Der", "Naranja_Con"),
        ("Incongruente Izq", "Verde_Con"),
        ("Incongruente Der", "Naranja_Pro"),
    ]:
        Valores_Ge = Resultados["Generales"][Clave]["Diferencias"]
        Valores_Ba = Resultados["Ballotage"][Clave]["Diferencias"]
        Valor_P = Test_Mann_Whitney(Valores_Ge, Valores_Ba)
        Marca = Obtener_Significancia(Valor_P)
        Linea = [
            Condicion,
            "GE vs BA",
            Formatear_P_Valor(Valor_P),
            Marca,
        ]
        Reporte.Agregar(" | ".join(Linea) + ".")

    Reporte.Agregar("")


# ------------------------------------------------------------
# SECTION 28. AREAS WITH BALANCED SAMPLE.
# ------------------------------------------------------------

def Calcular_Area_Muestra_Balanceada(
    Muestra: dict,
    Tipo_Poblacion: str,
    Items: list,
    Direccion: str
) -> dict:

    """
    Calculates area versus a balanced baseline.

    Parameters:
    - Muestra: Dictionary with subsamples.
    - Tipo_Poblacion: Izq or Der.
    - Items: List of items.
    - Direccion: Izq or Der.

    Returns:
    - Dictionary with differences, mean, error, and detail.

    Examples:
    - Calcular_Area_Muestra_Balanceada(Muestra, "Izq", Items, "Izq")
    """

    Datos_Bal = Muestra["Datos_Bal"]
    if Tipo_Poblacion == "Izq":
        Datos_Pob = Muestra["Datos_Izq_Bal"]
    else:
        Datos_Pob = Muestra["Datos_Der"]

    Diferencias = []
    Detalle = []

    for Item in Items:
        Col_Base = f"IP_Item_{Item}_Respuesta"
        Col_Asoc = f"IP_Item_{Item}_{Direccion}_Respuesta"
        if Col_Base in Datos_Bal.columns and Col_Asoc in (
            Datos_Pob.columns
        ):
            Media_Base = Datos_Bal[Col_Base].mean()
            Media_Asoc = Datos_Pob[Col_Asoc].mean()
            Diferencia = Media_Asoc - Media_Base
            Diferencias.append(Diferencia)
            Detalle.append({
                "Item": Item,
                "Media_Base": Media_Base,
                "Media_Asoc": Media_Asoc,
                "Diferencia": Diferencia,
            })

    Media, Error, _, Valores = Calcular_Media_Y_Error(Diferencias)

    return {
        "Diferencias": Valores,
        "Media": Media,
        "Error": Error,
        "Detalle": Detalle,
    }


def Agregar_Seccion_06(
    Reporte: Generador_Reporte,
    Datos: dict
) -> None:

    """
    Adds section 06 to the report.

    Parameters:
    - Reporte: Report generator instance.
    - Datos: Dictionary with dataframes.

    Returns:
    - None.

    Examples:
    - Agregar_Seccion_06(Reporte, Datos)
    """

    Reporte.Agregar_Separador("=")
    Reporte.Agregar("Seccion 06. Areas con muestra balanceada.")
    Reporte.Agregar_Separador("=")
    Reporte.Agregar("")


# ------------------------------------------------------------
# SECTION 07. THREE BALANCED POPULATIONS.
# ------------------------------------------------------------

def Crear_Muestra_Balanceada_Tres_Poblaciones(
    Datos_Entrada: Pandas.DataFrame
) -> dict:

    """
    Creates a balanced sample with three populations.

    Parameters:
    - Datos_Entrada: DataFrame with data.

    Returns:
    - Dictionary with subsamples.

    Examples:
    - Crear_Muestra_Balanceada_Tres_Poblaciones(Datos)
    """

    Datos_Izq = Filtrar_Por_Poblacion(
        Datos_Entrada,
        Poblacion_Izquierda
    )
    Datos_Centro = Filtrar_Por_Poblacion(
        Datos_Entrada,
        Poblacion_Centro
    )
    Datos_Der = Filtrar_Por_Poblacion(
        Datos_Entrada,
        Poblacion_Derecha
    )

    Cantidad_Der = len(Datos_Der)

    Datos_Izq_Bal = Datos_Izq.sample(
        n = Cantidad_Der, random_state = 42
    )
    Datos_Der_Bal = Datos_Der.sample(
        n = Cantidad_Der, random_state = 42
    )

    Datos_Bal = Pandas.concat(
        [Datos_Izq_Bal, Datos_Centro, Datos_Der_Bal],
        ignore_index = True
    )

    return {
        "Datos_Bal": Datos_Bal,
        "Datos_Izq": Datos_Izq_Bal,
        "Datos_Centro": Datos_Centro,
        "Datos_Der": Datos_Der_Bal,
        "Cantidad_Balanceo": Cantidad_Der,
    }


def Calcular_Medias_Tres_Poblaciones(
    Muestra: dict
) -> dict:

    """
    Calculates means for three populations with a balanced baseline.

    Parameters:
    - Muestra: Dictionary with subsamples.

    Returns:
    - Dictionary with means and data.

    Examples:
    - Calcular_Medias_Tres_Poblaciones(Muestra)
    """

    Datos_Bal = Muestra["Datos_Bal"]
    Datos_Izq = Muestra["Datos_Izq"]
    Datos_Centro = Muestra["Datos_Centro"]
    Datos_Der = Muestra["Datos_Der"]

    Medias_Base = {}
    Medias_Izq = {}
    Medias_Centro = {}
    Medias_Der = {}

    for Item in Todos_Los_Items:
        Col_Base = f"IP_Item_{Item}_Respuesta"
        Col_Izq = f"IP_Item_{Item}_Izq_Respuesta"
        Col_Der = f"IP_Item_{Item}_Der_Respuesta"

        if Col_Base in Datos_Bal.columns:
            Medias_Base[Item] = Datos_Bal[Col_Base].mean()
        if Col_Izq in Datos_Izq.columns:
            Medias_Izq[Item] = Datos_Izq[Col_Izq].mean()
        if Col_Izq in Datos_Centro.columns and Col_Der in (
            Datos_Centro.columns
        ):
            Promedio = (
                Datos_Centro[Col_Izq] + Datos_Centro[Col_Der]
            ) / 2
            Medias_Centro[Item] = Promedio.mean()
        if Col_Der in Datos_Der.columns:
            Medias_Der[Item] = Datos_Der[Col_Der].mean()

    Items_Con_Ordenados = sorted(
        Items_Conservadores,
        key = lambda Item: Medias_Base.get(Item, 0)
    )
    Items_Pro_Ordenados = sorted(
        Items_Progresistas,
        key = lambda Item: Medias_Base.get(Item, 0),
        reverse = True
    )
    Orden_Items = Items_Con_Ordenados + Items_Pro_Ordenados

    return {
        "Orden_Items": Orden_Items,
        "Medias_Base": Medias_Base,
        "Medias_Izq": Medias_Izq,
        "Medias_Centro": Medias_Centro,
        "Medias_Der": Medias_Der,
        "Datos_Izq": Datos_Izq,
        "Datos_Centro": Datos_Centro,
        "Datos_Der": Datos_Der,
        "Datos_Bal": Datos_Bal,
    }


def Calcular_Significancia_Tres_Poblaciones(
    Datos_Tres: dict
) -> dict:

    """
    Calculates Wilcoxon per item for three populations.

    Parameters:
    - Datos_Tres: Dictionary with data and means.

    Returns:
    - Dictionary with p-values.

    Examples:
    - Calcular_Significancia_Tres_Poblaciones(Datos_Tres)
    """

    Orden_Items = Datos_Tres["Orden_Items"]
    Datos_Izq = Datos_Tres["Datos_Izq"]
    Datos_Centro = Datos_Tres["Datos_Centro"]
    Datos_Der = Datos_Tres["Datos_Der"]

    P_Izq = {}
    P_Centro = {}
    P_Der = {}

    for Item in Orden_Items:
        Col_Base = f"IP_Item_{Item}_Respuesta"
        Col_Izq = f"IP_Item_{Item}_Izq_Respuesta"
        Col_Der = f"IP_Item_{Item}_Der_Respuesta"

        if Col_Base in Datos_Izq.columns and Col_Izq in (
            Datos_Izq.columns
        ):
            Valor_P = Test_Wilcoxon_Pareado(
                Datos_Izq[Col_Base], Datos_Izq[Col_Izq]
            )
        else:
            Valor_P = 1.0
        P_Izq[Item] = Valor_P

        if Col_Base in Datos_Centro.columns and (
            Col_Izq in Datos_Centro.columns
        ) and Col_Der in Datos_Centro.columns:
            Promedio = (
                Datos_Centro[Col_Izq] + Datos_Centro[Col_Der]
            ) / 2
            Valor_P = Test_Wilcoxon_Pareado(
                Datos_Centro[Col_Base], Promedio
            )
        else:
            Valor_P = 1.0
        P_Centro[Item] = Valor_P

        if Col_Base in Datos_Der.columns and Col_Der in (
            Datos_Der.columns
        ):
            Valor_P = Test_Wilcoxon_Pareado(
                Datos_Der[Col_Base], Datos_Der[Col_Der]
            )
        else:
            Valor_P = 1.0
        P_Der[Item] = Valor_P

    return {"P_Izq": P_Izq, "P_Centro": P_Centro, "P_Der": P_Der}


def Agregar_Seccion_07(
    Reporte: Generador_Reporte,
    Datos: dict
) -> None:

    """
    Adds section 07 to the report.

    Parameters:
    - Reporte: Report generator instance.
    - Datos: Dictionary with dataframes.

    Returns:
    - None.

    Examples:
    - Agregar_Seccion_07(Reporte, Datos)
    """

    Reporte.Agregar_Separador("=")
    Reporte.Agregar(
        "Seccion 07. Curvas tres poblaciones balanceadas."
    )
    Reporte.Agregar_Separador("=")
    Reporte.Agregar("")

    for Eleccion, Datos_Entrada in Datos.items():
        Reporte.Agregar_Separador("-")
        Reporte.Agregar(f"Eleccion {Eleccion}.")
        Reporte.Agregar_Separador("-")

        Muestra = Crear_Muestra_Balanceada_Tres_Poblaciones(
            Datos_Entrada
        )
        Datos_Tres = Calcular_Medias_Tres_Poblaciones(Muestra)
        Significancia = Calcular_Significancia_Tres_Poblaciones(
            Datos_Tres
        )

        Reporte.Agregar(
            "Cantidad balanceada por grupo: "
            f"{Muestra['Cantidad_Balanceo']}."
        )

        Encabezado = [
            "Item",
            "Tipo",
            "Baseline",
            "Dif_Izq",
            "Sig_Izq",
            "Dif_Centro",
            "Sig_Centro",
            "Dif_Der",
            "Sig_Der",
        ]
        Reporte.Agregar(" | ".join(Encabezado) + ".")

        Orden_Items = Datos_Tres["Orden_Items"]
        Medias_Base = Datos_Tres["Medias_Base"]
        Medias_Izq = Datos_Tres["Medias_Izq"]
        Medias_Centro = Datos_Tres["Medias_Centro"]
        Medias_Der = Datos_Tres["Medias_Der"]

        Suma_Izq = 0.0
        Suma_Centro = 0.0
        Suma_Der = 0.0

        for Item in Orden_Items:
            Tipo = "Con" if Item in Items_Conservadores else "Pro"
            Base = Medias_Base.get(Item, Numpy.nan)
            Izq = Medias_Izq.get(Item, Numpy.nan)
            Centro = Medias_Centro.get(Item, Numpy.nan)
            Der = Medias_Der.get(Item, Numpy.nan)

            Dif_Izq = Izq - Base
            Dif_Centro = Centro - Base
            Dif_Der = Der - Base

            Suma_Izq += Dif_Izq
            Suma_Centro += Dif_Centro
            Suma_Der += Dif_Der

            Sig_Izq = "*" if Significancia["P_Izq"].get(
                Item, 1.0
            ) < 0.05 else ""
            Sig_Centro = "*" if Significancia["P_Centro"].get(
                Item, 1.0
            ) < 0.05 else ""
            Sig_Der = "*" if Significancia["P_Der"].get(
                Item, 1.0
            ) < 0.05 else ""

            Fila = [
                str(Item),
                Tipo,
                Formatear_Numero(Base),
                Formatear_Numero(Dif_Izq, Mostrar_Signo = True),
                Sig_Izq,
                Formatear_Numero(Dif_Centro, Mostrar_Signo = True),
                Sig_Centro,
                Formatear_Numero(Dif_Der, Mostrar_Signo = True),
                Sig_Der,
            ]
            Reporte.Agregar(" | ".join(Fila) + ".")

        Cantidad = len(Orden_Items)
        Media_Izq = Suma_Izq / Cantidad if Cantidad else Numpy.nan
        Media_Centro = Suma_Centro / Cantidad if (
            Cantidad
        ) else Numpy.nan
        Media_Der = Suma_Der / Cantidad if Cantidad else Numpy.nan

        Reporte.Agregar(
            "Media diferencias izquierda: "
            f"{Formatear_Numero(Media_Izq)}."
        )
        Reporte.Agregar(
            "Media diferencias centro: "
            f"{Formatear_Numero(Media_Centro)}."
        )
        Reporte.Agregar(
            "Media diferencias derecha: "
            f"{Formatear_Numero(Media_Der)}."
        )
        Reporte.Agregar("")


# ------------------------------------------------------------
# SECTION 08. THREE-POPULATION QUANTIFICATION.
# ------------------------------------------------------------

def Calcular_Area_Tres_Poblaciones(
    Muestra: dict,
    Tipo_Poblacion: str,
    Items: list,
    Direccion: str
) -> dict:

    """
    Calculates area by population versus a balanced baseline.

    Parameters:
    - Muestra: Dictionary with subsamples.
    - Tipo_Poblacion: Izq, Centro or Der.
    - Items: List of items.
    - Direccion: Izq, Der or Average.

    Returns:
    - Dictionary with differences, mean, error, and detail.

    Examples:
    - Calcular_Area_Tres_Poblaciones(Muestra, "Izq", Items, "Izq")
    """

    Datos_Bal = Muestra["Datos_Bal"]
    if Tipo_Poblacion == "Izq":
        Datos_Pob = Muestra["Datos_Izq"]
    elif Tipo_Poblacion == "Centro":
        Datos_Pob = Muestra["Datos_Centro"]
    else:
        Datos_Pob = Muestra["Datos_Der"]

    Diferencias = []
    Detalle = []

    for Item in Items:
        Col_Base = f"IP_Item_{Item}_Respuesta"

        if Direccion == "Promedio":
            Col_Izq = f"IP_Item_{Item}_Izq_Respuesta"
            Col_Der = f"IP_Item_{Item}_Der_Respuesta"
            if Col_Base in Datos_Bal.columns and (
                Col_Izq in Datos_Pob.columns
            ) and Col_Der in Datos_Pob.columns:
                Media_Base = Datos_Bal[Col_Base].mean()
                Media_Asoc = (
                    (Datos_Pob[Col_Izq] + Datos_Pob[Col_Der]) / 2
                ).mean()
                Diferencia = Media_Asoc - Media_Base
                Diferencias.append(Diferencia)
                Detalle.append({
                    "Item": Item,
                    "Media_Base": Media_Base,
                    "Media_Asoc": Media_Asoc,
                    "Diferencia": Diferencia,
                })
        else:
            Col_Asoc = f"IP_Item_{Item}_{Direccion}_Respuesta"
            if Col_Base in Datos_Bal.columns and (
                Col_Asoc in Datos_Pob.columns
            ):
                Media_Base = Datos_Bal[Col_Base].mean()
                Media_Asoc = Datos_Pob[Col_Asoc].mean()
                Diferencia = Media_Asoc - Media_Base
                Diferencias.append(Diferencia)
                Detalle.append({
                    "Item": Item,
                    "Media_Base": Media_Base,
                    "Media_Asoc": Media_Asoc,
                    "Diferencia": Diferencia,
                })

    Media, Error, _, Valores = Calcular_Media_Y_Error(Diferencias)

    return {
        "Diferencias": Valores,
        "Media": Media,
        "Error": Error,
        "Detalle": Detalle,
    }


def Agregar_Seccion_08(
    Reporte: Generador_Reporte,
    Datos: dict
) -> None:

    """
    Adds section 08 to the report.

    Parameters:
    - Reporte: Report generator instance.
    - Datos: Dictionary with dataframes.

    Returns:
    - None.

    Examples:
    - Agregar_Seccion_08(Reporte, Datos)
    """

    Reporte.Agregar_Separador("=")
    Reporte.Agregar("Seccion 08. Cuantificacion tres poblaciones.")
    Reporte.Agregar_Separador("=")
    Reporte.Agregar("")


# ------------------------------------------------------------
# SECTION 09. CENTER WITH TWO ASSOCIATIONS.
# ------------------------------------------------------------

def Calcular_Medias_Centro_Desglosado(
    Muestra: dict
) -> dict:

    """
    Calculates means with the center split into two associations.

    Parameters:
    - Muestra: Dictionary with subsamples.

    Returns:
    - Dictionary with means and data.

    Examples:
    - Calcular_Medias_Centro_Desglosado(Muestra)
    """

    Datos_Bal = Muestra["Datos_Bal"]
    Datos_Izq = Muestra["Datos_Izq"]
    Datos_Centro = Muestra["Datos_Centro"]
    Datos_Der = Muestra["Datos_Der"]

    Medias_Base = {}
    Medias_Izq = {}
    Medias_Centro_Izq = {}
    Medias_Centro_Der = {}
    Medias_Der = {}

    for Item in Todos_Los_Items:
        Col_Base = f"IP_Item_{Item}_Respuesta"
        Col_Izq = f"IP_Item_{Item}_Izq_Respuesta"
        Col_Der = f"IP_Item_{Item}_Der_Respuesta"

        if Col_Base in Datos_Bal.columns:
            Medias_Base[Item] = Datos_Bal[Col_Base].mean()
        if Col_Izq in Datos_Izq.columns:
            Medias_Izq[Item] = Datos_Izq[Col_Izq].mean()
        if Col_Izq in Datos_Centro.columns:
            Medias_Centro_Izq[Item] = Datos_Centro[Col_Izq].mean()
        if Col_Der in Datos_Centro.columns:
            Medias_Centro_Der[Item] = Datos_Centro[Col_Der].mean()
        if Col_Der in Datos_Der.columns:
            Medias_Der[Item] = Datos_Der[Col_Der].mean()

    Items_Con_Ordenados = sorted(
        Items_Conservadores,
        key = lambda Item: Medias_Base.get(Item, 0)
    )
    Items_Pro_Ordenados = sorted(
        Items_Progresistas,
        key = lambda Item: Medias_Base.get(Item, 0),
        reverse = True
    )
    Orden_Items = Items_Con_Ordenados + Items_Pro_Ordenados

    return {
        "Orden_Items": Orden_Items,
        "Medias_Base": Medias_Base,
        "Medias_Izq": Medias_Izq,
        "Medias_Centro_Izq": Medias_Centro_Izq,
        "Medias_Centro_Der": Medias_Centro_Der,
        "Medias_Der": Medias_Der,
        "Datos_Izq": Datos_Izq,
        "Datos_Centro": Datos_Centro,
        "Datos_Der": Datos_Der,
    }


def Calcular_Significancia_Centro_Desglosado(
    Datos_Tres: dict
) -> dict:

    """
    Calculates Wilcoxon per item with the center split.

    Parameters:
    - Datos_Tres: Dictionary with data and means.

    Returns:
    - Dictionary with p-values.

    Examples:
    - Calcular_Significancia_Centro_Desglosado(Datos_Tres)
    """

    Orden_Items = Datos_Tres["Orden_Items"]
    Datos_Izq = Datos_Tres["Datos_Izq"]
    Datos_Centro = Datos_Tres["Datos_Centro"]
    Datos_Der = Datos_Tres["Datos_Der"]

    P_Izq = {}
    P_Centro_Izq = {}
    P_Centro_Der = {}
    P_Der = {}

    for Item in Orden_Items:
        Col_Base = f"IP_Item_{Item}_Respuesta"
        Col_Izq = f"IP_Item_{Item}_Izq_Respuesta"
        Col_Der = f"IP_Item_{Item}_Der_Respuesta"

        if Col_Base in Datos_Izq.columns and Col_Izq in (
            Datos_Izq.columns
        ):
            Valor_P = Test_Wilcoxon_Pareado(
                Datos_Izq[Col_Base], Datos_Izq[Col_Izq]
            )
        else:
            Valor_P = 1.0
        P_Izq[Item] = Valor_P

        if Col_Base in Datos_Centro.columns and Col_Izq in (
            Datos_Centro.columns
        ):
            Valor_P = Test_Wilcoxon_Pareado(
                Datos_Centro[Col_Base], Datos_Centro[Col_Izq]
            )
        else:
            Valor_P = 1.0
        P_Centro_Izq[Item] = Valor_P

        if Col_Base in Datos_Centro.columns and Col_Der in (
            Datos_Centro.columns
        ):
            Valor_P = Test_Wilcoxon_Pareado(
                Datos_Centro[Col_Base], Datos_Centro[Col_Der]
            )
        else:
            Valor_P = 1.0
        P_Centro_Der[Item] = Valor_P

        if Col_Base in Datos_Der.columns and Col_Der in (
            Datos_Der.columns
        ):
            Valor_P = Test_Wilcoxon_Pareado(
                Datos_Der[Col_Base], Datos_Der[Col_Der]
            )
        else:
            Valor_P = 1.0
        P_Der[Item] = Valor_P

    return {
        "P_Izq": P_Izq,
        "P_Centro_Izq": P_Centro_Izq,
        "P_Centro_Der": P_Centro_Der,
        "P_Der": P_Der,
    }


def Agregar_Seccion_09(
    Reporte: Generador_Reporte,
    Datos: dict
) -> None:

    """
    Adds section 09 to the report.

    Parameters:
    - Reporte: Report generator instance.
    - Datos: Dictionary with dataframes.

    Returns:
    - None.

    Examples:
    - Agregar_Seccion_09(Reporte, Datos)
    """

    Reporte.Agregar_Separador("=")
    Reporte.Agregar(
        "Seccion 09. Centro con dos asociaciones."
    )
    Reporte.Agregar_Separador("=")
    Reporte.Agregar("")

    for Eleccion, Datos_Entrada in Datos.items():
        Reporte.Agregar_Separador("-")
        Reporte.Agregar(f"Eleccion {Eleccion}.")
        Reporte.Agregar_Separador("-")

        Muestra = Crear_Muestra_Balanceada_Tres_Poblaciones(
            Datos_Entrada
        )
        Datos_Tres = Calcular_Medias_Centro_Desglosado(Muestra)
        Significancia = Calcular_Significancia_Centro_Desglosado(
            Datos_Tres
        )

        Orden_Items = Datos_Tres["Orden_Items"]
        Medias_Base = Datos_Tres["Medias_Base"]
        Medias_Izq = Datos_Tres["Medias_Izq"]
        Medias_Centro_Izq = Datos_Tres["Medias_Centro_Izq"]
        Medias_Centro_Der = Datos_Tres["Medias_Centro_Der"]
        Medias_Der = Datos_Tres["Medias_Der"]

        Encabezado = [
            "Item",
            "Tipo",
            "Baseline",
            "Dif_Izq",
            "Sig_Izq",
            "Dif_Centro_Izq",
            "Sig_Centro_Izq",
            "Dif_Centro_Der",
            "Sig_Centro_Der",
            "Dif_Der",
            "Sig_Der",
        ]
        Reporte.Agregar(" | ".join(Encabezado) + ".")

        for Item in Orden_Items:
            Tipo = "Con" if Item in Items_Conservadores else "Pro"
            Base = Medias_Base.get(Item, Numpy.nan)
            Izq = Medias_Izq.get(Item, Numpy.nan)
            Centro_Izq = Medias_Centro_Izq.get(Item, Numpy.nan)
            Centro_Der = Medias_Centro_Der.get(Item, Numpy.nan)
            Der = Medias_Der.get(Item, Numpy.nan)

            Dif_Izq = Izq - Base
            Dif_Centro_Izq = Centro_Izq - Base
            Dif_Centro_Der = Centro_Der - Base
            Dif_Der = Der - Base

            Sig_Izq = "*" if Significancia["P_Izq"].get(
                Item, 1.0
            ) < 0.05 else ""
            Sig_Centro_Izq = "*" if Significancia[
                "P_Centro_Izq"
            ].get(Item, 1.0) < 0.05 else ""
            Sig_Centro_Der = "*" if Significancia[
                "P_Centro_Der"
            ].get(Item, 1.0) < 0.05 else ""
            Sig_Der = "*" if Significancia["P_Der"].get(
                Item, 1.0
            ) < 0.05 else ""

            Fila = [
                str(Item),
                Tipo,
                Formatear_Numero(Base),
                Formatear_Numero(Dif_Izq, Mostrar_Signo = True),
                Sig_Izq,
                Formatear_Numero(
                    Dif_Centro_Izq, Mostrar_Signo = True
                ),
                Sig_Centro_Izq,
                Formatear_Numero(
                    Dif_Centro_Der, Mostrar_Signo = True
                ),
                Sig_Centro_Der,
                Formatear_Numero(Dif_Der, Mostrar_Signo = True),
                Sig_Der,
            ]
            Reporte.Agregar(" | ".join(Fila) + ".")

        Reporte.Agregar("")
        Reporte.Agregar("Comparacion centro izquierda vs derecha.")

        Datos_Centro = Datos_Tres["Datos_Centro"]
        Encabezado = [
            "Item",
            "Tipo",
            "Media_Centro_Izq",
            "Media_Centro_Der",
            "Diferencia",
            "p_valor",
            "Sig",
        ]
        Reporte.Agregar(" | ".join(Encabezado) + ".")

        Cant_Sig = 0
        Diferencias = []

        for Item in Orden_Items:
            Tipo = "Con" if Item in Items_Conservadores else "Pro"
            Col_Izq = f"IP_Item_{Item}_Izq_Respuesta"
            Col_Der = f"IP_Item_{Item}_Der_Respuesta"
            Media_Izq = Medias_Centro_Izq.get(Item, Numpy.nan)
            Media_Der = Medias_Centro_Der.get(Item, Numpy.nan)
            Diferencia = Media_Izq - Media_Der
            Diferencias.append(Diferencia)

            if Col_Izq in Datos_Centro.columns and (
                Col_Der in Datos_Centro.columns
            ):
                Valor_P = Test_Wilcoxon_Pareado(
                    Datos_Centro[Col_Izq],
                    Datos_Centro[Col_Der]
                )
            else:
                Valor_P = 1.0

            Marca = Obtener_Significancia(Valor_P)
            if Marca:
                Cant_Sig += 1

            Fila = [
                str(Item),
                Tipo,
                Formatear_Numero(Media_Izq),
                Formatear_Numero(Media_Der),
                Formatear_Numero(Diferencia, Mostrar_Signo = True),
                Formatear_P_Valor(Valor_P),
                Marca,
            ]
            Reporte.Agregar(" | ".join(Fila) + ".")

        Media_Diferencias = (
            sum(Diferencias) / len(Diferencias)
            if Diferencias else Numpy.nan
        )
        Reporte.Agregar(
            f"Items con diferencia significativa: {Cant_Sig}."
        )
        Reporte.Agregar(
            "Diferencia media centro Izq-Der: "
            f"{Formatear_Numero(Media_Diferencias)}."
        )
        Reporte.Agregar("")


# ------------------------------------------------------------
# SECTION 10. SPLIT-CENTER QUANTIFICATION.
# ------------------------------------------------------------

def Agregar_Seccion_10(
    Reporte: Generador_Reporte,
    Datos: dict
) -> None:

    """
    Adds section 10 to the report.

    Parameters:
    - Reporte: Report generator instance.
    - Datos: Dictionary with dataframes.

    Returns:
    - None.

    Examples:
    - Agregar_Seccion_10(Reporte, Datos)
    """

    Reporte.Agregar_Separador("=")
    Reporte.Agregar("Seccion 10. Cuantificacion centro desglosado.")
    Reporte.Agregar_Separador("=")
    Reporte.Agregar("")

    Resultados = {}
    for Nombre, Datos_Entrada in Datos.items():
        Muestra = Crear_Muestra_Balanceada_Tres_Poblaciones(
            Datos_Entrada
        )
        Resultados[Nombre] = {
            "Muestra": Muestra,
            "Izq_Pro": Calcular_Area_Tres_Poblaciones(
                Muestra, "Izq", Items_Progresistas, "Izq"
            ),
            "Izq_Con": Calcular_Area_Tres_Poblaciones(
                Muestra, "Izq", Items_Conservadores, "Izq"
            ),
            "Centro_Pro_Izq": Calcular_Area_Tres_Poblaciones(
                Muestra, "Centro", Items_Progresistas, "Izq"
            ),
            "Centro_Pro_Der": Calcular_Area_Tres_Poblaciones(
                Muestra, "Centro", Items_Progresistas, "Der"
            ),
            "Centro_Con_Izq": Calcular_Area_Tres_Poblaciones(
                Muestra, "Centro", Items_Conservadores, "Izq"
            ),
            "Centro_Con_Der": Calcular_Area_Tres_Poblaciones(
                Muestra, "Centro", Items_Conservadores, "Der"
            ),
            "Der_Con": Calcular_Area_Tres_Poblaciones(
                Muestra, "Der", Items_Conservadores, "Der"
            ),
            "Der_Pro": Calcular_Area_Tres_Poblaciones(
                Muestra, "Der", Items_Progresistas, "Der"
            ),
        }

    Reporte.Agregar("Tabla de resultados vs cero y GE vs BA.")
    Encabezado = [
        "Poblacion",
        "Tipo_Item",
        "Media_GE",
        "Error_GE",
        "p_vs_0_GE",
        "Sig_GE",
        "Media_BA",
        "Error_BA",
        "p_vs_0_BA",
        "Sig_BA",
        "p_GE_vs_BA",
        "Sig_GE_vs_BA",
    ]
    Reporte.Agregar(" | ".join(Encabezado) + ".")

    Areas = [
        ("Izq_Pro", "Izquierda", "Progresistas"),
        ("Izq_Con", "Izquierda", "Conservadores"),
        ("Centro_Pro_Izq", "Centro_Izq", "Progresistas"),
        ("Centro_Pro_Der", "Centro_Der", "Progresistas"),
        ("Centro_Con_Izq", "Centro_Izq", "Conservadores"),
        ("Centro_Con_Der", "Centro_Der", "Conservadores"),
        ("Der_Con", "Derecha", "Conservadores"),
        ("Der_Pro", "Derecha", "Progresistas"),
    ]

    for Clave, Poblacion, Tipo_Item in Areas:
        Res_Ge = Resultados["Generales"][Clave]
        Res_Ba = Resultados["Ballotage"][Clave]

        Valor_P_Ge = Test_Wilcoxon_Contra_Cero(
            Res_Ge["Diferencias"]
        )
        Valor_P_Ba = Test_Wilcoxon_Contra_Cero(
            Res_Ba["Diferencias"]
        )
        Valor_P_Ge_Ba = Test_Mann_Whitney(
            Res_Ge["Diferencias"], Res_Ba["Diferencias"]
        )

        Fila = [
            Poblacion,
            Tipo_Item,
            Formatear_Numero(Res_Ge["Media"], Mostrar_Signo = True),
            Formatear_Numero(Res_Ge["Error"]),
            Formatear_P_Valor(Valor_P_Ge),
            Obtener_Significancia(Valor_P_Ge),
            Formatear_Numero(Res_Ba["Media"], Mostrar_Signo = True),
            Formatear_Numero(Res_Ba["Error"]),
            Formatear_P_Valor(Valor_P_Ba),
            Obtener_Significancia(Valor_P_Ba),
            Formatear_P_Valor(Valor_P_Ge_Ba),
            Obtener_Significancia(Valor_P_Ge_Ba),
        ]
        Reporte.Agregar(" | ".join(Fila) + ".")

    Reporte.Agregar("")
    Reporte.Agregar("Comparaciones entre poblaciones.")

    for Eleccion in ["Generales", "Ballotage"]:
        for Tipo_Item, Clave_Pro, Clave_Con in [
            ("Progresistas", "Izq_Pro", "Der_Pro"),
            ("Conservadores", "Izq_Con", "Der_Con"),
        ]:
            Valores_Izq = Resultados[Eleccion][
                Clave_Pro
            ]["Diferencias"]
            Valores_Centro = Numpy.concatenate([
                Resultados[Eleccion][
                    "Centro_Pro_Izq" if Tipo_Item == "Progresistas"
                    else "Centro_Con_Izq"
                ]["Diferencias"],
                Resultados[Eleccion][
                    "Centro_Pro_Der" if Tipo_Item == "Progresistas"
                    else "Centro_Con_Der"
                ]["Diferencias"],
            ])
            Valores_Der = Resultados[Eleccion][
                Clave_Con
            ]["Diferencias"]

            Valor_P_Izq_Centro = Test_Mann_Whitney(
                Valores_Izq, Valores_Centro
            )
            Valor_P_Centro_Der = Test_Mann_Whitney(
                Valores_Centro, Valores_Der
            )
            Valor_P_Izq_Der = Test_Mann_Whitney(
                Valores_Izq, Valores_Der
            )

            Filas = [
                [
                    Eleccion,
                    Tipo_Item,
                    "Izq vs Centro",
                    Formatear_P_Valor(Valor_P_Izq_Centro),
                    Obtener_Significancia(Valor_P_Izq_Centro),
                ],
                [
                    Eleccion,
                    Tipo_Item,
                    "Centro vs Der",
                    Formatear_P_Valor(Valor_P_Centro_Der),
                    Obtener_Significancia(Valor_P_Centro_Der),
                ],
                [
                    Eleccion,
                    Tipo_Item,
                    "Izq vs Der",
                    Formatear_P_Valor(Valor_P_Izq_Der),
                    Obtener_Significancia(Valor_P_Izq_Der),
                ],
            ]
            for Fila in Filas:
                Reporte.Agregar(" | ".join(Fila) + ".")

    Reporte.Agregar("")


# ------------------------------------------------------------
# MAIN EXECUTION.
# ------------------------------------------------------------

def Ejecutar_Analisis_Completo() -> None:

    """
    Runs the full analysis and generates the final report.

    Parameters:
    - No parameters.

    Returns:
    - None.

    Examples:
    - Ejecutar_Analisis_Completo()
    """

    Ruta_Base = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    Ruta_Datos = os.path.join(Ruta_Base, "Processed_Data")
    Ruta_Reportes = os.path.join(Ruta_Base, "Results", "Reports")

    if not os.path.exists(Ruta_Reportes):
        os.makedirs(Ruta_Reportes)

    Datos = Cargar_Bases_Datos(Ruta_Datos)
    if "Generales" not in Datos or "Ballotage" not in Datos:
        raise FileNotFoundError(
            "No se encontraron bases de datos esperadas."
        )

    Reporte = Generador_Reporte()
    Reporte.Agregar_Separador("=")
    Reporte.Agregar(
        "Reporte de analisis de poblaciones segmentadas."
    )
    Reporte.Agregar("Sin graficos. Resultados replicables.")
    Reporte.Agregar_Separador("=")
    Reporte.Agregar("")

    Agregar_Seccion_01(Reporte, Datos)
    Agregar_Seccion_02(Reporte, Datos)
    Agregar_Seccion_03(Reporte, Datos)
    Agregar_Seccion_04(Reporte, Datos)
    Agregar_Seccion_05(Reporte, Datos)
    Agregar_Seccion_06(Reporte, Datos)
    Agregar_Seccion_07(Reporte, Datos)
    Agregar_Seccion_08(Reporte, Datos)
    Agregar_Seccion_09(Reporte, Datos)
    Agregar_Seccion_10(Reporte, Datos)

    Reporte.Agregar_Separador("=")
    Reporte.Agregar("Fin del reporte.")
    Reporte.Agregar_Separador("=")

    Ruta_Reporte = os.path.join(
        Ruta_Reportes,
        "Report_Segmented_Population_Analysis.txt"
    )
    Reporte.Guardar(Ruta_Reporte)


if __name__ == "__main__":
    Ejecutar_Analisis_Completo()
