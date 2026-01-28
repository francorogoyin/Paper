# -*- coding: utf-8 -*-
"""
Statistical analyses.

This script contains functions to perform statistical analyses
including Kruskal-Wallis, Dunn post-hoc, Wilcoxon, and Mann-Whitney
tests on the processed databases.

"""

import pandas as pd
import numpy as np
from scipy.stats import kruskal, wilcoxon, mannwhitneyu
import scikit_posthocs as sp
import statsmodels.api as sm
import os


# =============================================================================
# CONSTANTS AND CONFIGURATION.
# =============================================================================

# Progressive and conservative items.
Items_Progresistas = [5, 6, 9, 11, 16, 20, 24, 25, 27, 28]
Items_Conservadores = [3, 4, 7, 8, 10, 19, 22, 23, 29, 30]

# Valid categories for analysis.
Categorias_Validas = [
    'Left_Wing',
    'Progressivism',
    'Centre',
    'Moderate_Right_A',
    'Moderate_Right_B',
    'Right_Wing_Libertarian'
]

# Category labels.
Etiquetas_Categorias = {
    'Left_Wing': 'Left Wing',
    'Progressivism': 'Progressivism',
    'Centre': 'Centre',
    'Moderate_Right_A': 'Moderate Right A',
    'Moderate_Right_B': 'Moderate Right B',
    'Right_Wing_Libertarian': 'Right Wing Libertarian'
}

# Variable name mapping Spanish -> English.
Mapeo_Variables_Es_En = {
    # Self-perceptions.
    'Autopercepcion_Izq_Der': 'Left_Right_Self_Perception',
    'Autopercepcion_Con_Pro': 'Conservative_Progressive_Self_Perception',
    'Autopercepcion_Per_Antiper': 'Peronist_Anti_Peronist_Self_Perception',
    # Indices.
    'Indice_Progresismo': 'Progressivism_Index',
    'Indice_Conservadurismo': 'Conservatism_Index',
    'Indice_Positividad': 'Positivity_Index',
    'Indice_Progresismo_Tiempo': 'Progressivism_Index_Time',
    'Indice_Conservadurismo_Tiempo': 'Conservatism_Index_Time',
    # Closeness ratings.
    'Cercania_Bregman': 'Closeness_Bregman',
    'Cercania_Massa': 'Closeness_Massa',
    'Cercania_Schiaretti': 'Closeness_Schiaretti',
    'Cercania_Bullrich': 'Closeness_Bullrich',
    'Cercania_Milei': 'Closeness_Milei',
    # Influences.
    'Influencia_Prensa': 'Written_Media_Influence',
    'Influencia_Redes': 'Social_Networks_Influence',
    # Gender.
    'Genero_Masculino': 'Male',
    'Genero_Otro': 'Other_Gender',
    # Regions.
    'Region_Norte': 'North',
    'Region_Patagonia': 'Patagonia',
    'Region_Centro': 'Central',
    'Region_CABA': 'CABA',
    'Region_Cuyo': 'Cuyo',
}


def Traducir_Variable(Nombre_Variable: str) -> str:

    """
    Translates a variable name from Spanish to English.

    Parameters:
        Nombre_Variable (str): Variable name in Spanish.

    Returns:
        str: Translated variable name or original if not found.

    """

    return Mapeo_Variables_Es_En.get(Nombre_Variable, Nombre_Variable)


def Formatear_P_Valor(P_Valor: float) -> str:

    """
    Formats a p-value in scientific notation with 2 decimals.

    Parameters:
        P_Valor (float): The p-value to format.

    Returns:
        str: The formatted p-value as string (e.g., "2.12E-15").

    """

    if pd.isna(P_Valor):
        return "NaN"

    return f"{P_Valor:.2E}"


# =============================================================================
# DATA LOADING FUNCTIONS.
# =============================================================================

def Cargar_Bases_Datos(Ruta_Carpeta: str) -> dict:

    """
    Loads General and Ballotage databases from Excel files.

    Parameters:
        Ruta_Carpeta (str): Path to folder containing
            the Excel files.

    Returns:
        dict: Dictionary with loaded databases.

    """

    dfs_Finales = {}

    for Nombre in ['Generales', 'Ballotage']:
        Ruta_Archivo = os.path.join(Ruta_Carpeta, f'{Nombre}.xlsx')
        if os.path.exists(Ruta_Archivo):
            dfs_Finales[Nombre] = pd.read_excel(Ruta_Archivo)
            print(f"  Loaded: {Nombre} ({len(dfs_Finales[Nombre])} records)")
        else:
            print(f"  Warning: {Ruta_Archivo} not found")

    return dfs_Finales


# =============================================================================
# KRUSKAL-WALLIS FUNCTIONS.
# =============================================================================

def Kruskal_Wallis_Por_Categoria(
    df: pd.DataFrame,
    Columnas: list,
    Columna_Categoria: str = 'Categoria_PASO_2023'
) -> pd.DataFrame:

    """
    Applies Kruskal-Wallis test to compare multiple groups
    defined by a categorical column.

    Parameters:
        df (pd.DataFrame): DataFrame with data.
        Columnas (list): List of numeric columns to analyze.
        Columna_Categoria (str): Categorical column name
            for grouping.

    Returns:
        pd.DataFrame: Table with Kruskal-Wallis results.

    """

    # Filter only valid categories.
    df_Filtrado = df[df[Columna_Categoria].isin(Categorias_Validas)].copy()

    Resultados = {}

    for Nombre_Columna in Columnas:
        # Create groups by category.
        Grupos = [
            df_Filtrado[
                df_Filtrado[Columna_Categoria] == Categoria
            ][Nombre_Columna].dropna()
            for Categoria in Categorias_Validas
        ]

        # Verify that all groups have at least 2 data points.
        if all(len(Grupo) > 1 for Grupo in Grupos):
            Estadistico, P_Valor = kruskal(*Grupos)
            Resultados[Nombre_Columna] = {
                'Estadistico_H': Estadistico,
                'Valor_p': P_Valor
            }

    # Convert to DataFrame.
    Tabla_Resultados = pd.DataFrame.from_dict(
        Resultados, orient='index'
    ).reset_index().rename(columns={'index': 'Variable'})

    # Sort by p-value ascending.
    Tabla_Resultados = Tabla_Resultados.sort_values(
        by='Valor_p'
    ).reset_index(drop=True)

    # Add significance column.
    Tabla_Resultados['Significativo'] = \
        Tabla_Resultados['Valor_p'] < 0.05

    return Tabla_Resultados


def Obtener_Columnas_CO_Items() -> tuple:

    """
    Generates lists of Opinion Change columns by item.

    Returns:
        tuple: (Columnas_CO_Pro_Izq, Columnas_CO_Pro_Der,
                Columnas_CO_Con_Izq, Columnas_CO_Con_Der,
                Todas_Columnas_CO)

    """

    Columnas_CO_Pro_Izq = [
        f'CO_Item_{i}_Izq' for i in Items_Progresistas
    ]
    Columnas_CO_Pro_Der = [
        f'CO_Item_{i}_Der' for i in Items_Progresistas
    ]
    Columnas_CO_Con_Izq = [
        f'CO_Item_{i}_Izq' for i in Items_Conservadores
    ]
    Columnas_CO_Con_Der = [
        f'CO_Item_{i}_Der' for i in Items_Conservadores
    ]

    Todas_Columnas_CO = (
        Columnas_CO_Pro_Izq + Columnas_CO_Pro_Der +
        Columnas_CO_Con_Izq + Columnas_CO_Con_Der
    )

    return (
        Columnas_CO_Pro_Izq,
        Columnas_CO_Pro_Der,
        Columnas_CO_Con_Izq,
        Columnas_CO_Con_Der,
        Todas_Columnas_CO
    )


def Obtener_Columnas_CT_Items() -> tuple:

    """
    Generates lists of Time Change columns by item.

    Returns:
        tuple: (Columnas_CT_Pro_Izq, Columnas_CT_Pro_Der,
                Columnas_CT_Con_Izq, Columnas_CT_Con_Der,
                Todas_Columnas_CT)

    """

    Columnas_CT_Pro_Izq = [
        f'CT_Item_{i}_Izq' for i in Items_Progresistas
    ]
    Columnas_CT_Pro_Der = [
        f'CT_Item_{i}_Der' for i in Items_Progresistas
    ]
    Columnas_CT_Con_Izq = [
        f'CT_Item_{i}_Izq' for i in Items_Conservadores
    ]
    Columnas_CT_Con_Der = [
        f'CT_Item_{i}_Der' for i in Items_Conservadores
    ]

    Todas_Columnas_CT = (
        Columnas_CT_Pro_Izq + Columnas_CT_Pro_Der +
        Columnas_CT_Con_Izq + Columnas_CT_Con_Der
    )

    return (
        Columnas_CT_Pro_Izq,
        Columnas_CT_Pro_Der,
        Columnas_CT_Con_Izq,
        Columnas_CT_Con_Der,
        Todas_Columnas_CT
    )


# =============================================================================
# DUNN POST-HOC FUNCTIONS.
# =============================================================================

def Dunn_Post_Hoc(
    df: pd.DataFrame,
    Variable: str,
    Columna_Categoria: str = 'Categoria_PASO_2023',
    Metodo_Ajuste: str = 'bonferroni'
) -> pd.DataFrame:

    """
    Applies Dunn post-hoc test with Bonferroni correction
    for pairwise comparisons between categories.

    Parameters:
        df (pd.DataFrame): DataFrame with data.
        Variable (str): Name of column to analyze.
        Columna_Categoria (str): Categorical column name.
        Metodo_Ajuste (str): P-value adjustment method.

    Returns:
        pd.DataFrame: Matrix of adjusted p-values.

    """

    # Filter valid categories and remove NaN.
    df_Filtrado = df[
        df[Columna_Categoria].isin(Categorias_Validas)
    ][[Columna_Categoria, Variable]].dropna()

    # Apply Dunn test.
    Matriz_Dunn = sp.posthoc_dunn(
        df_Filtrado,
        val_col=Variable,
        group_col=Columna_Categoria,
        p_adjust=Metodo_Ajuste
    )

    return Matriz_Dunn


def Extraer_Comparaciones_Significativas(
    Matriz_Dunn: pd.DataFrame,
    Umbral_P: float = 0.05
) -> list:

    """
    Extracts significant pairwise comparisons from a
    Dunn matrix.

    Parameters:
        Matriz_Dunn (pd.DataFrame): P-value matrix.
        Umbral_P (float): Significance threshold.

    Returns:
        list: List of dictionaries with significant comparisons.

    """

    Comparaciones = []

    for Categoria_1 in Matriz_Dunn.index:
        for Categoria_2 in Matriz_Dunn.columns:
            # Avoid duplicates (upper triangle only).
            if Categoria_1 < Categoria_2:
                P_Valor = Matriz_Dunn.loc[Categoria_1, Categoria_2]
                if P_Valor < Umbral_P:
                    Comparaciones.append({
                        'Categoria_1': Categoria_1,
                        'Categoria_2': Categoria_2,
                        'Valor_p': P_Valor
                    })

    return Comparaciones


def Ejecutar_Dunn_Variables_Significativas(
    df: pd.DataFrame,
    Variables_Significativas: list,
    Columna_Categoria: str = 'Categoria_PASO_2023'
) -> dict:

    """
    Executes Dunn post-hoc for all significant Kruskal-Wallis
    variables.

    Parameters:
        df (pd.DataFrame): DataFrame with data.
        Variables_Significativas (list): List of variables.
        Columna_Categoria (str): Categorical column.

    Returns:
        dict: Dictionary with Dunn matrices by variable.

    """

    Resultados_Dunn = {}

    for Variable in Variables_Significativas:
        Matriz = Dunn_Post_Hoc(df, Variable, Columna_Categoria)
        Resultados_Dunn[Variable] = Matriz

    return Resultados_Dunn


# =============================================================================
# WILCOXON FUNCTIONS.
# =============================================================================

def Wilcoxon_Pareado(
    df: pd.DataFrame,
    Variable_1: str,
    Variable_2: str
) -> dict:

    """
    Applies paired Wilcoxon test between two variables.

    Parameters:
        df (pd.DataFrame): DataFrame with data.
        Variable_1 (str): First variable.
        Variable_2 (str): Second variable.

    Returns:
        dict: Dictionary with test results.

    """

    # Remove NaN in both variables.
    Datos_Pareados = df[[Variable_1, Variable_2]].dropna()

    if len(Datos_Pareados) < 2:
        return {
            'n': len(Datos_Pareados),
            'Estadistico_W': np.nan,
            'Valor_p': np.nan,
            'Significativo': False,
            'Error': 'Insufficient data'
        }

    try:
        Estadistico, P_Valor = wilcoxon(
            Datos_Pareados[Variable_1],
            Datos_Pareados[Variable_2]
        )

        # Determine significance level.
        if P_Valor < 0.001:
            Sig = '***'
        elif P_Valor < 0.01:
            Sig = '**'
        elif P_Valor < 0.05:
            Sig = '*'
        else:
            Sig = 'ns'

        return {
            'n': len(Datos_Pareados),
            'Media_1': Datos_Pareados[Variable_1].mean(),
            'Media_2': Datos_Pareados[Variable_2].mean(),
            'Estadistico_W': Estadistico,
            'Valor_p': P_Valor,
            'Significancia': Sig,
            'Significativo': P_Valor < 0.05
        }

    except Exception as e:
        return {
            'n': len(Datos_Pareados),
            'Estadistico_W': np.nan,
            'Valor_p': np.nan,
            'Significativo': False,
            'Error': str(e)
        }


def Wilcoxon_Congruencia(
    df: pd.DataFrame,
    Tipo: str = 'CO'
) -> dict:

    """
    Applies Wilcoxon to compare Congruent vs Incongruent.

    Parameters:
        df (pd.DataFrame): DataFrame with data.
        Tipo (str): 'CO' for Opinion Change or 'CT' for
            Time Change.

    Returns:
        dict: Test results.

    """

    Variable_1 = f'{Tipo}_Congruente'
    Variable_2 = f'{Tipo}_Incongruente'

    return Wilcoxon_Pareado(df, Variable_1, Variable_2)


def Wilcoxon_Congruencia_Por_Categoria(
    df: pd.DataFrame,
    Tipo: str = 'CO',
    Columna_Categoria: str = 'Categoria_PASO_2023'
) -> pd.DataFrame:

    """
    Applies Wilcoxon Congruent vs Incongruent for each category.

    Parameters:
        df (pd.DataFrame): DataFrame with data.
        Tipo (str): 'CO' or 'CT'.
        Columna_Categoria (str): Categorical column.

    Returns:
        pd.DataFrame: Results by category.

    """

    Resultados = []

    for Categoria in Categorias_Validas:
        df_Cat = df[df[Columna_Categoria] == Categoria].copy()

        Resultado = Wilcoxon_Congruencia(df_Cat, Tipo)
        Resultado['Categoria'] = Categoria
        Resultado['Etiqueta'] = Etiquetas_Categorias[Categoria]
        Resultado['Tipo'] = Tipo

        Resultados.append(Resultado)

    return pd.DataFrame(Resultados)


# =============================================================================
# WILCOXON FUNCTIONS FOR CURVES (UNASSOCIATED VS ASSOCIATED).
# =============================================================================

def Calcular_Significancia_Wilcoxon_Curvas_Asociacion(
    Datos: dict
) -> dict:

    """
    Calculates Wilcoxon between unassociated and associated items
    for right and left populations.

    Parameters:
        Datos (dict): Dictionary with 'Orden_Items', 'df_Derecha'
            and 'df_Izquierda'.

    Returns:
        dict: P-values by item for right and left.

    """

    Orden_Items = Datos['Orden_Items']
    df_Derecha = Datos['df_Derecha']
    df_Izquierda = Datos['df_Izquierda']

    P_Valores_Derecha = {}
    P_Valores_Izquierda = {}

    for Item in Orden_Items:
        Col_Base = f'IP_Item_{Item}_Respuesta'
        Col_Der = f'IP_Item_{Item}_Der_Respuesta'
        Col_Izq = f'IP_Item_{Item}_Izq_Respuesta'

        if Col_Base in df_Derecha.columns and Col_Der in df_Derecha.columns:
            Datos_Pares_Der = df_Derecha[[Col_Base, Col_Der]].dropna()
            if len(Datos_Pares_Der) > 10:
                try:
                    _, P_Valor = wilcoxon(
                        Datos_Pares_Der[Col_Base],
                        Datos_Pares_Der[Col_Der],
                        alternative='two-sided'
                    )
                    P_Valores_Derecha[Item] = P_Valor
                except Exception:
                    P_Valores_Derecha[Item] = 1.0
            else:
                P_Valores_Derecha[Item] = 1.0

        if Col_Base in df_Izquierda.columns and Col_Izq in df_Izquierda.columns:
            Datos_Pares_Izq = df_Izquierda[[Col_Base, Col_Izq]].dropna()
            if len(Datos_Pares_Izq) > 10:
                try:
                    _, P_Valor = wilcoxon(
                        Datos_Pares_Izq[Col_Base],
                        Datos_Pares_Izq[Col_Izq],
                        alternative='two-sided'
                    )
                    P_Valores_Izquierda[Item] = P_Valor
                except Exception:
                    P_Valores_Izquierda[Item] = 1.0
            else:
                P_Valores_Izquierda[Item] = 1.0

    return {
        'P_Valores_Derecha': P_Valores_Derecha,
        'P_Valores_Izquierda': P_Valores_Izquierda
    }


def Calcular_Significancia_Wilcoxon_Baseline_Propia(
    Datos: dict
) -> dict:

    """
    Calculates Wilcoxon for each population comparing its baseline
    (unassociated) vs associated.

    Parameters:
        Datos (dict): Dictionary with 'Orden_Items', 'df_Izq' and 'df_Der'.

    Returns:
        dict: P-values by item for left and right.

    """

    Orden_Items = Datos['Orden_Items']
    df_Izq = Datos['df_Izq']
    df_Der = Datos['df_Der']

    P_Valores_Izq = {}
    P_Valores_Der = {}

    for Item in Orden_Items:
        Col_Base = f'IP_Item_{Item}_Respuesta'
        Col_Izq = f'IP_Item_{Item}_Izq_Respuesta'
        Col_Der = f'IP_Item_{Item}_Der_Respuesta'

        if Col_Base in df_Izq.columns and Col_Izq in df_Izq.columns:
            Pares = df_Izq[[Col_Base, Col_Izq]].dropna()
            if len(Pares) > 10:
                try:
                    _, P_Valor = wilcoxon(Pares[Col_Base], Pares[Col_Izq])
                    P_Valores_Izq[Item] = P_Valor
                except Exception:
                    P_Valores_Izq[Item] = 1.0
            else:
                P_Valores_Izq[Item] = 1.0

        if Col_Base in df_Der.columns and Col_Der in df_Der.columns:
            Pares = df_Der[[Col_Base, Col_Der]].dropna()
            if len(Pares) > 10:
                try:
                    _, P_Valor = wilcoxon(Pares[Col_Base], Pares[Col_Der])
                    P_Valores_Der[Item] = P_Valor
                except Exception:
                    P_Valores_Der[Item] = 1.0
            else:
                P_Valores_Der[Item] = 1.0

    return {'P_Valores_Izq': P_Valores_Izq, 'P_Valores_Der': P_Valores_Der}


def Calcular_Significancia_Wilcoxon_Balanceada(
    Datos: dict
) -> dict:

    """
    Calculates Wilcoxon on balanced sample (same individuals)
    comparing baseline vs associated.

    Parameters:
        Datos (dict): Dictionary with 'Orden_Items', 'df_Izq' and 'df_Der'.

    Returns:
        dict: P-values by item for left and right.

    """

    Orden_Items = Datos['Orden_Items']
    df_Izq = Datos['df_Izq']
    df_Der = Datos['df_Der']

    P_Valores_Izq = {}
    P_Valores_Der = {}

    for Item in Orden_Items:
        Col_Base = f'IP_Item_{Item}_Respuesta'
        Col_Izq = f'IP_Item_{Item}_Izq_Respuesta'
        Col_Der = f'IP_Item_{Item}_Der_Respuesta'

        if Col_Base in df_Izq.columns and Col_Izq in df_Izq.columns:
            Pares = df_Izq[[Col_Base, Col_Izq]].dropna()
            if len(Pares) > 10:
                try:
                    _, P_Valor = wilcoxon(Pares[Col_Base], Pares[Col_Izq])
                    P_Valores_Izq[Item] = P_Valor
                except Exception:
                    P_Valores_Izq[Item] = 1.0
            else:
                P_Valores_Izq[Item] = 1.0
        else:
            P_Valores_Izq[Item] = 1.0

        if Col_Base in df_Der.columns and Col_Der in df_Der.columns:
            Pares = df_Der[[Col_Base, Col_Der]].dropna()
            if len(Pares) > 10:
                try:
                    _, P_Valor = wilcoxon(Pares[Col_Base], Pares[Col_Der])
                    P_Valores_Der[Item] = P_Valor
                except Exception:
                    P_Valores_Der[Item] = 1.0
            else:
                P_Valores_Der[Item] = 1.0
        else:
            P_Valores_Der[Item] = 1.0

    return {'P_Valores_Izq': P_Valores_Izq, 'P_Valores_Der': P_Valores_Der}


def Calcular_Significancia_Wilcoxon_Tres_Poblaciones(
    Datos: dict
) -> dict:

    """
    Calculates Wilcoxon for left, center and right populations.
    For center, compares baseline vs average of Left+Right.

    Parameters:
        Datos (dict): Dictionary with 'Orden_Items', 'df_Izq',
            'df_Centro' and 'df_Der'.

    Returns:
        dict: P-values by item for left, center and right.

    """

    Orden_Items = Datos['Orden_Items']
    df_Izq = Datos['df_Izq']
    df_Centro = Datos['df_Centro']
    df_Der = Datos['df_Der']

    P_Valores_Izq = {}
    P_Valores_Centro = {}
    P_Valores_Der = {}

    for Item in Orden_Items:
        Col_Base = f'IP_Item_{Item}_Respuesta'
        Col_Izq = f'IP_Item_{Item}_Izq_Respuesta'
        Col_Der = f'IP_Item_{Item}_Der_Respuesta'

        if Col_Base in df_Izq.columns and Col_Izq in df_Izq.columns:
            Pares = df_Izq[[Col_Base, Col_Izq]].dropna()
            if len(Pares) > 10:
                try:
                    _, P_Valor = wilcoxon(Pares[Col_Base], Pares[Col_Izq])
                    P_Valores_Izq[Item] = P_Valor
                except Exception:
                    P_Valores_Izq[Item] = 1.0
            else:
                P_Valores_Izq[Item] = 1.0
        else:
            P_Valores_Izq[Item] = 1.0

        if (Col_Base in df_Centro.columns and
                Col_Izq in df_Centro.columns and
                Col_Der in df_Centro.columns):
            df_Temp = df_Centro[[Col_Base, Col_Izq, Col_Der]].dropna()
            if len(df_Temp) > 10:
                try:
                    Promedio_Asoc = (df_Temp[Col_Izq] + df_Temp[Col_Der]) / 2
                    _, P_Valor = wilcoxon(df_Temp[Col_Base], Promedio_Asoc)
                    P_Valores_Centro[Item] = P_Valor
                except Exception:
                    P_Valores_Centro[Item] = 1.0
            else:
                P_Valores_Centro[Item] = 1.0
        else:
            P_Valores_Centro[Item] = 1.0

        if Col_Base in df_Der.columns and Col_Der in df_Der.columns:
            Pares = df_Der[[Col_Base, Col_Der]].dropna()
            if len(Pares) > 10:
                try:
                    _, P_Valor = wilcoxon(Pares[Col_Base], Pares[Col_Der])
                    P_Valores_Der[Item] = P_Valor
                except Exception:
                    P_Valores_Der[Item] = 1.0
            else:
                P_Valores_Der[Item] = 1.0
        else:
            P_Valores_Der[Item] = 1.0

    return {
        'P_Valores_Izq': P_Valores_Izq,
        'P_Valores_Centro': P_Valores_Centro,
        'P_Valores_Der': P_Valores_Der
    }


def Calcular_Significancia_Wilcoxon_Centro_Dos_Asociaciones(
    Datos: dict
) -> dict:

    """
    Calculates Wilcoxon for left, center (Left and Right associations)
    and right populations.

    Parameters:
        Datos (dict): Dictionary with 'Orden_Items', 'df_Izq',
            'df_Centro' and 'df_Der'.

    Returns:
        dict: P-values by item for left, center and right.

    """

    Orden_Items = Datos['Orden_Items']
    df_Izq = Datos['df_Izq']
    df_Centro = Datos['df_Centro']
    df_Der = Datos['df_Der']

    P_Valores_Izq = {}
    P_Valores_Centro_Izq = {}
    P_Valores_Centro_Der = {}
    P_Valores_Der = {}

    for Item in Orden_Items:
        Col_Base = f'IP_Item_{Item}_Respuesta'
        Col_Izq = f'IP_Item_{Item}_Izq_Respuesta'
        Col_Der = f'IP_Item_{Item}_Der_Respuesta'

        if Col_Base in df_Izq.columns and Col_Izq in df_Izq.columns:
            Pares = df_Izq[[Col_Base, Col_Izq]].dropna()
            if len(Pares) > 10:
                try:
                    _, P_Valor = wilcoxon(Pares[Col_Base], Pares[Col_Izq])
                    P_Valores_Izq[Item] = P_Valor
                except Exception:
                    P_Valores_Izq[Item] = 1.0
            else:
                P_Valores_Izq[Item] = 1.0
        else:
            P_Valores_Izq[Item] = 1.0

        if Col_Base in df_Centro.columns and Col_Izq in df_Centro.columns:
            Pares = df_Centro[[Col_Base, Col_Izq]].dropna()
            if len(Pares) > 10:
                try:
                    _, P_Valor = wilcoxon(Pares[Col_Base], Pares[Col_Izq])
                    P_Valores_Centro_Izq[Item] = P_Valor
                except Exception:
                    P_Valores_Centro_Izq[Item] = 1.0
            else:
                P_Valores_Centro_Izq[Item] = 1.0
        else:
            P_Valores_Centro_Izq[Item] = 1.0

        if Col_Base in df_Centro.columns and Col_Der in df_Centro.columns:
            Pares = df_Centro[[Col_Base, Col_Der]].dropna()
            if len(Pares) > 10:
                try:
                    _, P_Valor = wilcoxon(Pares[Col_Base], Pares[Col_Der])
                    P_Valores_Centro_Der[Item] = P_Valor
                except Exception:
                    P_Valores_Centro_Der[Item] = 1.0
            else:
                P_Valores_Centro_Der[Item] = 1.0
        else:
            P_Valores_Centro_Der[Item] = 1.0

        if Col_Base in df_Der.columns and Col_Der in df_Der.columns:
            Pares = df_Der[[Col_Base, Col_Der]].dropna()
            if len(Pares) > 10:
                try:
                    _, P_Valor = wilcoxon(Pares[Col_Base], Pares[Col_Der])
                    P_Valores_Der[Item] = P_Valor
                except Exception:
                    P_Valores_Der[Item] = 1.0
            else:
                P_Valores_Der[Item] = 1.0
        else:
            P_Valores_Der[Item] = 1.0

    return {
        'P_Valores_Izq': P_Valores_Izq,
        'P_Valores_Centro_Izq': P_Valores_Centro_Izq,
        'P_Valores_Centro_Der': P_Valores_Centro_Der,
        'P_Valores_Der': P_Valores_Der
    }


# =============================================================================
# MANN-WHITNEY U FUNCTIONS.
# =============================================================================

def Test_Mann_Whitney(Valores_1: list, Valores_2: list) -> float:

    """
    Performs Mann-Whitney U test to compare two groups.

    """

    if len(Valores_1) < 3 or len(Valores_2) < 3:
        return 1.0

    try:
        _, P_Valor = mannwhitneyu(
            Valores_1,
            Valores_2,
            alternative='two-sided'
        )
        return P_Valor
    except Exception:
        return 1.0


def Test_Wilcoxon_Contra_Cero(Valores: list) -> float:

    """
    Performs Wilcoxon signed-rank test against zero.
    Tests if differences are significantly different from 0.

    """

    if len(Valores) < 3:
        return 1.0

    try:
        _, P_Valor = wilcoxon(Valores, alternative='two-sided')
        return P_Valor
    except Exception:
        return 1.0


def Obtener_Significancia_Str(P_Valor: float) -> str:

    """
    Converts p-value to significance string.

    """

    if P_Valor < 0.001:
        return '***'
    elif P_Valor < 0.01:
        return '**'
    elif P_Valor < 0.05:
        return '*'
    else:
        return ''


def Mann_Whitney_Entre_Datasets(
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
    Variable: str,
    Nombre_1: str = 'Generales',
    Nombre_2: str = 'Ballotage'
) -> dict:

    """
    Applies Mann-Whitney U to compare a variable between
    two datasets (General vs Ballotage).

    Parameters:
        df_1 (pd.DataFrame): First DataFrame.
        df_2 (pd.DataFrame): Second DataFrame.
        Variable (str): Variable to compare.
        Nombre_1 (str): First dataset name.
        Nombre_2 (str): Second dataset name.

    Returns:
        dict: Test results.

    """

    Datos_1 = df_1[Variable].dropna()
    Datos_2 = df_2[Variable].dropna()

    if len(Datos_1) < 2 or len(Datos_2) < 2:
        return {
            'Variable': Variable,
            f'n_{Nombre_1}': len(Datos_1),
            f'n_{Nombre_2}': len(Datos_2),
            'Estadistico_U': np.nan,
            'Valor_p': np.nan,
            'Significativo': False,
            'Error': 'Insufficient data'
        }

    try:
        Estadistico, P_Valor = mannwhitneyu(
            Datos_1,
            Datos_2,
            alternative='two-sided'
        )

        if P_Valor < 0.001:
            Sig = '***'
        elif P_Valor < 0.01:
            Sig = '**'
        elif P_Valor < 0.05:
            Sig = '*'
        else:
            Sig = 'ns'

        return {
            'Variable': Variable,
            f'n_{Nombre_1}': len(Datos_1),
            f'n_{Nombre_2}': len(Datos_2),
            f'Media_{Nombre_1}': Datos_1.mean(),
            f'Media_{Nombre_2}': Datos_2.mean(),
            'Estadistico_U': Estadistico,
            'Valor_p': P_Valor,
            'Significancia': Sig,
            'Significativo': P_Valor < 0.05
        }

    except Exception as e:
        return {
            'Variable': Variable,
            f'n_{Nombre_1}': len(Datos_1),
            f'n_{Nombre_2}': len(Datos_2),
            'Estadistico_U': np.nan,
            'Valor_p': np.nan,
            'Significativo': False,
            'Error': str(e)
        }


# =============================================================================
# COMPLETE ANALYSIS FUNCTIONS.
# =============================================================================

def Analisis_Kruskal_CO_Items(dfs_Finales: dict) -> dict:

    """
    Executes Kruskal-Wallis for all individual CO items
    in both datasets.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.

    Returns:
        dict: Results by dataset.

    """

    _, _, _, _, Todas_Columnas_CO = Obtener_Columnas_CO_Items()

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Kruskal-Wallis CO Items - {Nombre_df}")
        print('='*60)

        Tabla = Kruskal_Wallis_Por_Categoria(df, Todas_Columnas_CO)
        Resultados[Nombre_df] = Tabla

        # Show summary.
        N_Significativas = Tabla['Significativo'].sum()
        print(f"Significant variables: {N_Significativas}/{len(Tabla)}")

        if N_Significativas > 0:
            print("\nSignificant variables (p < 0.05):")
            Sig = Tabla[Tabla['Significativo']]
            for _, Fila in Sig.iterrows():
                print(f"  - {Fila['Variable']}: "
                      f"H={Fila['Estadistico_H']:.4f}, "
                      f"p={Formatear_P_Valor(Fila['Valor_p'])}")

    return Resultados


def Analisis_Kruskal_CT_Items(dfs_Finales: dict) -> dict:

    """
    Executes Kruskal-Wallis for all individual CT items
    in both datasets.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.

    Returns:
        dict: Results by dataset.

    """

    _, _, _, _, Todas_Columnas_CT = Obtener_Columnas_CT_Items()

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Kruskal-Wallis CT Items - {Nombre_df}")
        print('='*60)

        Tabla = Kruskal_Wallis_Por_Categoria(df, Todas_Columnas_CT)
        Resultados[Nombre_df] = Tabla

        # Show summary.
        N_Significativas = Tabla['Significativo'].sum()
        print(f"Significant variables: {N_Significativas}/{len(Tabla)}")

        if N_Significativas > 0:
            print("\nSignificant variables (p < 0.05):")
            Sig = Tabla[Tabla['Significativo']]
            for _, Fila in Sig.iterrows():
                print(f"  - {Fila['Variable']}: "
                      f"H={Fila['Estadistico_H']:.4f}, "
                      f"p={Formatear_P_Valor(Fila['Valor_p'])}")

    return Resultados


def Obtener_Columnas_Agregadas() -> list:

    """
    Returns the list of all aggregated CO and CT columns.

    Returns:
        list: List of aggregated column names.

    """

    # Aggregated CO variables.
    Columnas_CO_Agregadas = [
        'CO_Pro_Izq', 'CO_Con_Izq', 'CO_Pro_Der', 'CO_Con_Der',
        'CO_Congruente', 'CO_Incongruente'
    ]

    # Aggregated CT variables.
    Columnas_CT_Agregadas = [
        'CT_Pro_Izq', 'CT_Con_Izq', 'CT_Pro_Der', 'CT_Con_Der',
        'CT_Congruente', 'CT_Incongruente'
    ]

    return Columnas_CO_Agregadas + Columnas_CT_Agregadas


def Analisis_Kruskal_Variables_Agregadas(dfs_Finales: dict) -> dict:

    """
    Executes Kruskal-Wallis for aggregated CO and CT variables.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.

    Returns:
        dict: Results by dataset.

    """

    Todas_Agregadas = Obtener_Columnas_Agregadas()

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Kruskal-Wallis Variables Agregadas - {Nombre_df}")
        print('='*60)

        # Filter only existing columns.
        Columnas_Existentes = [
            c for c in Todas_Agregadas if c in df.columns
        ]

        if not Columnas_Existentes:
            print("  No aggregated variables found.")
            continue

        Tabla = Kruskal_Wallis_Por_Categoria(df, Columnas_Existentes)
        Resultados[Nombre_df] = Tabla

        # Show summary.
        N_Significativas = Tabla['Significativo'].sum()
        print(f"Significant variables: {N_Significativas}/{len(Tabla)}")

        print("\nResults:")
        for _, Fila in Tabla.iterrows():
            Marca = "***" if Fila['Valor_p'] < 0.001 else \
                    "**" if Fila['Valor_p'] < 0.01 else \
                    "*" if Fila['Valor_p'] < 0.05 else ""
            print(f"  {Fila['Variable']}: "
                  f"H={Fila['Estadistico_H']:.4f}, "
                  f"p={Formatear_P_Valor(Fila['Valor_p'])} {Marca}")

    return Resultados


def Analisis_Dunn_Variables_Agregadas(
    dfs_Finales: dict,
    Resultados_Kruskal_Agregadas: dict
) -> dict:

    """
    Executes Dunn post-hoc for significant Kruskal-Wallis
    aggregated variables.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.
        Resultados_Kruskal_Agregadas (dict): Kruskal-Wallis results
            for aggregated variables.

    Returns:
        dict: Dunn results by dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Dunn Post-Hoc Variables Agregadas - {Nombre_df}")
        print('='*60)

        Tabla_Kruskal = Resultados_Kruskal_Agregadas.get(Nombre_df)

        if Tabla_Kruskal is None:
            print("  No Kruskal-Wallis results available.")
            continue

        # Get significant variables.
        Variables_Sig = Tabla_Kruskal[
            Tabla_Kruskal['Significativo']
        ]['Variable'].tolist()

        if not Variables_Sig:
            print("  No significant variables for post-hoc.")
            continue

        print(f"Variables for post-hoc: {len(Variables_Sig)}")

        Resultados_Dunn = Ejecutar_Dunn_Variables_Significativas(
            df, Variables_Sig
        )

        Resultados[Nombre_df] = Resultados_Dunn

        # Show summary of significant comparisons.
        for Variable, Matriz in Resultados_Dunn.items():
            Comparaciones = Extraer_Comparaciones_Significativas(Matriz)

            if Comparaciones:
                print(f"\n  {Variable}:")
                for Comp in Comparaciones:
                    print(f"    - {Comp['Categoria_1']} vs "
                          f"{Comp['Categoria_2']}: "
                          f"p={Formatear_P_Valor(Comp['Valor_p'])}")
            else:
                print(f"\n  {Variable}: No significant comparisons")

    return Resultados


def Analisis_Dunn_Post_Hoc(
    dfs_Finales: dict,
    Resultados_Kruskal: dict
) -> dict:

    """
    Executes Dunn post-hoc for significant Kruskal-Wallis
    variables.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.
        Resultados_Kruskal (dict): Kruskal-Wallis results.

    Returns:
        dict: Dunn results by dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Dunn Post-Hoc - {Nombre_df}")
        print('='*60)

        Tabla_Kruskal = Resultados_Kruskal.get(Nombre_df)

        if Tabla_Kruskal is None:
            print("  No Kruskal-Wallis results available.")
            continue

        # Get significant variables.
        Variables_Sig = Tabla_Kruskal[
            Tabla_Kruskal['Significativo']
        ]['Variable'].tolist()

        if not Variables_Sig:
            print("  No significant variables for post-hoc.")
            continue

        print(f"Variables for post-hoc: {len(Variables_Sig)}")

        Resultados_Dunn = Ejecutar_Dunn_Variables_Significativas(
            df, Variables_Sig
        )

        Resultados[Nombre_df] = Resultados_Dunn

        # Show summary of significant comparisons.
        for Variable, Matriz in Resultados_Dunn.items():
            Comparaciones = Extraer_Comparaciones_Significativas(Matriz)

            if Comparaciones:
                print(f"\n  {Variable}:")
                for Comp in Comparaciones:
                    print(f"    - {Comp['Categoria_1']} vs "
                          f"{Comp['Categoria_2']}: "
                          f"p={Formatear_P_Valor(Comp['Valor_p'])}")
            else:
                print(f"\n  {Variable}: No significant comparisons")

    return Resultados


def Analisis_Wilcoxon_Congruencia(dfs_Finales: dict) -> dict:

    """
    Executes Wilcoxon to compare Congruent vs Incongruent.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.

    Returns:
        dict: Results by dataset and type.

    """

    Resultados = {'General': [], 'Por_Categoria': []}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Wilcoxon Congruencia - {Nombre_df}")
        print('='*60)

        # General analysis (all populations).
        for Tipo in ['CO', 'CT']:
            Resultado = Wilcoxon_Congruencia(df, Tipo)
            Resultado['Dataset'] = Nombre_df
            Resultado['Tipo'] = Tipo
            Resultados['General'].append(Resultado)

            print(f"\n  {Tipo} Congruente vs Incongruente:")
            print(f"    n = {Resultado['n']}")
            if 'Media_1' in Resultado:
                print(f"    Media Congruente = {Resultado['Media_1']:.4f}")
                print(f"    Media Incongruente = {Resultado['Media_2']:.4f}")
                print(f"    W = {Resultado['Estadistico_W']:.4f}")
                print(f"    p = {Formatear_P_Valor(Resultado['Valor_p'])} "
                      f"({Resultado['Significancia']})")

        # Analysis by category.
        print("\n  By Category:")
        for Tipo in ['CO', 'CT']:
            Tabla_Cat = Wilcoxon_Congruencia_Por_Categoria(df, Tipo)
            Tabla_Cat['Dataset'] = Nombre_df

            for _, Fila in Tabla_Cat.iterrows():
                Resultados['Por_Categoria'].append(Fila.to_dict())

                if Fila.get('Significativo', False):
                    print(f"    {Fila['Etiqueta']} ({Tipo}): "
                          f"p={Formatear_P_Valor(Fila['Valor_p'])} "
                          f"({Fila['Significancia']})")

    return Resultados


def Analisis_Mann_Whitney_Entre_Elecciones(
    dfs_Finales: dict,
    Variables: list
) -> pd.DataFrame:

    """
    Compares variables between General and Ballotage using
    Mann-Whitney U.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.
        Variables (list): List of variables to compare.

    Returns:
        pd.DataFrame: Comparison results.

    """

    if 'Generales' not in dfs_Finales or 'Ballotage' not in dfs_Finales:
        print("Error: Both datasets are required.")
        return pd.DataFrame()

    df_Gen = dfs_Finales['Generales']
    df_Bal = dfs_Finales['Ballotage']

    print(f"\n{'='*60}")
    print("Mann-Whitney U: Generales vs Ballotage")
    print('='*60)

    Resultados = []

    for Variable in Variables:
        if Variable in df_Gen.columns and Variable in df_Bal.columns:
            Resultado = Mann_Whitney_Entre_Datasets(
                df_Gen, df_Bal, Variable
            )
            Resultados.append(Resultado)

            if Resultado.get('Significativo', False):
                print(f"  {Variable}: "
                      f"U={Resultado['Estadistico_U']:.4f}, "
                      f"p={Formatear_P_Valor(Resultado['Valor_p'])} "
                      f"({Resultado['Significancia']})")

    return pd.DataFrame(Resultados)


# =============================================================================
# LEFT VS RIGHT COMPARISON FUNCTIONS BY ITEM.
# =============================================================================

def Wilcoxon_Izq_vs_Der_Por_Item(
    df: pd.DataFrame,
    Items: list = None
) -> pd.DataFrame:

    """
    Compares CO_Item_X_Izq vs CO_Item_X_Der using paired Wilcoxon
    for each item (same participant, different association).

    Parameters:
        df (pd.DataFrame): DataFrame with data.
        Items (list): List of item numbers to analyze.

    Returns:
        pd.DataFrame: Results by item.

    """

    if Items is None:
        Items = Items_Progresistas + Items_Conservadores

    Resultados = []

    for Item in Items:
        Variable_Izq = f'CO_Item_{Item}_Izq'
        Variable_Der = f'CO_Item_{Item}_Der'

        if Variable_Izq not in df.columns or \
           Variable_Der not in df.columns:
            continue

        # Paired data (no NaN in either variable).
        Datos_Pareados = df[[Variable_Izq, Variable_Der]].dropna()

        if len(Datos_Pareados) < 10:
            continue

        try:
            Estadistico, P_Valor = wilcoxon(
                Datos_Pareados[Variable_Izq],
                Datos_Pareados[Variable_Der]
            )

            if P_Valor < 0.001:
                Sig = '***'
            elif P_Valor < 0.01:
                Sig = '**'
            elif P_Valor < 0.05:
                Sig = '*'
            else:
                Sig = 'ns'

            # Determine item type.
            Tipo = 'Progresista' if Item in Items_Progresistas \
                   else 'Conservador'

            Resultados.append({
                'Item': Item,
                'Tipo_Item': Tipo,
                'n': len(Datos_Pareados),
                'Media_Izq': Datos_Pareados[Variable_Izq].mean(),
                'Media_Der': Datos_Pareados[Variable_Der].mean(),
                'Diferencia': (
                    Datos_Pareados[Variable_Izq].mean() -
                    Datos_Pareados[Variable_Der].mean()
                ),
                'Estadistico_W': Estadistico,
                'Valor_p': P_Valor,
                'Significancia': Sig,
                'Significativo': P_Valor < 0.05
            })

        except Exception:
            continue

    df_Resultados = pd.DataFrame(Resultados)

    if len(df_Resultados) > 0:
        df_Resultados = df_Resultados.sort_values(
            'Valor_p'
        ).reset_index(drop=True)

    return df_Resultados


def Analisis_Wilcoxon_Izq_vs_Der(dfs_Finales: dict) -> dict:

    """
    Executes paired Wilcoxon Left vs Right for all items
    in both datasets.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.

    Returns:
        dict: Results by dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Wilcoxon Izq vs Der por Ítem - {Nombre_df}")
        print('='*60)

        Tabla = Wilcoxon_Izq_vs_Der_Por_Item(df)
        Resultados[Nombre_df] = Tabla

        N_Sig = Tabla['Significativo'].sum() if len(Tabla) > 0 else 0
        print(f"Items with significant difference: {N_Sig}/{len(Tabla)}")

        if N_Sig > 0:
            print("\nSignificant items (Left ≠ Right):")
            Sig = Tabla[Tabla['Significativo']]
            for _, Fila in Sig.iterrows():
                print(f"  - Item {Fila['Item']} ({Fila['Tipo_Item']}): "
                      f"Δ={Fila['Diferencia']:.4f}, "
                      f"p={Formatear_P_Valor(Fila['Valor_p'])} "
                      f"({Fila['Significancia']})")

    return Resultados


def Calcular_Diferencias_Izq_Der(
    df: pd.DataFrame,
    Items: list = None
) -> pd.DataFrame:

    """
    Calculates the difference (Left - Right) for each item and participant.

    Parameters:
        df (pd.DataFrame): DataFrame with data.
        Items (list): List of item numbers.

    Returns:
        pd.DataFrame: DataFrame with Diff_CO_Item_X columns.

    """

    if Items is None:
        Items = Items_Progresistas + Items_Conservadores

    df_Resultado = df.copy()

    for Item in Items:
        Variable_Izq = f'CO_Item_{Item}_Izq'
        Variable_Der = f'CO_Item_{Item}_Der'
        Variable_Diff = f'Diff_CO_Item_{Item}'

        if Variable_Izq in df.columns and Variable_Der in df.columns:
            df_Resultado[Variable_Diff] = (
                df[Variable_Izq] - df[Variable_Der]
            )

    return df_Resultado


def Kruskal_Wallis_Diferencias_Entre_Poblaciones(
    df: pd.DataFrame,
    Items: list = None,
    Columna_Categoria: str = 'Categoria_PASO_2023'
) -> pd.DataFrame:

    """
    Compares the difference (Left - Right) among the 6 electoral
    categories using Kruskal-Wallis.

    Parameters:
        df (pd.DataFrame): DataFrame with calculated differences.
        Items (list): List of item numbers.
        Columna_Categoria (str): Category column.

    Returns:
        pd.DataFrame: Kruskal-Wallis results by item.

    """

    if Items is None:
        Items = Items_Progresistas + Items_Conservadores

    # Filter valid categories.
    df_Filtrado = df[df[Columna_Categoria].isin(Categorias_Validas)]

    Resultados = []

    for Item in Items:
        Variable_Diff = f'Diff_CO_Item_{Item}'

        if Variable_Diff not in df_Filtrado.columns:
            continue

        # Create groups by category.
        Grupos = [
            df_Filtrado[
                df_Filtrado[Columna_Categoria] == Cat
            ][Variable_Diff].dropna()
            for Cat in Categorias_Validas
        ]

        # Verify sufficient data.
        if all(len(Grupo) >= 2 for Grupo in Grupos):
            try:
                H_Stat, P_Valor = kruskal(*Grupos)

                if P_Valor < 0.001:
                    Sig = '***'
                elif P_Valor < 0.01:
                    Sig = '**'
                elif P_Valor < 0.05:
                    Sig = '*'
                else:
                    Sig = 'ns'

                Tipo = 'Progresista' if Item in Items_Progresistas \
                       else 'Conservador'

                Resultados.append({
                    'Item': Item,
                    'Tipo_Item': Tipo,
                    'Variable': Variable_Diff,
                    'Estadistico_H': H_Stat,
                    'Valor_p': P_Valor,
                    'Significancia': Sig,
                    'Significativo': P_Valor < 0.05
                })

            except Exception:
                continue

    df_Resultados = pd.DataFrame(Resultados)

    if len(df_Resultados) > 0:
        df_Resultados = df_Resultados.sort_values(
            'Valor_p'
        ).reset_index(drop=True)

    return df_Resultados


def Analisis_Diferencias_Entre_Poblaciones(dfs_Finales: dict) -> dict:

    """
    Analyzes if the difference (Left - Right) varies among populations
    using Kruskal-Wallis and Dunn post-hoc.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.

    Returns:
        dict: Results with Kruskal-Wallis and Dunn.

    """

    Resultados = {'Kruskal': {}, 'Dunn': {}}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Differences (Left-Right) Among Populations - {Nombre_df}")
        print('='*60)

        # Calculate differences.
        df_Con_Diff = Calcular_Diferencias_Izq_Der(df)

        # Kruskal-Wallis.
        Tabla_Kruskal = Kruskal_Wallis_Diferencias_Entre_Poblaciones(
            df_Con_Diff
        )
        Resultados['Kruskal'][Nombre_df] = Tabla_Kruskal

        N_Sig = Tabla_Kruskal['Significativo'].sum() \
                if len(Tabla_Kruskal) > 0 else 0
        print(f"Items with differences among populations: "
              f"{N_Sig}/{len(Tabla_Kruskal)}")

        # Dunn post-hoc for significant items.
        Items_Sig = Tabla_Kruskal[
            Tabla_Kruskal['Significativo']
        ]['Item'].tolist() if len(Tabla_Kruskal) > 0 else []

        Resultados_Dunn = {}

        if Items_Sig:
            print(f"\nPost-hoc Dunn for {len(Items_Sig)} items:")

            for Item in Items_Sig:
                Variable_Diff = f'Diff_CO_Item_{Item}'

                Datos = df_Con_Diff[
                    df_Con_Diff['Categoria_PASO_2023'].isin(
                        Categorias_Validas
                    )
                ][['Categoria_PASO_2023', Variable_Diff]].dropna()

                if len(Datos) < 10:
                    continue

                try:
                    Matriz = sp.posthoc_dunn(
                        Datos,
                        val_col=Variable_Diff,
                        group_col='Categoria_PASO_2023',
                        p_adjust='bonferroni'
                    )
                    Resultados_Dunn[Item] = Matriz

                    # Extract significant comparisons.
                    Comparaciones = Extraer_Comparaciones_Significativas(
                        Matriz
                    )

                    if Comparaciones:
                        print(f"\n  Item {Item}:")
                        for Comp in Comparaciones:
                            print(f"    - {Comp['Categoria_1']} vs "
                                  f"{Comp['Categoria_2']}: "
                                  f"p={Formatear_P_Valor(Comp['Valor_p'])}")
                    else:
                        print(f"\n  Item {Item}: No significant comparisons "
                              f"post-Bonferroni")

                except Exception:
                    continue

        Resultados['Dunn'][Nombre_df] = Resultados_Dunn

    return Resultados


# =============================================================================
# FUNCTIONS FOR SELF-PERCEPTIONS, INDICES AND CLOSENESS.
# =============================================================================

# Self-perception columns.
Columnas_Autopercepciones = [
    'Autopercepcion_Izq_Der',
    'Autopercepcion_Con_Pro',
    'Autopercepcion_Per_Antiper'
]

# Index columns.
Columnas_Indices = [
    'Indice_Progresismo',
    'Indice_Conservadurismo',
    'Indice_Positividad'
]

# Candidate closeness columns.
Columnas_Cercanias = [
    'Cercania_Massa',
    'Cercania_Bullrich',
    'Cercania_Bregman',
    'Cercania_Milei',
    'Cercania_Schiaretti'
]

Columnas_Influencias = [
    'Influencia_Redes',
    'Influencia_Prensa'
]


def Analisis_Kruskal_Autopercepciones(dfs_Finales: dict) -> dict:

    """
    Executes Kruskal-Wallis for self-perceptions among electoral
    categories.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.

    Returns:
        dict: Results by dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Kruskal-Wallis Autopercepciones - {Nombre_df}")
        print('='*60)

        # Filter existing columns.
        Columnas_Existentes = [
            c for c in Columnas_Autopercepciones if c in df.columns
        ]

        if not Columnas_Existentes:
            print("  No self-perception columns found.")
            continue

        Tabla = Kruskal_Wallis_Por_Categoria(df, Columnas_Existentes)
        Resultados[Nombre_df] = Tabla

        print("\nResults:")
        for _, Fila in Tabla.iterrows():
            Marca = "***" if Fila['Valor_p'] < 0.001 else \
                    "**" if Fila['Valor_p'] < 0.01 else \
                    "*" if Fila['Valor_p'] < 0.05 else ""
            print(f"  {Fila['Variable']}: "
                  f"H={Fila['Estadistico_H']:.4f}, "
                  f"p={Formatear_P_Valor(Fila['Valor_p'])} {Marca}")

    return Resultados


def Analisis_Kruskal_Indices(dfs_Finales: dict) -> dict:

    """
    Executes Kruskal-Wallis for indices (Progressivism, Conservatism,
    Positivity) among electoral categories.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.

    Returns:
        dict: Results by dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Kruskal-Wallis Índices - {Nombre_df}")
        print('='*60)

        # Filter existing columns.
        Columnas_Existentes = [
            c for c in Columnas_Indices if c in df.columns
        ]

        if not Columnas_Existentes:
            print("  No index columns found.")
            continue

        Tabla = Kruskal_Wallis_Por_Categoria(df, Columnas_Existentes)
        Resultados[Nombre_df] = Tabla

        print("\nResults:")
        for _, Fila in Tabla.iterrows():
            Marca = "***" if Fila['Valor_p'] < 0.001 else \
                    "**" if Fila['Valor_p'] < 0.01 else \
                    "*" if Fila['Valor_p'] < 0.05 else ""
            print(f"  {Fila['Variable']}: "
                  f"H={Fila['Estadistico_H']:.4f}, "
                  f"p={Formatear_P_Valor(Fila['Valor_p'])} {Marca}")

    return Resultados


def Analisis_Kruskal_Cercanias(dfs_Finales: dict) -> dict:

    """
    Executes Kruskal-Wallis for candidate closeness among
    electoral categories.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.

    Returns:
        dict: Results by dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Kruskal-Wallis Cercanías a Candidatos - {Nombre_df}")
        print('='*60)

        # Filter existing columns.
        Columnas_Existentes = [
            c for c in Columnas_Cercanias if c in df.columns
        ]

        if not Columnas_Existentes:
            print("  No closeness columns found.")
            continue

        Tabla = Kruskal_Wallis_Por_Categoria(df, Columnas_Existentes)
        Resultados[Nombre_df] = Tabla

        print("\nResults:")
        for _, Fila in Tabla.iterrows():
            Marca = "***" if Fila['Valor_p'] < 0.001 else \
                    "**" if Fila['Valor_p'] < 0.01 else \
                    "*" if Fila['Valor_p'] < 0.05 else ""
            print(f"  {Fila['Variable']}: "
                  f"H={Fila['Estadistico_H']:.4f}, "
                  f"p={Formatear_P_Valor(Fila['Valor_p'])} {Marca}")

    return Resultados


def Analisis_Kruskal_Influencias(dfs_Finales: dict) -> dict:

    """
    Executes Kruskal-Wallis for social media and press influence
    among electoral categories.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.

    Returns:
        dict: Results by dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Kruskal-Wallis Influencias - {Nombre_df}")
        print('='*60)

        # Filter existing columns.
        Columnas_Existentes = [
            c for c in Columnas_Influencias if c in df.columns
        ]

        if not Columnas_Existentes:
            print("  No influence columns found.")
            continue

        Tabla = Kruskal_Wallis_Por_Categoria(df, Columnas_Existentes)
        Resultados[Nombre_df] = Tabla

        print("\nResults:")
        for _, Fila in Tabla.iterrows():
            Marca = "***" if Fila['Valor_p'] < 0.001 else \
                    "**" if Fila['Valor_p'] < 0.01 else \
                    "*" if Fila['Valor_p'] < 0.05 else ""
            print(f"  {Fila['Variable']}: "
                  f"H={Fila['Estadistico_H']:.4f}, "
                  f"p={Formatear_P_Valor(Fila['Valor_p'])} {Marca}")

    return Resultados


def Analisis_Dunn_Autopercepciones(
    dfs_Finales: dict,
    Resultados_Kruskal_Autopercepciones: dict
) -> dict:

    """
    Executes Dunn post-hoc for significant Kruskal-Wallis
    self-perceptions.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.
        Resultados_Kruskal_Autopercepciones (dict): Kruskal-Wallis
            results for self-perceptions.

    Returns:
        dict: Dunn results by dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Dunn Post-Hoc Autopercepciones - {Nombre_df}")
        print('='*60)

        Tabla_Kruskal = Resultados_Kruskal_Autopercepciones.get(Nombre_df)

        if Tabla_Kruskal is None:
            print("  No Kruskal-Wallis results available.")
            continue

        # Get significant variables.
        Variables_Sig = Tabla_Kruskal[
            Tabla_Kruskal['Significativo']
        ]['Variable'].tolist()

        if not Variables_Sig:
            print("  No significant variables for post-hoc.")
            continue

        print(f"Variables for post-hoc: {len(Variables_Sig)}")

        Resultados_Dunn = Ejecutar_Dunn_Variables_Significativas(
            df, Variables_Sig
        )

        Resultados[Nombre_df] = Resultados_Dunn

        # Show summary of significant comparisons.
        for Variable, Matriz in Resultados_Dunn.items():
            Comparaciones = Extraer_Comparaciones_Significativas(Matriz)

            if Comparaciones:
                print(f"\n  {Variable}:")
                for Comp in Comparaciones:
                    print(f"    - {Comp['Categoria_1']} vs "
                          f"{Comp['Categoria_2']}: "
                          f"p={Formatear_P_Valor(Comp['Valor_p'])}")
            else:
                print(f"\n  {Variable}: No significant comparisons")

    return Resultados


def Analisis_Dunn_Indices(
    dfs_Finales: dict,
    Resultados_Kruskal_Indices: dict
) -> dict:

    """
    Executes Dunn post-hoc for significant Kruskal-Wallis
    indices.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.
        Resultados_Kruskal_Indices (dict): Kruskal-Wallis
            results for indices.

    Returns:
        dict: Dunn results by dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Dunn Post-Hoc Índices - {Nombre_df}")
        print('='*60)

        Tabla_Kruskal = Resultados_Kruskal_Indices.get(Nombre_df)

        if Tabla_Kruskal is None:
            print("  No Kruskal-Wallis results available.")
            continue

        # Get significant variables.
        Variables_Sig = Tabla_Kruskal[
            Tabla_Kruskal['Significativo']
        ]['Variable'].tolist()

        if not Variables_Sig:
            print("  No significant variables for post-hoc.")
            continue

        print(f"Variables for post-hoc: {len(Variables_Sig)}")

        Resultados_Dunn = Ejecutar_Dunn_Variables_Significativas(
            df, Variables_Sig
        )

        Resultados[Nombre_df] = Resultados_Dunn

        # Show summary of significant comparisons.
        for Variable, Matriz in Resultados_Dunn.items():
            Comparaciones = Extraer_Comparaciones_Significativas(Matriz)

            if Comparaciones:
                print(f"\n  {Variable}:")
                for Comp in Comparaciones:
                    print(f"    - {Comp['Categoria_1']} vs "
                          f"{Comp['Categoria_2']}: "
                          f"p={Formatear_P_Valor(Comp['Valor_p'])}")
            else:
                print(f"\n  {Variable}: No significant comparisons")

    return Resultados


def Analisis_Dunn_Cercanias(
    dfs_Finales: dict,
    Resultados_Kruskal_Cercanias: dict
) -> dict:

    """
    Executes Dunn post-hoc for significant Kruskal-Wallis
    candidate closeness ratings.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.
        Resultados_Kruskal_Cercanias (dict): Kruskal-Wallis
            results for closeness.

    Returns:
        dict: Dunn results by dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Dunn Post-Hoc Cercanías a Candidatos - {Nombre_df}")
        print('='*60)

        Tabla_Kruskal = Resultados_Kruskal_Cercanias.get(Nombre_df)

        if Tabla_Kruskal is None:
            print("  No Kruskal-Wallis results available.")
            continue

        # Get significant variables.
        Variables_Sig = Tabla_Kruskal[
            Tabla_Kruskal['Significativo']
        ]['Variable'].tolist()

        if not Variables_Sig:
            print("  No significant variables for post-hoc.")
            continue

        print(f"Variables for post-hoc: {len(Variables_Sig)}")

        Resultados_Dunn = Ejecutar_Dunn_Variables_Significativas(
            df, Variables_Sig
        )

        Resultados[Nombre_df] = Resultados_Dunn

        # Show summary of significant comparisons.
        for Variable, Matriz in Resultados_Dunn.items():
            Comparaciones = Extraer_Comparaciones_Significativas(Matriz)

            if Comparaciones:
                print(f"\n  {Variable}:")
                for Comp in Comparaciones:
                    print(f"    - {Comp['Categoria_1']} vs "
                          f"{Comp['Categoria_2']}: "
                          f"p={Formatear_P_Valor(Comp['Valor_p'])}")
            else:
                print(f"\n  {Variable}: No significant comparisons")

    return Resultados


def Analisis_Dunn_Influencias(
    dfs_Finales: dict,
    Resultados_Kruskal_Influencias: dict
) -> dict:

    """
    Executes Dunn post-hoc for significant Kruskal-Wallis
    influences.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.
        Resultados_Kruskal_Influencias (dict): Kruskal-Wallis
            results for influences.

    Returns:
        dict: Dunn results by dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Dunn Post-Hoc Influencias - {Nombre_df}")
        print('='*60)

        Tabla_Kruskal = Resultados_Kruskal_Influencias.get(Nombre_df)

        if Tabla_Kruskal is None:
            print("  No Kruskal-Wallis results available.")
            continue

        # Get significant variables.
        Variables_Sig = Tabla_Kruskal[
            Tabla_Kruskal['Significativo']
        ]['Variable'].tolist()

        if not Variables_Sig:
            print("  No significant variables for post-hoc.")
            continue

        print(f"Variables for post-hoc: {len(Variables_Sig)}")

        Resultados_Dunn = Ejecutar_Dunn_Variables_Significativas(
            df, Variables_Sig
        )

        Resultados[Nombre_df] = Resultados_Dunn

        # Show summary of significant comparisons.
        for Variable, Matriz in Resultados_Dunn.items():
            Comparaciones = Extraer_Comparaciones_Significativas(Matriz)

            if Comparaciones:
                print(f"\n  {Variable}:")
                for Comp in Comparaciones:
                    print(f"    - {Comp['Categoria_1']} vs "
                          f"{Comp['Categoria_2']}: "
                          f"p={Formatear_P_Valor(Comp['Valor_p'])}")
            else:
                print(f"\n  {Variable}: No significant comparisons")

    return Resultados


# =============================================================================
# DEMOGRAPHIC COVARIATE CONTROL.
# =============================================================================

def Analisis_Control_Demografico(dfs_Finales: dict) -> dict:

    """
    Performs regression analysis to verify if the Congruent vs
    Incongruent effect persists after controlling for Gender and Age.

    Method: OLS regression of the difference (CO_Congruent - CO_Incongruent)
    on Gender and Age. If the intercept is significant, the effect
    persists when controlling for demographic covariates.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames for each election.

    Returns:
        dict: Regression results for each election and type (CO, CT).

    """

    print("\nAnalysis controlling for Gender and Age...")
    print("-" * 70)

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{Nombre_df}:")
        print("-" * 50)

        Resultados[Nombre_df] = {}

        for Tipo in ['CO', 'CT']:
            Variable_Congruente = f'{Tipo}_Congruente'
            Variable_Incongruente = f'{Tipo}_Incongruente'

            # Verify required columns exist.
            Columnas_Requeridas = [
                Variable_Congruente, Variable_Incongruente,
                'Genero', 'Edad'
            ]

            if not all(Col in df.columns for Col in Columnas_Requeridas):
                print(f"  {Tipo}: Missing required columns.")
                continue

            # Create DataFrame for regression.
            df_Regresion = df[Columnas_Requeridas].copy()
            df_Regresion = df_Regresion.dropna()

            if len(df_Regresion) < 10:
                print(f"  {Tipo}: Insufficient data (n={len(df_Regresion)}).")
                continue

            # Calculate Congruent - Incongruent difference.
            df_Regresion['Diferencia'] = (
                df_Regresion[Variable_Congruente] -
                df_Regresion[Variable_Incongruente]
            )

            # Create dummy variables for Gender.
            # Using Female as reference category.
            df_Regresion['Genero_Masculino'] = (
                df_Regresion['Genero'] == 'Masculino'
            ).astype(int)
            df_Regresion['Genero_Otro'] = (
                df_Regresion['Genero'] == 'Otro'
            ).astype(int)

            # Convert Age to numeric.
            df_Regresion['Edad'] = pd.to_numeric(
                df_Regresion['Edad'], errors='coerce'
            )
            df_Regresion = df_Regresion.dropna()

            if len(df_Regresion) < 10:
                print(f"  {Tipo}: Insufficient data after cleaning.")
                continue

            # Prepare variables for regression.
            Y = df_Regresion['Diferencia']
            X = df_Regresion[['Genero_Masculino', 'Genero_Otro', 'Edad']]
            X = sm.add_constant(X)

            # Fit OLS model.
            Modelo = sm.OLS(Y, X).fit()

            # Extract results.
            Intercepto = Modelo.params['const']
            P_Intercepto = Modelo.pvalues['const']
            R2 = Modelo.rsquared
            R2_Ajustado = Modelo.rsquared_adj
            N = len(df_Regresion)

            # Covariate coefficients.
            Coef_Masculino = Modelo.params['Genero_Masculino']
            P_Masculino = Modelo.pvalues['Genero_Masculino']
            Coef_Otro = Modelo.params.get('Genero_Otro', np.nan)
            P_Otro = Modelo.pvalues.get('Genero_Otro', np.nan)
            Coef_Edad = Modelo.params['Edad']
            P_Edad = Modelo.pvalues['Edad']

            # Save results.
            Resultados[Nombre_df][Tipo] = {
                'N': N,
                'Intercepto': Intercepto,
                'P_Intercepto': P_Intercepto,
                'Intercepto_Significativo': P_Intercepto < 0.05,
                'Coef_Genero_Masculino': Coef_Masculino,
                'P_Genero_Masculino': P_Masculino,
                'Coef_Genero_Otro': Coef_Otro,
                'P_Genero_Otro': P_Otro,
                'Coef_Edad': Coef_Edad,
                'P_Edad': P_Edad,
                'R2': R2,
                'R2_Ajustado': R2_Ajustado,
                'Modelo': Modelo
            }

            # Determine intercept significance.
            if P_Intercepto < 0.001:
                Sig_Int = '***'
            elif P_Intercepto < 0.01:
                Sig_Int = '**'
            elif P_Intercepto < 0.05:
                Sig_Int = '*'
            else:
                Sig_Int = 'ns'

            print(f"\n  {Tipo} Congruente - Incongruente (N={N}):")
            print(f"    Intercepto: {Intercepto:.4f} (p={P_Intercepto:.4f}) "
                  f"{Sig_Int}")
            print(f"    Male Gender: β={Coef_Masculino:.4f} "
                  f"(p={P_Masculino:.4f})")
            print(f"    Age: β={Coef_Edad:.4f} (p={P_Edad:.4f})")
            print(f"    R² ajustado: {R2_Ajustado:.4f}")

            if P_Intercepto < 0.05:
                print(f"    → The {Tipo} Congruent vs Incongruent effect "
                      f"PERSISTS when controlling for Gender and Age.")
            else:
                print(f"    → The {Tipo} Congruent vs Incongruent effect "
                      f"does NOT persist when controlling for Gender and Age.")

    return Resultados


# =============================================================================
# MAIN EXECUTION.
# =============================================================================

def Ejecutar_Todos_Los_Analisis(Ruta_Datos: str) -> dict:

    """
    Executes all statistical analyses and returns results.

    Execution order:
    - First: All Kruskal-Wallis (1-8)
    - Second: All Dunn post-hoc (9-16)
    - Third: Paired tests - Wilcoxon and Mann-Whitney (17-19)

    Parameters:
        Ruta_Datos (str): Path to folder with data.

    Returns:
        dict: Dictionary with all results.

    """

    print("="*70)
    print("COMPLETE STATISTICAL ANALYSES")
    print("="*70)

    # Load data.
    print("\nLoading data...")
    dfs_Finales = Cargar_Bases_Datos(Ruta_Datos)

    if not dfs_Finales:
        print("Error: Could not load data.")
        return {}

    Todos_Resultados = {}

    # =========================================================================
    # SECTION A: KRUSKAL-WALLIS TESTS.
    # =========================================================================

    print("\n" + "#"*70)
    print("# SECTION A: KRUSKAL-WALLIS TESTS")
    print("#"*70)

    # 1. Kruskal-Wallis para ítems CO individuales.
    print("\n" + "="*70)
    print("1. KRUSKAL-WALLIS: ÍTEMS CO INDIVIDUALES")
    print("="*70)
    Todos_Resultados['Kruskal_CO_Items'] = \
        Analisis_Kruskal_CO_Items(dfs_Finales)

    # 2. Kruskal-Wallis para ítems CT individuales.
    print("\n" + "="*70)
    print("2. KRUSKAL-WALLIS: ÍTEMS CT INDIVIDUALES")
    print("="*70)
    Todos_Resultados['Kruskal_CT_Items'] = \
        Analisis_Kruskal_CT_Items(dfs_Finales)

    # 3. Kruskal-Wallis para variables agregadas.
    print("\n" + "="*70)
    print("3. KRUSKAL-WALLIS: VARIABLES AGREGADAS")
    print("="*70)
    Todos_Resultados['Kruskal_Agregadas'] = \
        Analisis_Kruskal_Variables_Agregadas(dfs_Finales)

    # 4. Kruskal-Wallis diferencias (Izq-Der) entre poblaciones.
    print("\n" + "="*70)
    print("4. KRUSKAL-WALLIS: DIFERENCIAS (IZQ-DER) ENTRE POBLACIONES")
    print("="*70)
    Todos_Resultados['Diferencias_Entre_Poblaciones'] = \
        Analisis_Diferencias_Entre_Poblaciones(dfs_Finales)

    # 5. Kruskal-Wallis para autopercepciones.
    print("\n" + "="*70)
    print("5. KRUSKAL-WALLIS: AUTOPERCEPCIONES POR CATEGORÍA")
    print("="*70)
    Todos_Resultados['Kruskal_Autopercepciones'] = \
        Analisis_Kruskal_Autopercepciones(dfs_Finales)

    # 6. Kruskal-Wallis para índices.
    print("\n" + "="*70)
    print("6. KRUSKAL-WALLIS: ÍNDICES POR CATEGORÍA")
    print("="*70)
    Todos_Resultados['Kruskal_Indices'] = \
        Analisis_Kruskal_Indices(dfs_Finales)

    # 7. Kruskal-Wallis para cercanías a candidatos.
    print("\n" + "="*70)
    print("7. KRUSKAL-WALLIS: CERCANÍAS A CANDIDATOS POR CATEGORÍA")
    print("="*70)
    Todos_Resultados['Kruskal_Cercanias'] = \
        Analisis_Kruskal_Cercanias(dfs_Finales)

    # 8. Kruskal-Wallis para influencias (redes y prensa).
    print("\n" + "="*70)
    print("8. KRUSKAL-WALLIS: INFLUENCIAS POR CATEGORÍA")
    print("="*70)
    Todos_Resultados['Kruskal_Influencias'] = \
        Analisis_Kruskal_Influencias(dfs_Finales)

    # =========================================================================
    # SECTION B: DUNN POST-HOC TESTS.
    # =========================================================================

    print("\n" + "#"*70)
    print("# SECTION B: DUNN POST-HOC TESTS (Corrección Bonferroni)")
    print("#"*70)

    # 8. Dunn post-hoc para CO items.
    print("\n" + "="*70)
    print("8. DUNN POST-HOC: ÍTEMS CO")
    print("="*70)
    Todos_Resultados['Dunn_CO'] = Analisis_Dunn_Post_Hoc(
        dfs_Finales,
        Todos_Resultados['Kruskal_CO_Items']
    )

    # 9. Dunn post-hoc para CT items.
    print("\n" + "="*70)
    print("9. DUNN POST-HOC: ÍTEMS CT")
    print("="*70)
    Todos_Resultados['Dunn_CT'] = Analisis_Dunn_Post_Hoc(
        dfs_Finales,
        Todos_Resultados['Kruskal_CT_Items']
    )

    # 10. Dunn post-hoc para variables agregadas.
    print("\n" + "="*70)
    print("10. DUNN POST-HOC: VARIABLES AGREGADAS")
    print("="*70)
    Todos_Resultados['Dunn_Agregadas'] = Analisis_Dunn_Variables_Agregadas(
        dfs_Finales,
        Todos_Resultados['Kruskal_Agregadas']
    )

    # 11. Dunn post-hoc para autopercepciones.
    print("\n" + "="*70)
    print("11. DUNN POST-HOC: AUTOPERCEPCIONES")
    print("="*70)
    Todos_Resultados['Dunn_Autopercepciones'] = Analisis_Dunn_Autopercepciones(
        dfs_Finales,
        Todos_Resultados['Kruskal_Autopercepciones']
    )

    # 12. Dunn post-hoc para índices.
    print("\n" + "="*70)
    print("12. DUNN POST-HOC: ÍNDICES")
    print("="*70)
    Todos_Resultados['Dunn_Indices'] = Analisis_Dunn_Indices(
        dfs_Finales,
        Todos_Resultados['Kruskal_Indices']
    )

    # 13. Dunn post-hoc para cercanías a candidatos.
    print("\n" + "="*70)
    print("13. DUNN POST-HOC: CERCANÍAS A CANDIDATOS")
    print("="*70)
    Todos_Resultados['Dunn_Cercanias'] = Analisis_Dunn_Cercanias(
        dfs_Finales,
        Todos_Resultados['Kruskal_Cercanias']
    )

    # 14. Dunn post-hoc para influencias.
    print("\n" + "="*70)
    print("14. DUNN POST-HOC: INFLUENCIAS")
    print("="*70)
    Todos_Resultados['Dunn_Influencias'] = Analisis_Dunn_Influencias(
        dfs_Finales,
        Todos_Resultados['Kruskal_Influencias']
    )

    # =========================================================================
    # SECTION C: PAIRED TESTS (WILCOXON AND MANN-WHITNEY).
    # =========================================================================

    print("\n" + "#"*70)
    print("# SECTION C: PAIRED TESTS (WILCOXON AND MANN-WHITNEY)")
    print("#"*70)

    # 14. Wilcoxon para congruencia ideológica.
    print("\n" + "="*70)
    print("14. WILCOXON: CONGRUENCIA IDEOLÓGICA")
    print("="*70)
    Todos_Resultados['Wilcoxon_Congruencia'] = \
        Analisis_Wilcoxon_Congruencia(dfs_Finales)

    # 15. Wilcoxon Izq vs Der por ítem.
    print("\n" + "="*70)
    print("17. WILCOXON: IZQ VS DER POR ÍTEM (PAREADO)")
    print("="*70)
    Todos_Resultados['Wilcoxon_Izq_vs_Der'] = \
        Analisis_Wilcoxon_Izq_vs_Der(dfs_Finales)

    # 16. Mann-Whitney entre elecciones.
    print("\n" + "="*70)
    print("18. MANN-WHITNEY: GENERALES VS BALLOTAGE")
    print("="*70)

    # Variables to compare between elections.
    Variables_Comparar = [
        'CO_Congruente', 'CO_Incongruente',
        'CT_Congruente', 'CT_Incongruente',
        'CO_Pro_Izq', 'CO_Con_Izq', 'CO_Pro_Der', 'CO_Con_Der',
        'CT_Pro_Izq', 'CT_Con_Izq', 'CT_Pro_Der', 'CT_Con_Der'
    ]

    Todos_Resultados['Mann_Whitney'] = \
        Analisis_Mann_Whitney_Entre_Elecciones(
            dfs_Finales,
            Variables_Comparar
        )

    # 17. Control por covariables demográficas.
    print("\n" + "="*70)
    print("19. CONTROL POR GÉNERO Y EDAD (REGRESIÓN)")
    print("="*70)
    Todos_Resultados['Control_Demografico'] = \
        Analisis_Control_Demografico(dfs_Finales)

    print("\n" + "="*70)
    print("ANALYSES COMPLETED")
    print("="*70)

    return Todos_Resultados, dfs_Finales


def Generar_Reporte_TXT(
    Todos_Resultados: dict,
    dfs_Finales: dict,
    Ruta_Salida: str
) -> None:

    """
    Generates a TXT file with all results formatted
    in a readable manner.

    Report order:
    - Section A: All Kruskal-Wallis (1-7)
    - Section B: All Dunn post-hoc (8-13)
    - Section C: Paired Wilcoxon and Mann-Whitney tests (14-16)

    Parameters:
        Todos_Resultados (dict): Dictionary with all results.
        dfs_Finales (dict): Dictionary with DataFrames.
        Ruta_Salida (str): Path to output TXT file.

    """

    Lineas = []

    def Agregar(Texto: str = ""):
        Lineas.append(Texto)

    def Separador(Caracter: str = "=", Largo: int = 80):
        Lineas.append(Caracter * Largo)

    def Escribir_Kruskal(
        Nombre_Seccion: str,
        Clave_Resultado: str,
        Descripcion: str = ""
    ):
        """
        Helper function to write Kruskal-Wallis sections.

        """

        Separador("=")
        Agregar(Nombre_Seccion)
        Separador("=")
        Agregar("")
        if Descripcion:
            Agregar(Descripcion)
            Agregar("")

        for Nombre_df, Tabla in Todos_Resultados.get(
            Clave_Resultado, {}
        ).items():
            Separador("-")
            Agregar(f"Dataset: {Nombre_df}")
            Separador("-")

            N_Sig = Tabla['Significativo'].sum()
            Agregar(f"Variables significativas (p < 0.05): "
                    f"{N_Sig}/{len(Tabla)}")
            Agregar("")

            Agregar("Resultados:")
            for _, Fila in Tabla.iterrows():
                Marca = "***" if Fila['Valor_p'] < 0.001 else \
                        "**" if Fila['Valor_p'] < 0.01 else \
                        "*" if Fila['Valor_p'] < 0.05 else ""
                Var_Traducida = Traducir_Variable(Fila['Variable'])
                Agregar(f"  {Var_Traducida:30s} "
                        f"H = {Fila['Estadistico_H']:8.4f}  "
                        f"p = {Formatear_P_Valor(Fila['Valor_p'])} {Marca}")
            Agregar("")

    def Escribir_Dunn(Nombre_Seccion: str, Clave_Resultado: str):
        """
        Helper function to write Dunn post-hoc sections.

        """

        Separador("=")
        Agregar(Nombre_Seccion)
        Separador("=")
        Agregar("")

        for Nombre_df, Resultados_Dunn in Todos_Resultados.get(
            Clave_Resultado, {}
        ).items():
            Separador("-")
            Agregar(f"Dataset: {Nombre_df}")
            Separador("-")

            if not Resultados_Dunn:
                Agregar("  No hay variables significativas para post-hoc.")
                Agregar("")
                continue

            for Variable, Matriz in Resultados_Dunn.items():
                Comparaciones = Extraer_Comparaciones_Significativas(Matriz)
                Var_Traducida = Traducir_Variable(Variable)

                if Comparaciones:
                    Agregar(f"\n  {Var_Traducida}:")
                    for Comp in Comparaciones:
                        Agregar(f"    - {Comp['Categoria_1']:25s} vs "
                                f"{Comp['Categoria_2']:25s} "
                                f"p = {Formatear_P_Valor(Comp['Valor_p'])}")
                else:
                    Agregar(f"\n  {Var_Traducida}: "
                            f"Sin comparaciones significativas")
            Agregar("")

    # =========================================================================
    # ENCABEZADO.
    # =========================================================================
    Separador("=")
    Agregar("RESULTADOS DE ANÁLISIS ESTADÍSTICOS")
    Agregar("Proyecto de Tesis - Experimento Electoral Argentina 2023")
    Separador("=")
    Agregar("")

    # =========================================================================
    # SECTION A: KRUSKAL-WALLIS TESTS.
    # =========================================================================
    Separador("#")
    Agregar("# SECTION A: KRUSKAL-WALLIS TESTS")
    Separador("#")
    Agregar("")

    # 1. Kruskal-Wallis CO Items.
    Escribir_Kruskal(
        "1. KRUSKAL-WALLIS: ÍTEMS CO INDIVIDUALES POR CATEGORÍA",
        'Kruskal_CO_Items',
        "Compara cambios de opinión entre las 6 categorías electorales."
    )

    # 2. Kruskal-Wallis CT Items.
    Escribir_Kruskal(
        "2. KRUSKAL-WALLIS: ÍTEMS CT INDIVIDUALES POR CATEGORÍA",
        'Kruskal_CT_Items',
        "Compara cambios de tiempo entre las 6 categorías electorales."
    )

    # 3. Kruskal-Wallis Variables Agregadas.
    Escribir_Kruskal(
        "3. KRUSKAL-WALLIS: VARIABLES AGREGADAS POR CATEGORÍA",
        'Kruskal_Agregadas',
        "Variables agregadas de CO y CT (promedios por tipo y dirección)."
    )

    # 4. Kruskal-Wallis Diferencias (Izq-Der).
    Separador("=")
    Agregar("4. KRUSKAL-WALLIS: DIFERENCIAS (IZQ-DER) ENTRE POBLACIONES")
    Separador("=")
    Agregar("")
    Agregar("Compara la diferencia (CO_Izq - CO_Der) entre categorías.")
    Agregar("")

    Resultados_Diff = Todos_Resultados.get(
        'Diferencias_Entre_Poblaciones', {}
    )

    for Nombre_df in ['Generales', 'Ballotage']:
        Tabla_Kruskal = Resultados_Diff.get('Kruskal', {}).get(Nombre_df)

        if Tabla_Kruskal is not None and len(Tabla_Kruskal) > 0:
            Separador("-")
            Agregar(f"Dataset: {Nombre_df}")
            Separador("-")

            N_Sig = Tabla_Kruskal['Significativo'].sum()
            Agregar(f"Ítems con diferencias significativas: "
                    f"{N_Sig}/{len(Tabla_Kruskal)}")
            Agregar("")

            if N_Sig > 0:
                Agregar("Ítems significativos:")
                Sig = Tabla_Kruskal[Tabla_Kruskal['Significativo']]
                for _, Fila in Sig.iterrows():
                    Agregar(f"  Item {Fila['Item']:2d} "
                            f"({Fila['Tipo_Item']:12s}): "
                            f"H = {Fila['Estadistico_H']:8.4f}, "
                            f"p = {Formatear_P_Valor(Fila['Valor_p'])} "
                            f"({Fila['Significancia']})")
            Agregar("")

    # 5. Kruskal-Wallis Autopercepciones.
    Escribir_Kruskal(
        "5. KRUSKAL-WALLIS: AUTOPERCEPCIONES POR CATEGORÍA",
        'Kruskal_Autopercepciones',
        "Escalas: Izq-Der, Con-Pro, Per-Antiper."
    )

    # 6. Kruskal-Wallis Índices.
    Escribir_Kruskal(
        "6. KRUSKAL-WALLIS: ÍNDICES POR CATEGORÍA",
        'Kruskal_Indices',
        "Índices de Progresismo, Conservadurismo y Positividad."
    )

    # 7. Kruskal-Wallis Cercanías.
    Escribir_Kruskal(
        "7. KRUSKAL-WALLIS: CERCANÍAS A CANDIDATOS POR CATEGORÍA",
        'Kruskal_Cercanias',
        "Cercanía a Massa, Bullrich, Bregman, Milei, Schiaretti."
    )

    # 8. Kruskal-Wallis Influencias.
    Escribir_Kruskal(
        "8. KRUSKAL-WALLIS: INFLUENCIAS POR CATEGORÍA",
        'Kruskal_Influencias',
        "Influencia percibida de Redes Sociales y Prensa."
    )

    # =========================================================================
    # SECTION B: DUNN POST-HOC TESTS.
    # =========================================================================
    Separador("#")
    Agregar("# SECTION B: DUNN POST-HOC TESTS (Corrección Bonferroni)")
    Separador("#")
    Agregar("")

    # 9. Dunn CO Items.
    Escribir_Dunn(
        "9. DUNN POST-HOC: ÍTEMS CO",
        'Dunn_CO'
    )

    # 10. Dunn CT Items.
    Escribir_Dunn(
        "10. DUNN POST-HOC: ÍTEMS CT",
        'Dunn_CT'
    )

    # 11. Dunn Variables Agregadas.
    Escribir_Dunn(
        "11. DUNN POST-HOC: VARIABLES AGREGADAS",
        'Dunn_Agregadas'
    )

    # 12. Dunn Autopercepciones.
    Escribir_Dunn(
        "12. DUNN POST-HOC: AUTOPERCEPCIONES",
        'Dunn_Autopercepciones'
    )

    # 13. Dunn Índices.
    Escribir_Dunn(
        "13. DUNN POST-HOC: ÍNDICES",
        'Dunn_Indices'
    )

    # 14. Dunn Cercanías.
    Escribir_Dunn(
        "14. DUNN POST-HOC: CERCANÍAS A CANDIDATOS",
        'Dunn_Cercanias'
    )

    # 15. Dunn Influencias.
    Escribir_Dunn(
        "15. DUNN POST-HOC: INFLUENCIAS",
        'Dunn_Influencias'
    )

    # =========================================================================
    # SECCIÓN C: TESTS PAREADOS.
    # =========================================================================
    Separador("#")
    Agregar("# SECTION C: PAIRED TESTS (WILCOXON AND MANN-WHITNEY)")
    Separador("#")
    Agregar("")

    # 16. Wilcoxon Congruencia.
    Separador("=")
    Agregar("16. WILCOXON PAREADO: CONGRUENCIA IDEOLÓGICA")
    Separador("=")
    Agregar("")
    Agregar("Comparación Congruente vs Incongruente (mismo participante)")
    Agregar("")
    Agregar("Definiciones:")
    Agregar("  CONGRUENTE: Ítems Progresistas→Izq + Conservadores→Der")
    Agregar("  INCONGRUENTE: Ítems Progresistas→Der + Conservadores→Izq")
    Agregar("")

    Resultados_Wilcoxon = Todos_Resultados.get('Wilcoxon_Congruencia', {})

    # Resultados generales.
    Agregar("14.1 ANÁLISIS GENERAL (Todas las poblaciones)")
    Separador("-", 50)

    for Resultado in Resultados_Wilcoxon.get('General', []):
        Dataset = Resultado.get('Dataset', '')
        Tipo = Resultado.get('Tipo', '')
        N = Resultado.get('n', 0)
        Media_1 = Resultado.get('Media_1', np.nan)
        Media_2 = Resultado.get('Media_2', np.nan)
        W = Resultado.get('Estadistico_W', np.nan)
        P_Val = Resultado.get('Valor_p', np.nan)
        Sig = Resultado.get('Significancia', 'ns')

        Agregar(f"\n  {Dataset} - {Tipo} Congruente vs Incongruente:")
        Agregar(f"    n pareados         = {N}")
        Agregar(f"    Media Congruente   = {Media_1:.4f}")
        Agregar(f"    Media Incongruente = {Media_2:.4f}")
        Agregar(f"    Estadístico W      = {W:.4f}")
        Agregar(f"    p-valor            = {Formatear_P_Valor(P_Val)} ({Sig})")

        if Sig != 'ns':
            Agregar(f"    --> DIFERENCIA SIGNIFICATIVA")
    Agregar("")

    # Resultados por categoría.
    Agregar("14.2 ANÁLISIS POR CATEGORÍA")
    Separador("-", 50)

    Resultados_Cat = Resultados_Wilcoxon.get('Por_Categoria', [])
    df_Cat = pd.DataFrame(Resultados_Cat)

    if len(df_Cat) > 0:
        for Dataset in df_Cat['Dataset'].unique():
            Agregar(f"\n  {Dataset}:")

            df_Sub = df_Cat[df_Cat['Dataset'] == Dataset]

            for Tipo in ['CO', 'CT']:
                df_Tipo = df_Sub[df_Sub['Tipo'] == Tipo]
                Sig_Tipo = df_Tipo[df_Tipo['Significativo'] == True]

                if len(Sig_Tipo) > 0:
                    Agregar(f"\n    {Tipo} - Comparaciones significativas:")
                    for _, Fila in Sig_Tipo.iterrows():
                        Agregar(f"      - {Fila['Etiqueta']:25s} "
                                f"p = {Formatear_P_Valor(Fila['Valor_p'])} "
                                f"({Fila['Significancia']})")
                else:
                    Agregar(f"\n    {Tipo} - Sin comparaciones significativas")
    Agregar("")

    # 15. Wilcoxon Izq vs Der por ítem.
    Separador("=")
    Agregar("17. WILCOXON PAREADO: IZQ VS DER POR ÍTEM")
    Separador("=")
    Agregar("")
    Agregar("Compara CO_Item_X_Izq vs CO_Item_X_Der (mismo participante)")
    Agregar("")

    for Nombre_df, Tabla in Todos_Resultados.get(
        'Wilcoxon_Izq_vs_Der', {}
    ).items():
        Separador("-")
        Agregar(f"Dataset: {Nombre_df}")
        Separador("-")

        if len(Tabla) == 0:
            Agregar("  Sin datos disponibles.")
            Agregar("")
            continue

        N_Sig = Tabla['Significativo'].sum()
        Agregar(f"Ítems con diferencia significativa: {N_Sig}/{len(Tabla)}")
        Agregar("")

        if N_Sig > 0:
            Agregar("Ítems significativos (Izq ≠ Der):")
            Sig = Tabla[Tabla['Significativo']]
            for _, Fila in Sig.iterrows():
                Agregar(f"  Item {Fila['Item']:2d} ({Fila['Tipo_Item']:12s}): "
                        f"Δ = {Fila['Diferencia']:+.4f}, "
                        f"p = {Formatear_P_Valor(Fila['Valor_p'])} "
                        f"({Fila['Significancia']})")
        Agregar("")

    # 16. Mann-Whitney entre elecciones.
    Separador("=")
    Agregar("18. MANN-WHITNEY U: GENERALES VS BALLOTAGE")
    Separador("=")
    Agregar("")
    Agregar("Comparación de variables entre ambas elecciones (no pareado)")
    Agregar("")

    df_MW = Todos_Resultados.get('Mann_Whitney', pd.DataFrame())

    if len(df_MW) > 0:
        # Significativas.
        Sig_MW = df_MW[df_MW['Significativo'] == True]

        if len(Sig_MW) > 0:
            Agregar("Variables con diferencias significativas:")
            for _, Fila in Sig_MW.iterrows():
                Var_Traducida = Traducir_Variable(Fila['Variable'])
                Agregar(f"\n  {Var_Traducida}:")
                Agregar(f"    n Generales  = {Fila['n_Generales']}, "
                        f"Media = {Fila['Media_Generales']:.4f}")
                Agregar(f"    n Ballotage  = {Fila['n_Ballotage']}, "
                        f"Media = {Fila['Media_Ballotage']:.4f}")
                Agregar(f"    U = {Fila['Estadistico_U']:.4f}, "
                        f"p = {Formatear_P_Valor(Fila['Valor_p'])} "
                        f"({Fila['Significancia']})")
            Agregar("")

        # No significativas.
        No_Sig_MW = df_MW[df_MW['Significativo'] == False]

        if len(No_Sig_MW) > 0:
            Agregar("Variables sin diferencias significativas:")
            for _, Fila in No_Sig_MW.iterrows():
                Var_Traducida = Traducir_Variable(Fila['Variable'])
                Agregar(f"  {Var_Traducida:20s} "
                        f"U = {Fila['Estadistico_U']:.4f}, "
                        f"p = {Formatear_P_Valor(Fila['Valor_p'])}")
    Agregar("")

    # 17. Control por covariables demográficas.
    Separador("=")
    Agregar("19. CONTROL POR GÉNERO Y EDAD (REGRESIÓN OLS)")
    Separador("=")
    Agregar("")
    Agregar("Verificación de que el efecto Congruente vs Incongruente")
    Agregar("se mantiene después de controlar por Género y Edad.")
    Agregar("")
    Agregar("Método: Regresión OLS de (Congruente - Incongruente) ~ Género + Edad")
    Agregar("Un intercepto significativo indica que el efecto se mantiene.")
    Agregar("")

    Resultados_Control = Todos_Resultados.get('Control_Demografico', {})

    for Nombre_df in ['Generales', 'Ballotage']:
        if Nombre_df not in Resultados_Control:
            continue

        Agregar(f"{Nombre_df}:")
        Separador("-", 50)

        for Tipo in ['CO', 'CT']:
            if Tipo not in Resultados_Control[Nombre_df]:
                continue

            Res = Resultados_Control[Nombre_df][Tipo]
            N = Res['N']
            Intercepto = Res['Intercepto']
            P_Int = Res['P_Intercepto']
            Coef_Masc = Res['Coef_Genero_Masculino']
            P_Masc = Res['P_Genero_Masculino']
            Coef_Edad = Res['Coef_Edad']
            P_Edad = Res['P_Edad']
            R2_Aj = Res['R2_Ajustado']

            # Determinar significancia.
            if P_Int < 0.001:
                Sig_Int = '***'
            elif P_Int < 0.01:
                Sig_Int = '**'
            elif P_Int < 0.05:
                Sig_Int = '*'
            else:
                Sig_Int = 'ns'

            Agregar(f"\n  {Tipo} Congruente - Incongruente (N={N}):")
            Agregar(f"    Intercepto:      β = {Intercepto:7.4f}, "
                    f"p = {Formatear_P_Valor(P_Int)} {Sig_Int}")
            Agregar(f"    Género (Masc):   β = {Coef_Masc:7.4f}, "
                    f"p = {Formatear_P_Valor(P_Masc)}")
            Agregar(f"    Edad:            β = {Coef_Edad:7.4f}, "
                    f"p = {Formatear_P_Valor(P_Edad)}")
            Agregar(f"    R² ajustado:     {R2_Aj:.4f}")

            if Res['Intercepto_Significativo']:
                Agregar(f"    → El efecto {Tipo} Cong. vs Incong. SE MANTIENE "
                        f"controlando por Género y Edad.")
            else:
                Agregar(f"    → El efecto {Tipo} Cong. vs Incong. NO se "
                        f"mantiene controlando por Género y Edad.")

        Agregar("")

    Agregar("Notas:")
    Agregar("  - Categoría de referencia para Género: Femenino")
    Agregar("  - *** p<.001, ** p<.01, * p<.05, ns = no significativo")
    Agregar("")

    # =========================================================================
    # RESUMEN FINAL.
    # =========================================================================
    Separador("=")
    Agregar("RESUMEN DE RESULTADOS")
    Separador("=")
    Agregar("")

    for Nombre_df in ['Generales', 'Ballotage']:
        Agregar(f"{Nombre_df}:")
        Separador("-", 40)

        # Kruskal-Wallis.
        for Clave, Nombre in [
            ('Kruskal_CO_Items', 'CO Items'),
            ('Kruskal_CT_Items', 'CT Items'),
            ('Kruskal_Agregadas', 'Agregadas'),
            ('Kruskal_Autopercepciones', 'Autopercepciones'),
            ('Kruskal_Indices', 'Índices'),
            ('Kruskal_Cercanias', 'Cercanías'),
            ('Kruskal_Influencias', 'Influencias')
        ]:
            if Nombre_df in Todos_Resultados.get(Clave, {}):
                Tabla = Todos_Resultados[Clave][Nombre_df]
                N_Sig = Tabla['Significativo'].sum()
                Agregar(f"  KW {Nombre:20s}: {N_Sig}/{len(Tabla)} sig.")

        Agregar("")

    # Wilcoxon general.
    Agregar("Wilcoxon Congruencia (General):")
    Separador("-", 40)
    for Resultado in Resultados_Wilcoxon.get('General', []):
        Dataset = Resultado.get('Dataset', '')
        Tipo = Resultado.get('Tipo', '')
        Sig = Resultado.get('Significancia', 'ns')
        P_Val = Resultado.get('Valor_p', np.nan)
        Agregar(f"  {Dataset} {Tipo}: p = {Formatear_P_Valor(P_Val)} ({Sig})")
    Agregar("")

    # Mann-Whitney.
    if len(df_MW) > 0:
        N_Sig_MW = df_MW['Significativo'].sum()
        Agregar(f"Mann-Whitney Generales vs Ballotage: "
                f"{N_Sig_MW}/{len(df_MW)} significativas")
    Agregar("")

    Separador("=")
    Agregar("FIN DEL REPORTE")
    Separador("=")

    # Escribir archivo.
    with open(Ruta_Salida, 'w', encoding='utf-8') as Archivo:
        Archivo.write('\n'.join(Lineas))

    print(f"\nReporte guardado en: {Ruta_Salida}")


def Crear_Carpeta_Tablas(Ruta_Base: str) -> str:

    """
    Crear carpeta de tablas en Results/Tables.

    Parametros:
    - Ruta_Base: Ruta base del proyecto.

    Retorna:
    - Ruta_Tablas: Ruta a la carpeta de tablas.

    Ejemplos:
    - Ruta_Tablas = Crear_Carpeta_Tablas(Ruta_Base)
    """

    Ruta_Tablas = os.path.join(Ruta_Base, 'Results', 'Tables')
    if not os.path.exists(Ruta_Tablas):
        os.makedirs(Ruta_Tablas)
    return Ruta_Tablas


def Determinar_Asteriscos(P_Valor: float) -> str:

    """
    Determinar asteriscos de significancia por p-valor.

    Parametros:
    - P_Valor: Valor p numerico.

    Retorna:
    - Asteriscos: Cadena de asteriscos.

    Ejemplos:
    - Asteriscos = Determinar_Asteriscos(0.03)
    """

    if pd.isna(P_Valor):
        return ""
    if P_Valor < 0.001:
        return "***"
    if P_Valor < 0.01:
        return "**"
    if P_Valor < 0.05:
        return "*"
    return ""


def Convertir_Matriz_Dunn_A_Filas(
    Matriz_Dunn: pd.DataFrame,
    Dataset: str,
    Variable: str,
    Item: int = None
) -> list:

    """
    Convertir matriz Dunn a tabla larga de comparaciones.

    Parametros:
    - Matriz_Dunn: Matriz de p-valores ajustados.
    - Dataset: Nombre del dataset.
    - Variable: Nombre de la variable analizada.
    - Item: Numero de item si aplica.

    Retorna:
    - Filas: Lista de diccionarios con comparaciones.

    Ejemplos:
    - Filas = Convertir_Matriz_Dunn_A_Filas(
      Matriz_Dunn,
      Dataset = "Generales",
      Variable = "CO_Item_3_Izq",
    )
    """

    Filas = []
    for Categoria_1 in Matriz_Dunn.index:
        for Categoria_2 in Matriz_Dunn.columns:
            if Categoria_1 >= Categoria_2:
                continue
            P_Valor = Matriz_Dunn.loc[Categoria_1, Categoria_2]
            Filas.append({
                'Dataset': Dataset,
                'Variable': Variable,
                'Item': Item,
                'Categoria_1': Categoria_1,
                'Categoria_2': Categoria_2,
                'Valor_p': P_Valor,
                'Significancia': Determinar_Asteriscos(P_Valor)
            })
    return Filas


def Exportar_Tabla_Kruskal(
    Resultados: dict,
    Ruta_Salida: str,
    Tipo_Analisis: str
) -> None:

    """
    Exportar resultados de Kruskal a un xlsx.

    Parametros:
    - Resultados: Diccionario por dataset con DataFrames.
    - Ruta_Salida: Ruta del archivo xlsx.
    - Tipo_Analisis: Etiqueta del analisis.

    Retorna:
    - Ninguno.

    Ejemplos:
    - Exportar_Tabla_Kruskal(
      Resultados_Kruskal,
      Ruta_Salida = "Kruskal_CO_Items.xlsx",
      Tipo_Analisis = "CO_Items",
    )
    """

    Tablas = []
    for Nombre_df, Tabla in Resultados.items():
        if Tabla is None or len(Tabla) == 0:
            continue
        Tabla_Copia = Tabla.copy()
        Tabla_Copia['Dataset'] = Nombre_df
        Tabla_Copia['Tipo_Analisis'] = Tipo_Analisis
        Tablas.append(Tabla_Copia)

    if not Tablas:
        return

    Tabla_Final = pd.concat(Tablas, ignore_index = True)
    Tabla_Final = Tabla_Final.sort_values(
        ['Dataset', 'Valor_p']
    ).reset_index(drop = True)
    Tabla_Final.to_excel(Ruta_Salida, index = False)


def Exportar_Tabla_Dunn(
    Resultados_Dunn: dict,
    Ruta_Salida: str,
    Tipo_Analisis: str
) -> None:

    """
    Exportar resultados Dunn a un xlsx en formato largo.

    Parametros:
    - Resultados_Dunn: Diccionario por dataset y variable.
    - Ruta_Salida: Ruta del archivo xlsx.
    - Tipo_Analisis: Etiqueta del analisis.

    Retorna:
    - Ninguno.

    Ejemplos:
    - Exportar_Tabla_Dunn(
      Resultados_Dunn,
      Ruta_Salida = "Dunn_CO_Items.xlsx",
      Tipo_Analisis = "CO_Items",
    )
    """

    Filas = []
    for Nombre_df, Dicc_Matrices in Resultados_Dunn.items():
        if not Dicc_Matrices:
            continue
        for Clave, Matriz in Dicc_Matrices.items():
            Item = None
            Variable = Clave
            if isinstance(Clave, int):
                Item = Clave
                Variable = f'Diff_CO_Item_{Clave}'
            Filas.extend(
                Convertir_Matriz_Dunn_A_Filas(
                    Matriz,
                    Dataset = Nombre_df,
                    Variable = Variable,
                    Item = Item
                )
            )

    if not Filas:
        return

    Tabla_Final = pd.DataFrame(Filas)
    Tabla_Final['Tipo_Analisis'] = Tipo_Analisis
    Tabla_Final = Tabla_Final.sort_values(
        ['Dataset', 'Variable', 'Valor_p']
    ).reset_index(drop = True)
    Tabla_Final.to_excel(Ruta_Salida, index = False)


def Exportar_Tablas_Estadisticos(
    Todos_Resultados: dict,
    Ruta_Tablas: str
) -> None:

    """
    Exportar tablas xlsx de analisis estadisticos.

    Parametros:
    - Todos_Resultados: Diccionario con todos los resultados.
    - Ruta_Tablas: Carpeta de salida.

    Retorna:
    - Ninguno.

    Ejemplos:
    - Exportar_Tablas_Estadisticos(Resultados, Ruta_Tablas)
    """

    Exportar_Tabla_Kruskal(
        Todos_Resultados.get('Kruskal_CO_Items', {}),
        os.path.join(Ruta_Tablas, 'Kruskal_CO_Items.xlsx'),
        'CO_Items'
    )
    Exportar_Tabla_Kruskal(
        Todos_Resultados.get('Kruskal_CT_Items', {}),
        os.path.join(Ruta_Tablas, 'Kruskal_CT_Items.xlsx'),
        'CT_Items'
    )
    Exportar_Tabla_Kruskal(
        Todos_Resultados.get('Kruskal_Agregadas', {}),
        os.path.join(Ruta_Tablas, 'Kruskal_Aggregated.xlsx'),
        'Agregadas'
    )
    Exportar_Tabla_Kruskal(
        Todos_Resultados.get('Kruskal_Autopercepciones', {}),
        os.path.join(Ruta_Tablas, 'Kruskal_Self_Perceptions.xlsx'),
        'Autopercepciones'
    )
    Exportar_Tabla_Kruskal(
        Todos_Resultados.get('Kruskal_Indices', {}),
        os.path.join(Ruta_Tablas, 'Kruskal_Indices.xlsx'),
        'Indices'
    )
    Exportar_Tabla_Kruskal(
        Todos_Resultados.get('Kruskal_Cercanias', {}),
        os.path.join(Ruta_Tablas, 'Kruskal_Closeness.xlsx'),
        'Cercanias'
    )
    Exportar_Tabla_Kruskal(
        Todos_Resultados.get('Kruskal_Influencias', {}),
        os.path.join(Ruta_Tablas, 'Kruskal_Influences.xlsx'),
        'Influencias'
    )

    Resultados_Diff = Todos_Resultados.get(
        'Diferencias_Entre_Poblaciones', {}
    )
    Exportar_Tabla_Kruskal(
        Resultados_Diff.get('Kruskal', {}),
        os.path.join(
            Ruta_Tablas,
            'Kruskal_Differences_Between_Populations.xlsx'
        ),
        'Diferencias_Entre_Poblaciones'
    )

    Exportar_Tabla_Dunn(
        Todos_Resultados.get('Dunn_CO', {}),
        os.path.join(Ruta_Tablas, 'Dunn_CO_Items.xlsx'),
        'CO_Items'
    )
    Exportar_Tabla_Dunn(
        Todos_Resultados.get('Dunn_CT', {}),
        os.path.join(Ruta_Tablas, 'Dunn_CT_Items.xlsx'),
        'CT_Items'
    )
    Exportar_Tabla_Dunn(
        Todos_Resultados.get('Dunn_Agregadas', {}),
        os.path.join(Ruta_Tablas, 'Dunn_Aggregated.xlsx'),
        'Agregadas'
    )
    Exportar_Tabla_Dunn(
        Todos_Resultados.get('Dunn_Autopercepciones', {}),
        os.path.join(Ruta_Tablas, 'Dunn_Self_Perceptions.xlsx'),
        'Autopercepciones'
    )
    Exportar_Tabla_Dunn(
        Todos_Resultados.get('Dunn_Indices', {}),
        os.path.join(Ruta_Tablas, 'Dunn_Indices.xlsx'),
        'Indices'
    )
    Exportar_Tabla_Dunn(
        Todos_Resultados.get('Dunn_Cercanias', {}),
        os.path.join(Ruta_Tablas, 'Dunn_Closeness.xlsx'),
        'Cercanias'
    )
    Exportar_Tabla_Dunn(
        Todos_Resultados.get('Dunn_Influencias', {}),
        os.path.join(Ruta_Tablas, 'Dunn_Influences.xlsx'),
        'Influencias'
    )
    Exportar_Tabla_Dunn(
        Resultados_Diff.get('Dunn', {}),
        os.path.join(
            Ruta_Tablas,
            'Dunn_Differences_Between_Populations.xlsx'
        ),
        'Diferencias_Entre_Poblaciones'
    )


if __name__ == '__main__':
    # Ruta a la carpeta con los datos definitivos.
    Ruta_Base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    Ruta_Datos = os.path.join(Ruta_Base, 'Processed_Data')

    # Ejecutar todos los análisis.
    Resultados, dfs = Ejecutar_Todos_Los_Analisis(Ruta_Datos)

    # Crear carpeta Results/Reports si no existe.
    Ruta_Reportes = os.path.join(Ruta_Base, 'Results', 'Reports')
    if not os.path.exists(Ruta_Reportes):
        os.makedirs(Ruta_Reportes)

    # Generar reporte TXT.
    Ruta_Reporte = os.path.join(
        Ruta_Reportes,
        'Report_Statistical_Analysis.txt'
    )
    Generar_Reporte_TXT(Resultados, dfs, Ruta_Reporte)

    # Exportar tablas xlsx.
    Ruta_Tablas = Crear_Carpeta_Tablas(Ruta_Base)
    Exportar_Tablas_Estadisticos(Resultados, Ruta_Tablas)

    print("\nResultados guardados en el diccionario 'Resultados'.")
