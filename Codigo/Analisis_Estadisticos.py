# -*- coding: utf-8 -*-
"""
Análisis estadísticos para el proyecto de tesis.

Este script contiene funciones para realizar análisis estadísticos
de Kruskal-Wallis, Dunn post-hoc, Wilcoxon y Mann-Whitney sobre
las bases de datos procesadas.

"""

import pandas as pd
import numpy as np
from scipy.stats import kruskal, wilcoxon, mannwhitneyu
import scikit_posthocs as sp
import os


# =============================================================================
# CONSTANTES Y CONFIGURACIÓN.
# =============================================================================

# Ítems progresistas y conservadores.
Items_Progresistas = [5, 6, 9, 11, 16, 20, 24, 25, 27, 28]
Items_Conservadores = [3, 4, 7, 8, 10, 19, 22, 23, 29, 30]

# Categorías válidas para análisis.
Categorias_Validas = [
    'Left_Wing',
    'Progressivism',
    'Centre',
    'Moderate_Right_A',
    'Moderate_Right_B',
    'Right_Wing_Libertarian'
]

# Etiquetas para categorías.
Etiquetas_Categorias = {
    'Left_Wing': 'Left Wing',
    'Progressivism': 'Progressivism',
    'Centre': 'Centre',
    'Moderate_Right_A': 'Moderate Right A',
    'Moderate_Right_B': 'Moderate Right B',
    'Right_Wing_Libertarian': 'Right Wing Libertarian'
}


def Formatear_P_Valor(P_Valor: float) -> str:

    """
    Formatea un p-valor en notación científica con 2 decimales.

    Parámetros:
        P_Valor (float): El p-valor a formatear.

    Retorna:
        str: El p-valor formateado como string (ej: "2.12E-15").

    """

    if pd.isna(P_Valor):
        return "NaN"

    return f"{P_Valor:.2E}"


# =============================================================================
# FUNCIONES DE CARGA DE DATOS.
# =============================================================================

def Cargar_Bases_Datos(Ruta_Carpeta: str) -> dict:

    """
    Carga las bases de datos de Generales y Ballotage desde
    archivos Excel.

    Parámetros:
        Ruta_Carpeta (str): Ruta a la carpeta que contiene
            los archivos Excel.

    Retorna:
        dict: Diccionario con las bases de datos cargadas.

    """

    dfs_Finales = {}

    for Nombre in ['Generales', 'Ballotage']:
        Ruta_Archivo = os.path.join(Ruta_Carpeta, f'{Nombre}.xlsx')
        if os.path.exists(Ruta_Archivo):
            dfs_Finales[Nombre] = pd.read_excel(Ruta_Archivo)
            print(f"  Cargado: {Nombre} ({len(dfs_Finales[Nombre])} registros)")
        else:
            print(f"  Advertencia: No se encontró {Ruta_Archivo}")

    return dfs_Finales


# =============================================================================
# FUNCIONES DE KRUSKAL-WALLIS.
# =============================================================================

def Kruskal_Wallis_Por_Categoria(
    df: pd.DataFrame,
    Columnas: list,
    Columna_Categoria: str = 'Categoria_PASO_2023'
) -> pd.DataFrame:

    """
    Aplica el test de Kruskal-Wallis para comparar múltiples grupos
    definidos por una columna categórica.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        Columnas (list): Lista de columnas numéricas a analizar.
        Columna_Categoria (str): Nombre de la columna categórica
            para agrupar.

    Retorna:
        pd.DataFrame: Tabla con resultados de Kruskal-Wallis.

    """

    # Filtrar solo las categorías válidas.
    df_Filtrado = df[df[Columna_Categoria].isin(Categorias_Validas)].copy()

    Resultados = {}

    for Nombre_Columna in Columnas:
        # Crear grupos por categoría.
        Grupos = [
            df_Filtrado[
                df_Filtrado[Columna_Categoria] == Categoria
            ][Nombre_Columna].dropna()
            for Categoria in Categorias_Validas
        ]

        # Verificar que todos los grupos tengan al menos 2 datos.
        if all(len(Grupo) > 1 for Grupo in Grupos):
            Estadistico, P_Valor = kruskal(*Grupos)
            Resultados[Nombre_Columna] = {
                'Estadistico_H': Estadistico,
                'Valor_p': P_Valor
            }

    # Convertir a DataFrame.
    Tabla_Resultados = pd.DataFrame.from_dict(
        Resultados, orient='index'
    ).reset_index().rename(columns={'index': 'Variable'})

    # Ordenar por p-valor ascendente.
    Tabla_Resultados = Tabla_Resultados.sort_values(
        by='Valor_p'
    ).reset_index(drop=True)

    # Agregar columna de significancia.
    Tabla_Resultados['Significativo'] = \
        Tabla_Resultados['Valor_p'] < 0.05

    return Tabla_Resultados


def Obtener_Columnas_CO_Items() -> tuple:

    """
    Genera las listas de columnas de Cambio de Opinión por ítem.

    Retorna:
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
    Genera las listas de columnas de Cambio de Tiempo por ítem.

    Retorna:
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
# FUNCIONES DE DUNN POST-HOC.
# =============================================================================

def Dunn_Post_Hoc(
    df: pd.DataFrame,
    Variable: str,
    Columna_Categoria: str = 'Categoria_PASO_2023',
    Metodo_Ajuste: str = 'bonferroni'
) -> pd.DataFrame:

    """
    Aplica la prueba post-hoc de Dunn con corrección de Bonferroni
    para comparaciones pareadas entre categorías.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        Variable (str): Nombre de la columna a analizar.
        Columna_Categoria (str): Nombre de la columna categórica.
        Metodo_Ajuste (str): Método de ajuste de p-valores.

    Retorna:
        pd.DataFrame: Matriz de p-valores ajustados.

    """

    # Filtrar categorías válidas y eliminar NaN.
    df_Filtrado = df[
        df[Columna_Categoria].isin(Categorias_Validas)
    ][[Columna_Categoria, Variable]].dropna()

    # Aplicar prueba de Dunn.
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
    Extrae las comparaciones pareadas significativas de una
    matriz de Dunn.

    Parámetros:
        Matriz_Dunn (pd.DataFrame): Matriz de p-valores.
        Umbral_P (float): Umbral de significancia.

    Retorna:
        list: Lista de diccionarios con comparaciones significativas.

    """

    Comparaciones = []

    for Categoria_1 in Matriz_Dunn.index:
        for Categoria_2 in Matriz_Dunn.columns:
            # Evitar duplicados (solo triángulo superior).
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
    Ejecuta Dunn post-hoc para todas las variables significativas
    de Kruskal-Wallis.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        Variables_Significativas (list): Lista de variables.
        Columna_Categoria (str): Columna categórica.

    Retorna:
        dict: Diccionario con matrices de Dunn por variable.

    """

    Resultados_Dunn = {}

    for Variable in Variables_Significativas:
        Matriz = Dunn_Post_Hoc(df, Variable, Columna_Categoria)
        Resultados_Dunn[Variable] = Matriz

    return Resultados_Dunn


# =============================================================================
# FUNCIONES DE WILCOXON.
# =============================================================================

def Wilcoxon_Pareado(
    df: pd.DataFrame,
    Variable_1: str,
    Variable_2: str
) -> dict:

    """
    Aplica el test de Wilcoxon pareado entre dos variables.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        Variable_1 (str): Primera variable.
        Variable_2 (str): Segunda variable.

    Retorna:
        dict: Diccionario con resultados del test.

    """

    # Eliminar NaN en ambas variables.
    Datos_Pareados = df[[Variable_1, Variable_2]].dropna()

    if len(Datos_Pareados) < 2:
        return {
            'n': len(Datos_Pareados),
            'Estadistico_W': np.nan,
            'Valor_p': np.nan,
            'Significativo': False,
            'Error': 'Datos insuficientes'
        }

    try:
        Estadistico, P_Valor = wilcoxon(
            Datos_Pareados[Variable_1],
            Datos_Pareados[Variable_2]
        )

        # Determinar nivel de significancia.
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
    Aplica Wilcoxon para comparar Congruente vs Incongruente.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        Tipo (str): 'CO' para Cambio de Opinión o 'CT' para
            Cambio de Tiempo.

    Retorna:
        dict: Resultados del test.

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
    Aplica Wilcoxon Congruente vs Incongruente por cada categoría.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        Tipo (str): 'CO' o 'CT'.
        Columna_Categoria (str): Columna categórica.

    Retorna:
        pd.DataFrame: Resultados por categoría.

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
# FUNCIONES DE WILCOXON PARA CURVAS (SIN ASOCIAR VS ASOCIADA).
# =============================================================================

def Calcular_Significancia_Wilcoxon_Curvas_Asociacion(
    Datos: dict
) -> dict:

    """
    Calcula Wilcoxon entre ítems sin asociar y asociados para
    poblaciones de derecha e izquierda.

    Parámetros:
        Datos (dict): Diccionario con 'Orden_Items', 'df_Derecha'
            y 'df_Izquierda'.

    Retorna:
        dict: P-valores por ítem para derecha e izquierda.

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
    Calcula Wilcoxon para cada población comparando su baseline
    (sin asociar) vs asociada.

    Parámetros:
        Datos (dict): Diccionario con 'Orden_Items', 'df_Izq' y 'df_Der'.

    Retorna:
        dict: P-valores por ítem para izquierda y derecha.

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
    Calcula Wilcoxon en muestra balanceada (mismas personas)
    comparando baseline vs asociado.

    Parámetros:
        Datos (dict): Diccionario con 'Orden_Items', 'df_Izq' y 'df_Der'.

    Retorna:
        dict: P-valores por ítem para izquierda y derecha.

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
    Calcula Wilcoxon para izquierda, centro y derecha.
    En centro compara baseline vs promedio de Izq+Der.

    Parámetros:
        Datos (dict): Diccionario con 'Orden_Items', 'df_Izq',
            'df_Centro' y 'df_Der'.

    Retorna:
        dict: P-valores por ítem para izquierda, centro y derecha.

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
    Calcula Wilcoxon para izquierda, centro (asoc Izq y Der)
    y derecha.

    Parámetros:
        Datos (dict): Diccionario con 'Orden_Items', 'df_Izq',
            'df_Centro' y 'df_Der'.

    Retorna:
        dict: P-valores por ítem para izquierda, centro y derecha.

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
# FUNCIONES DE MANN-WHITNEY U.
# =============================================================================

def Test_Mann_Whitney(Valores_1: list, Valores_2: list) -> float:

    """
    Realiza test de Mann-Whitney U para comparar dos grupos.

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
    Realiza test de Wilcoxon signed-rank contra cero.
    Testea si las diferencias son significativamente distintas de 0.

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
    Convierte p-valor a string de significancia.

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
    Aplica Mann-Whitney U para comparar una variable entre
    dos datasets (Generales vs Ballotage).

    Parámetros:
        df_1 (pd.DataFrame): Primer DataFrame.
        df_2 (pd.DataFrame): Segundo DataFrame.
        Variable (str): Variable a comparar.
        Nombre_1 (str): Nombre del primer dataset.
        Nombre_2 (str): Nombre del segundo dataset.

    Retorna:
        dict: Resultados del test.

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
            'Error': 'Datos insuficientes'
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
# FUNCIONES DE ANÁLISIS COMPLETO.
# =============================================================================

def Analisis_Kruskal_CO_Items(dfs_Finales: dict) -> dict:

    """
    Ejecuta Kruskal-Wallis para todos los ítems CO individuales
    en ambos datasets.

    Parámetros:
        dfs_Finales (dict): Diccionario con DataFrames.

    Retorna:
        dict: Resultados por dataset.

    """

    _, _, _, _, Todas_Columnas_CO = Obtener_Columnas_CO_Items()

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Kruskal-Wallis CO Items - {Nombre_df}")
        print('='*60)

        Tabla = Kruskal_Wallis_Por_Categoria(df, Todas_Columnas_CO)
        Resultados[Nombre_df] = Tabla

        # Mostrar resumen.
        N_Significativas = Tabla['Significativo'].sum()
        print(f"Variables significativas: {N_Significativas}/{len(Tabla)}")

        if N_Significativas > 0:
            print("\nVariables significativas (p < 0.05):")
            Sig = Tabla[Tabla['Significativo']]
            for _, Fila in Sig.iterrows():
                print(f"  - {Fila['Variable']}: "
                      f"H={Fila['Estadistico_H']:.4f}, "
                      f"p={Formatear_P_Valor(Fila['Valor_p'])}")

    return Resultados


def Analisis_Kruskal_CT_Items(dfs_Finales: dict) -> dict:

    """
    Ejecuta Kruskal-Wallis para todos los ítems CT individuales
    en ambos datasets.

    Parámetros:
        dfs_Finales (dict): Diccionario con DataFrames.

    Retorna:
        dict: Resultados por dataset.

    """

    _, _, _, _, Todas_Columnas_CT = Obtener_Columnas_CT_Items()

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Kruskal-Wallis CT Items - {Nombre_df}")
        print('='*60)

        Tabla = Kruskal_Wallis_Por_Categoria(df, Todas_Columnas_CT)
        Resultados[Nombre_df] = Tabla

        # Mostrar resumen.
        N_Significativas = Tabla['Significativo'].sum()
        print(f"Variables significativas: {N_Significativas}/{len(Tabla)}")

        if N_Significativas > 0:
            print("\nVariables significativas (p < 0.05):")
            Sig = Tabla[Tabla['Significativo']]
            for _, Fila in Sig.iterrows():
                print(f"  - {Fila['Variable']}: "
                      f"H={Fila['Estadistico_H']:.4f}, "
                      f"p={Formatear_P_Valor(Fila['Valor_p'])}")

    return Resultados


def Obtener_Columnas_Agregadas() -> list:

    """
    Retorna la lista de todas las columnas agregadas de CO y CT.

    Retorna:
        list: Lista de nombres de columnas agregadas.

    """

    # Variables agregadas de CO.
    Columnas_CO_Agregadas = [
        'CO_Pro_Izq', 'CO_Con_Izq', 'CO_Pro_Der', 'CO_Con_Der',
        'CO_Congruente', 'CO_Incongruente'
    ]

    # Variables agregadas de CT.
    Columnas_CT_Agregadas = [
        'CT_Pro_Izq', 'CT_Con_Izq', 'CT_Pro_Der', 'CT_Con_Der',
        'CT_Congruente', 'CT_Incongruente'
    ]

    return Columnas_CO_Agregadas + Columnas_CT_Agregadas


def Analisis_Kruskal_Variables_Agregadas(dfs_Finales: dict) -> dict:

    """
    Ejecuta Kruskal-Wallis para variables agregadas de CO y CT.

    Parámetros:
        dfs_Finales (dict): Diccionario con DataFrames.

    Retorna:
        dict: Resultados por dataset.

    """

    Todas_Agregadas = Obtener_Columnas_Agregadas()

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Kruskal-Wallis Variables Agregadas - {Nombre_df}")
        print('='*60)

        # Filtrar solo columnas que existen.
        Columnas_Existentes = [
            c for c in Todas_Agregadas if c in df.columns
        ]

        if not Columnas_Existentes:
            print("  No se encontraron variables agregadas.")
            continue

        Tabla = Kruskal_Wallis_Por_Categoria(df, Columnas_Existentes)
        Resultados[Nombre_df] = Tabla

        # Mostrar resumen.
        N_Significativas = Tabla['Significativo'].sum()
        print(f"Variables significativas: {N_Significativas}/{len(Tabla)}")

        print("\nResultados:")
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
    Ejecuta Dunn post-hoc para variables agregadas significativas
    de Kruskal-Wallis.

    Parámetros:
        dfs_Finales (dict): Diccionario con DataFrames.
        Resultados_Kruskal_Agregadas (dict): Resultados de Kruskal-Wallis
            para variables agregadas.

    Retorna:
        dict: Resultados de Dunn por dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Dunn Post-Hoc Variables Agregadas - {Nombre_df}")
        print('='*60)

        Tabla_Kruskal = Resultados_Kruskal_Agregadas.get(Nombre_df)

        if Tabla_Kruskal is None:
            print("  No hay resultados de Kruskal-Wallis.")
            continue

        # Obtener variables significativas.
        Variables_Sig = Tabla_Kruskal[
            Tabla_Kruskal['Significativo']
        ]['Variable'].tolist()

        if not Variables_Sig:
            print("  No hay variables significativas para post-hoc.")
            continue

        print(f"Variables para post-hoc: {len(Variables_Sig)}")

        Resultados_Dunn = Ejecutar_Dunn_Variables_Significativas(
            df, Variables_Sig
        )

        Resultados[Nombre_df] = Resultados_Dunn

        # Mostrar resumen de comparaciones significativas.
        for Variable, Matriz in Resultados_Dunn.items():
            Comparaciones = Extraer_Comparaciones_Significativas(Matriz)

            if Comparaciones:
                print(f"\n  {Variable}:")
                for Comp in Comparaciones:
                    print(f"    - {Comp['Categoria_1']} vs "
                          f"{Comp['Categoria_2']}: "
                          f"p={Formatear_P_Valor(Comp['Valor_p'])}")
            else:
                print(f"\n  {Variable}: Sin comparaciones significativas")

    return Resultados


def Analisis_Dunn_Post_Hoc(
    dfs_Finales: dict,
    Resultados_Kruskal: dict
) -> dict:

    """
    Ejecuta Dunn post-hoc para variables significativas
    de Kruskal-Wallis.

    Parámetros:
        dfs_Finales (dict): Diccionario con DataFrames.
        Resultados_Kruskal (dict): Resultados de Kruskal-Wallis.

    Retorna:
        dict: Resultados de Dunn por dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Dunn Post-Hoc - {Nombre_df}")
        print('='*60)

        Tabla_Kruskal = Resultados_Kruskal.get(Nombre_df)

        if Tabla_Kruskal is None:
            print("  No hay resultados de Kruskal-Wallis.")
            continue

        # Obtener variables significativas.
        Variables_Sig = Tabla_Kruskal[
            Tabla_Kruskal['Significativo']
        ]['Variable'].tolist()

        if not Variables_Sig:
            print("  No hay variables significativas para post-hoc.")
            continue

        print(f"Variables para post-hoc: {len(Variables_Sig)}")

        Resultados_Dunn = Ejecutar_Dunn_Variables_Significativas(
            df, Variables_Sig
        )

        Resultados[Nombre_df] = Resultados_Dunn

        # Mostrar resumen de comparaciones significativas.
        for Variable, Matriz in Resultados_Dunn.items():
            Comparaciones = Extraer_Comparaciones_Significativas(Matriz)

            if Comparaciones:
                print(f"\n  {Variable}:")
                for Comp in Comparaciones:
                    print(f"    - {Comp['Categoria_1']} vs "
                          f"{Comp['Categoria_2']}: "
                          f"p={Formatear_P_Valor(Comp['Valor_p'])}")
            else:
                print(f"\n  {Variable}: Sin comparaciones significativas")

    return Resultados


def Analisis_Wilcoxon_Congruencia(dfs_Finales: dict) -> dict:

    """
    Ejecuta Wilcoxon para comparar Congruente vs Incongruente.

    Parámetros:
        dfs_Finales (dict): Diccionario con DataFrames.

    Retorna:
        dict: Resultados por dataset y tipo.

    """

    Resultados = {'General': [], 'Por_Categoria': []}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Wilcoxon Congruencia - {Nombre_df}")
        print('='*60)

        # Análisis general (todas las poblaciones).
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

        # Análisis por categoría.
        print("\n  Por Categoría:")
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
    Compara variables entre Generales y Ballotage usando
    Mann-Whitney U.

    Parámetros:
        dfs_Finales (dict): Diccionario con DataFrames.
        Variables (list): Lista de variables a comparar.

    Retorna:
        pd.DataFrame: Resultados de las comparaciones.

    """

    if 'Generales' not in dfs_Finales or 'Ballotage' not in dfs_Finales:
        print("Error: Se necesitan ambos datasets.")
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
# FUNCIONES DE COMPARACIÓN IZQ VS DER POR ÍTEM.
# =============================================================================

def Wilcoxon_Izq_vs_Der_Por_Item(
    df: pd.DataFrame,
    Items: list = None
) -> pd.DataFrame:

    """
    Compara CO_Item_X_Izq vs CO_Item_X_Der usando Wilcoxon pareado
    para cada ítem (mismo participante, diferente asociación).

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        Items (list): Lista de números de ítems a analizar.

    Retorna:
        pd.DataFrame: Resultados por ítem.

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

        # Datos pareados (sin NaN en ninguna de las dos).
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

            # Determinar tipo de ítem.
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
    Ejecuta Wilcoxon pareado Izq vs Der para todos los ítems
    en ambos datasets.

    Parámetros:
        dfs_Finales (dict): Diccionario con DataFrames.

    Retorna:
        dict: Resultados por dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Wilcoxon Izq vs Der por Ítem - {Nombre_df}")
        print('='*60)

        Tabla = Wilcoxon_Izq_vs_Der_Por_Item(df)
        Resultados[Nombre_df] = Tabla

        N_Sig = Tabla['Significativo'].sum() if len(Tabla) > 0 else 0
        print(f"Ítems con diferencia significativa: {N_Sig}/{len(Tabla)}")

        if N_Sig > 0:
            print("\nÍtems significativos (Izq ≠ Der):")
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
    Calcula la diferencia (Izq - Der) para cada ítem y participante.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        Items (list): Lista de números de ítems.

    Retorna:
        pd.DataFrame: DataFrame con columnas Diff_CO_Item_X.

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
    Compara la diferencia (Izq - Der) entre las 6 categorías
    electorales usando Kruskal-Wallis.

    Parámetros:
        df (pd.DataFrame): DataFrame con diferencias calculadas.
        Items (list): Lista de números de ítems.
        Columna_Categoria (str): Columna de categoría.

    Retorna:
        pd.DataFrame: Resultados de Kruskal-Wallis por ítem.

    """

    if Items is None:
        Items = Items_Progresistas + Items_Conservadores

    # Filtrar categorías válidas.
    df_Filtrado = df[df[Columna_Categoria].isin(Categorias_Validas)]

    Resultados = []

    for Item in Items:
        Variable_Diff = f'Diff_CO_Item_{Item}'

        if Variable_Diff not in df_Filtrado.columns:
            continue

        # Crear grupos por categoría.
        Grupos = [
            df_Filtrado[
                df_Filtrado[Columna_Categoria] == Cat
            ][Variable_Diff].dropna()
            for Cat in Categorias_Validas
        ]

        # Verificar datos suficientes.
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
    Analiza si la diferencia (Izq - Der) varía entre poblaciones
    usando Kruskal-Wallis y Dunn post-hoc.

    Parámetros:
        dfs_Finales (dict): Diccionario con DataFrames.

    Retorna:
        dict: Resultados con Kruskal-Wallis y Dunn.

    """

    Resultados = {'Kruskal': {}, 'Dunn': {}}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Diferencias (Izq-Der) Entre Poblaciones - {Nombre_df}")
        print('='*60)

        # Calcular diferencias.
        df_Con_Diff = Calcular_Diferencias_Izq_Der(df)

        # Kruskal-Wallis.
        Tabla_Kruskal = Kruskal_Wallis_Diferencias_Entre_Poblaciones(
            df_Con_Diff
        )
        Resultados['Kruskal'][Nombre_df] = Tabla_Kruskal

        N_Sig = Tabla_Kruskal['Significativo'].sum() \
                if len(Tabla_Kruskal) > 0 else 0
        print(f"Ítems con diferencias entre poblaciones: "
              f"{N_Sig}/{len(Tabla_Kruskal)}")

        # Dunn post-hoc para significativos.
        Items_Sig = Tabla_Kruskal[
            Tabla_Kruskal['Significativo']
        ]['Item'].tolist() if len(Tabla_Kruskal) > 0 else []

        Resultados_Dunn = {}

        if Items_Sig:
            print(f"\nPost-hoc Dunn para {len(Items_Sig)} ítems:")

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

                    # Extraer comparaciones significativas.
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
                        print(f"\n  Item {Item}: Sin comparaciones "
                              f"significativas post-Bonferroni")

                except Exception:
                    continue

        Resultados['Dunn'][Nombre_df] = Resultados_Dunn

    return Resultados


# =============================================================================
# FUNCIONES PARA AUTOPERCEPCIONES, ÍNDICES Y CERCANÍAS.
# =============================================================================

# Columnas de autopercepciones.
Columnas_Autopercepciones = [
    'Autopercepcion_Izq_Der',
    'Autopercepcion_Con_Pro',
    'Autopercepcion_Per_Antiper'
]

# Columnas de índices.
Columnas_Indices = [
    'Indice_Progresismo',
    'Indice_Conservadurismo',
    'Indice_Positividad'
]

# Columnas de cercanías a candidatos.
Columnas_Cercanias = [
    'Cercania_Massa',
    'Cercania_Bullrich',
    'Cercania_Bregman',
    'Cercania_Milei',
    'Cercania_Schiaretti'
]


def Analisis_Kruskal_Autopercepciones(dfs_Finales: dict) -> dict:

    """
    Ejecuta Kruskal-Wallis para autopercepciones entre las
    categorías electorales.

    Parámetros:
        dfs_Finales (dict): Diccionario con DataFrames.

    Retorna:
        dict: Resultados por dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Kruskal-Wallis Autopercepciones - {Nombre_df}")
        print('='*60)

        # Filtrar columnas existentes.
        Columnas_Existentes = [
            c for c in Columnas_Autopercepciones if c in df.columns
        ]

        if not Columnas_Existentes:
            print("  No se encontraron columnas de autopercepciones.")
            continue

        Tabla = Kruskal_Wallis_Por_Categoria(df, Columnas_Existentes)
        Resultados[Nombre_df] = Tabla

        print("\nResultados:")
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
    Ejecuta Kruskal-Wallis para índices (Progresismo, Conservadurismo,
    Positividad) entre las categorías electorales.

    Parámetros:
        dfs_Finales (dict): Diccionario con DataFrames.

    Retorna:
        dict: Resultados por dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Kruskal-Wallis Índices - {Nombre_df}")
        print('='*60)

        # Filtrar columnas existentes.
        Columnas_Existentes = [
            c for c in Columnas_Indices if c in df.columns
        ]

        if not Columnas_Existentes:
            print("  No se encontraron columnas de índices.")
            continue

        Tabla = Kruskal_Wallis_Por_Categoria(df, Columnas_Existentes)
        Resultados[Nombre_df] = Tabla

        print("\nResultados:")
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
    Ejecuta Kruskal-Wallis para cercanías a candidatos entre
    las categorías electorales.

    Parámetros:
        dfs_Finales (dict): Diccionario con DataFrames.

    Retorna:
        dict: Resultados por dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Kruskal-Wallis Cercanías a Candidatos - {Nombre_df}")
        print('='*60)

        # Filtrar columnas existentes.
        Columnas_Existentes = [
            c for c in Columnas_Cercanias if c in df.columns
        ]

        if not Columnas_Existentes:
            print("  No se encontraron columnas de cercanías.")
            continue

        Tabla = Kruskal_Wallis_Por_Categoria(df, Columnas_Existentes)
        Resultados[Nombre_df] = Tabla

        print("\nResultados:")
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
    Ejecuta Dunn post-hoc para autopercepciones significativas
    de Kruskal-Wallis.

    Parámetros:
        dfs_Finales (dict): Diccionario con DataFrames.
        Resultados_Kruskal_Autopercepciones (dict): Resultados de
            Kruskal-Wallis para autopercepciones.

    Retorna:
        dict: Resultados de Dunn por dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Dunn Post-Hoc Autopercepciones - {Nombre_df}")
        print('='*60)

        Tabla_Kruskal = Resultados_Kruskal_Autopercepciones.get(Nombre_df)

        if Tabla_Kruskal is None:
            print("  No hay resultados de Kruskal-Wallis.")
            continue

        # Obtener variables significativas.
        Variables_Sig = Tabla_Kruskal[
            Tabla_Kruskal['Significativo']
        ]['Variable'].tolist()

        if not Variables_Sig:
            print("  No hay variables significativas para post-hoc.")
            continue

        print(f"Variables para post-hoc: {len(Variables_Sig)}")

        Resultados_Dunn = Ejecutar_Dunn_Variables_Significativas(
            df, Variables_Sig
        )

        Resultados[Nombre_df] = Resultados_Dunn

        # Mostrar resumen de comparaciones significativas.
        for Variable, Matriz in Resultados_Dunn.items():
            Comparaciones = Extraer_Comparaciones_Significativas(Matriz)

            if Comparaciones:
                print(f"\n  {Variable}:")
                for Comp in Comparaciones:
                    print(f"    - {Comp['Categoria_1']} vs "
                          f"{Comp['Categoria_2']}: "
                          f"p={Formatear_P_Valor(Comp['Valor_p'])}")
            else:
                print(f"\n  {Variable}: Sin comparaciones significativas")

    return Resultados


def Analisis_Dunn_Indices(
    dfs_Finales: dict,
    Resultados_Kruskal_Indices: dict
) -> dict:

    """
    Ejecuta Dunn post-hoc para índices significativos
    de Kruskal-Wallis.

    Parámetros:
        dfs_Finales (dict): Diccionario con DataFrames.
        Resultados_Kruskal_Indices (dict): Resultados de
            Kruskal-Wallis para índices.

    Retorna:
        dict: Resultados de Dunn por dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Dunn Post-Hoc Índices - {Nombre_df}")
        print('='*60)

        Tabla_Kruskal = Resultados_Kruskal_Indices.get(Nombre_df)

        if Tabla_Kruskal is None:
            print("  No hay resultados de Kruskal-Wallis.")
            continue

        # Obtener variables significativas.
        Variables_Sig = Tabla_Kruskal[
            Tabla_Kruskal['Significativo']
        ]['Variable'].tolist()

        if not Variables_Sig:
            print("  No hay variables significativas para post-hoc.")
            continue

        print(f"Variables para post-hoc: {len(Variables_Sig)}")

        Resultados_Dunn = Ejecutar_Dunn_Variables_Significativas(
            df, Variables_Sig
        )

        Resultados[Nombre_df] = Resultados_Dunn

        # Mostrar resumen de comparaciones significativas.
        for Variable, Matriz in Resultados_Dunn.items():
            Comparaciones = Extraer_Comparaciones_Significativas(Matriz)

            if Comparaciones:
                print(f"\n  {Variable}:")
                for Comp in Comparaciones:
                    print(f"    - {Comp['Categoria_1']} vs "
                          f"{Comp['Categoria_2']}: "
                          f"p={Formatear_P_Valor(Comp['Valor_p'])}")
            else:
                print(f"\n  {Variable}: Sin comparaciones significativas")

    return Resultados


def Analisis_Dunn_Cercanias(
    dfs_Finales: dict,
    Resultados_Kruskal_Cercanias: dict
) -> dict:

    """
    Ejecuta Dunn post-hoc para cercanías a candidatos significativas
    de Kruskal-Wallis.

    Parámetros:
        dfs_Finales (dict): Diccionario con DataFrames.
        Resultados_Kruskal_Cercanias (dict): Resultados de
            Kruskal-Wallis para cercanías.

    Retorna:
        dict: Resultados de Dunn por dataset.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"Dunn Post-Hoc Cercanías a Candidatos - {Nombre_df}")
        print('='*60)

        Tabla_Kruskal = Resultados_Kruskal_Cercanias.get(Nombre_df)

        if Tabla_Kruskal is None:
            print("  No hay resultados de Kruskal-Wallis.")
            continue

        # Obtener variables significativas.
        Variables_Sig = Tabla_Kruskal[
            Tabla_Kruskal['Significativo']
        ]['Variable'].tolist()

        if not Variables_Sig:
            print("  No hay variables significativas para post-hoc.")
            continue

        print(f"Variables para post-hoc: {len(Variables_Sig)}")

        Resultados_Dunn = Ejecutar_Dunn_Variables_Significativas(
            df, Variables_Sig
        )

        Resultados[Nombre_df] = Resultados_Dunn

        # Mostrar resumen de comparaciones significativas.
        for Variable, Matriz in Resultados_Dunn.items():
            Comparaciones = Extraer_Comparaciones_Significativas(Matriz)

            if Comparaciones:
                print(f"\n  {Variable}:")
                for Comp in Comparaciones:
                    print(f"    - {Comp['Categoria_1']} vs "
                          f"{Comp['Categoria_2']}: "
                          f"p={Formatear_P_Valor(Comp['Valor_p'])}")
            else:
                print(f"\n  {Variable}: Sin comparaciones significativas")

    return Resultados


# =============================================================================
# EJECUCIÓN PRINCIPAL.
# =============================================================================

def Ejecutar_Todos_Los_Analisis(Ruta_Datos: str) -> dict:

    """
    Ejecuta todos los análisis estadísticos y retorna los resultados.

    Orden de ejecución:
    - Primero: Todos los Kruskal-Wallis (1-8)
    - Segundo: Todos los Dunn post-hoc (9-16)
    - Tercero: Tests pareados - Wilcoxon y Mann-Whitney (17-19)

    Parámetros:
        Ruta_Datos (str): Ruta a la carpeta con los datos.

    Retorna:
        dict: Diccionario con todos los resultados.

    """

    print("="*70)
    print("ANÁLISIS ESTADÍSTICOS COMPLETOS")
    print("="*70)

    # Cargar datos.
    print("\nCargando datos...")
    dfs_Finales = Cargar_Bases_Datos(Ruta_Datos)

    if not dfs_Finales:
        print("Error: No se pudieron cargar los datos.")
        return {}

    Todos_Resultados = {}

    # =========================================================================
    # SECCIÓN A: TESTS KRUSKAL-WALLIS.
    # =========================================================================

    print("\n" + "#"*70)
    print("# SECCIÓN A: TESTS KRUSKAL-WALLIS")
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

    # =========================================================================
    # SECCIÓN B: TESTS DUNN POST-HOC.
    # =========================================================================

    print("\n" + "#"*70)
    print("# SECCIÓN B: TESTS DUNN POST-HOC (Corrección Bonferroni)")
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

    # =========================================================================
    # SECCIÓN C: TESTS PAREADOS (WILCOXON Y MANN-WHITNEY).
    # =========================================================================

    print("\n" + "#"*70)
    print("# SECCIÓN C: TESTS PAREADOS (WILCOXON Y MANN-WHITNEY)")
    print("#"*70)

    # 14. Wilcoxon para congruencia ideológica.
    print("\n" + "="*70)
    print("14. WILCOXON: CONGRUENCIA IDEOLÓGICA")
    print("="*70)
    Todos_Resultados['Wilcoxon_Congruencia'] = \
        Analisis_Wilcoxon_Congruencia(dfs_Finales)

    # 15. Wilcoxon Izq vs Der por ítem.
    print("\n" + "="*70)
    print("15. WILCOXON: IZQ VS DER POR ÍTEM (PAREADO)")
    print("="*70)
    Todos_Resultados['Wilcoxon_Izq_vs_Der'] = \
        Analisis_Wilcoxon_Izq_vs_Der(dfs_Finales)

    # 16. Mann-Whitney entre elecciones.
    print("\n" + "="*70)
    print("16. MANN-WHITNEY: GENERALES VS BALLOTAGE")
    print("="*70)

    # Variables a comparar entre elecciones.
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

    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADOS")
    print("="*70)

    return Todos_Resultados, dfs_Finales


def Generar_Reporte_TXT(
    Todos_Resultados: dict,
    dfs_Finales: dict,
    Ruta_Salida: str
) -> None:

    """
    Genera un archivo TXT con todos los resultados formateados
    de manera legible.

    Orden del reporte:
    - Sección A: Todos los Kruskal-Wallis (1-7)
    - Sección B: Todos los Dunn post-hoc (8-13)
    - Sección C: Tests pareados Wilcoxon y Mann-Whitney (14-16)

    Parámetros:
        Todos_Resultados (dict): Diccionario con todos los resultados.
        dfs_Finales (dict): Diccionario con los DataFrames.
        Ruta_Salida (str): Ruta al archivo TXT de salida.

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
        Función auxiliar para escribir secciones de Kruskal-Wallis.

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
                Agregar(f"  {Fila['Variable']:30s} "
                        f"H = {Fila['Estadistico_H']:8.4f}  "
                        f"p = {Formatear_P_Valor(Fila['Valor_p'])} {Marca}")
            Agregar("")

    def Escribir_Dunn(Nombre_Seccion: str, Clave_Resultado: str):
        """
        Función auxiliar para escribir secciones de Dunn post-hoc.

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

                if Comparaciones:
                    Agregar(f"\n  {Variable}:")
                    for Comp in Comparaciones:
                        Agregar(f"    - {Comp['Categoria_1']:25s} vs "
                                f"{Comp['Categoria_2']:25s} "
                                f"p = {Formatear_P_Valor(Comp['Valor_p'])}")
                else:
                    Agregar(f"\n  {Variable}: "
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
    # SECCIÓN A: TESTS KRUSKAL-WALLIS.
    # =========================================================================
    Separador("#")
    Agregar("# SECCIÓN A: TESTS KRUSKAL-WALLIS")
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

    # =========================================================================
    # SECCIÓN B: TESTS DUNN POST-HOC.
    # =========================================================================
    Separador("#")
    Agregar("# SECCIÓN B: TESTS DUNN POST-HOC (Corrección Bonferroni)")
    Separador("#")
    Agregar("")

    # 8. Dunn CO Items.
    Escribir_Dunn(
        "8. DUNN POST-HOC: ÍTEMS CO",
        'Dunn_CO'
    )

    # 9. Dunn CT Items.
    Escribir_Dunn(
        "9. DUNN POST-HOC: ÍTEMS CT",
        'Dunn_CT'
    )

    # 10. Dunn Variables Agregadas.
    Escribir_Dunn(
        "10. DUNN POST-HOC: VARIABLES AGREGADAS",
        'Dunn_Agregadas'
    )

    # 11. Dunn Autopercepciones.
    Escribir_Dunn(
        "11. DUNN POST-HOC: AUTOPERCEPCIONES",
        'Dunn_Autopercepciones'
    )

    # 12. Dunn Índices.
    Escribir_Dunn(
        "12. DUNN POST-HOC: ÍNDICES",
        'Dunn_Indices'
    )

    # 13. Dunn Cercanías.
    Escribir_Dunn(
        "13. DUNN POST-HOC: CERCANÍAS A CANDIDATOS",
        'Dunn_Cercanias'
    )

    # =========================================================================
    # SECCIÓN C: TESTS PAREADOS.
    # =========================================================================
    Separador("#")
    Agregar("# SECCIÓN C: TESTS PAREADOS (WILCOXON Y MANN-WHITNEY)")
    Separador("#")
    Agregar("")

    # 14. Wilcoxon Congruencia.
    Separador("=")
    Agregar("14. WILCOXON PAREADO: CONGRUENCIA IDEOLÓGICA")
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
    Agregar("15. WILCOXON PAREADO: IZQ VS DER POR ÍTEM")
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
    Agregar("16. MANN-WHITNEY U: GENERALES VS BALLOTAGE")
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
                Agregar(f"\n  {Fila['Variable']}:")
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
                Agregar(f"  {Fila['Variable']:20s} "
                        f"U = {Fila['Estadistico_U']:.4f}, "
                        f"p = {Formatear_P_Valor(Fila['Valor_p'])}")
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
            ('Kruskal_Cercanias', 'Cercanías')
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
    Crear carpeta de tablas en Resultados/Tablas.

    Parametros:
    - Ruta_Base: Ruta base del proyecto.

    Retorna:
    - Ruta_Tablas: Ruta a la carpeta de tablas.

    Ejemplos:
    - Ruta_Tablas = Crear_Carpeta_Tablas(Ruta_Base)
    """

    Ruta_Tablas = os.path.join(Ruta_Base, 'Resultados', 'Tablas')
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
        os.path.join(Ruta_Tablas, 'Kruskal_Agregadas.xlsx'),
        'Agregadas'
    )
    Exportar_Tabla_Kruskal(
        Todos_Resultados.get('Kruskal_Autopercepciones', {}),
        os.path.join(Ruta_Tablas, 'Kruskal_Autopercepciones.xlsx'),
        'Autopercepciones'
    )
    Exportar_Tabla_Kruskal(
        Todos_Resultados.get('Kruskal_Indices', {}),
        os.path.join(Ruta_Tablas, 'Kruskal_Indices.xlsx'),
        'Indices'
    )
    Exportar_Tabla_Kruskal(
        Todos_Resultados.get('Kruskal_Cercanias', {}),
        os.path.join(Ruta_Tablas, 'Kruskal_Cercanias.xlsx'),
        'Cercanias'
    )

    Resultados_Diff = Todos_Resultados.get(
        'Diferencias_Entre_Poblaciones', {}
    )
    Exportar_Tabla_Kruskal(
        Resultados_Diff.get('Kruskal', {}),
        os.path.join(
            Ruta_Tablas,
            'Kruskal_Diferencias_Entre_Poblaciones.xlsx'
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
        os.path.join(Ruta_Tablas, 'Dunn_Agregadas.xlsx'),
        'Agregadas'
    )
    Exportar_Tabla_Dunn(
        Todos_Resultados.get('Dunn_Autopercepciones', {}),
        os.path.join(Ruta_Tablas, 'Dunn_Autopercepciones.xlsx'),
        'Autopercepciones'
    )
    Exportar_Tabla_Dunn(
        Todos_Resultados.get('Dunn_Indices', {}),
        os.path.join(Ruta_Tablas, 'Dunn_Indices.xlsx'),
        'Indices'
    )
    Exportar_Tabla_Dunn(
        Todos_Resultados.get('Dunn_Cercanias', {}),
        os.path.join(Ruta_Tablas, 'Dunn_Cercanias.xlsx'),
        'Cercanias'
    )
    Exportar_Tabla_Dunn(
        Resultados_Diff.get('Dunn', {}),
        os.path.join(
            Ruta_Tablas,
            'Dunn_Diferencias_Entre_Poblaciones.xlsx'
        ),
        'Diferencias_Entre_Poblaciones'
    )


if __name__ == '__main__':
    # Ruta a la carpeta con los datos definitivos.
    Ruta_Base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    Ruta_Datos = os.path.join(Ruta_Base, 'Datos definitivos')

    # Ejecutar todos los análisis.
    Resultados, dfs = Ejecutar_Todos_Los_Analisis(Ruta_Datos)

    # Crear carpeta Resultados si no existe.
    Ruta_Resultados = os.path.join(Ruta_Base, 'Resultados')
    if not os.path.exists(Ruta_Resultados):
        os.makedirs(Ruta_Resultados)

    # Generar reporte TXT.
    Ruta_Reporte = os.path.join(
        Ruta_Resultados,
        'Resultados_Analisis_Estadisticos.txt'
    )
    Generar_Reporte_TXT(Resultados, dfs, Ruta_Reporte)

    # Exportar tablas xlsx.
    Ruta_Tablas = Crear_Carpeta_Tablas(Ruta_Base)
    Exportar_Tablas_Estadisticos(Resultados, Ruta_Tablas)

    print("\nResultados guardados en el diccionario 'Resultados'.")
