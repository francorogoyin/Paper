# -*- coding: utf-8 -*-
"""
Análisis de correlaciones para el proyecto de tesis.

Este script genera matrices de correlación para las variables
numéricas del estudio, tanto a nivel general como segmentado
por categoría electoral (Categoria_PASO_2023).

"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTES Y CONFIGURACIÓN.
# =============================================================================

# Categorías electorales válidas.
Categorias_Validas = [
    'Left_Wing',
    'Progressivism',
    'Centre',
    'Moderate_Right_A',
    'Moderate_Right_B',
    'Right_Wing_Libertarian'
]

# Mapeo de nombres de variables español -> inglés.
Mapeo_Variables_Es_En = {
    # Autopercepciones.
    'Autopercepcion_Izq_Der': 'Left_Right_Self_Perception',
    'Autopercepcion_Con_Pro': 'Conservative_Progressive_Self_Perception',
    'Autopercepcion_Per_Antiper': 'Peronist_Anti_Peronist_Self_Perception',
    # Índices.
    'Indice_Progresismo': 'Progressivism_Index',
    'Indice_Conservadurismo': 'Conservatism_Index',
    'Indice_Positividad': 'Positivity_Index',
    'Indice_Progresismo_Tiempo': 'Progressivism_Index_Time',
    'Indice_Conservadurismo_Tiempo': 'Conservatism_Index_Time',
    # Cercanías.
    'Cercania_Bregman': 'Closeness_Bregman',
    'Cercania_Massa': 'Closeness_Massa',
    'Cercania_Schiaretti': 'Closeness_Schiaretti',
    'Cercania_Bullrich': 'Closeness_Bullrich',
    'Cercania_Milei': 'Closeness_Milei',
    # Influencias.
    'Influencia_Prensa': 'Written_Media_Influence',
    'Influencia_Redes': 'Social_Networks_Influence',
    # Género.
    'Genero_Masculino': 'Male',
    'Genero_Otro': 'Other_Gender',
    # Regiones.
    'Region_Norte': 'North',
    'Region_Patagonia': 'Patagonia',
    'Region_Centro': 'Central',
    'Region_CABA': 'CABA',
    'Region_Cuyo': 'Cuyo',
}


def Traducir_Variable(Nombre_Variable: str) -> str:

    """
    Traduce el nombre de una variable de español a inglés.

    Parámetros:
        Nombre_Variable (str): Nombre de la variable en español.

    Retorna:
        str: Nombre de la variable traducido o el original si no existe.

    """

    return Mapeo_Variables_Es_En.get(Nombre_Variable, Nombre_Variable)


# Variables a excluir del análisis (identificadores, listas, etc.).
Variables_Excluir = [
    'ID',
    'Fecha',
    'Timestamp',
    'index',
    'Orden_IP_Items',
    'Orden_IP_Items_Asociados'
]

# Prefijos de variables a excluir (variables de tipo lista).
Prefijos_Excluir = [
    'Orden_'
]

# Variables seleccionadas para correlaciones.
Columnas_Autopercepciones = [
    'Autopercepcion_Izq_Der',
    'Autopercepcion_Con_Pro',
    'Autopercepcion_Per_Antiper'
]

Columnas_Cercanias = [
    'Cercania_Massa',
    'Cercania_Bullrich',
    'Cercania_Bregman',
    'Cercania_Milei',
    'Cercania_Schiaretti'
]

Columnas_Indices = [
    'Indice_Progresismo',
    'Indice_Conservadurismo',
    'Indice_Positividad'
]

Columnas_CO_Agrupados = [
    'CO_Pro',
    'CO_Pro_Izq',
    'CO_Pro_Der',
    'CO_Con',
    'CO_Con_Izq',
    'CO_Con_Der',
    'CO_Congruente',
    'CO_Incongruente'
]


# =============================================================================
# FUNCIONES AUXILIARES.
# =============================================================================

def Formatear_Numero(Valor: float, Decimales: int = 4) -> str:

    """
    Formatea un número con la cantidad de decimales especificada.
    Usa notación científica solo para valores muy pequeños.

    Parámetros:
        Valor (float): El número a formatear.
        Decimales (int): Cantidad de decimales a mostrar.

    Retorna:
        str: El número formateado como string.

    """

    if pd.isna(Valor):
        return "NaN"

    if isinstance(Valor, (int, np.integer)):
        return str(Valor)

    if abs(Valor) < 0.001 and Valor != 0:
        return f"{Valor:.2E}"
    else:
        return f"{Valor:.{Decimales}f}"


def Formatear_P_Valor(P_Valor: float) -> str:

    """
    Formatea un p-valor con notación científica para valores
    muy pequeños.

    Parámetros:
        P_Valor (float): El p-valor a formatear.

    Retorna:
        str: El p-valor formateado.

    """

    if pd.isna(P_Valor):
        return "NaN"

    if P_Valor < 0.001:
        return f"{P_Valor:.2E}"
    else:
        return f"{P_Valor:.4f}"


def Es_Variable_Valida(Nombre_Variable: str, Serie: pd.Series) -> bool:

    """
    Verifica si una variable es válida para el análisis de
    correlaciones (numérica y no es lista).

    Parámetros:
        Nombre_Variable (str): Nombre de la variable.
        Serie (pd.Series): Serie con los datos.

    Retorna:
        bool: True si es válida, False en caso contrario.

    """

    # Excluir por nombre exacto.
    if Nombre_Variable in Variables_Excluir:
        return False

    # Excluir por prefijo.
    for Prefijo in Prefijos_Excluir:
        if Nombre_Variable.startswith(Prefijo):
            return False

    # Verificar que sea numérica.
    if not np.issubdtype(Serie.dtype, np.number):
        return False

    # Verificar que tenga variabilidad.
    Serie_Limpia = Serie.dropna()
    if len(Serie_Limpia) < 3:
        return False

    if Serie_Limpia.std() == 0:
        return False

    return True


def Obtener_Variables_Numericas(df: pd.DataFrame) -> list:

    """
    Obtiene la lista de variables numéricas válidas para
    el análisis de correlaciones.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.

    Retorna:
        list: Lista de nombres de variables numéricas válidas.

    """

    Variables_Validas = []

    for Columna in df.columns:
        if Es_Variable_Valida(Columna, df[Columna]):
            Variables_Validas.append(Columna)

    return Variables_Validas


def Obtener_Variables_Seleccionadas(df: pd.DataFrame) -> list:

    """
    Obtiene variables seleccionadas para correlaciones.

    ParamÇ­metros:
        df (pd.DataFrame): DataFrame con los datos.

    Retorna:
        list: Lista de variables seleccionadas.

    """

    Variables_Base = (
        Columnas_Autopercepciones
        + Columnas_Cercanias
        + Columnas_CO_Agrupados
        + Columnas_Indices
    )

    Variables_Validas = []
    Vistas = set()

    for Variable in Variables_Base:
        if Variable in Vistas:
            continue
        if Variable not in df.columns:
            continue
        if Es_Variable_Valida(Variable, df[Variable]):
            Variables_Validas.append(Variable)
            Vistas.add(Variable)

    return Variables_Validas


# =============================================================================
# FUNCIONES DE CÁLCULO DE CORRELACIONES.
# =============================================================================

def Calcular_Correlacion_Pearson(
    X: pd.Series,
    Y: pd.Series
) -> dict:

    """
    Calcula la correlación de Pearson entre dos variables.

    Parámetros:
        X (pd.Series): Primera variable.
        Y (pd.Series): Segunda variable.

    Retorna:
        dict: Diccionario con coeficiente r y p-valor.

    """

    # Eliminar valores faltantes pareados.
    Datos_Validos = pd.DataFrame({'X': X, 'Y': Y}).dropna()

    if len(Datos_Validos) < 3:
        return {'R': np.nan, 'P_Valor': np.nan, 'N': 0}

    R, P_Valor = stats.pearsonr(Datos_Validos['X'], Datos_Validos['Y'])

    return {
        'R': R,
        'P_Valor': P_Valor,
        'N': len(Datos_Validos)
    }


def Calcular_Correlacion_Spearman(
    X: pd.Series,
    Y: pd.Series
) -> dict:

    """
    Calcula la correlación de Spearman entre dos variables.

    Parámetros:
        X (pd.Series): Primera variable.
        Y (pd.Series): Segunda variable.

    Retorna:
        dict: Diccionario con coeficiente rho y p-valor.

    """

    # Eliminar valores faltantes pareados.
    Datos_Validos = pd.DataFrame({'X': X, 'Y': Y}).dropna()

    if len(Datos_Validos) < 3:
        return {'Rho': np.nan, 'P_Valor': np.nan, 'N': 0}

    Rho, P_Valor = stats.spearmanr(Datos_Validos['X'], Datos_Validos['Y'])

    return {
        'Rho': Rho,
        'P_Valor': P_Valor,
        'N': len(Datos_Validos)
    }


def Calcular_Matriz_Correlacion(
    df: pd.DataFrame,
    Variables: list,
    Metodo: str = 'pearson'
) -> dict:

    """
    Calcula la matriz de correlación para un conjunto de variables.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        Variables (list): Lista de variables a correlacionar.
        Metodo (str): 'pearson' o 'spearman'.

    Retorna:
        dict: Diccionario con matrices de coeficientes y p-valores.

    """

    N_Variables = len(Variables)

    # Inicializar matrices.
    Matriz_Coef = pd.DataFrame(
        np.zeros((N_Variables, N_Variables)),
        index=Variables,
        columns=Variables
    )

    Matriz_P = pd.DataFrame(
        np.zeros((N_Variables, N_Variables)),
        index=Variables,
        columns=Variables
    )

    Matriz_N = pd.DataFrame(
        np.zeros((N_Variables, N_Variables)),
        index=Variables,
        columns=Variables
    )

    # Calcular correlaciones.
    for i, Var_1 in enumerate(Variables):
        for j, Var_2 in enumerate(Variables):

            if i == j:
                # Diagonal: correlación perfecta.
                Matriz_Coef.iloc[i, j] = 1.0
                Matriz_P.iloc[i, j] = 0.0
                Matriz_N.iloc[i, j] = df[Var_1].dropna().shape[0]

            elif i < j:
                # Triangular superior: calcular.
                if Metodo == 'pearson':
                    Resultado = Calcular_Correlacion_Pearson(
                        df[Var_1], df[Var_2]
                    )
                    Coef = Resultado['R']
                else:
                    Resultado = Calcular_Correlacion_Spearman(
                        df[Var_1], df[Var_2]
                    )
                    Coef = Resultado['Rho']

                Matriz_Coef.iloc[i, j] = Coef
                Matriz_Coef.iloc[j, i] = Coef
                Matriz_P.iloc[i, j] = Resultado['P_Valor']
                Matriz_P.iloc[j, i] = Resultado['P_Valor']
                Matriz_N.iloc[i, j] = Resultado['N']
                Matriz_N.iloc[j, i] = Resultado['N']

    return {
        'Coeficientes': Matriz_Coef,
        'P_Valores': Matriz_P,
        'N': Matriz_N
    }


def Calcular_Correlaciones_Par_A_Par(
    df: pd.DataFrame,
    Variables: list,
    Metodo: str = 'pearson'
) -> pd.DataFrame:

    """
    Calcula las correlaciones par a par y las devuelve en formato
    de tabla larga (para facilitar el filtrado).

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        Variables (list): Lista de variables a correlacionar.
        Metodo (str): 'pearson' o 'spearman'.

    Retorna:
        pd.DataFrame: Tabla con Variable_1, Variable_2, Coef, P, N.

    """

    Resultados = []

    for i, Var_1 in enumerate(Variables):
        for j, Var_2 in enumerate(Variables):

            if i < j:
                if Metodo == 'pearson':
                    Resultado = Calcular_Correlacion_Pearson(
                        df[Var_1], df[Var_2]
                    )
                    Coef = Resultado['R']
                else:
                    Resultado = Calcular_Correlacion_Spearman(
                        df[Var_1], df[Var_2]
                    )
                    Coef = Resultado['Rho']

                Resultados.append({
                    'Variable_1': Var_1,
                    'Variable_2': Var_2,
                    'Coeficiente': Coef,
                    'P_Valor': Resultado['P_Valor'],
                    'N': Resultado['N'],
                    'Significativo': Resultado['P_Valor'] < 0.05
                })

    df_Resultados = pd.DataFrame(Resultados)

    # Ordenar por valor absoluto del coeficiente.
    if len(df_Resultados) > 0:
        df_Resultados['Abs_Coef'] = df_Resultados['Coeficiente'].abs()
        df_Resultados = df_Resultados.sort_values(
            'Abs_Coef', ascending=False
        )
        df_Resultados = df_Resultados.drop('Abs_Coef', axis=1)
        df_Resultados = df_Resultados.reset_index(drop=True)

    return df_Resultados


# =============================================================================
# FUNCIÓN DE CARGA DE DATOS.
# =============================================================================

def Cargar_Bases_Datos(Ruta_Carpeta: str) -> dict:

    """
    Carga las bases de datos de Generales y Ballotage desde
    archivos Excel.

    Parámetros:
        Ruta_Carpeta (str): Ruta a la carpeta con los archivos.

    Retorna:
        dict: Diccionario con DataFrames cargados.

    """

    dfs = {}

    Ruta_Generales = os.path.join(Ruta_Carpeta, 'Generales.xlsx')
    Ruta_Ballotage = os.path.join(Ruta_Carpeta, 'Ballotage.xlsx')

    if os.path.exists(Ruta_Generales):
        dfs['Generales'] = pd.read_excel(Ruta_Generales)
        print(f"Generales: {len(dfs['Generales'])} observaciones")

    if os.path.exists(Ruta_Ballotage):
        dfs['Ballotage'] = pd.read_excel(Ruta_Ballotage)
        print(f"Ballotage: {len(dfs['Ballotage'])} observaciones")

    return dfs


# =============================================================================
# FUNCIÓN DE EJECUCIÓN COMPLETA.
# =============================================================================

def Ejecutar_Analisis_Correlaciones(
    Ruta_Datos: str,
    Metodo: str = 'spearman'
) -> dict:

    """
    Ejecuta el análisis de correlaciones completo para todas las
    variables numéricas en ambos datasets.

    Parámetros:
        Ruta_Datos (str): Ruta a la carpeta con los datos.
        Metodo (str): 'pearson' o 'spearman'.

    Retorna:
        dict: Diccionario con todos los resultados del análisis.

    """

    print("=" * 70)
    print(f"ANÁLISIS DE CORRELACIONES ({Metodo.upper()})")
    print("=" * 70)

    # Cargar datos.
    print("\nCargando datos...")
    dfs = Cargar_Bases_Datos(Ruta_Datos)

    if not dfs:
        print("Error: No se pudieron cargar los datos.")
        return {}

    Todos_Resultados = {}

    for Nombre_df, df in dfs.items():
        print(f"\n{'='*60}")
        print(f"ANALIZANDO: {Nombre_df}")
        print('='*60)

        Resultados_df = {
            'General': {},
            'Por_Categoria': {}
        }

        # Obtener variables seleccionadas.
        Variables = Obtener_Variables_Seleccionadas(df)
        print(f"Variables seleccionadas v?lidas: {len(Variables)}")

        if len(Variables) < 2:
            print("  Advertencia: Variables insuficientes")
            Todos_Resultados[Nombre_df] = Resultados_df
            continue

        # Análisis general.
        print("\nCalculando correlaciones generales...")
        Resultados_df['General']['Matriz'] = Calcular_Matriz_Correlacion(
            df, Variables, Metodo
        )
        Resultados_df['General']['Par_A_Par'] = Calcular_Correlaciones_Par_A_Par(
            df, Variables, Metodo
        )
        print(f"  Pares calculados: "
              f"{len(Resultados_df['General']['Par_A_Par'])}")

        # Análisis por categoría.
        if 'Categoria_PASO_2023' in df.columns:
            print("\nCalculando correlaciones por categoría electoral...")

            for Categoria in Categorias_Validas:
                df_Categoria = df[df['Categoria_PASO_2023'] == Categoria]

                if len(df_Categoria) < 10:
                    print(f"  {Categoria}: Insuficientes datos")
                    continue

                # Obtener variables válidas para esta categoría.
                Variables_Cat = [
                    v for v in Variables
                    if Es_Variable_Valida(v, df_Categoria[v])
                ]

                if len(Variables_Cat) < 2:
                    continue

                Resultados_df['Por_Categoria'][Categoria] = {
                    'Matriz': Calcular_Matriz_Correlacion(
                        df_Categoria, Variables_Cat, Metodo
                    ),
                    'Par_A_Par': Calcular_Correlaciones_Par_A_Par(
                        df_Categoria, Variables_Cat, Metodo
                    )
                }

                print(f"  {Categoria}: {len(df_Categoria)} obs, "
                      f"{len(Variables_Cat)} variables")

        Todos_Resultados[Nombre_df] = Resultados_df

    print("\n" + "="*70)
    print("ANÁLISIS DE CORRELACIONES COMPLETADO")
    print("="*70)

    return Todos_Resultados, dfs


# =============================================================================
# GENERACIÓN DE REPORTE TXT.
# =============================================================================

# Orden temático de las variables para el reporte.
# El orden es: Índices → Autopercepciones → Cercanías → CO_Agrupados.
Orden_Tematico_Variables = (
    Columnas_Indices
    + Columnas_Autopercepciones
    + Columnas_Cercanias
    + Columnas_CO_Agrupados
)


def Obtener_Indice_Tematico(Variable: str) -> int:

    """
    Obtiene el índice temático de una variable para ordenamiento.

    Parámetros:
        Variable (str): Nombre de la variable.

    Retorna:
        int: Índice en el orden temático (999 si no se encuentra).

    """

    if Variable in Orden_Tematico_Variables:
        return Orden_Tematico_Variables.index(Variable)
    return 999


def Agrupar_Correlaciones_Por_Variable(
    df_Correlaciones: pd.DataFrame,
    Solo_Significativas: bool = True
) -> dict:

    """
    Agrupa las correlaciones por variable principal, ordenando las
    variables correlacionadas según el orden temático establecido.

    Parámetros:
        df_Correlaciones (pd.DataFrame): DataFrame con correlaciones.
        Solo_Significativas (bool): Si True, solo incluye significativas.

    Retorna:
        dict: Diccionario con variable como clave y lista de
              correlaciones ordenadas temáticamente como valor.

    """

    if Solo_Significativas:
        df_Filtrado = df_Correlaciones[
            df_Correlaciones['Significativo'] == True
        ].copy()
    else:
        df_Filtrado = df_Correlaciones.copy()

    if len(df_Filtrado) == 0:
        return {}

    # Crear diccionario para agrupar por variable.
    Correlaciones_Por_Variable = {}

    for Variable in Orden_Tematico_Variables:
        # Buscar correlaciones donde esta variable aparece.
        Correlaciones_Variable = []

        for _, Fila in df_Filtrado.iterrows():
            if Fila['Variable_1'] == Variable:
                Correlaciones_Variable.append({
                    'Variable': Fila['Variable_2'],
                    'Coeficiente': Fila['Coeficiente'],
                    'P_Valor': Fila['P_Valor']
                })
            elif Fila['Variable_2'] == Variable:
                Correlaciones_Variable.append({
                    'Variable': Fila['Variable_1'],
                    'Coeficiente': Fila['Coeficiente'],
                    'P_Valor': Fila['P_Valor']
                })

        if len(Correlaciones_Variable) > 0:
            # Ordenar por orden temático de la variable correlacionada.
            Correlaciones_Variable.sort(
                key=lambda x: Obtener_Indice_Tematico(x['Variable'])
            )
            Correlaciones_Por_Variable[Variable] = Correlaciones_Variable

    return Correlaciones_Por_Variable


def Generar_Reporte_Correlaciones_TXT(
    Todos_Resultados: dict,
    Ruta_Salida: str,
    Metodo: str = 'spearman',
    Top_N: int = 50
) -> None:

    """
    Genera un archivo TXT con el reporte de correlaciones,
    agrupadas por variable y ordenadas temáticamente.

    Parámetros:
        Todos_Resultados (dict): Diccionario con los resultados.
        Ruta_Salida (str): Ruta al archivo de salida.
        Metodo (str): Método de correlación usado.
        Top_N (int): Cantidad de correlaciones top a mostrar.

    """

    Lineas = []

    def Agregar(Texto: str = ""):
        Lineas.append(Texto)

    def Separador(Caracter: str = "=", Largo: int = 80):
        Lineas.append(Caracter * Largo)

    # Nombre del coeficiente.
    Nombre_Coef = "r" if Metodo == 'pearson' else "ρ"

    # =========================================================================
    # ENCABEZADO.
    # =========================================================================
    Separador("=")
    Agregar(f"ANÁLISIS DE CORRELACIONES ({Metodo.upper()}) - RESULTADOS")
    Agregar("Proyecto de Tesis - Experimento Electoral Argentina 2023")
    Separador("=")
    Agregar("")

    for Nombre_df, Resultados_df in Todos_Resultados.items():
        Separador("#")
        Agregar(f"# DATASET: {Nombre_df.upper()}")
        Separador("#")
        Agregar("")

        # =================================================================
        # SECCIÓN 1: CORRELACIONES SIGNIFICATIVAS POR VARIABLE.
        # =================================================================
        Separador("=")
        Agregar("SECCIÓN 1: CORRELACIONES SIGNIFICATIVAS POR VARIABLE")
        Agregar("(Ordenadas temáticamente: Índices → Autopercepciones "
                "→ Cercanías → CO)")
        Separador("=")
        Agregar("")

        Resultados_General = Resultados_df.get('General', {})
        Par_A_Par = Resultados_General.get('Par_A_Par', pd.DataFrame())

        if len(Par_A_Par) > 0:
            # Filtrar significativas.
            Par_A_Par_Sig = Par_A_Par[Par_A_Par['Significativo'] == True]

            Agregar(f"Total de pares: {len(Par_A_Par)}")
            Agregar(f"Pares significativos (p < 0.05): {len(Par_A_Par_Sig)}")
            Agregar("")

            # Agrupar por variable.
            Correlaciones_Agrupadas = Agrupar_Correlaciones_Por_Variable(
                Par_A_Par,
                Solo_Significativas=True
            )

            for Variable, Correlaciones in Correlaciones_Agrupadas.items():
                Var_Traducida = Traducir_Variable(Variable)
                Agregar(f"► {Var_Traducida}")
                Agregar("-" * 70)

                for Corr in Correlaciones:
                    Sig_Marca = "***" if Corr['P_Valor'] < 0.001 else (
                        "**" if Corr['P_Valor'] < 0.01 else "*"
                    )
                    Corr_Var_Traducida = Traducir_Variable(Corr['Variable'])
                    Agregar(
                        f"  {Corr_Var_Traducida:<40} "
                        f"{Nombre_Coef}={Corr['Coeficiente']:>7.4f}  "
                        f"p={Formatear_P_Valor(Corr['P_Valor']):>12}  "
                        f"{Sig_Marca}"
                    )

                Agregar("")

            Agregar("Significancia: *** p < 0.001, ** p < 0.01, * p < 0.05")
            Agregar("")

        # =================================================================
        # SECCIÓN 2: CORRELACIONES POR CATEGORÍA ELECTORAL.
        # =================================================================
        Separador("=")
        Agregar("SECCIÓN 2: CORRELACIONES POR CATEGORÍA ELECTORAL")
        Separador("=")
        Agregar("")

        Resultados_Por_Cat = Resultados_df.get('Por_Categoria', {})

        for Categoria in Categorias_Validas:
            if Categoria not in Resultados_Por_Cat:
                continue

            Res_Cat = Resultados_Por_Cat[Categoria]
            Par_A_Par_Cat = Res_Cat.get('Par_A_Par', pd.DataFrame())

            if len(Par_A_Par_Cat) == 0:
                continue

            Separador("-")
            Agregar(f"► {Categoria}")
            Separador("-")
            Agregar("")

            # Filtrar significativas.
            Par_A_Par_Sig = Par_A_Par_Cat[
                Par_A_Par_Cat['Significativo'] == True
            ]

            Agregar(f"Total de pares: {len(Par_A_Par_Cat)}")
            Agregar(f"Pares significativos (p < 0.05): {len(Par_A_Par_Sig)}")
            Agregar("")

            # Agrupar por variable.
            Correlaciones_Agrupadas = Agrupar_Correlaciones_Por_Variable(
                Par_A_Par_Cat,
                Solo_Significativas=True
            )

            for Variable, Correlaciones in Correlaciones_Agrupadas.items():
                Var_Traducida = Traducir_Variable(Variable)
                Agregar(f"  ▸ {Var_Traducida}")

                for Corr in Correlaciones:
                    Sig_Marca = "***" if Corr['P_Valor'] < 0.001 else (
                        "**" if Corr['P_Valor'] < 0.01 else "*"
                    )
                    Corr_Var_Traducida = Traducir_Variable(Corr['Variable'])
                    Agregar(
                        f"      {Corr_Var_Traducida:<36} "
                        f"{Nombre_Coef}={Corr['Coeficiente']:>7.4f}  "
                        f"p={Formatear_P_Valor(Corr['P_Valor']):>12}  "
                        f"{Sig_Marca}"
                    )

                Agregar("")

        Agregar("")

    Separador("=")
    Agregar("FIN DEL REPORTE")
    Separador("=")

    # Escribir archivo.
    with open(Ruta_Salida, 'w', encoding='utf-8') as Archivo:
        Archivo.write('\n'.join(Lineas))

    print(f"\nReporte guardado en: {Ruta_Salida}")


# =============================================================================
# EJECUCIÓN PRINCIPAL.
# =============================================================================

if __name__ == '__main__':
    # Ruta a la carpeta con los datos definitivos.
    Ruta_Base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    Ruta_Datos = os.path.join(Ruta_Base, 'Processed_Data')

    # Ejecutar análisis de correlaciones (Spearman por defecto).
    Resultados, dfs = Ejecutar_Analisis_Correlaciones(
        Ruta_Datos,
        Metodo='spearman'
    )

    # Crear carpeta Results/Reports si no existe.
    Ruta_Reportes = os.path.join(Ruta_Base, 'Results', 'Reports')
    if not os.path.exists(Ruta_Reportes):
        os.makedirs(Ruta_Reportes)

    # Generar reporte TXT.
    Ruta_Reporte = os.path.join(
        Ruta_Reportes,
        'Report_Correlation_Analysis.txt'
    )
    Generar_Reporte_Correlaciones_TXT(
        Resultados,
        Ruta_Reporte,
        Metodo='spearman'
    )

    print("\nAnálisis completado.")
