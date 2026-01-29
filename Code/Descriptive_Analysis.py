# -*- coding: utf-8 -*-
"""
Descriptive Analysis.

This script generates complete descriptive statistics for all
variables in the study, both at a general level and segmented
by electoral category (Categoria_PASO_2023).

"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTS AND CONFIGURATION.
# =============================================================================

# Valid electoral categories.
Categorias_Validas = [
    'Left_Wing',
    'Progressivism',
    'Centre',
    'Moderate_Right_A',
    'Moderate_Right_B',
    'Right_Wing_Libertarian'
]

# Variables to exclude from analysis (identifiers, etc.).
Variables_Excluir = [
    'ID',
    'Fecha',
    'Timestamp',
    'index'
]

Candidatos_Cdc = ['Massa', 'Bullrich', 'Schiaretti', 'Milei', 'Bregman']
Columnas_Cdc_Izq_Der = [f'CDC_{Candidato}_0' for Candidato in Candidatos_Cdc]
Columnas_Cdc_Con_Pro = [f'CDC_{Candidato}_1' for Candidato in Candidatos_Cdc]
Columnas_Cdc_Todas = Columnas_Cdc_Izq_Der + Columnas_Cdc_Con_Pro


# =============================================================================
# AUXILIARY FUNCTIONS.
# =============================================================================

def Formatear_Numero(Valor: float, Decimales: int = 4) -> str:

    """
    Formats a number with the specified number of decimals.
    Uses scientific notation only for very small values.

    Parameters:
        Valor (float): The number to format.
        Decimales (int): Number of decimals to display.

    Returns:
        str: The formatted number as a string.

    """

    if pd.isna(Valor):
        return "NaN"

    if isinstance(Valor, (int, np.integer)):
        return str(Valor)

    if abs(Valor) < 0.001 and Valor != 0:
        return f"{Valor:.2E}"
    else:
        return f"{Valor:.{Decimales}f}"


def Determinar_Tipo_Variable(Serie: pd.Series) -> str:

    """
    Determines the type of a variable to apply the appropriate
    descriptive analysis.

    Parameters:
        Serie (pd.Series): Series with the variable data.

    Returns:
        str: Variable type ('continua', 'discreta', 'categorica',
            'binaria', 'booleana', 'lista').

    """

    # Remove null values for analysis.
    Serie_Limpia = Serie.dropna()

    if len(Serie_Limpia) == 0:
        return 'vacia'

    # Check if it contains lists or complex structures.
    if len(Serie_Limpia) > 0:
        Primer_Valor = Serie_Limpia.iloc[0]
        if isinstance(Primer_Valor, (list, dict, tuple)):
            return 'lista'

    # Booleans.
    if Serie.dtype == bool:
        return 'booleana'

    # Categorical (object/string).
    if Serie.dtype == 'object' or Serie.dtype.name == 'category':
        # Check if they are strings that look like lists.
        if len(Serie_Limpia) > 0:
            Primer_Valor = Serie_Limpia.iloc[0]
            if isinstance(Primer_Valor, str):
                if Primer_Valor.startswith('[') and Primer_Valor.endswith(']'):
                    return 'lista'
        return 'categorica'

    # Numeric.
    Valores_Unicos = Serie_Limpia.nunique()

    # Binary (only 2 values).
    if Valores_Unicos == 2:
        return 'binaria'

    # Discrete (few unique values or all integers).
    if Valores_Unicos <= 10:
        return 'discreta'

    # Check if all are integers.
    if Serie_Limpia.dtype in ['int64', 'int32']:
        if Valores_Unicos <= 20:
            return 'discreta'

    # Continuous.
    return 'continua'


def Traducir_Variable_Cdc(Nombre_Variable: str) -> str:

    """
    Converts a CDC variable name to a legible label.

    Parameters:
        Nombre_Variable (str): Name like CDC_Milei_0.

    Returns:
        str: Human-readable label (e.g., 'Milei (Izq-Der)').
    """

    if not Nombre_Variable.startswith('CDC_'):
        return Nombre_Variable

    Partes = Nombre_Variable.split('_')
    if len(Partes) != 3:
        return Nombre_Variable

    Candidato = Partes[1]
    Dimension = 'Izq-Der' if Partes[2] == '0' else \
                'Con-Pro' if Partes[2] == '1' else Partes[2]
    return f"{Candidato} ({Dimension})"


# =============================================================================
# DESCRIPTIVE STATISTICS FUNCTIONS.
# =============================================================================

def Calcular_Estadisticos_Continuos(Serie: pd.Series) -> dict:

    """
    Calculates descriptive statistics for continuous variables.

    Parameters:
        Serie (pd.Series): Series with numeric data.

    Returns:
        dict: Dictionary with all calculated statistics.

    """

    Serie_Limpia = Serie.dropna()
    N_Total = len(Serie)
    N_Validos = len(Serie_Limpia)
    N_Faltantes = N_Total - N_Validos

    if N_Validos == 0:
        return {
            'N_Total': N_Total,
            'N_Validos': 0,
            'N_Faltantes': N_Faltantes,
            'Porcentaje_Faltantes': 100.0
        }

    # Central tendency statistics.
    Media = Serie_Limpia.mean()
    Mediana = Serie_Limpia.median()
    Moda_Resultado = Serie_Limpia.mode()
    Moda = Moda_Resultado.iloc[0] if len(Moda_Resultado) > 0 else np.nan

    # Dispersion statistics.
    Desviacion_Estandar = Serie_Limpia.std()
    Varianza = Serie_Limpia.var()
    Minimo = Serie_Limpia.min()
    Maximo = Serie_Limpia.max()
    Rango = Maximo - Minimo

    # Distribution shape.
    Asimetria = Serie_Limpia.skew()
    Curtosis = Serie_Limpia.kurtosis()

    return {
        'N_Total': N_Total,
        'N_Validos': N_Validos,
        'N_Faltantes': N_Faltantes,
        'Porcentaje_Faltantes': (N_Faltantes / N_Total * 100),
        'Media': Media,
        'Mediana': Mediana,
        'Moda': Moda,
        'Desviacion_Estandar': Desviacion_Estandar,
        'Minimo': Minimo,
        'Maximo': Maximo,
        'Asimetria': Asimetria,
        'Curtosis': Curtosis
    }


def Calcular_Estadisticos_Discretos(Serie: pd.Series) -> dict:

    """
    Calculates descriptive statistics for discrete variables,
    including unique value counts.

    Parameters:
        Serie (pd.Series): Series with discrete data.

    Returns:
        dict: Dictionary with statistics and frequencies.

    """

    # First calculate statistics as if continuous.
    Estadisticos = Calcular_Estadisticos_Continuos(Serie)

    # Add unique value count.
    Serie_Limpia = Serie.dropna()
    Valores_Unicos = Serie_Limpia.nunique()

    # Frequencies.
    Frecuencias = Serie_Limpia.value_counts().sort_index()
    Frecuencias_Relativas = Serie_Limpia.value_counts(
        normalize=True
    ).sort_index() * 100

    # Build frequency dictionary.
    Conteo_Valores = {}
    for Valor in Frecuencias.index:
        Conteo_Valores[Valor] = {
            'Frecuencia': int(Frecuencias[Valor]),
            'Porcentaje': Frecuencias_Relativas[Valor]
        }

    Estadisticos['Valores_Unicos'] = Valores_Unicos
    Estadisticos['Conteo_Valores'] = Conteo_Valores

    return Estadisticos


def Calcular_Estadisticos_Categoricos(Serie: pd.Series) -> dict:

    """
    Calculates descriptive statistics for categorical variables.

    Parameters:
        Serie (pd.Series): Series with categorical data.

    Returns:
        dict: Dictionary with frequencies and percentages.

    """

    Serie_Limpia = Serie.dropna()
    N_Total = len(Serie)
    N_Validos = len(Serie_Limpia)
    N_Faltantes = N_Total - N_Validos

    if N_Validos == 0:
        return {
            'N_Total': N_Total,
            'N_Validos': 0,
            'N_Faltantes': N_Faltantes,
            'Porcentaje_Faltantes': 100.0
        }

    # Unique values.
    Valores_Unicos = Serie_Limpia.nunique()

    # Mode.
    Moda_Resultado = Serie_Limpia.mode()
    Moda = Moda_Resultado.iloc[0] if len(Moda_Resultado) > 0 else None

    # Frequencies.
    Frecuencias = Serie_Limpia.value_counts()
    Frecuencias_Relativas = Serie_Limpia.value_counts(normalize=True) * 100

    # Build frequency dictionary.
    Conteo_Valores = {}
    for Valor in Frecuencias.index:
        Conteo_Valores[str(Valor)] = {
            'Frecuencia': int(Frecuencias[Valor]),
            'Porcentaje': Frecuencias_Relativas[Valor]
        }

    return {
        'N_Total': N_Total,
        'N_Validos': N_Validos,
        'N_Faltantes': N_Faltantes,
        'Porcentaje_Faltantes': (N_Faltantes / N_Total * 100),
        'Valores_Unicos': Valores_Unicos,
        'Moda': Moda,
        'Conteo_Valores': Conteo_Valores
    }


def Calcular_Estadisticos_Binarios(Serie: pd.Series) -> dict:

    """
    Calculates descriptive statistics for binary variables
    (including booleans).

    Parameters:
        Serie (pd.Series): Series with binary data.

    Returns:
        dict: Dictionary with frequencies and proportions.

    """

    Serie_Limpia = Serie.dropna()
    N_Total = len(Serie)
    N_Validos = len(Serie_Limpia)
    N_Faltantes = N_Total - N_Validos

    if N_Validos == 0:
        return {
            'N_Total': N_Total,
            'N_Validos': 0,
            'N_Faltantes': N_Faltantes,
            'Porcentaje_Faltantes': 100.0
        }

    # Frequencies.
    Frecuencias = Serie_Limpia.value_counts()
    Frecuencias_Relativas = Serie_Limpia.value_counts(normalize=True) * 100

    # Identify values.
    Valores = sorted(Serie_Limpia.unique())

    # Proportion of "positive" value (True, 1, or the highest).
    if Serie.dtype == bool:
        Proporcion_Positivo = Serie_Limpia.mean() * 100
    else:
        Proporcion_Positivo = (
            Serie_Limpia == max(Valores)
        ).mean() * 100

    # Build frequency dictionary.
    Conteo_Valores = {}
    for Valor in Frecuencias.index:
        Conteo_Valores[str(Valor)] = {
            'Frecuencia': int(Frecuencias[Valor]),
            'Porcentaje': Frecuencias_Relativas[Valor]
        }

    return {
        'N_Total': N_Total,
        'N_Validos': N_Validos,
        'N_Faltantes': N_Faltantes,
        'Porcentaje_Faltantes': (N_Faltantes / N_Total * 100),
        'Proporcion_Positivo': Proporcion_Positivo,
        'Conteo_Valores': Conteo_Valores
    }


# =============================================================================
# MAIN ANALYSIS FUNCTION.
# =============================================================================

def Analizar_Variable(
    df: pd.DataFrame,
    Nombre_Variable: str
) -> dict:

    """
    Analyzes a variable and returns its descriptive statistics
    according to its type.

    Parameters:
        df (pd.DataFrame): DataFrame with the data.
        Nombre_Variable (str): Name of the variable to analyze.

    Returns:
        dict: Dictionary with variable type and statistics.

    """

    if Nombre_Variable not in df.columns:
        return None

    Serie = df[Nombre_Variable]
    Tipo = Determinar_Tipo_Variable(Serie)

    if Tipo == 'vacia':
        return {
            'Tipo': 'vacia',
            'Estadisticos': {'N_Total': len(Serie), 'N_Validos': 0}
        }

    # Ignore variables that contain lists.
    elif Tipo == 'lista':
        return None

    elif Tipo == 'continua':
        return {
            'Tipo': 'continua',
            'Estadisticos': Calcular_Estadisticos_Continuos(Serie)
        }

    elif Tipo == 'discreta':
        return {
            'Tipo': 'discreta',
            'Estadisticos': Calcular_Estadisticos_Discretos(Serie)
        }

    elif Tipo == 'categorica':
        return {
            'Tipo': 'categorica',
            'Estadisticos': Calcular_Estadisticos_Categoricos(Serie)
        }

    elif Tipo in ['binaria', 'booleana']:
        return {
            'Tipo': Tipo,
            'Estadisticos': Calcular_Estadisticos_Binarios(Serie)
        }

    return None


def Analizar_Variable_Por_Categoria(
    df: pd.DataFrame,
    Nombre_Variable: str,
    Columna_Categoria: str = 'Categoria_PASO_2023'
) -> dict:

    """
    Analyzes a variable segmented by electoral category.

    Parameters:
        df (pd.DataFrame): DataFrame with the data.
        Nombre_Variable (str): Name of the variable to analyze.
        Columna_Categoria (str): Name of the category column.

    Returns:
        dict: Dictionary with statistics for each category.

    """

    if Nombre_Variable not in df.columns:
        return None

    if Columna_Categoria not in df.columns:
        return None

    Resultados_Por_Categoria = {}

    for Categoria in Categorias_Validas:
        df_Categoria = df[df[Columna_Categoria] == Categoria]

        if len(df_Categoria) == 0:
            continue

        Resultado = Analizar_Variable(df_Categoria, Nombre_Variable)

        if Resultado is not None:
            Resultados_Por_Categoria[Categoria] = Resultado

    return Resultados_Por_Categoria


# =============================================================================
# DATA LOADING FUNCTION.
# =============================================================================

def Cargar_Bases_Datos(Ruta_Carpeta: str) -> dict:

    """
    Loads the Generales and Ballotage databases from
    Excel files.

    Parameters:
        Ruta_Carpeta (str): Path to the folder with the files.

    Returns:
        dict: Dictionary with loaded DataFrames.

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
# COMPLETE EXECUTION FUNCTION.
# =============================================================================

def Ejecutar_Analisis_Descriptivo(Ruta_Datos: str) -> dict:

    """
    Executes the complete descriptive analysis for all
    variables in both datasets.

    Parameters:
        Ruta_Datos (str): Path to the folder with the data.

    Returns:
        dict: Dictionary with all analysis results.

    """

    print("=" * 70)
    print("ANÁLISIS DESCRIPTIVO COMPLETO")
    print("=" * 70)

    # Load data.
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

        # Get list of variables to analyze.
        Variables = [
            col for col in df.columns
            if col not in Variables_Excluir
        ]

        print(f"Variables a analizar: {len(Variables)}")

        # General analysis of each variable.
        print("\nAnálisis general...")
        for i, Variable in enumerate(Variables):
            Resultado = Analizar_Variable(df, Variable)
            if Resultado is not None:
                Resultados_df['General'][Variable] = Resultado

            if (i + 1) % 50 == 0:
                print(f"  Procesadas: {i + 1}/{len(Variables)}")

        print(f"  Completado: {len(Variables)} variables")

        # Analysis by category.
        if 'Categoria_PASO_2023' in df.columns:
            print("\nAnálisis por categoría electoral...")

            for i, Variable in enumerate(Variables):
                if Variable == 'Categoria_PASO_2023':
                    continue

                Resultado = Analizar_Variable_Por_Categoria(
                    df, Variable
                )

                if Resultado is not None:
                    Resultados_df['Por_Categoria'][Variable] = Resultado

                if (i + 1) % 50 == 0:
                    print(f"  Procesadas: {i + 1}/{len(Variables)}")

            print(f"  Completado: {len(Variables) - 1} variables")

        Todos_Resultados[Nombre_df] = Resultados_df

    print("\n" + "="*70)
    print("ANÁLISIS DESCRIPTIVO COMPLETADO")
    print("="*70)

    return Todos_Resultados, dfs


# =============================================================================
# TXT REPORT GENERATION.
# =============================================================================

def Generar_Reporte_Descriptivo_TXT(
    Todos_Resultados: dict,
    Ruta_Salida: str
) -> None:

    """
    Generates a TXT file with the complete descriptive analysis
    report.

    Parameters:
        Todos_Resultados (dict): Dictionary with the results.
        Ruta_Salida (str): Path to the output file.

    """

    Lineas = []

    def Agregar(Texto: str = ""):
        Lineas.append(Texto)

    def Separador(Caracter: str = "=", Largo: int = 80):
        Lineas.append(Caracter * Largo)

    # =========================================================================
    # HEADER.
    # =========================================================================
    Separador("=")
    Agregar("ANÁLISIS DESCRIPTIVO - RESULTADOS COMPLETOS")
    Agregar("Proyecto de Tesis - Experimento Electoral Argentina 2023")
    Separador("=")
    Agregar("")

    for Nombre_df, Resultados_df in Todos_Resultados.items():
        Separador("#")
        Agregar(f"# DATASET: {Nombre_df.upper()}")
        Separador("#")
        Agregar("")

        # =================================================================
        # SECTION 1: GENERAL ANALYSIS.
        # =================================================================
        Separador("=")
        Agregar("SECCIÓN 1: ESTADÍSTICOS GENERALES POR VARIABLE")
        Separador("=")
        Agregar("")

        Resultados_General = Resultados_df.get('General', {})

        # Group by variable type.
        Variables_Por_Tipo = {
            'continua': [],
            'discreta': [],
            'categorica': [],
            'binaria': [],
            'booleana': []
        }

        for Variable, Resultado in Resultados_General.items():
            Tipo = Resultado.get('Tipo', 'otro')
            if Tipo in Variables_Por_Tipo:
                Variables_Por_Tipo[Tipo].append((Variable, Resultado))

        # Continuous variables.
        if Variables_Por_Tipo['continua']:
            Separador("-")
            Agregar("VARIABLES CONTINUAS")
            Separador("-")
            Agregar("")

            for Variable, Resultado in Variables_Por_Tipo['continua']:
                Est = Resultado['Estadisticos']
                Agregar(f"► {Variable}")
                Agregar(f"  N válidos: {Est['N_Validos']} "
                        f"(faltantes: {Est['N_Faltantes']}, "
                        f"{Est['Porcentaje_Faltantes']:.1f}%)")
                Agregar(f"  Media: {Formatear_Numero(Est['Media'])}  |  "
                        f"Mediana: {Formatear_Numero(Est['Mediana'])}  |  "
                        f"Moda: {Formatear_Numero(Est['Moda'])}")
                Agregar(f"  DE: {Formatear_Numero(Est['Desviacion_Estandar'])}")
                Agregar(f"  Mín: {Formatear_Numero(Est['Minimo'])}  |  "
                        f"Máx: {Formatear_Numero(Est['Maximo'])}")
                Agregar(f"  Asimetría: {Formatear_Numero(Est['Asimetria'])}  |  "
                        f"Curtosis: {Formatear_Numero(Est['Curtosis'])}")
                Agregar("")

        # Discrete variables.
        if Variables_Por_Tipo['discreta']:
            Separador("-")
            Agregar("VARIABLES DISCRETAS")
            Separador("-")
            Agregar("")

            for Variable, Resultado in Variables_Por_Tipo['discreta']:
                Est = Resultado['Estadisticos']
                Agregar(f"► {Variable}")
                Agregar(f"  N válidos: {Est['N_Validos']} "
                        f"(faltantes: {Est['N_Faltantes']}, "
                        f"{Est['Porcentaje_Faltantes']:.1f}%)")
                Agregar(f"  Media: {Formatear_Numero(Est['Media'])}  |  "
                        f"Mediana: {Formatear_Numero(Est['Mediana'])}  |  "
                        f"DE: {Formatear_Numero(Est['Desviacion_Estandar'])}")
                Agregar(f"  Mín: {Formatear_Numero(Est['Minimo'])}  |  "
                        f"Máx: {Formatear_Numero(Est['Maximo'])}  |  "
                        f"Valores únicos: {Est['Valores_Unicos']}")
                Agregar(f"  Distribución de valores:")

                Conteo = Est.get('Conteo_Valores', {})
                for Valor, Info in sorted(Conteo.items()):
                    Agregar(f"    {Valor}: {Info['Frecuencia']} "
                            f"({Info['Porcentaje']:.1f}%)")

                Agregar("")

        # Categorical variables.
        if Variables_Por_Tipo['categorica']:
            Separador("-")
            Agregar("VARIABLES CATEGÓRICAS")
            Separador("-")
            Agregar("")

            for Variable, Resultado in Variables_Por_Tipo['categorica']:
                Est = Resultado['Estadisticos']
                Agregar(f"► {Variable}")
                Agregar(f"  N válidos: {Est['N_Validos']} "
                        f"(faltantes: {Est['N_Faltantes']}, "
                        f"{Est['Porcentaje_Faltantes']:.1f}%)")
                Agregar(f"  Valores únicos: {Est['Valores_Unicos']}  |  "
                        f"Moda: {Est['Moda']}")
                Agregar(f"  Distribución de valores:")

                Conteo = Est.get('Conteo_Valores', {})
                # Sort by descending frequency.
                Conteo_Ordenado = sorted(
                    Conteo.items(),
                    key=lambda x: x[1]['Frecuencia'],
                    reverse=True
                )
                for Valor, Info in Conteo_Ordenado:
                    Agregar(f"    {Valor}: {Info['Frecuencia']} "
                            f"({Info['Porcentaje']:.1f}%)")

                Agregar("")

        # Binary and boolean variables.
        Binarias_Y_Bool = (
            Variables_Por_Tipo['binaria'] +
            Variables_Por_Tipo['booleana']
        )

        if Binarias_Y_Bool:
            Separador("-")
            Agregar("VARIABLES BINARIAS/BOOLEANAS")
            Separador("-")
            Agregar("")

            for Variable, Resultado in Binarias_Y_Bool:
                Est = Resultado['Estadisticos']
                Agregar(f"► {Variable} ({Resultado['Tipo']})")
                Agregar(f"  N válidos: {Est['N_Validos']} "
                        f"(faltantes: {Est['N_Faltantes']}, "
                        f"{Est['Porcentaje_Faltantes']:.1f}%)")

                Conteo = Est.get('Conteo_Valores', {})
                for Valor, Info in sorted(Conteo.items()):
                    Agregar(f"    {Valor}: {Info['Frecuencia']} "
                            f"({Info['Porcentaje']:.1f}%)")

                Agregar("")

        # =================================================================
        # SECTION 2: ANALYSIS BY CATEGORY.
        # =================================================================
        Separador("=")
        Agregar("SECCIÓN 2: ESTADÍSTICOS POR CATEGORÍA ELECTORAL")
        Separador("=")
        Agregar("")

        Resultados_Por_Cat = Resultados_df.get('Por_Categoria', {})

        # Only show numeric variables by category.
        for Variable, Resultados_Cats in Resultados_Por_Cat.items():

            # Verify that there are results.
            if not Resultados_Cats:
                continue

            # Verify if it is numeric.
            Primer_Resultado = list(Resultados_Cats.values())[0]
            Tipo = Primer_Resultado.get('Tipo', '')

            if Tipo not in ['continua', 'discreta']:
                continue

            Separador("-")
            Agregar(f"► {Variable}")
            Separador("-")
            Agregar("")

            # Table header.
            Agregar(f"{'Categoría':<25} {'N':>6} {'Media':>10} "
                    f"{'Mediana':>10} {'DE':>10} {'Mín':>8} {'Máx':>8}")
            Agregar("-" * 80)

            for Categoria in Categorias_Validas:
                if Categoria not in Resultados_Cats:
                    continue

                Res = Resultados_Cats[Categoria]
                Est = Res['Estadisticos']

                # Verify that the numeric statistics exist.
                if 'Media' not in Est:
                    continue

                Agregar(
                    f"{Categoria:<25} "
                    f"{Est['N_Validos']:>6} "
                    f"{Est['Media']:>10.4f} "
                    f"{Est['Mediana']:>10.4f} "
                    f"{Est['Desviacion_Estandar']:>10.4f} "
                    f"{Est['Minimo']:>8.2f} "
                    f"{Est['Maximo']:>8.2f}"
                )

        Agregar("")

        Agregar("")

        Columnas_Cdc_Existentes = [
            c for c in Columnas_Cdc_Todas if c in Resultados_General
        ]

        if Columnas_Cdc_Existentes:
            Separador("=")
            Agregar("SECCIÓN 3: PERCEPCIONES CDC POR CANDIDATO")
            Separador("=")
            Agregar("")

            Resultados_Por_Cat = Resultados_df.get('Por_Categoria', {})

            for Variable in Columnas_Cdc_Existentes:
                Est = Resultados_General.get(Variable, {}).get('Estadisticos', {})
                Agregar(f"► {Traducir_Variable_Cdc(Variable)}")
                Agregar(f"  N válidos: {Est.get('N_Validos', 0)} "
                        f"(faltantes: {Est.get('N_Faltantes', 0)}, "
                        f"{Formatear_Numero(Est.get('Porcentaje_Faltantes'))}%)")
                Agregar(f"  Media: {Formatear_Numero(Est.get('Media'))}  |  "
                        f"DE: {Formatear_Numero(Est.get('Desviacion_Estandar'))}")

                Comparaciones_Cat = Resultados_Por_Cat.get(Variable, {})
                if Comparaciones_Cat:
                    Agregar("  Por categoría (Media ± DE):")
                    for Categoria in Categorias_Validas:
                        Res_Cat = Comparaciones_Cat.get(Categoria)
                        if not Res_Cat:
                            continue
                        Est_Cat = Res_Cat.get('Estadisticos', {})
                        if 'Media' not in Est_Cat:
                            continue
                        Media_Cat = Est_Cat.get('Media')
                        DE_Cat = Est_Cat.get('Desviacion_Estandar')
                        Agregar(f"    {Categoria:<25} "
                                f"{Formatear_Numero(Media_Cat)} ± "
                                f"{Formatear_Numero(DE_Cat)}")
                Agregar("")

            Agregar("")

    Separador("=")
    Agregar("FIN DEL REPORTE")
    Separador("=")

    # Write file.
    with open(Ruta_Salida, 'w', encoding='utf-8') as Archivo:
        Archivo.write('\n'.join(Lineas))

    print(f"\nReporte guardado en: {Ruta_Salida}")


# =============================================================================
# MAIN EXECUTION.
# =============================================================================

if __name__ == '__main__':
    # Path to the folder with the final data.
    Ruta_Base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    Ruta_Datos = os.path.join(Ruta_Base, 'Processed_Data')

    # Execute descriptive analysis.
    Resultados, dfs = Ejecutar_Analisis_Descriptivo(Ruta_Datos)

    # Create Results/Reports folder if it doesn't exist.
    Ruta_Reportes = os.path.join(Ruta_Base, 'Results', 'Reports')
    if not os.path.exists(Ruta_Reportes):
        os.makedirs(Ruta_Reportes)

    # Generate TXT report.
    Ruta_Reporte = os.path.join(
        Ruta_Reportes,
        'Report_Descriptive_Analysis.txt'
    )
    Generar_Reporte_Descriptivo_TXT(Resultados, Ruta_Reporte)

    print("\nAnálisis completado.")
