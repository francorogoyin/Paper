# -*- coding: utf-8 -*-
"""
Análisis descriptivo para el proyecto de tesis.

Este script genera estadísticos descriptivos completos para todas
las variables del estudio, tanto a nivel general como segmentado
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

# Variables a excluir del análisis (identificadores, etc.).
Variables_Excluir = [
    'ID',
    'Fecha',
    'Timestamp',
    'index'
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


def Determinar_Tipo_Variable(Serie: pd.Series) -> str:

    """
    Determina el tipo de una variable para aplicar el análisis
    descriptivo apropiado.

    Parámetros:
        Serie (pd.Series): Serie con los datos de la variable.

    Retorna:
        str: Tipo de variable ('continua', 'discreta', 'categorica',
            'binaria', 'booleana', 'lista').

    """

    # Eliminar valores nulos para el análisis.
    Serie_Limpia = Serie.dropna()

    if len(Serie_Limpia) == 0:
        return 'vacia'

    # Verificar si contiene listas o estructuras complejas.
    if len(Serie_Limpia) > 0:
        Primer_Valor = Serie_Limpia.iloc[0]
        if isinstance(Primer_Valor, (list, dict, tuple)):
            return 'lista'

    # Booleanos.
    if Serie.dtype == bool:
        return 'booleana'

    # Categóricas (objeto/string).
    if Serie.dtype == 'object' or Serie.dtype.name == 'category':
        # Verificar si son strings que parecen listas.
        if len(Serie_Limpia) > 0:
            Primer_Valor = Serie_Limpia.iloc[0]
            if isinstance(Primer_Valor, str):
                if Primer_Valor.startswith('[') and Primer_Valor.endswith(']'):
                    return 'lista'
        return 'categorica'

    # Numéricas.
    Valores_Unicos = Serie_Limpia.nunique()

    # Binarias (solo 2 valores).
    if Valores_Unicos == 2:
        return 'binaria'

    # Discretas (pocos valores únicos o todos enteros).
    if Valores_Unicos <= 10:
        return 'discreta'

    # Verificar si todos son enteros.
    if Serie_Limpia.dtype in ['int64', 'int32']:
        if Valores_Unicos <= 20:
            return 'discreta'

    # Continuas.
    return 'continua'


# =============================================================================
# FUNCIONES DE ESTADÍSTICOS DESCRIPTIVOS.
# =============================================================================

def Calcular_Estadisticos_Continuos(Serie: pd.Series) -> dict:

    """
    Calcula estadísticos descriptivos para variables continuas.

    Parámetros:
        Serie (pd.Series): Serie con los datos numéricos.

    Retorna:
        dict: Diccionario con todos los estadísticos calculados.

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

    # Estadísticos de tendencia central.
    Media = Serie_Limpia.mean()
    Mediana = Serie_Limpia.median()
    Moda_Resultado = Serie_Limpia.mode()
    Moda = Moda_Resultado.iloc[0] if len(Moda_Resultado) > 0 else np.nan

    # Estadísticos de dispersión.
    Desviacion_Estandar = Serie_Limpia.std()
    Varianza = Serie_Limpia.var()
    Minimo = Serie_Limpia.min()
    Maximo = Serie_Limpia.max()
    Rango = Maximo - Minimo

    # Forma de la distribución.
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
    Calcula estadísticos descriptivos para variables discretas,
    incluyendo conteo de valores únicos.

    Parámetros:
        Serie (pd.Series): Serie con los datos discretos.

    Retorna:
        dict: Diccionario con estadísticos y frecuencias.

    """

    # Primero calcular estadísticos como si fuera continua.
    Estadisticos = Calcular_Estadisticos_Continuos(Serie)

    # Agregar conteo de valores únicos.
    Serie_Limpia = Serie.dropna()
    Valores_Unicos = Serie_Limpia.nunique()

    # Frecuencias.
    Frecuencias = Serie_Limpia.value_counts().sort_index()
    Frecuencias_Relativas = Serie_Limpia.value_counts(
        normalize=True
    ).sort_index() * 100

    # Construir diccionario de frecuencias.
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
    Calcula estadísticos descriptivos para variables categóricas.

    Parámetros:
        Serie (pd.Series): Serie con los datos categóricos.

    Retorna:
        dict: Diccionario con frecuencias y porcentajes.

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

    # Valores únicos.
    Valores_Unicos = Serie_Limpia.nunique()

    # Moda.
    Moda_Resultado = Serie_Limpia.mode()
    Moda = Moda_Resultado.iloc[0] if len(Moda_Resultado) > 0 else None

    # Frecuencias.
    Frecuencias = Serie_Limpia.value_counts()
    Frecuencias_Relativas = Serie_Limpia.value_counts(normalize=True) * 100

    # Construir diccionario de frecuencias.
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
    Calcula estadísticos descriptivos para variables binarias
    (incluyendo booleanas).

    Parámetros:
        Serie (pd.Series): Serie con los datos binarios.

    Retorna:
        dict: Diccionario con frecuencias y proporciones.

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

    # Frecuencias.
    Frecuencias = Serie_Limpia.value_counts()
    Frecuencias_Relativas = Serie_Limpia.value_counts(normalize=True) * 100

    # Identificar valores.
    Valores = sorted(Serie_Limpia.unique())

    # Proporción del valor "positivo" (True, 1, o el mayor).
    if Serie.dtype == bool:
        Proporcion_Positivo = Serie_Limpia.mean() * 100
    else:
        Proporcion_Positivo = (
            Serie_Limpia == max(Valores)
        ).mean() * 100

    # Construir diccionario de frecuencias.
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
# FUNCIÓN PRINCIPAL DE ANÁLISIS.
# =============================================================================

def Analizar_Variable(
    df: pd.DataFrame,
    Nombre_Variable: str
) -> dict:

    """
    Analiza una variable y retorna sus estadísticos descriptivos
    según su tipo.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        Nombre_Variable (str): Nombre de la variable a analizar.

    Retorna:
        dict: Diccionario con tipo de variable y estadísticos.

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

    # Ignorar variables que contienen listas.
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
    Analiza una variable segmentada por categoría electoral.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        Nombre_Variable (str): Nombre de la variable a analizar.
        Columna_Categoria (str): Nombre de la columna de categoría.

    Retorna:
        dict: Diccionario con estadísticos por cada categoría.

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

def Ejecutar_Analisis_Descriptivo(Ruta_Datos: str) -> dict:

    """
    Ejecuta el análisis descriptivo completo para todas las
    variables en ambos datasets.

    Parámetros:
        Ruta_Datos (str): Ruta a la carpeta con los datos.

    Retorna:
        dict: Diccionario con todos los resultados del análisis.

    """

    print("=" * 70)
    print("ANÁLISIS DESCRIPTIVO COMPLETO")
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

        # Obtener lista de variables a analizar.
        Variables = [
            col for col in df.columns
            if col not in Variables_Excluir
        ]

        print(f"Variables a analizar: {len(Variables)}")

        # Análisis general de cada variable.
        print("\nAnálisis general...")
        for i, Variable in enumerate(Variables):
            Resultado = Analizar_Variable(df, Variable)
            if Resultado is not None:
                Resultados_df['General'][Variable] = Resultado

            if (i + 1) % 50 == 0:
                print(f"  Procesadas: {i + 1}/{len(Variables)}")

        print(f"  Completado: {len(Variables)} variables")

        # Análisis por categoría.
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
# GENERACIÓN DE REPORTE TXT.
# =============================================================================

def Generar_Reporte_Descriptivo_TXT(
    Todos_Resultados: dict,
    Ruta_Salida: str
) -> None:

    """
    Genera un archivo TXT con el reporte completo del análisis
    descriptivo.

    Parámetros:
        Todos_Resultados (dict): Diccionario con los resultados.
        Ruta_Salida (str): Ruta al archivo de salida.

    """

    Lineas = []

    def Agregar(Texto: str = ""):
        Lineas.append(Texto)

    def Separador(Caracter: str = "=", Largo: int = 80):
        Lineas.append(Caracter * Largo)

    # =========================================================================
    # ENCABEZADO.
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
        # SECCIÓN 1: ANÁLISIS GENERAL.
        # =================================================================
        Separador("=")
        Agregar("SECCIÓN 1: ESTADÍSTICOS GENERALES POR VARIABLE")
        Separador("=")
        Agregar("")

        Resultados_General = Resultados_df.get('General', {})

        # Agrupar por tipo de variable.
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

        # Variables continuas.
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

        # Variables discretas.
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

        # Variables categóricas.
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
                # Ordenar por frecuencia descendente.
                Conteo_Ordenado = sorted(
                    Conteo.items(),
                    key=lambda x: x[1]['Frecuencia'],
                    reverse=True
                )
                for Valor, Info in Conteo_Ordenado:
                    Agregar(f"    {Valor}: {Info['Frecuencia']} "
                            f"({Info['Porcentaje']:.1f}%)")

                Agregar("")

        # Variables binarias y booleanas.
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
        # SECCIÓN 2: ANÁLISIS POR CATEGORÍA.
        # =================================================================
        Separador("=")
        Agregar("SECCIÓN 2: ESTADÍSTICOS POR CATEGORÍA ELECTORAL")
        Separador("=")
        Agregar("")

        Resultados_Por_Cat = Resultados_df.get('Por_Categoria', {})

        # Solo mostrar variables numéricas por categoría.
        for Variable, Resultados_Cats in Resultados_Por_Cat.items():

            # Verificar que haya resultados.
            if not Resultados_Cats:
                continue

            # Verificar si es numérica.
            Primer_Resultado = list(Resultados_Cats.values())[0]
            Tipo = Primer_Resultado.get('Tipo', '')

            if Tipo not in ['continua', 'discreta']:
                continue

            Separador("-")
            Agregar(f"► {Variable}")
            Separador("-")
            Agregar("")

            # Encabezado de tabla.
            Agregar(f"{'Categoría':<25} {'N':>6} {'Media':>10} "
                    f"{'Mediana':>10} {'DE':>10} {'Mín':>8} {'Máx':>8}")
            Agregar("-" * 80)

            for Categoria in Categorias_Validas:
                if Categoria not in Resultados_Cats:
                    continue

                Res = Resultados_Cats[Categoria]
                Est = Res['Estadisticos']

                # Verificar que los estadísticos numéricos existan.
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
    Ruta_Datos = os.path.join(Ruta_Base, 'Datos definitivos')

    # Ejecutar análisis descriptivo.
    Resultados, dfs = Ejecutar_Analisis_Descriptivo(Ruta_Datos)

    # Crear carpeta Resultados si no existe.
    Ruta_Resultados = os.path.join(Ruta_Base, 'Resultados')
    if not os.path.exists(Ruta_Resultados):
        os.makedirs(Ruta_Resultados)

    # Generar reporte TXT.
    Ruta_Reporte = os.path.join(
        Ruta_Resultados,
        'Analisis_Descriptivo.txt'
    )
    Generar_Reporte_Descriptivo_TXT(Resultados, Ruta_Reporte)

    print("\nAnálisis completado.")
