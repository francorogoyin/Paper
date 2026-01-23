# -*- coding: utf-8 -*-
"""
Clustering de participantes basado en respuestas a items IP.

Este script realiza clustering jerarquico aglomerativo sobre las
respuestas a items conservadores y progresistas, utilizando
distancia de Hamming como metrica de similitud.

"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import hamming
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
import os


# =============================================================================
# CONFIGURACION DE ITEMS.
# =============================================================================

# Items conservadores (10 items).
Items_Conservadores = [
    'IP_Item_3_Respuesta',
    'IP_Item_4_Respuesta',
    'IP_Item_7_Respuesta',
    'IP_Item_8_Respuesta',
    'IP_Item_10_Respuesta',
    'IP_Item_19_Respuesta',
    'IP_Item_22_Respuesta',
    'IP_Item_23_Respuesta',
    'IP_Item_29_Respuesta',
    'IP_Item_30_Respuesta'
]

# Items progresistas (10 items).
Items_Progresistas = [
    'IP_Item_5_Respuesta',
    'IP_Item_6_Respuesta',
    'IP_Item_9_Respuesta',
    'IP_Item_11_Respuesta',
    'IP_Item_16_Respuesta',
    'IP_Item_20_Respuesta',
    'IP_Item_24_Respuesta',
    'IP_Item_25_Respuesta',
    'IP_Item_27_Respuesta',
    'IP_Item_28_Respuesta'
]


# =============================================================================
# FUNCIONES DE PROCESAMIENTO.
# =============================================================================

def Cargar_Bases_Definitivas(Ruta_Datos: str) -> tuple:

    """
    Carga las bases de datos definitivas de Generales y Ballotage.

    Parametros:
        Ruta_Datos (str): Ruta a la carpeta con los datos definitivos.

    Retorna:
        tuple: (DataFrame Generales, DataFrame Ballotage).

    """

    Ruta_Generales = os.path.join(Ruta_Datos, 'Generales.xlsx')
    Ruta_Ballotage = os.path.join(Ruta_Datos, 'Ballotage.xlsx')

    print("Cargando bases de datos definitivas...")
    print(f"  Generales: {Ruta_Generales}")
    print(f"  Ballotage: {Ruta_Ballotage}")

    Df_Generales = pd.read_excel(Ruta_Generales)
    Df_Ballotage = pd.read_excel(Ruta_Ballotage)

    print(f"  OK Generales: {len(Df_Generales)} registros")
    print(f"  OK Ballotage: {len(Df_Ballotage)} registros")

    return Df_Generales, Df_Ballotage


def Preparar_Dataframes_Clustering(
    Df_Generales: pd.DataFrame,
    Df_Ballotage: pd.DataFrame
) -> pd.DataFrame:

    """
    Prepara los DataFrames para el clustering extrayendo las columnas
    necesarias y agregando la columna de eleccion.

    Parametros:
        Df_Generales (pd.DataFrame): DataFrame de Generales.
        Df_Ballotage (pd.DataFrame): DataFrame de Ballotage.

    Retorna:
        pd.DataFrame: DataFrame combinado con columnas para clustering.

    """

    print("\nPreparando DataFrames para clustering...")

    # Columnas necesarias para clustering.
    Columnas_Clustering = ['ID'] + Items_Conservadores + Items_Progresistas

    # Extraer columnas relevantes.
    Df_Items_Generales = Df_Generales[Columnas_Clustering].copy()
    Df_Items_Ballotage = Df_Ballotage[Columnas_Clustering].copy()

    # Agregar columna de eleccion.
    Df_Items_Generales['Election'] = 'General'
    Df_Items_Ballotage['Election'] = 'Ballotage'

    # Combinar ambos DataFrames.
    Df_Combinado = pd.concat(
        [Df_Items_Generales, Df_Items_Ballotage],
        ignore_index=True
    )

    print(f"  Total de registros combinados: {len(Df_Combinado)}")

    return Df_Combinado


def Crear_Vectores_Respuestas(Df: pd.DataFrame) -> pd.DataFrame:

    """
    Crea vectores de respuestas conservadoras y progresistas
    para cada participante.

    Parametros:
        Df (pd.DataFrame): DataFrame con las respuestas a items.

    Retorna:
        pd.DataFrame: DataFrame con columnas de vectores agregadas.

    """

    print("\nCreando vectores de respuestas...")

    Df['Conservative_Vector'] = Df[Items_Conservadores].values.tolist()
    Df['Progressive_Vector'] = Df[Items_Progresistas].values.tolist()

    print("  OK Vectores creados.")

    return Df


def Calcular_Distancias_Hamming(Df: pd.DataFrame) -> tuple:

    """
    Calcula las matrices de distancia de Hamming para los vectores
    conservadores y progresistas.

    Parametros:
        Df (pd.DataFrame): DataFrame con los vectores de respuestas.

    Retorna:
        tuple: (Matriz conservadora, Matriz progresista, IDs de sujetos).

    """

    print("\nCalculando distancias de Hamming...")

    Ids_Sujetos = Df['ID'].tolist()
    Num_Sujetos = len(Ids_Sujetos)

    # Crear indice de ID a posicion.
    Indice_Id = {Id: Idx for Idx, Id in enumerate(Ids_Sujetos)}

    # Calcular distancias para cada par de sujetos.
    Distancias_Conservadoras = []
    Distancias_Progresistas = []

    Total_Pares = len(list(combinations(Ids_Sujetos, 2)))
    print(f"  Calculando {Total_Pares} pares de distancias...")

    for Id_1, Id_2 in combinations(Ids_Sujetos, 2):
        # Obtener vectores.
        Vector_Cons_1 = Df.loc[
            Df['ID'] == Id_1, 'Conservative_Vector'
        ].values[0]
        Vector_Cons_2 = Df.loc[
            Df['ID'] == Id_2, 'Conservative_Vector'
        ].values[0]
        Vector_Prog_1 = Df.loc[
            Df['ID'] == Id_1, 'Progressive_Vector'
        ].values[0]
        Vector_Prog_2 = Df.loc[
            Df['ID'] == Id_2, 'Progressive_Vector'
        ].values[0]

        # Calcular distancias de Hamming.
        Distancia_Conservadora = hamming(Vector_Cons_1, Vector_Cons_2)
        Distancia_Progresista = hamming(Vector_Prog_1, Vector_Prog_2)

        Distancias_Conservadoras.append((Id_1, Id_2, Distancia_Conservadora))
        Distancias_Progresistas.append((Id_1, Id_2, Distancia_Progresista))

    print("  OK Distancias calculadas.")

    # Crear matrices de distancia simetricas.
    print("  Construyendo matrices de distancia...")

    Matriz_Conservadora = np.zeros((Num_Sujetos, Num_Sujetos))
    Matriz_Progresista = np.zeros((Num_Sujetos, Num_Sujetos))

    for Id_1, Id_2, Distancia in Distancias_Conservadoras:
        Idx_1 = Indice_Id[Id_1]
        Idx_2 = Indice_Id[Id_2]
        Matriz_Conservadora[Idx_1, Idx_2] = Distancia
        Matriz_Conservadora[Idx_2, Idx_1] = Distancia

    for Id_1, Id_2, Distancia in Distancias_Progresistas:
        Idx_1 = Indice_Id[Id_1]
        Idx_2 = Indice_Id[Id_2]
        Matriz_Progresista[Idx_1, Idx_2] = Distancia
        Matriz_Progresista[Idx_2, Idx_1] = Distancia

    print("  OK Matrices construidas.")

    return Matriz_Conservadora, Matriz_Progresista, Ids_Sujetos


def Aplicar_Clustering(
    Matriz_Conservadora: np.ndarray,
    Matriz_Progresista: np.ndarray,
    N_Clusters_Conservador: int = 3,
    N_Clusters_Progresista: int = 3
) -> tuple:

    """
    Aplica clustering jerarquico aglomerativo sobre las matrices
    de distancia.

    Parametros:
        Matriz_Conservadora (np.ndarray): Matriz de distancia conservadora.
        Matriz_Progresista (np.ndarray): Matriz de distancia progresista.
        N_Clusters_Conservador (int): Numero de clusters conservadores.
        N_Clusters_Progresista (int): Numero de clusters progresistas.

    Retorna:
        tuple: (Labels conservadores, Labels progresistas).

    """

    print("\nAplicando clustering jerarquico aglomerativo...")

    # Clustering conservador.
    Clustering_Conservador = AgglomerativeClustering(
        n_clusters=N_Clusters_Conservador,
        linkage='average',
        metric='precomputed'
    )
    Labels_Conservadores = Clustering_Conservador.fit_predict(
        Matriz_Conservadora
    )

    # Clustering progresista.
    Clustering_Progresista = AgglomerativeClustering(
        n_clusters=N_Clusters_Progresista,
        linkage='average',
        metric='precomputed'
    )
    Labels_Progresistas = Clustering_Progresista.fit_predict(
        Matriz_Progresista
    )

    print(f"  Clusters conservadores: {N_Clusters_Conservador}")
    print(f"  Clusters progresistas: {N_Clusters_Progresista}")

    return Labels_Conservadores, Labels_Progresistas


def Calcular_Medianas(Df: pd.DataFrame) -> pd.DataFrame:

    """
    Calcula las medianas de los vectores conservadores y progresistas
    para cada participante.

    Parametros:
        Df (pd.DataFrame): DataFrame con los vectores.

    Retorna:
        pd.DataFrame: DataFrame con columnas de medianas agregadas.

    """

    print("\nCalculando medianas de vectores...")

    Df['Conservative_Median'] = Df['Conservative_Vector'].apply(np.median)
    Df['Progressive_Median'] = Df['Progressive_Vector'].apply(np.median)

    print("  OK Medianas calculadas.")

    return Df


def Exportar_Resultados(
    Df: pd.DataFrame,
    Ruta_Salida: str
) -> None:

    """
    Exporta el DataFrame con los resultados del clustering a Excel.

    Parametros:
        Df (pd.DataFrame): DataFrame con los resultados.
        Ruta_Salida (str): Ruta al archivo Excel de salida.

    """

    print(f"\nExportando resultados a: {Ruta_Salida}")

    Df.to_excel(Ruta_Salida, index=False)

    print("  OK Archivo exportado.")


def Comparar_Con_Original(
    Df_Nuevo: pd.DataFrame,
    Ruta_Original: str
) -> dict:

    """
    Compara el DataFrame nuevo con el archivo original para verificar
    que los resultados sean identicos.

    Parametros:
        Df_Nuevo (pd.DataFrame): DataFrame generado por este script.
        Ruta_Original (str): Ruta al archivo Clusters.xlsx original.

    Retorna:
        dict: Diccionario con los resultados de la comparacion.

    """

    print("\n" + "="*70)
    print("CONTROL DE COMPARACION CON ARCHIVO ORIGINAL")
    print("="*70)

    Resultados_Comparacion = {
        'Archivo_Original': Ruta_Original,
        'Identico': True,
        'Diferencias': []
    }

    # Cargar archivo original.
    print(f"\nCargando archivo original: {Ruta_Original}")

    if not os.path.exists(Ruta_Original):
        print("  ERROR: Archivo original no encontrado.")
        Resultados_Comparacion['Identico'] = False
        Resultados_Comparacion['Diferencias'].append(
            "Archivo original no encontrado"
        )
        return Resultados_Comparacion

    Df_Original = pd.read_excel(Ruta_Original)

    # Comparar dimensiones.
    print("\n1. COMPARACION DE DIMENSIONES")
    print("-"*50)

    Shape_Nuevo = Df_Nuevo.shape
    Shape_Original = Df_Original.shape

    print(f"  Nuevo:    {Shape_Nuevo[0]} filas x {Shape_Nuevo[1]} columnas")
    print(f"  Original: {Shape_Original[0]} filas x {Shape_Original[1]} columnas")

    if Shape_Nuevo != Shape_Original:
        Resultados_Comparacion['Identico'] = False
        Resultados_Comparacion['Diferencias'].append(
            f"Dimensiones diferentes: {Shape_Nuevo} vs {Shape_Original}"
        )
        print("  DIFERENCIA DETECTADA")
    else:
        print("  OK Dimensiones identicas")

    # Comparar columnas.
    print("\n2. COMPARACION DE COLUMNAS")
    print("-"*50)

    Columnas_Nuevo = set(Df_Nuevo.columns)
    Columnas_Original = set(Df_Original.columns)

    Faltantes_Nuevo = Columnas_Original - Columnas_Nuevo
    Faltantes_Original = Columnas_Nuevo - Columnas_Original

    if Faltantes_Nuevo:
        print(f"  Columnas faltantes en nuevo: {Faltantes_Nuevo}")
        Resultados_Comparacion['Diferencias'].append(
            f"Columnas faltantes en nuevo: {Faltantes_Nuevo}"
        )
        Resultados_Comparacion['Identico'] = False

    if Faltantes_Original:
        print(f"  Columnas extra en nuevo: {Faltantes_Original}")
        Resultados_Comparacion['Diferencias'].append(
            f"Columnas extra en nuevo: {Faltantes_Original}"
        )
        Resultados_Comparacion['Identico'] = False

    if not Faltantes_Nuevo and not Faltantes_Original:
        print("  OK Columnas identicas")

    # Comparar IDs.
    print("\n3. COMPARACION DE IDs")
    print("-"*50)

    Ids_Nuevo = set(Df_Nuevo['ID'].tolist())
    Ids_Original = set(Df_Original['ID'].tolist())

    Ids_Faltantes_Nuevo = Ids_Original - Ids_Nuevo
    Ids_Faltantes_Original = Ids_Nuevo - Ids_Original

    if Ids_Faltantes_Nuevo:
        print(f"  IDs faltantes en nuevo: {len(Ids_Faltantes_Nuevo)}")
        Resultados_Comparacion['Diferencias'].append(
            f"IDs faltantes en nuevo: {len(Ids_Faltantes_Nuevo)}"
        )
        Resultados_Comparacion['Identico'] = False

    if Ids_Faltantes_Original:
        print(f"  IDs extra en nuevo: {len(Ids_Faltantes_Original)}")
        Resultados_Comparacion['Diferencias'].append(
            f"IDs extra en nuevo: {len(Ids_Faltantes_Original)}"
        )
        Resultados_Comparacion['Identico'] = False

    if not Ids_Faltantes_Nuevo and not Ids_Faltantes_Original:
        print(f"  OK {len(Ids_Nuevo)} IDs identicos")

    # Comparar estadisticas de columnas numericas.
    print("\n4. COMPARACION DE ESTADISTICAS POR COLUMNA")
    print("-"*50)

    Columnas_Numericas = [
        'Conservative_Cluster', 'Progressive_Cluster',
        'Conservative_Median', 'Progressive_Median'
    ] + Items_Conservadores + Items_Progresistas

    for Columna in Columnas_Numericas:
        if Columna not in Df_Nuevo.columns or Columna not in Df_Original.columns:
            continue

        # Ordenar por ID para comparar correctamente.
        Df_Nuevo_Ordenado = Df_Nuevo.sort_values('ID').reset_index(drop=True)
        Df_Original_Ordenado = Df_Original.sort_values('ID').reset_index(drop=True)

        Valores_Nuevo = Df_Nuevo_Ordenado[Columna]
        Valores_Original = Df_Original_Ordenado[Columna]

        # Convertir a numerico si es posible.
        Valores_Nuevo = pd.to_numeric(Valores_Nuevo, errors='coerce')
        Valores_Original = pd.to_numeric(Valores_Original, errors='coerce')

        Media_Nuevo = Valores_Nuevo.mean()
        Media_Original = Valores_Original.mean()
        Mediana_Nuevo = Valores_Nuevo.median()
        Mediana_Original = Valores_Original.median()
        Suma_Nuevo = Valores_Nuevo.sum()
        Suma_Original = Valores_Original.sum()

        Diferencia_Media = abs(Media_Nuevo - Media_Original)
        Diferencia_Mediana = abs(Mediana_Nuevo - Mediana_Original)
        Diferencia_Suma = abs(Suma_Nuevo - Suma_Original)

        if Diferencia_Media > 0.0001 or Diferencia_Mediana > 0.0001:
            print(f"\n  {Columna}:")
            print(f"    Media:   Nuevo={Media_Nuevo:.4f}, "
                  f"Original={Media_Original:.4f}, Dif={Diferencia_Media:.4f}")
            print(f"    Mediana: Nuevo={Mediana_Nuevo:.4f}, "
                  f"Original={Mediana_Original:.4f}, Dif={Diferencia_Mediana:.4f}")
            Resultados_Comparacion['Diferencias'].append(
                f"{Columna}: Media diff={Diferencia_Media:.4f}, "
                f"Mediana diff={Diferencia_Mediana:.4f}"
            )
            Resultados_Comparacion['Identico'] = False

    if Resultados_Comparacion['Identico']:
        print("  OK Todas las estadisticas son identicas")

    # Comparar distribucion de clusters.
    print("\n5. COMPARACION DE DISTRIBUCION DE CLUSTERS")
    print("-"*50)

    for Tipo_Cluster in ['Conservative_Cluster', 'Progressive_Cluster']:
        if Tipo_Cluster not in Df_Nuevo.columns:
            continue

        Conteo_Nuevo = Df_Nuevo[Tipo_Cluster].value_counts().sort_index()
        Conteo_Original = Df_Original[Tipo_Cluster].value_counts().sort_index()

        print(f"\n  {Tipo_Cluster}:")
        print(f"    Nuevo:    {Conteo_Nuevo.to_dict()}")
        print(f"    Original: {Conteo_Original.to_dict()}")

        if not Conteo_Nuevo.equals(Conteo_Original):
            Resultados_Comparacion['Diferencias'].append(
                f"{Tipo_Cluster}: Distribucion diferente"
            )
            Resultados_Comparacion['Identico'] = False
            print("    DIFERENCIA DETECTADA")
        else:
            print("    OK Distribucion identica")

    # Comparar Election.
    print("\n6. COMPARACION DE COLUMNA ELECTION")
    print("-"*50)

    Conteo_Election_Nuevo = Df_Nuevo['Election'].value_counts().sort_index()
    Conteo_Election_Original = Df_Original['Election'].value_counts().sort_index()

    print(f"  Nuevo:    {Conteo_Election_Nuevo.to_dict()}")
    print(f"  Original: {Conteo_Election_Original.to_dict()}")

    if not Conteo_Election_Nuevo.equals(Conteo_Election_Original):
        Resultados_Comparacion['Diferencias'].append(
            "Election: Distribucion diferente"
        )
        Resultados_Comparacion['Identico'] = False
        print("  DIFERENCIA DETECTADA")
    else:
        print("  OK Distribucion identica")

    # Resumen final.
    print("\n" + "="*70)
    print("RESUMEN DE COMPARACION")
    print("="*70)

    if Resultados_Comparacion['Identico']:
        print("\n  RESULTADO: Los archivos son IDENTICOS")
    else:
        print("\n  RESULTADO: Se encontraron DIFERENCIAS")
        print("\n  Diferencias detectadas:")
        for Diferencia in Resultados_Comparacion['Diferencias']:
            print(f"    - {Diferencia}")

    return Resultados_Comparacion


def Ejecutar_Clustering_Completo(
    Ruta_Datos_Definitivos: str,
    Ruta_Salida: str,
    Ruta_Original: str = None
) -> dict:

    """
    Ejecuta el pipeline completo de clustering.

    Parametros:
        Ruta_Datos_Definitivos (str): Ruta a la carpeta con datos definitivos.
        Ruta_Salida (str): Ruta al archivo Excel de salida.
        Ruta_Original (str): Ruta al archivo original para comparacion.

    Retorna:
        dict: Diccionario con resultados y comparacion.

    """

    print("="*70)
    print("CLUSTERING DE PARTICIPANTES")
    print("="*70)

    # 1. Cargar bases definitivas.
    Df_Generales, Df_Ballotage = Cargar_Bases_Definitivas(
        Ruta_Datos_Definitivos
    )

    # 2. Preparar DataFrames para clustering.
    Df_Combinado = Preparar_Dataframes_Clustering(Df_Generales, Df_Ballotage)

    # 3. Crear vectores de respuestas.
    Df_Combinado = Crear_Vectores_Respuestas(Df_Combinado)

    # 4. Calcular distancias de Hamming.
    Matriz_Conservadora, Matriz_Progresista, Ids = Calcular_Distancias_Hamming(
        Df_Combinado
    )

    # 5. Aplicar clustering.
    Labels_Conservadores, Labels_Progresistas = Aplicar_Clustering(
        Matriz_Conservadora,
        Matriz_Progresista,
        N_Clusters_Conservador=3,
        N_Clusters_Progresista=3
    )

    # 6. Agregar labels al DataFrame.
    Df_Combinado['Conservative_Cluster'] = Labels_Conservadores
    Df_Combinado['Progressive_Cluster'] = Labels_Progresistas

    # 7. Calcular medianas.
    Df_Combinado = Calcular_Medianas(Df_Combinado)

    # 8. Exportar resultados.
    Exportar_Resultados(Df_Combinado, Ruta_Salida)

    # 9. Comparar con original si se proporciona.
    Resultados_Comparacion = None
    if Ruta_Original:
        Resultados_Comparacion = Comparar_Con_Original(
            Df_Combinado,
            Ruta_Original
        )

    print("\n" + "="*70)
    print("CLUSTERING COMPLETADO")
    print("="*70)

    return {
        'DataFrame': Df_Combinado,
        'Comparacion': Resultados_Comparacion
    }


# =============================================================================
# EJECUCION PRINCIPAL.
# =============================================================================

if __name__ == "__main__":

    # Configurar rutas.
    Ruta_Base = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    Ruta_Datos_Definitivos = os.path.join(Ruta_Base, "Datos definitivos")
    Ruta_Salida = os.path.join(Ruta_Base, "Datos crudos", "Clustering.xlsx")
    Ruta_Original = os.path.join(Ruta_Base, "Datos crudos", "Clusters.xlsx")

    # Ejecutar clustering.
    Resultados = Ejecutar_Clustering_Completo(
        Ruta_Datos_Definitivos,
        Ruta_Salida,
        Ruta_Original
    )
