# -*- coding: utf-8 -*-
"""
Participant clustering based on IP item responses.

This script performs hierarchical agglomerative clustering on
responses to conservative and progressive items, using Hamming
distance as the similarity metric.

"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import hamming
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
import os


# =============================================================================
# ITEM CONFIGURATION.
# =============================================================================

# Conservative items (10 items).
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

# Progressive items (10 items).
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
# PROCESSING FUNCTIONS.
# =============================================================================

def Cargar_Bases_Definitivas(Ruta_Datos: str) -> tuple:

    """
    Loads the definitive databases for General and Ballotage elections.

    Parameters:
        Ruta_Datos (str): Path to folder containing definitive data.

    Returns:
        tuple: (DataFrame General, DataFrame Ballotage).

    """

    Ruta_Generales = os.path.join(Ruta_Datos, 'Generales.xlsx')
    Ruta_Ballotage = os.path.join(Ruta_Datos, 'Ballotage.xlsx')

    print("Loading definitive databases...")
    print(f"  General: {Ruta_Generales}")
    print(f"  Ballotage: {Ruta_Ballotage}")

    Df_Generales = pd.read_excel(Ruta_Generales)
    Df_Ballotage = pd.read_excel(Ruta_Ballotage)

    print(f"  OK General: {len(Df_Generales)} records")
    print(f"  OK Ballotage: {len(Df_Ballotage)} records")

    return Df_Generales, Df_Ballotage


def Preparar_Dataframes_Clustering(
    Df_Generales: pd.DataFrame,
    Df_Ballotage: pd.DataFrame
) -> pd.DataFrame:

    """
    Prepares DataFrames for clustering by extracting necessary columns
    and adding the election column.

    Parameters:
        Df_Generales (pd.DataFrame): General election DataFrame.
        Df_Ballotage (pd.DataFrame): Ballotage election DataFrame.

    Returns:
        pd.DataFrame: Combined DataFrame with clustering columns.

    """

    print("\nPreparing DataFrames for clustering...")

    # Columns required for clustering.
    Columnas_Clustering = ['ID'] + Items_Conservadores + Items_Progresistas

    # Extract relevant columns.
    Df_Items_Generales = Df_Generales[Columnas_Clustering].copy()
    Df_Items_Ballotage = Df_Ballotage[Columnas_Clustering].copy()

    Df_Items_Generales = Construir_Columnas_Cambio_Tiempo(
        Construir_Columnas_Cambio_Opinion(Df_Items_Generales)
    )
    Df_Items_Ballotage = Construir_Columnas_Cambio_Tiempo(
        Construir_Columnas_Cambio_Opinion(Df_Items_Ballotage)
    )

    # Add election column.
    Df_Items_Generales['Election'] = 'General'
    Df_Items_Ballotage['Election'] = 'Ballotage'

    # Combine both DataFrames.
    Df_Combinado = pd.concat(
        [Df_Items_Generales, Df_Items_Ballotage],
        ignore_index=True
    )

    print(f"  Total combined records: {len(Df_Combinado)}")

    return Df_Combinado


def Crear_Vectores_Respuestas(Df: pd.DataFrame) -> pd.DataFrame:

    """
    Creates conservative and progressive response vectors
    for each participant.

    Parameters:
        Df (pd.DataFrame): DataFrame with item responses.

    Returns:
        pd.DataFrame: DataFrame with vector columns added.

    """

    print("\nCreating response vectors...")

    Df['Conservative_Vector'] = Df[Items_Conservadores].values.tolist()
    Df['Progressive_Vector'] = Df[Items_Progresistas].values.tolist()

    print("  OK Vectors created.")

    return Df


def Construir_Columnas_Cambio_Opinion(Df: pd.DataFrame) -> pd.DataFrame:

    """
    Generates CO_Item fields for each IP item from base, left,
    and right responses.

    Parameters:
        Df (pd.DataFrame): DataFrame with raw responses.

    Returns:
        pd.DataFrame: DataFrame with CO_Item_* columns added.

    """

    Items_IP = Items_Conservadores + Items_Progresistas

    for Numero_Item in Items_IP:

        Columna_Base = f'IP_Item_{Numero_Item}_Respuesta'
        Columna_Izq = f'IP_Item_{Numero_Item}_Izq_Respuesta'
        Columna_Der = f'IP_Item_{Numero_Item}_Der_Respuesta'

        Nueva_Columna_Izq = f'CO_Item_{Numero_Item}_Izq'
        Nueva_Columna_Der = f'CO_Item_{Numero_Item}_Der'

        if all(
            Columna in Df.columns
            for Columna in [Columna_Base, Columna_Izq, Columna_Der]
        ):

            Base_Numerica = pd.to_numeric(
                Df[Columna_Base],
                errors='coerce'
            )
            Izq_Numerica = pd.to_numeric(
                Df[Columna_Izq],
                errors='coerce'
            )
            Der_Numerica = pd.to_numeric(
                Df[Columna_Der],
                errors='coerce'
            )

            Df[Nueva_Columna_Izq] = Izq_Numerica - Base_Numerica
            Df[Nueva_Columna_Der] = Der_Numerica - Base_Numerica

    return Df


def Construir_Columnas_Cambio_Tiempo(Df: pd.DataFrame) -> pd.DataFrame:

    """
    Generates CT_Item fields for each IP item from base, left,
    and right response times.

    Parameters:
        Df (pd.DataFrame): DataFrame with raw times.

    Returns:
        pd.DataFrame: DataFrame with CT_Item_* columns added.

    """

    Items_IP = Items_Conservadores + Items_Progresistas

    for Numero_Item in Items_IP:

        Columna_Base = f'IP_Item_{Numero_Item}_Tiempo'
        Columna_Izq = f'IP_Item_{Numero_Item}_Izq_Tiempo'
        Columna_Der = f'IP_Item_{Numero_Item}_Der_Tiempo'

        Nueva_Columna_Izq = f'CT_Item_{Numero_Item}_Izq'
        Nueva_Columna_Der = f'CT_Item_{Numero_Item}_Der'

        if all(
            Columna in Df.columns
            for Columna in [Columna_Base, Columna_Izq, Columna_Der]
        ):

            Base_Numerica = pd.to_numeric(
                Df[Columna_Base],
                errors='coerce'
            )
            Izq_Numerica = pd.to_numeric(
                Df[Columna_Izq],
                errors='coerce'
            )
            Der_Numerica = pd.to_numeric(
                Df[Columna_Der],
                errors='coerce'
            )

            Df[Nueva_Columna_Izq] = Izq_Numerica - Base_Numerica
            Df[Nueva_Columna_Der] = Der_Numerica - Base_Numerica

    return Df


def Agregar_Variables_Co_Ct(Df: pd.DataFrame) -> pd.DataFrame:

    """
    Calculates CO/CT aggregates (means, sums, congruence and
    incongruence) used in the original processing.

    Parameters:
        Df (pd.DataFrame): DataFrame with vectors and CO/CT columns.

    Returns:
        pd.DataFrame: DataFrame enriched with new variables.

    """

    def Columnas_Disponibles(Nombres):
        return [Nombre for Nombre in Nombres if Nombre in Df.columns]

    def Contar_No_Nulos(Columnas):
        if not Columnas:
            return pd.Series(0, index=Df.index)
        return Df[Columnas].notna().sum(axis=1)

    Cols_CO_Pro_Izq = Columnas_Disponibles(
        [f'CO_Item_{Num}_Izq' for Num in Items_Progresistas]
    )
    Cols_CO_Pro_Der = Columnas_Disponibles(
        [f'CO_Item_{Num}_Der' for Num in Items_Progresistas]
    )
    Cols_CO_Con_Izq = Columnas_Disponibles(
        [f'CO_Item_{Num}_Izq' for Num in Items_Conservadores]
    )
    Cols_CO_Con_Der = Columnas_Disponibles(
        [f'CO_Item_{Num}_Der' for Num in Items_Conservadores]
    )
    Cols_CO_Pro = Cols_CO_Pro_Izq + Cols_CO_Pro_Der
    Cols_CO_Con = Cols_CO_Con_Izq + Cols_CO_Con_Der

    Cols_CT_Pro_Izq = Columnas_Disponibles(
        [f'CT_Item_{Num}_Izq' for Num in Items_Progresistas]
    )
    Cols_CT_Pro_Der = Columnas_Disponibles(
        [f'CT_Item_{Num}_Der' for Num in Items_Progresistas]
    )
    Cols_CT_Con_Izq = Columnas_Disponibles(
        [f'CT_Item_{Num}_Izq' for Num in Items_Conservadores]
    )
    Cols_CT_Con_Der = Columnas_Disponibles(
        [f'CT_Item_{Num}_Der' for Num in Items_Conservadores]
    )
    Cols_CT_Pro = Cols_CT_Pro_Izq + Cols_CT_Pro_Der
    Cols_CT_Con = Cols_CT_Con_Izq + Cols_CT_Con_Der

    if Cols_CO_Pro:
        Df['CO_Pro'] = Df[Cols_CO_Pro].mean(axis=1)
    if Cols_CO_Pro_Izq:
        Df['CO_Pro_Izq'] = Df[Cols_CO_Pro_Izq].mean(axis=1)
    if Cols_CO_Pro_Der:
        Df['CO_Pro_Der'] = Df[Cols_CO_Pro_Der].mean(axis=1)
    if Cols_CO_Con:
        Df['CO_Con'] = Df[Cols_CO_Con].mean(axis=1)
    if Cols_CO_Con_Izq:
        Df['CO_Con_Izq'] = Df[Cols_CO_Con_Izq].mean(axis=1)
    if Cols_CO_Con_Der:
        Df['CO_Con_Der'] = Df[Cols_CO_Con_Der].mean(axis=1)

    if Cols_CS := Columnas_Disponibles(Cols_CO_Pro_Izq):
        Df['Cambio_Op_Sum_Pro_Izq'] = Df[Cols_CS].sum(axis=1)
    if Cols := Columnas_Disponibles(Cols_CO_Pro_Der):
        Df['Cambio_Op_Sum_Pro_Der'] = Df[Cols].sum(axis=1)
    if Cols := Columnas_Disponibles(Cols_CO_Con_Izq):
        Df['Cambio_Op_Sum_Con_Izq'] = Df[Cols].sum(axis=1)
    if Cols := Columnas_Disponibles(Cols_CO_Con_Der):
        Df['Cambio_Op_Sum_Con_Der'] = Df[Cols].sum(axis=1)

    if Cols_CT_Pro:
        Df['CT_Pro'] = Df[Cols_CT_Pro].mean(axis=1)
    if Cols_CT_Pro_Izq:
        Df['CT_Pro_Izq'] = Df[Cols_CT_Pro_Izq].mean(axis=1)
    if Cols_CT_Pro_Der:
        Df['CT_Pro_Der'] = Df[Cols_CT_Pro_Der].mean(axis=1)
    if Cols_CT_Con:
        Df['CT_Con'] = Df[Cols_CT_Con].mean(axis=1)
    if Cols_CT_Con_Izq:
        Df['CT_Con_Izq'] = Df[Cols_CT_Con_Izq].mean(axis=1)
    if Cols_CT_Con_Der:
        Df['CT_Con_Der'] = Df[Cols_CT_Con_Der].mean(axis=1)

    if Cols := Columnas_Disponibles(Cols_CT_Pro_Izq):
        Df['Cambio_Tiempo_Sum_Pro_Izq'] = Df[Cols].sum(axis=1)
    if Cols := Columnas_Disponibles(Cols_CT_Pro_Der):
        Df['Cambio_Tiempo_Sum_Pro_Der'] = Df[Cols].sum(axis=1)
    if Cols := Columnas_Disponibles(Cols_CT_Con_Izq):
        Df['Cambio_Tiempo_Sum_Con_Izq'] = Df[Cols].sum(axis=1)
    if Cols := Columnas_Disponibles(Cols_CT_Con_Der):
        Df['Cambio_Tiempo_Sum_Con_Der'] = Df[Cols].sum(axis=1)

    Suma_Congruente_CO = (
        Df.get('Cambio_Op_Sum_Pro_Izq', 0) +
        Df.get('Cambio_Op_Sum_Con_Der', 0)
    )
    N_Pro_Izq = Contar_No_Nulos(Cols_CO_Pro_Izq)
    N_Con_Der = Contar_No_Nulos(Cols_CO_Con_Der)
    N_Congruente_CO = N_Pro_Izq + N_Con_Der
    if 'Cambio_Op_Sum_Pro_Izq' in Df.columns \
            and 'Cambio_Op_Sum_Con_Der' in Df.columns:
        Df['CO_Congruente'] = np.where(
            N_Congruente_CO > 0,
            Suma_Congruente_CO / N_Congruente_CO,
            np.nan
        )

    Suma_Incongruente_CO = (
        Df.get('Cambio_Op_Sum_Pro_Der', 0) +
        Df.get('Cambio_Op_Sum_Con_Izq', 0)
    )
    N_Pro_Der = Contar_No_Nulos(Cols_CO_Pro_Der)
    N_Con_Izq = Contar_No_Nulos(Cols_CO_Con_Izq)
    N_Incongruente_CO = N_Pro_Der + N_Con_Izq
    if 'Cambio_Op_Sum_Pro_Der' in Df.columns \
            and 'Cambio_Op_Sum_Con_Izq' in Df.columns:
        Df['CO_Incongruente'] = np.where(
            N_Incongruente_CO > 0,
            Suma_Incongruente_CO / N_Incongruente_CO,
            np.nan
        )

    Suma_Congruente_CT = (
        Df.get('Cambio_Tiempo_Sum_Pro_Izq', 0) +
        Df.get('Cambio_Tiempo_Sum_Con_Der', 0)
    )
    N_CT_Pro_Izq = Contar_No_Nulos(Cols_CT_Pro_Izq)
    N_CT_Con_Der = Contar_No_Nulos(Cols_CT_Con_Der)
    N_Congruente_CT = N_CT_Pro_Izq + N_CT_Con_Der
    if 'Cambio_Tiempo_Sum_Pro_Izq' in Df.columns \
            and 'Cambio_Tiempo_Sum_Con_Der' in Df.columns:
        Df['CT_Congruente'] = np.where(
            N_Congruente_CT > 0,
            Suma_Congruente_CT / N_Congruente_CT,
            np.nan
        )

    Suma_Incongruente_CT = (
        Df.get('Cambio_Tiempo_Sum_Pro_Der', 0) +
        Df.get('Cambio_Tiempo_Sum_Con_Izq', 0)
    )
    N_CT_Pro_Der = Contar_No_Nulos(Cols_CT_Pro_Der)
    N_CT_Con_Izq = Contar_No_Nulos(Cols_CT_Con_Izq)
    N_Incongruente_CT = N_CT_Pro_Der + N_CT_Con_Izq
    if 'Cambio_Tiempo_Sum_Pro_Der' in Df.columns \
            and 'Cambio_Tiempo_Sum_Con_Izq' in Df.columns:
        Df['CT_Incongruente'] = np.where(
            N_Incongruente_CT > 0,
            Suma_Incongruente_CT / N_Incongruente_CT,
            np.nan
        )

    return Df


def Calcular_Distancias_Hamming(Df: pd.DataFrame) -> tuple:

    """
    Calculates Hamming distance matrices for conservative and
    progressive vectors.

    Parameters:
        Df (pd.DataFrame): DataFrame with response vectors.

    Returns:
        tuple: (Conservative matrix, Progressive matrix, Subject IDs).

    """

    print("\nCalculating Hamming distances...")

    Ids_Sujetos = Df['ID'].tolist()
    Num_Sujetos = len(Ids_Sujetos)

    # Create ID to position index.
    Indice_Id = {Id: Idx for Idx, Id in enumerate(Ids_Sujetos)}

    # Calculate distances for each pair of subjects.
    Distancias_Conservadoras = []
    Distancias_Progresistas = []

    Total_Pares = len(list(combinations(Ids_Sujetos, 2)))
    print(f"  Calculating {Total_Pares} distance pairs...")

    for Id_1, Id_2 in combinations(Ids_Sujetos, 2):
        # Get vectors.
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

        # Calculate Hamming distances.
        Distancia_Conservadora = hamming(Vector_Cons_1, Vector_Cons_2)
        Distancia_Progresista = hamming(Vector_Prog_1, Vector_Prog_2)

        Distancias_Conservadoras.append((Id_1, Id_2, Distancia_Conservadora))
        Distancias_Progresistas.append((Id_1, Id_2, Distancia_Progresista))

    print("  OK Distances calculated.")

    # Create symmetric distance matrices.
    print("  Building distance matrices...")

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

    print("  OK Matrices built.")

    return Matriz_Conservadora, Matriz_Progresista, Ids_Sujetos


def Aplicar_Clustering(
    Matriz_Conservadora: np.ndarray,
    Matriz_Progresista: np.ndarray,
    N_Clusters_Conservador: int = 3,
    N_Clusters_Progresista: int = 3
) -> tuple:

    """
    Applies hierarchical agglomerative clustering on the
    distance matrices.

    Parameters:
        Matriz_Conservadora (np.ndarray): Conservative distance matrix.
        Matriz_Progresista (np.ndarray): Progressive distance matrix.
        N_Clusters_Conservador (int): Number of conservative clusters.
        N_Clusters_Progresista (int): Number of progressive clusters.

    Returns:
        tuple: (Conservative labels, Progressive labels).

    """

    print("\nApplying hierarchical agglomerative clustering...")

    # Conservative clustering.
    Clustering_Conservador = AgglomerativeClustering(
        n_clusters=N_Clusters_Conservador,
        linkage='average',
        metric='precomputed'
    )
    Labels_Conservadores = Clustering_Conservador.fit_predict(
        Matriz_Conservadora
    )

    # Progressive clustering.
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
    Calculates medians of conservative and progressive vectors
    for each participant.

    Parameters:
        Df (pd.DataFrame): DataFrame with vectors.

    Returns:
        pd.DataFrame: DataFrame with median columns added.

    """

    print("\nCalculating vector medians...")

    Df['Conservative_Median'] = Df['Conservative_Vector'].apply(np.median)
    Df['Progressive_Median'] = Df['Progressive_Vector'].apply(np.median)

    print("  OK Medians calculated.")

    return Df


def Exportar_Resultados(
    Df: pd.DataFrame,
    Ruta_Salida: str
) -> None:

    """
    Exports the DataFrame with clustering results to Excel.

    Parameters:
        Df (pd.DataFrame): DataFrame with results.
        Ruta_Salida (str): Path to output Excel file.

    """

    print(f"\nExporting results to: {Ruta_Salida}")
    Carpeta_Salida = os.path.dirname(Ruta_Salida)
    if Carpeta_Salida:
        os.makedirs(Carpeta_Salida, exist_ok=True)

    Df.to_excel(Ruta_Salida, index=False)

    print("  OK File exported.")


def Generar_Resumen_Clusters(
    Df: pd.DataFrame,
    Ruta_Excel: str,
    Ruta_Reporte: str
) -> str:

    """
    Builds a text summary of clusters by election.

    Parameters:
        Df (pd.DataFrame): DataFrame with clustering results.
        Ruta_Excel (str): Path to exported Excel file.
        Ruta_Reporte (str): Path where text report will be saved.

    Returns:
        str: Formatted report content.

    """

    def Formatear_Flotante(Valor):
        return f"{Valor:.3f}" if pd.notna(Valor) else "N/A"

    Lineas = [
        "Clustering general summary.",
        f"  Exported Excel: {Ruta_Excel}",
        f"  Saved report: {Ruta_Reporte}",
        f"  Total records: {len(Df)}"
    ]

    for Eleccion in ['General', 'Ballotage']:
        Df_Eleccion = Df.loc[Df['Election'] == Eleccion]
        Lineas.append(f"{Eleccion}: {len(Df_Eleccion)} records")

        for Tipo_Cluster in ['Conservative_Cluster', 'Progressive_Cluster']:
            if Tipo_Cluster not in Df.columns:
                continue

            Conteo = (
                Df_Eleccion[Tipo_Cluster]
                .value_counts()
                .sort_index()
            )

            if Conteo.empty:
                continue

            Texto_Conteo = ", ".join(
                f"{int(Etiqueta)}={int(Cantidad)}"
                for Etiqueta, Cantidad in Conteo.items()
            )

            Lineas.append(
                f"  {Tipo_Cluster}: {Texto_Conteo}"
            )

        if 'Conservative_Median' in Df.columns:
            Mediana_Cons = Df_Eleccion['Conservative_Median'].median()
            Mediana_Prog = Df_Eleccion['Progressive_Median'].median()
            Lineas.append(
                f"  Conservative median: {Formatear_Flotante(Mediana_Cons)}"
            )
            Lineas.append(
                f"  Progressive median: {Formatear_Flotante(Mediana_Prog)}"
            )

    return "\n".join(Lineas) + "\n"


def Guardar_Reporte_Clusters(
    Ruta_Reporte: str,
    Contenido: str
) -> None:

    """
    Saves the cluster report to disk.

    Parameters:
        Ruta_Reporte (str): Destination file path.
        Contenido (str): Text to be written.

    Returns:
        None.

    """

    Carpeta_Reporte = os.path.dirname(Ruta_Reporte)
    if Carpeta_Reporte:
        os.makedirs(Carpeta_Reporte, exist_ok=True)

    with open(Ruta_Reporte, 'w', encoding='utf-8') as Archivo:
        Archivo.write(Contenido)


def Actualizar_Bases_Definitivas(
    Ruta_Datos_Definitivos: str,
    Df_Clusters: pd.DataFrame
) -> None:

    """
    Adds cluster columns to definitive databases and overwrites
    General and Ballotage files.

    Parameters:
        Ruta_Datos_Definitivos (str): Folder containing the databases.
        Df_Clusters (pd.DataFrame): Combined DataFrame with cluster columns.

    Returns:
        None.

    """

    Columnas_Cluster = [
        'Conservative_Cluster',
        'Progressive_Cluster',
        'Conservative_Median',
        'Progressive_Median'
    ]

    Mapeo_Eleccion = {
        'Generales.xlsx': 'General',
        'Ballotage.xlsx': 'Ballotage'
    }

    for Archivo_Base, Eleccion in Mapeo_Eleccion.items():
        Ruta_Archivo = os.path.join(Ruta_Datos_Definitivos, Archivo_Base)
        if not os.path.exists(Ruta_Archivo):
            print(f"  Warning: {Ruta_Archivo} does not exist.")
            continue

        Df_Base = pd.read_excel(Ruta_Archivo)
        if Columnas_Cluster:
            Df_Base = Df_Base.drop(
                columns=[
                    Columna for Columna in Columnas_Cluster
                    if Columna in Df_Base.columns
                ],
                errors='ignore'
            )

        Df_Cluster_Eleccion = Df_Clusters.loc[
            Df_Clusters['Election'] == Eleccion,
            ['ID'] + Columnas_Cluster
        ].copy()

        Df_Actualizado = Df_Base.merge(
            Df_Cluster_Eleccion,
            on='ID',
            how='left'
        )

        Df_Actualizado.to_excel(
            Ruta_Archivo,
            index=False,
            engine='openpyxl'
        )

        print(
            f"  OK {Archivo_Base}: cluster columns added "
            f"({len(Df_Cluster_Eleccion)} records with cluster)."
        )

def Ejecutar_Clustering_Completo(
    Ruta_Datos_Definitivos: str,
    Ruta_Salida: str,
    Ruta_Reporte: str
) -> dict:

    """
    Executes the complete clustering pipeline.

    If Clusters.xlsx file already exists at Ruta_Salida, uses that data
    instead of recalculating clustering. Only recalculates if not present.

    Parameters:
        Ruta_Datos_Definitivos (str): Path to folder with definitive data.
        Ruta_Salida (str): Path to output Excel file.
        Ruta_Reporte (str): Path to text report file.

    Returns:
        dict: Dictionary with generated DataFrame and report path.

    """

    print("="*70)
    print("PARTICIPANT CLUSTERING")
    print("="*70)

    # Check if cluster file already exists.
    if os.path.exists(Ruta_Salida):
        print(f"\nExisting cluster file detected: {Ruta_Salida}")
        print("Using existing cluster data (not recalculating).")
        print("-"*70)

        # Load existing clusters.
        Df_Combinado = pd.read_excel(Ruta_Salida)
        print(f"  OK Clusters loaded: {len(Df_Combinado)} records.")

    else:
        print("\nCluster file does not exist. Calculating clustering...")
        print("-"*70)

        # 1. Load definitive databases.
        Df_Generales, Df_Ballotage = Cargar_Bases_Definitivas(
            Ruta_Datos_Definitivos
        )

        # 2. Prepare DataFrames for clustering.
        Df_Combinado = Preparar_Dataframes_Clustering(
            Df_Generales, Df_Ballotage
        )

        # 3. Create response vectors.
        Df_Combinado = Crear_Vectores_Respuestas(Df_Combinado)
        Df_Combinado = Agregar_Variables_Co_Ct(Df_Combinado)

        # 4. Calculate Hamming distances.
        Matriz_Conservadora, Matriz_Progresista, Ids = \
            Calcular_Distancias_Hamming(Df_Combinado)

        # 5. Apply clustering.
        Labels_Conservadores, Labels_Progresistas = Aplicar_Clustering(
            Matriz_Conservadora,
            Matriz_Progresista,
            N_Clusters_Conservador=3,
            N_Clusters_Progresista=3
        )

        # 6. Add labels to DataFrame.
        Df_Combinado['Conservative_Cluster'] = Labels_Conservadores
        Df_Combinado['Progressive_Cluster'] = Labels_Progresistas

        # 7. Calculate medians.
        Df_Combinado = Calcular_Medianas(Df_Combinado)

        # 8. Export results.
        Exportar_Resultados(Df_Combinado, Ruta_Salida)

    # 9. Update definitive databases (always executed).
    print("\nUpdating definitive databases with cluster columns...")
    Actualizar_Bases_Definitivas(
        Ruta_Datos_Definitivos,
        Df_Combinado
    )

    # 10. Generate text summary.
    Contenido_Reporte = Generar_Resumen_Clusters(
        Df_Combinado,
        Ruta_Salida,
        Ruta_Reporte
    )
    Guardar_Reporte_Clusters(Ruta_Reporte, Contenido_Reporte)

    print("\n" + "="*70)
    print("CLUSTERING COMPLETED")
    print("="*70)

    return {
        'DataFrame': Df_Combinado,
        'Ruta_Reporte': Ruta_Reporte
    }


# =============================================================================
# MAIN EXECUTION.
# =============================================================================

if __name__ == "__main__":

    # Configure paths.
    Ruta_Base = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    Ruta_Datos_Definitivos = os.path.join(Ruta_Base, "Processed_Data")
    Ruta_Salida = os.path.join(
        Ruta_Base,
        "Results",
        "Tables",
        "Clustering.xlsx"
    )

    # Execute clustering.
    Ruta_Reporte = os.path.join(
        Ruta_Base,
        "Results",
        "Reports",
        "Report_Clusters.txt"
    )

    # Execute clustering.
    Resultados = Ejecutar_Clustering_Completo(
        Ruta_Datos_Definitivos,
        Ruta_Salida,
        Ruta_Reporte
    )
