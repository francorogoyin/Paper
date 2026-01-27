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

    Df_Items_Generales = Construir_Columnas_Cambio_Tiempo(
        Construir_Columnas_Cambio_Opinion(Df_Items_Generales)
    )
    Df_Items_Ballotage = Construir_Columnas_Cambio_Tiempo(
        Construir_Columnas_Cambio_Opinion(Df_Items_Ballotage)
    )

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


def Construir_Columnas_Cambio_Opinion(Df: pd.DataFrame) -> pd.DataFrame:

    """
    Genera los campos CO_Item para cada ítem IP a partir de las
    respuestas base, de izquierda y de derecha.

    Parametros:
        Df (pd.DataFrame): DataFrame con las respuestas raw.

    Retorna:
        pd.DataFrame: DataFrame con columnas CO_Item_* agregadas.

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
    Genera los campos CT_Item para cada ítem IP a partir de los
    tiempos base, de izquierda y de derecha.

    Parametros:
        Df (pd.DataFrame): DataFrame con los tiempos raw.

    Retorna:
        pd.DataFrame: DataFrame con columnas CT_Item_* agregadas.

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
    Calcula los agregados CO/CT (promedios, sumas, congruencia e
    incongruencia) que usaba el procesamiento original.

    Parametros:
        Df (pd.DataFrame): DataFrame con los vectores y columnas CO/CT.

    Retorna:
        pd.DataFrame: DataFrame enriquecido con las nuevas variables.

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
    Carpeta_Salida = os.path.dirname(Ruta_Salida)
    if Carpeta_Salida:
        os.makedirs(Carpeta_Salida, exist_ok=True)

    Df.to_excel(Ruta_Salida, index=False)

    print("  OK Archivo exportado.")


def Generar_Resumen_Clusters(
    Df: pd.DataFrame,
    Ruta_Excel: str,
    Ruta_Reporte: str
) -> str:

    """
    Arma un resumen textual de los clusters por eleccion.

    Parametros:
        Df (pd.DataFrame): DataFrame con los resultados del clustering.
        Ruta_Excel (str): Ruta al Excel exportado.
        Ruta_Reporte (str): Ruta donde se guardara el reporte de texto.

    Retorna:
        str: Contenido formateado del reporte.

    """

    def Formatear_Flotante(Valor):
        return f"{Valor:.3f}" if pd.notna(Valor) else "N/A"

    Lineas = [
        "Resumen general del clustering.",
        f"  Excel exportado: {Ruta_Excel}",
        f"  Reporte guardado: {Ruta_Reporte}",
        f"  Total de registros: {len(Df)}"
    ]

    for Eleccion in ['General', 'Ballotage']:
        Df_Eleccion = Df.loc[Df['Election'] == Eleccion]
        Lineas.append(f"{Eleccion}: {len(Df_Eleccion)} registros")

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
                f"  Mediana conservador: {Formatear_Flotante(Mediana_Cons)}"
            )
            Lineas.append(
                f"  Mediana progresista: {Formatear_Flotante(Mediana_Prog)}"
            )

    return "\n".join(Lineas) + "\n"


def Guardar_Reporte_Clusters(
    Ruta_Reporte: str,
    Contenido: str
) -> None:

    """
    Guarda el reporte de clusters en disco.

    Parametros:
        Ruta_Reporte (str): Ruta de destino del archivo.
        Contenido (str): Texto que se escribira.

    Retorna:
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
    Agrega las columnas de cluster a las bases definitivas y sobrescribe
    los archivos de Generales y Ballotage.

    Parametros:
        Ruta_Datos_Definitivos (str): Carpeta donde se encuentran las bases.
        Df_Clusters (pd.DataFrame): DataFrame combinado con columnas de clusters.

    Retorna:
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
            print(f"  Advertencia: no existe {Ruta_Archivo}.")
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
            f"  OK {Archivo_Base}: columnas de cluster agregadas "
            f"({len(Df_Cluster_Eleccion)} registros con cluster)."
        )

def Ejecutar_Clustering_Completo(
    Ruta_Datos_Definitivos: str,
    Ruta_Salida: str,
    Ruta_Reporte: str
) -> dict:

    """
    Ejecuta el pipeline completo de clustering.

    Si el archivo Clusters.xlsx ya existe en Ruta_Salida, usa esos datos
    en lugar de recalcular el clustering. Solo recalcula si no existe.

    Parametros:
        Ruta_Datos_Definitivos (str): Ruta a la carpeta con datos definitivos.
        Ruta_Salida (str): Ruta al archivo Excel de salida.
        Ruta_Reporte (str): Ruta al archivo de texto de reporte.

    Retorna:
        dict: Diccionario con el DataFrame generado y la ruta del reporte.

    """

    print("="*70)
    print("CLUSTERING DE PARTICIPANTES")
    print("="*70)

    # Verificar si ya existe el archivo de clusters.
    if os.path.exists(Ruta_Salida):
        print(f"\nArchivo de clusters existente detectado: {Ruta_Salida}")
        print("Usando datos de clusters existentes (no se recalcula).")
        print("-"*70)

        # Cargar clusters existentes.
        Df_Combinado = pd.read_excel(Ruta_Salida)
        print(f"  OK Clusters cargados: {len(Df_Combinado)} registros.")

    else:
        print("\nNo existe archivo de clusters. Calculando clustering...")
        print("-"*70)

        # 1. Cargar bases definitivas.
        Df_Generales, Df_Ballotage = Cargar_Bases_Definitivas(
            Ruta_Datos_Definitivos
        )

        # 2. Preparar DataFrames para clustering.
        Df_Combinado = Preparar_Dataframes_Clustering(
            Df_Generales, Df_Ballotage
        )

        # 3. Crear vectores de respuestas.
        Df_Combinado = Crear_Vectores_Respuestas(Df_Combinado)
        Df_Combinado = Agregar_Variables_Co_Ct(Df_Combinado)

        # 4. Calcular distancias de Hamming.
        Matriz_Conservadora, Matriz_Progresista, Ids = \
            Calcular_Distancias_Hamming(Df_Combinado)

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

    # 9. Actualizar bases definitivas (siempre se ejecuta).
    print("\nActualizando bases definitivas con columnas de cluster...")
    Actualizar_Bases_Definitivas(
        Ruta_Datos_Definitivos,
        Df_Combinado
    )

    # 10. Generar resumen en texto.
    Contenido_Reporte = Generar_Resumen_Clusters(
        Df_Combinado,
        Ruta_Salida,
        Ruta_Reporte
    )
    Guardar_Reporte_Clusters(Ruta_Reporte, Contenido_Reporte)

    print("\n" + "="*70)
    print("CLUSTERING COMPLETADO")
    print("="*70)

    return {
        'DataFrame': Df_Combinado,
        'Ruta_Reporte': Ruta_Reporte
    }


# =============================================================================
# EJECUCION PRINCIPAL.
# =============================================================================

if __name__ == "__main__":

    # Configurar rutas.
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

    # Ejecutar clustering.
    Ruta_Reporte = os.path.join(
        Ruta_Base,
        "Results",
        "Reports",
        "Report_Clusters.txt"
    )

    # Ejecutar clustering.
    Resultados = Ejecutar_Clustering_Completo(
        Ruta_Datos_Definitivos,
        Ruta_Salida,
        Ruta_Reporte
    )
