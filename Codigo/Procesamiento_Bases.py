"""
Script consolidado de procesamiento de datos del experimento de tesis.

Este script combina los notebooks 1-19 del directorio Código en un único
pipeline de procesamiento que transforma los datos crudos del experimento
electoral argentino 2023 en bases de datos limpias y procesadas listas para
análisis estadístico.

Autor: Patricio Nogueroles
Proyecto: Tesis de Licenciatura - Experimento de Toma de Decisiones Electorales
"""

import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Any
from unidecode import unidecode
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def Crear_Variables_De_Orden_IP_Items(df: pd.DataFrame) -> pd.DataFrame:

    """
    Procesa todo el DataFrame para extraer orden de IP Items y último Item.

    """

    def Extraer_Datos_JSON(Fila_JSON):
        try:
            Datos_Sujeto = json.loads(Fila_JSON)

            Orden_IP_Items = [
                int(Clave.split('_')[-1]) for Clave in
                Datos_Sujeto['results'][1]['fase_3']['IP'].keys()
                if Clave.startswith('IP_item_')
            ]

            Ultimo_IP_Item = (
                Orden_IP_Items[-1] if Orden_IP_Items else None
            )

            return Orden_IP_Items, Ultimo_IP_Item

        except (KeyError, IndexError, json.JSONDecodeError):
            return [], None

    Resultados_Procesados = df['results'].apply(Extraer_Datos_JSON)

    df['Orden_IP_Items'] = [
        Resultado[0] for Resultado in Resultados_Procesados
    ]
    df['Ultimo_IP_Item'] = [
        Resultado[1] for Resultado in Resultados_Procesados
    ]

    return df


def Crear_Variables_De_Orden_IP_Items_Asociados(
    df: pd.DataFrame
) -> pd.DataFrame:

    """
    Procesa todo el DataFrame para extraer orden de IP Items asociados
    a candidatos y último Item.

    """

    Lista_Orden_IP_Items = []
    Lista_Ultimo_IP_Item = []

    for Fila_JSON in df['results']:
        try:
            Datos_Sujeto = json.loads(Fila_JSON)

            Orden_IP_Items = [
                Clave.split('_')[2] + '_' + Clave.split('_')[3]
                for Clave in
                Datos_Sujeto['results'][1]['fase_3'][
                    'IP_modificada'
                ].keys()
                if Clave.startswith('IP_item_') and
                len(Clave.split('_')) > 3
            ]

            Ultimo_IP_Item = (
                Orden_IP_Items[-1] if Orden_IP_Items else None
            )

            Lista_Orden_IP_Items.append(Orden_IP_Items)
            Lista_Ultimo_IP_Item.append(Ultimo_IP_Item)

        except (KeyError, IndexError, json.JSONDecodeError):
            Lista_Orden_IP_Items.append([])
            Lista_Ultimo_IP_Item.append(None)

    df['Orden_IP_Items_Asociados'] = Lista_Orden_IP_Items
    df['Ultimo_IP_Item_Asociado'] = Lista_Ultimo_IP_Item

    return df


def Crear_Primeros_IP_Items_Asociados(
    df: pd.DataFrame,
    n: int
) -> pd.DataFrame:

    """
    Crea una nueva columna con los primeros N elementos de
    'Orden_IP_Items_Asociados'.

    """

    df['Primeros_IP_Items_Asociados'] = (
        df['Orden_IP_Items_Asociados'].apply(
            lambda lista: lista[:n] if isinstance(lista, list) and
            len(lista) >= n else lista
        )
    )

    return df


def Aplanar_Diccionario(
    Diccionario: Dict,
    Prefijo: str = ''
) -> Dict:

    """
    Convierte un diccionario anidado en uno plano, usando puntos para
    separar los niveles de anidamiento.

    """

    Diccionario_Plano = {}

    for Clave, Valor in Diccionario.items():
        Nueva_Clave = f"{Prefijo}.{Clave}" if Prefijo else Clave

        if isinstance(Valor, dict):
            Diccionario_Plano.update(
                Aplanar_Diccionario(Valor, Nueva_Clave)
            )
        elif isinstance(Valor, list):
            Diccionario_Plano[Nueva_Clave] = ', '.join(map(str, Valor))
        else:
            Diccionario_Plano[Nueva_Clave] = Valor

    return Diccionario_Plano


def Procesar_Columna_Results(df: pd.DataFrame) -> pd.DataFrame:

    """
    Extrae y procesa la columna 'results' de un DataFrame, convirtiendo
    el contenido JSON en un DataFrame de pandas.

    """

    Columna_Results = df['results']
    Lista_Datos_Procesados = []

    for Indice, Contenido_JSON in enumerate(Columna_Results):
        try:
            if pd.isna(Contenido_JSON) or Contenido_JSON is None:
                continue

            if not isinstance(Contenido_JSON, str):
                continue

            Datos_JSON = json.loads(Contenido_JSON)
            Array_Results = Datos_JSON.get('results', [])

            Fila_Procesada = {
                'id': Datos_JSON.get('subject')
            }

            for Item in Array_Results:
                if isinstance(Item, dict):
                    Datos_Aplanados = Aplanar_Diccionario(Item)
                    Fila_Procesada.update(Datos_Aplanados)

            Lista_Datos_Procesados.append(Fila_Procesada)

        except json.JSONDecodeError:
            continue
        except Exception:
            continue

    if not Lista_Datos_Procesados:
        return pd.DataFrame()

    DataFrame_Final = pd.DataFrame(Lista_Datos_Procesados)

    return DataFrame_Final


def Rellenar_IP_Items_Asociados_Faltantes(df: pd.DataFrame) -> None:

    """
    Rellena los valores faltantes en columnas IP_Item_X_Izq/Der cuando
    uno tiene valor y el otro es NaN, usando la mediana por categoría.

    """

    Numeros_IP = set()
    for Columna in df.columns:
        if 'IP_Item_' in Columna and (
            '_Izq_' in Columna or '_Der_' in Columna
        ):
            Partes = Columna.split('_')
            if len(Partes) >= 3:
                Numero = Partes[2]
                Numeros_IP.add(Numero)

    for Numero in sorted(Numeros_IP):
        for Tipo in ['Respuesta', 'Tiempo']:
            Col_Izq = f'IP_Item_{Numero}_Izq_{Tipo}'
            Col_Der = f'IP_Item_{Numero}_Der_{Tipo}'

            if Col_Izq in df.columns and Col_Der in df.columns:

                df[Col_Izq] = pd.to_numeric(df[Col_Izq], errors='coerce')
                df[Col_Der] = pd.to_numeric(df[Col_Der], errors='coerce')

                Mask_Izq_Lleno_Der_Vacio = (
                    df[Col_Izq].notna() & df[Col_Der].isna()
                )
                Mask_Der_Lleno_Izq_Vacio = (
                    df[Col_Der].notna() & df[Col_Izq].isna()
                )

                Medianas_Izq = df.groupby('Categoria_PASO_2023')[
                    Col_Izq
                ].median()
                Medianas_Der = df.groupby('Categoria_PASO_2023')[
                    Col_Der
                ].median()

                for Indice in df[Mask_Izq_Lleno_Der_Vacio].index:
                    Categoria = df.loc[Indice, 'Categoria_PASO_2023']

                    if (Categoria in Medianas_Der.index and
                        pd.notna(Medianas_Der[Categoria])):
                        Valor_Mediana = Medianas_Der[Categoria]
                        df.loc[Indice, Col_Der] = Valor_Mediana

                for Indice in df[Mask_Der_Lleno_Izq_Vacio].index:
                    Categoria = df.loc[Indice, 'Categoria_PASO_2023']

                    if (Categoria in Medianas_Izq.index and
                        pd.notna(Medianas_Izq[Categoria])):
                        Valor_Mediana = Medianas_Izq[Categoria]
                        df.loc[Indice, Col_Izq] = Valor_Mediana


def Eliminar_Primeros_Datos_IP_Items_Asociados(
    df: pd.DataFrame
) -> pd.DataFrame:

    """
    Elimina los datos de los primeros 3 IP Items únicos basándose en
    'Orden_IP_Items_Asociados'. Para cada IP Item, elimina tanto la
    versión _Izq como _Der.

    """

    df_Modificado = df.copy()
    Sufijos = ['_Respuesta', '_Candidato', '_Tiempo']

    for Index, Fila in df_Modificado.iterrows():

        Orden_IP_Items = Fila['Orden_IP_Items_Asociados']

        if isinstance(Orden_IP_Items, list) and len(Orden_IP_Items) > 0:

            Numeros_IP_Unicos = set()

            for Item in Orden_IP_Items:
                if isinstance(Item, str) and '_' in Item:
                    Numero = Item.split('_')[0]
                    Numeros_IP_Unicos.add(Numero)

                    if len(Numeros_IP_Unicos) >= 3:
                        break

            for Numero in Numeros_IP_Unicos:
                for Direccion in ['_Izq', '_Der']:
                    for Sufijo in Sufijos:
                        Columna = f'IP_Item_{Numero}{Direccion}{Sufijo}'

                        if Columna in df_Modificado.columns:
                            df_Modificado.at[Index, Columna] = pd.NA

    return df_Modificado


def Eliminar_Filas_Por_Desviacion_Estandar(
    Data_Frame: pd.DataFrame,
    Columnas_Tiempo: List[str],
    Numero_Desviaciones: int = 3
) -> pd.DataFrame:

    """
    Elimina filas donde algún valor de tiempo exceda el número
    especificado de desvíos estándar desde la media global de todas
    las columnas de tiempo.

    """

    Columnas_Existentes = [
        Columna for Columna in Columnas_Tiempo
        if Columna in Data_Frame.columns
    ]

    if len(Columnas_Existentes) == 0:
        print("ADVERTENCIA Ninguna columna de tiempo encontrada.")
        return Data_Frame

    Todos_Los_Valores = []
    for Columna in Columnas_Existentes:
        Valores_Limpios = pd.to_numeric(
            Data_Frame[Columna],
            errors='coerce'
        ).dropna().tolist()
        Todos_Los_Valores.extend(Valores_Limpios)

    if len(Todos_Los_Valores) == 0:
        print("ADVERTENCIA No hay valores numéricos para calcular estadísticos.")
        return Data_Frame

    Media_Global = sum(Todos_Los_Valores) / len(Todos_Los_Valores)
    Varianza = sum(
        (Valor - Media_Global) ** 2 for Valor in Todos_Los_Valores
    ) / len(Todos_Los_Valores)
    Desviacion_Estandar_Global = Varianza ** 0.5
    Limite_Superior = Media_Global + (
        Numero_Desviaciones * Desviacion_Estandar_Global
    )

    print(f"  Media global: {Media_Global:.2f}")
    print(
        f"  Desvío estándar global: "
        f"{Desviacion_Estandar_Global:.2f}"
    )
    print(
        f"  Límite superior ({Numero_Desviaciones} desvíos): "
        f"{Limite_Superior:.2f}"
    )

    Filas_Validas = []
    Contador_Eliminados = 0

    for Indice, Fila in Data_Frame.iterrows():
        Fila_Valida = True
        for Columna in Columnas_Existentes:
            Valor = pd.to_numeric(Fila[Columna], errors='coerce')

            if isinstance(Valor, pd.Series):
                if not Valor.empty:
                    Valor = Valor.iloc[0]
                else:
                    continue

            if pd.notna(Valor) and Valor > Limite_Superior:
                Fila_Valida = False
                Contador_Eliminados += 1
                break

        if Fila_Valida:
            Filas_Validas.append(Indice)

    print(f"  Participantes eliminados: {Contador_Eliminados}")
    Data_Frame_Resultante = Data_Frame.loc[Filas_Validas]

    return Data_Frame_Resultante


def Crear_Columnas_Cambio_Opinion(
    Diccionario_Dataframes: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:

    """
    Crea columnas de cambio de opinión para cada ítem IP comparando
    las respuestas con candidatos de izquierda/derecha versus las
    respuestas base de cada ítem.

    """

    Items_IP = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 19, 20, 22, 23, 24,
        25, 27, 28, 29, 30
    ]

    for Nombre_Df, Dataframe in Diccionario_Dataframes.items():

        for Numero_Item in Items_IP:

            Columna_Base = f'IP_Item_{Numero_Item}_Respuesta'
            Columna_Izq = f'IP_Item_{Numero_Item}_Izq_Respuesta'
            Columna_Der = f'IP_Item_{Numero_Item}_Der_Respuesta'

            Nueva_Columna_Izq = f'CO_Item_{Numero_Item}_Izq'
            Nueva_Columna_Der = f'CO_Item_{Numero_Item}_Der'

            if all(Columna in Dataframe.columns for Columna in
                   [Columna_Base, Columna_Izq, Columna_Der]):

                Base_Numerica = pd.to_numeric(
                    Dataframe[Columna_Base],
                    errors='coerce'
                )
                Izq_Numerica = pd.to_numeric(
                    Dataframe[Columna_Izq],
                    errors='coerce'
                )
                Der_Numerica = pd.to_numeric(
                    Dataframe[Columna_Der],
                    errors='coerce'
                )

                Dataframe[Nueva_Columna_Izq] = (
                    Izq_Numerica - Base_Numerica
                )
                Dataframe[Nueva_Columna_Der] = (
                    Der_Numerica - Base_Numerica
                )

    return Diccionario_Dataframes


def Crear_Columnas_Cambio_Tiempo(
    Diccionario_Dataframes: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:

    """
    Crea columnas de cambio de tiempo para cada ítem IP comparando
    los tiempos de respuesta con candidatos de izquierda/derecha versus
    los tiempos base de cada ítem.

    """

    Items_IP = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 19, 20, 22, 23, 24,
        25, 27, 28, 29, 30
    ]

    for Nombre_Df, Dataframe in Diccionario_Dataframes.items():

        for Numero_Item in Items_IP:

            Columna_Base = f'IP_Item_{Numero_Item}_Tiempo'
            Columna_Izq = f'IP_Item_{Numero_Item}_Izq_Tiempo'
            Columna_Der = f'IP_Item_{Numero_Item}_Der_Tiempo'

            Nueva_Columna_Izq = f'CT_Item_{Numero_Item}_Izq'
            Nueva_Columna_Der = f'CT_Item_{Numero_Item}_Der'

            if all(Columna in Dataframe.columns for Columna in
                   [Columna_Base, Columna_Izq, Columna_Der]):

                Base_Numerica = pd.to_numeric(
                    Dataframe[Columna_Base],
                    errors='coerce'
                )
                Izq_Numerica = pd.to_numeric(
                    Dataframe[Columna_Izq],
                    errors='coerce'
                )
                Der_Numerica = pd.to_numeric(
                    Dataframe[Columna_Der],
                    errors='coerce'
                )

                Dataframe[Nueva_Columna_Izq] = (
                    Izq_Numerica - Base_Numerica
                )
                Dataframe[Nueva_Columna_Der] = (
                    Der_Numerica - Base_Numerica
                )

    return Diccionario_Dataframes


# ============================================================================
# DICCIONARIO DE MAPEO DE NOMBRES DE COLUMNAS
# ============================================================================

def Obtener_Mapeo_Nombres_Columnas() -> Dict[str, str]:

    """
    Retorna el diccionario completo de mapeo de nombres de columnas
    desde el formato JSON expandido al formato legible.

    """

    Mapeo = {
        # Metadatos.
        'id': 'ID',
        'fase_0.url_params.PROLIFIC_PID': 'Prolific_PID',
        'fase_0.url_params.STUDY_ID': 'Study_ID',
        'fase_0.url_params.SESSION_ID': 'Session_ID',

        # Caracterización personal.
        'fase_1.caracterizacion_personal.genero': 'Genero',
        'fase_1.caracterizacion_personal.edad': 'Edad',
        'fase_1.caracterizacion_personal.nacionalidad': 'Nacionalidad',
        'fase_1.caracterizacion_personal.provincia': 'Provincia',
        'fase_1.caracterizacion_personal.e_social': 'Estrato_Social',
        'fase_1.caracterizacion_personal.niv_educativo': (
            'Nivel_Educativo'
        ),
        'fase_1.caracterizacion_personal.f_ingreso': (
            'Fuente_Ingreso'
        ),
        'fase_1.caracterizacion_personal.inmueble_res': (
            'Inmueble_Residencia'
        ),

        # Caracterización política.
        'fase_1.caracterizacion_politica.voto_2019': 'Voto_2019',
        'fase_1.caracterizacion_politica.voto_PASO_2023': (
            'Voto_PASO_2023'
        ),
        'fase_1.caracterizacion_politica.candidato_PASO_2023': (
            'Candidato_PASO_2023'
        ),
        'fase_1.caracterizacion_politica.votara_2023': 'Votara_2023',
        'fase_1.caracterizacion_politica.afiliacion_pol': (
            'Afiliacion_Politica'
        ),
        'fase_1.caracterizacion_politica.autopercepcion_0': (
            'Autopercepcion_Izq_Der'
        ),
        'fase_1.caracterizacion_politica.autopercepcion_1': (
            'Autopercepcion_Con_Pro'
        ),
        'fase_1.caracterizacion_politica.autopercepcion_2': (
            'Autopercepcion_Per_Antiper'
        ),

        # Cercanía a candidatos.
        'fase_1.caracterizacion_politica.cercania_Sergio_MassaSergio Massa': (
            'Cercania_Massa'
        ),
        'fase_1.caracterizacion_politica.cercania_Patricia_BullrichPatricia Bullrich': (
            'Cercania_Bullrich'
        ),
        'fase_1.caracterizacion_politica.cercania_Myriam_BregmanMyriam Bregman': (
            'Cercania_Bregman'
        ),
        'fase_1.caracterizacion_politica.cercania_Javier_MileiJavier Milei': (
            'Cercania_Milei'
        ),
        'fase_1.caracterizacion_politica.cercania_Juan_SchiarettiJuan Schiaretti': (
            'Cercania_Schiaretti'
        ),

        # Medios y redes sociales.
        'fase_1.caracterizacion_medios_comunicacion.medios_informacion': (
            'Medios_Informacion'
        ),
        'fase_1.caracterizacion_medios_comunicacion.red_social': (
            'Red_Social'
        ),
        'fase_1.caracterizacion_medios_comunicacion.influencia_redes_0': (
            'Influencia_Redes'
        ),
        'fase_1.caracterizacion_medios_comunicacion.medios_prensa': (
            'Medios_Prensa'
        ),
        'fase_1.caracterizacion_medios_comunicacion.influencia_prensa_0': (
            'Influencia_Prensa'
        ),

        # Clima electoral.
        'fase_2.caracterizacion_electoral.ece_item_1': 'ECE_Item_1',
        'fase_2.caracterizacion_electoral.ece_item_2': 'ECE_Item_2',
        'fase_2.caracterizacion_electoral.ece_item_3': 'ECE_Item_3',
        'fase_2.caracterizacion_electoral.ece_item_4': 'ECE_Item_4',
        'fase_2.caracterizacion_electoral.ece_item_5': 'ECE_Item_5',
        'fase_2.caracterizacion_electoral.ece_item_6': 'ECE_Item_6',
        'fase_2.caracterizacion_electoral.ece_item_7': 'ECE_Item_7',

        # Caracterización de candidatos.
        'fase_2.caracterizacion_candidatos.Sergio Massa.cdc_0': (
            'CDC_Massa_0'
        ),
        'fase_2.caracterizacion_candidatos.Sergio Massa.cdc_1': (
            'CDC_Massa_1'
        ),
        'fase_2.caracterizacion_candidatos.Sergio Massa.time': (
            'CDC_Massa_Tiempo'
        ),
        'fase_2.caracterizacion_candidatos.Patricia Bullrich.cdc_0': (
            'CDC_Bullrich_0'
        ),
        'fase_2.caracterizacion_candidatos.Patricia Bullrich.cdc_1': (
            'CDC_Bullrich_1'
        ),
        'fase_2.caracterizacion_candidatos.Patricia Bullrich.time': (
            'CDC_Bullrich_Tiempo'
        ),
        'fase_2.caracterizacion_candidatos.Juan Schiaretti.cdc_0': (
            'CDC_Schiaretti_0'
        ),
        'fase_2.caracterizacion_candidatos.Juan Schiaretti.cdc_1': (
            'CDC_Schiaretti_1'
        ),
        'fase_2.caracterizacion_candidatos.Juan Schiaretti.time': (
            'CDC_Schiaretti_Tiempo'
        ),
        'fase_2.caracterizacion_candidatos.Javier Milei.cdc_0': (
            'CDC_Milei_0'
        ),
        'fase_2.caracterizacion_candidatos.Javier Milei.cdc_1': (
            'CDC_Milei_1'
        ),
        'fase_2.caracterizacion_candidatos.Javier Milei.time': (
            'CDC_Milei_Tiempo'
        ),
        'fase_2.caracterizacion_candidatos.Myriam Bregman.cdc_0': (
            'CDC_Bregman_0'
        ),
        'fase_2.caracterizacion_candidatos.Myriam Bregman.cdc_1': (
            'CDC_Bregman_1'
        ),
        'fase_2.caracterizacion_candidatos.Myriam Bregman.time': (
            'CDC_Bregman_Tiempo'
        ),

        # Fase 4 - Cuestionario final.
        'fase_4.sentimiento_fase_final': 'Sentimiento_Fase_Final',
        'fase_4.evento_estresante': 'Evento_Estresante',
        'fase_4.sabia_del_experimento': 'Sabia_Del_Experimento',
        'fase_4.usa_sustancias': 'Usa_Sustancias',
        'fase_4.tabaco': 'Tabaco',
        'fase_4.tomaste_drogas_recreativas': 'Drogas_Recreativas',
        'fase_4.alcohol': 'Alcohol',
        'fase_4.marihuana': 'Marihuana',
        'fase_4.medicamentos': 'Medicamentos',
        'fase_4.cual_medicamento': 'Cual_Medicamento',
        'fase_4.otras_sustancias': 'Otras_Sustancias',
        'fase_4.cual_droga_recreativa': 'Cual_Droga_Recreativa',
        'fase_4.horas_sueño_promedio': 'Horas_Sueno_Promedio',
        'fase_4.sueño_noche_anterior': 'Sueno_Noche_Anterior',
    }

    # IP Items sin asociar (20 ítems × Respuesta + Tiempo).
    Numeros_IP = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 19, 20, 22, 23, 24, 25,
        27, 28, 29, 30
    ]

    for Numero in Numeros_IP:
        Mapeo[f'fase_3.IP.IP_item_{Numero}.IP_respuesta'] = (
            f'IP_Item_{Numero}_Respuesta'
        )
        Mapeo[f'fase_3.IP.IP_item_{Numero}.time'] = (
            f'IP_Item_{Numero}_Tiempo'
        )

    # IP Items asociados (20 ítems × 2 direcciones × Respuesta +
    # Candidato + Tiempo).
    for Numero in Numeros_IP:
        for Direccion in ['Izq', 'Der']:
            Mapeo[
                f'fase_3.IP_modificada.IP_item_{Numero}_{Direccion}.'
                f'IP_respuesta'
            ] = f'IP_Item_{Numero}_{Direccion}_Respuesta'

            Mapeo[
                f'fase_3.IP_modificada.IP_item_{Numero}_{Direccion}.'
                f'candidato'
            ] = f'IP_Item_{Numero}_{Direccion}_Candidato'

            Mapeo[
                f'fase_3.IP_modificada.IP_item_{Numero}_{Direccion}.'
                f'tiempo'
            ] = f'IP_Item_{Numero}_{Direccion}_Tiempo'

    return Mapeo


# ============================================================================
# PIPELINE PRINCIPAL DE PROCESAMIENTO
# ============================================================================

def Procesar_Datos_Completo(
    Ruta_Datos_Crudos: str,
    Ruta_Salida: str
) -> Dict[str, pd.DataFrame]:

    """
    Pipeline completo de procesamiento de datos del experimento.
    Combina los notebooks 1-19 en un único flujo de transformación.

    Parámetros:
    - Ruta_Datos_Crudos: Carpeta donde están los archivos CSV crudos.
    - Ruta_Salida: Carpeta donde se guardarán los Excel finales.

    Retorna:
    - Diccionario con DataFrames procesados {'Generales': df,
      'Ballotage': df}.

    """

    print("="*70)
    print("PROCESAMIENTO DE DATOS DEL EXPERIMENTO ELECTORAL 2023")
    print("="*70)
    print()

    # ========================================================================
    # 1: Armar databases de datos crudos
    # ========================================================================
    print("PASO 1: Cargando datos crudos...")
    print("-"*70)

    # Cargar archivos de Generales.
    Archivos_Generales = [
        f"Generales {i}.csv" for i in range(1, 6)
    ]

    Dfs_Generales = []
    for Archivo in Archivos_Generales:
        Ruta_Completa = os.path.join(Ruta_Datos_Crudos, Archivo)
        if os.path.exists(Ruta_Completa):
            df_Temp = pd.read_csv(Ruta_Completa)
            Dfs_Generales.append(df_Temp)
            print(f"  OK Cargado: {Archivo} ({len(df_Temp)} filas)")
        else:
            print(f"  ✗ No encontrado: {Archivo}")

    Df_Generales = pd.concat(Dfs_Generales, ignore_index=True)
    print(f"\n  Total Generales: {len(Df_Generales)} filas")

    # Cargar archivos de Ballotage.
    Archivos_Ballotage = [
        f"Ballotage {i}.csv" for i in range(1, 3)
    ]

    Dfs_Ballotage = []
    for Archivo in Archivos_Ballotage:
        Ruta_Completa = os.path.join(Ruta_Datos_Crudos, Archivo)
        if os.path.exists(Ruta_Completa):
            df_Temp = pd.read_csv(Ruta_Completa)
            Dfs_Ballotage.append(df_Temp)
            print(f"  OK Cargado: {Archivo} ({len(df_Temp)} filas)")
        else:
            print(f"  ✗ No encontrado: {Archivo}")

    Df_Ballotage = pd.concat(Dfs_Ballotage, ignore_index=True)
    print(f"\n  Total Ballotage: {len(Df_Ballotage)} filas")
    print()

    Dataframes_Procesados = [Df_Generales, Df_Ballotage]

    # ========================================================================
    # 1.1: Variables de orden de ítems sin asociar
    # ========================================================================
    print("PASO 1.1: Creando variables de orden de ítems...")
    print("-"*70)

    for i, df in enumerate(Dataframes_Procesados):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        Filas_Iniciales = len(df)

        # Filtrar solo columnas necesarias.
        df = df[['id', 'results']].copy()

        # Eliminar filas con results nulos.
        df = df[df['results'].notna()].reset_index(drop=True)

        Filas_Validas = len(df)
        print(
            f"    Filas válidas: {Filas_Validas}/{Filas_Iniciales} "
            f"({Filas_Validas/Filas_Iniciales*100:.1f}%)"
        )

        df = Crear_Variables_De_Orden_IP_Items(df)
        df = Crear_Variables_De_Orden_IP_Items_Asociados(df)
        df = Crear_Primeros_IP_Items_Asociados(df, 3)

        # Conservar solo columnas de interés.
        df = df[[
            'id', 'results',
            'Orden_IP_Items', 'Ultimo_IP_Item',
            'Orden_IP_Items_Asociados', 'Primeros_IP_Items_Asociados',
            'Ultimo_IP_Item_Asociado'
        ]]

        Dataframes_Procesados[i] = df
        print(f"  OK Variables de orden creadas para {Nombre}")

    print()

    # ========================================================================
    # 2: Formatear databases
    # ========================================================================
    print("PASO 2: Formateando databases...")
    print("-"*70)

    Dfs_Finales = []
    Mapeo_Nombres = Obtener_Mapeo_Nombres_Columnas()

    for i, df in enumerate(Dataframes_Procesados):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        # Guardar columnas de orden antes de procesar results.
        Columnas_Orden = df[[
            'id',
            'Orden_IP_Items',
            'Ultimo_IP_Item',
            'Orden_IP_Items_Asociados',
            'Ultimo_IP_Item_Asociado',
            'Primeros_IP_Items_Asociados'
        ]].copy()

        # Renombrar 'id' a 'ID' en columnas de orden.
        Columnas_Orden = Columnas_Orden.rename(columns={'id': 'ID'})

        # Procesar columna results.
        df_Expandido = Procesar_Columna_Results(df)

        # Renombrar columnas según mapeo.
        df_Expandido = df_Expandido.rename(columns=Mapeo_Nombres)

        # Convertir ID a int.
        df_Expandido['ID'] = df_Expandido['ID'].astype(int)
        Columnas_Orden['ID'] = Columnas_Orden['ID'].astype(int)

        # Hacer merge por ID (igual que el notebook).
        df_Final = pd.merge(
            df_Expandido,
            Columnas_Orden,
            on='ID',
            how='left'
        )

        print(
            f"  OK {Nombre}: {len(df_Final)} filas, "
            f"{len(df_Final.columns)} columnas"
        )
        Dfs_Finales.append(df_Final)

    print()

    # ========================================================================
    # 3: Índice de positividad
    # ========================================================================
    print("PASO 3: Calculando índice de positividad...")
    print("-"*70)

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        # Convertir columnas ECE a numérico.
        Columnas_ECE = [
            'ECE_Item_1', 'ECE_Item_2', 'ECE_Item_3', 'ECE_Item_4',
            'ECE_Item_5', 'ECE_Item_6', 'ECE_Item_7'
        ]

        for Columna in Columnas_ECE:
            df[Columna] = pd.to_numeric(df[Columna])

        # Invertir ECE_Item_5 (escala 1-5 → 5-1).
        df['ECE_Item_5_Negativo'] = df['ECE_Item_5'].map({
            1: 5, 2: 4, 3: 3, 4: 2, 5: 1
        })

        # Calcular suma de 7 ítems ECE.
        Columnas_ECE_Con_Negativo = [
            'ECE_Item_1', 'ECE_Item_2', 'ECE_Item_3', 'ECE_Item_4',
            'ECE_Item_5_Negativo', 'ECE_Item_6', 'ECE_Item_7'
        ]

        df['Indice_Positividad'] = df[
            Columnas_ECE_Con_Negativo
        ].sum(axis=1)

        Media = df['Indice_Positividad'].mean()
        Mediana = df['Indice_Positividad'].median()
        Min_Val = df['Indice_Positividad'].min()
        Max_Val = df['Indice_Positividad'].max()

        print(
            f"  OK {Nombre}: Media={Media:.2f}, Mediana={Mediana:.2f}, "
            f"Rango={Min_Val}-{Max_Val}"
        )

    print()

    # ========================================================================
    # 4: Categorías de candidatos
    # ========================================================================
    print("PASO 4: Asignando categorías electorales...")
    print("-"*70)

    Mapeo_Categorias = {
        'Sergio Massa': 'Progressivism',
        'Juan Grabois': 'Progressivism',
        'no aplica': 'No apply',
        'Myriam Bregman': 'Left wing',
        'Patricia Bullrich': 'Moderate right B',
        'Horacio Rodriguez Larreta': 'Moderate right A',
        'Juan Schiaretti': 'Centre',
        'Prefiero no decirlo': 'No response',
        'Guillermo Moreno': 'Progressivism',
        'Voto en blanco': 'Blank',
        'Gabriel Solano': 'Left wing',
        'Manuela Castañeira': 'Left wing',
        'Javier Milei': 'Right wing libertarian'
    }

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        df['Categoria_PASO_2023'] = df['Candidato_PASO_2023'].map(
            Mapeo_Categorias
        ).fillna('Other')

        Distribucion = df['Categoria_PASO_2023'].value_counts()
        print(f"\n  Distribución de categorías en {Nombre}:")
        for Cat, Cantidad in Distribucion.items():
            Porcentaje = (Cantidad / len(df)) * 100
            print(f"    {Cat}: {Cantidad} ({Porcentaje:.1f}%)")

    print()

    # ========================================================================
    # 5: Relleno de medianas de IP no asociados
    # ========================================================================
    print("PASO 5: Rellenando valores faltantes en IP Items...")
    print("-"*70)

    Numeros_IP = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 19, 20, 22, 23, 24, 25,
        27, 28, 29, 30
    ]

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        # Paso 1: Guardar valores originales (solo respuestas).
        for Numero in Numeros_IP:
            Col_Respuesta = f'IP_Item_{Numero}_Respuesta'
            if Col_Respuesta in df.columns:
                Columna_Original = f'{Col_Respuesta}_Original'
                if Columna_Original not in df.columns:
                    df[Columna_Original] = df[Col_Respuesta]

        # Paso 2: Convertir todas las columnas a numérico.
        for Numero in Numeros_IP:
            Col_Respuesta = f'IP_Item_{Numero}_Respuesta'
            Col_Tiempo = f'IP_Item_{Numero}_Tiempo'

            if Col_Respuesta in df.columns:
                df[Col_Respuesta] = pd.to_numeric(
                    df[Col_Respuesta], errors='coerce'
                )
            if Col_Tiempo in df.columns:
                df[Col_Tiempo] = pd.to_numeric(
                    df[Col_Tiempo], errors='coerce'
                )

        # Contar faltantes antes de imputar.
        Total_Faltantes_Inicial = sum(
            df[f'IP_Item_{Num}_Respuesta'].isna().sum()
            for Num in Numeros_IP
            if f'IP_Item_{Num}_Respuesta' in df.columns
        )

        # Paso 3: Imputar con mediana por categoría.
        for Numero in Numeros_IP:
            Col_Respuesta = f'IP_Item_{Numero}_Respuesta'
            Col_Tiempo = f'IP_Item_{Numero}_Tiempo'

            if Col_Respuesta in df.columns:
                df[Col_Respuesta] = df.groupby('Categoria_PASO_2023')[
                    Col_Respuesta
                ].transform(
                    lambda x: x.fillna(x.median())
                )

            if Col_Tiempo in df.columns:
                df[Col_Tiempo] = df.groupby('Categoria_PASO_2023')[
                    Col_Tiempo
                ].transform(
                    lambda x: x.fillna(x.median())
                )

        Total_Faltantes_Final = sum(
            df[f'IP_Item_{Num}_Respuesta'].isna().sum()
            for Num in Numeros_IP
            if f'IP_Item_{Num}_Respuesta' in df.columns
        )

        print(
            f"  OK {Nombre}: Valores faltantes antes="
            f"{Total_Faltantes_Inicial}, después={Total_Faltantes_Final}"
        )

    print()

    # ========================================================================
    # 6: Relleno de medianas de IP asociados
    # ========================================================================
    print("PASO 6: Rellenando IP Items asociados faltantes...")
    print("-"*70)

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        Rellenar_IP_Items_Asociados_Faltantes(df)

        print(f"  OK {Nombre}: Parejas de IP Items completadas")

    print()

    # ========================================================================
    # 7: Índice de progresismo
    # ========================================================================
    print("PASO 7: Calculando índice de progresismo...")
    print("-"*70)

    Items_Progresistas = [5, 6, 9, 11, 16, 20, 24, 25, 27, 28]

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        # Construir listas de nombres de columnas.
        Columnas_Progresistas = [
            f'IP_Item_{Num}_Respuesta' for Num in Items_Progresistas
        ]
        Columnas_Tiempos = [
            f'IP_Item_{Num}_Tiempo' for Num in Items_Progresistas
        ]

        # Invertir ítems progresistas (1→5, 5→1).
        df[Columnas_Progresistas] = df[Columnas_Progresistas].apply(
            lambda x: 6 - pd.to_numeric(x, errors='coerce')
        )

        # Convertir a float.
        df[Columnas_Progresistas] = df[Columnas_Progresistas].astype(float)
        df[Columnas_Tiempos] = df[Columnas_Tiempos].astype(float)

        # Calcular Indice de progresismo.
        df['Indice_Progresismo'] = df[Columnas_Progresistas].mean(axis=1)

        # Calcular Indice de tiempo en items de progresismo.
        df['Indice_Progresismo_Tiempo'] = df[Columnas_Tiempos].mean(axis=1)

        Media = df['Indice_Progresismo'].mean()
        Mediana = df['Indice_Progresismo'].median()

        print(
            f"  OK {Nombre}: Índice Progresismo "
            f"Media={Media:.2f}, Mediana={Mediana:.2f}"
        )

    print()

    # ========================================================================
    # 8: Índice de conservadurismo
    # ========================================================================
    print("PASO 8: Calculando índice de conservadurismo...")
    print("-"*70)

    Items_Conservadores = [3, 4, 7, 8, 10, 19, 22, 23, 29, 30]

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        # Construir listas de nombres de columnas.
        Columnas_Conservadores = [
            f'IP_Item_{Num}_Respuesta' for Num in Items_Conservadores
        ]
        Columnas_Tiempos = [
            f'IP_Item_{Num}_Tiempo' for Num in Items_Conservadores
        ]

        # Invertir ítems conservadores.
        df[Columnas_Conservadores] = df[Columnas_Conservadores].apply(
            lambda x: 6 - pd.to_numeric(x, errors='coerce')
        )

        # Convertir a float.
        df[Columnas_Conservadores] = df[Columnas_Conservadores].astype(float)
        df[Columnas_Tiempos] = df[Columnas_Tiempos].astype(float)

        # Calcular Indice de conservadurismo.
        df['Indice_Conservadurismo'] = df[Columnas_Conservadores].mean(axis=1)

        # Calcular Indice de tiempo en items de conservadurismo.
        df['Indice_Conservadurismo_Tiempo'] = df[Columnas_Tiempos].mean(axis=1)

        Media = df['Indice_Conservadurismo'].mean()
        Mediana = df['Indice_Conservadurismo'].median()

        print(
            f"  OK {Nombre}: Índice Conservadurismo "
            f"Media={Media:.2f}, Mediana={Mediana:.2f}"
        )

    print()

    # ========================================================================
    # 9: Promedios de tiempos de respuesta
    # ========================================================================
    print("PASO 9: Calculando promedios de tiempos de respuesta...")
    print("-"*70)

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        # Tiempos progresistas.
        Columnas_Pro_Izq = [
            f'IP_Item_{Num}_Izq_Tiempo' for Num in Items_Progresistas
        ]
        Columnas_Pro_Der = [
            f'IP_Item_{Num}_Der_Tiempo' for Num in Items_Progresistas
        ]

        df['Tiempos_Respuesta_Promedio_Pro_Izq'] = df[
            Columnas_Pro_Izq
        ].mean(axis=1)
        df['Tiempos_Respuesta_Promedio_Pro_Der'] = df[
            Columnas_Pro_Der
        ].mean(axis=1)
        df['Tiempos_Respuesta_Promedio_Pro'] = (
            df['Tiempos_Respuesta_Promedio_Pro_Izq'] +
            df['Tiempos_Respuesta_Promedio_Pro_Der']
        ) / 2

        # Tiempos conservadores.
        Columnas_Con_Izq = [
            f'IP_Item_{Num}_Izq_Tiempo' for Num in Items_Conservadores
        ]
        Columnas_Con_Der = [
            f'IP_Item_{Num}_Der_Tiempo' for Num in Items_Conservadores
        ]

        df['Tiempos_Respuesta_Promedio_Con_Izq'] = df[
            Columnas_Con_Izq
        ].mean(axis=1)
        df['Tiempos_Respuesta_Promedio_Con_Der'] = df[
            Columnas_Con_Der
        ].mean(axis=1)
        df['Tiempos_Respuesta_Promedio_Con'] = (
            df['Tiempos_Respuesta_Promedio_Con_Izq'] +
            df['Tiempos_Respuesta_Promedio_Con_Der']
        ) / 2

        Media_Pro = df['Tiempos_Respuesta_Promedio_Pro'].mean()
        Media_Con = df['Tiempos_Respuesta_Promedio_Con'].mean()

        print(
            f"  OK {Nombre}: Tiempo Promedio Pro={Media_Pro:.2f}s, "
            f"Con={Media_Con:.2f}s"
        )

    print()

    # ========================================================================
    # 10: Redes sociales
    # ========================================================================
    print("PASO 10: Procesando redes sociales...")
    print("-"*70)

    Redes_Sociales = [
        'twitter', 'facebook', 'instagram', 'threads', 'tiktok',
        'youtube', 'whatsapp', 'telegram'
    ]

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        # Normalizar columna Red_Social.
        df['Red_Social'] = df['Red_Social'].apply(
            lambda x: unidecode(str(x)).lower() if pd.notna(x) else x
        )

        # Crear columna binaria para cada red social.
        for Red in Redes_Sociales:
            Nombre_Columna = f"{Red}"
            df[Nombre_Columna] = df['Red_Social'].str.contains(
                Red, case=False, na=False
            ).astype(int)

        # Renombrar columnas a nombres capitalizados.
        Mapeo_Redes = {Red: Red.capitalize() for Red in Redes_Sociales}
        df.rename(columns=Mapeo_Redes, inplace=True)

        # Calcular promedio de redes por usuario.
        Redes_Capitalizadas = [Red.capitalize() for Red in Redes_Sociales]
        Promedio_Redes = sum(
            df[Red].sum() for Red in Redes_Capitalizadas
        ) / len(df)

        print(
            f"  OK {Nombre}: Promedio de redes por usuario="
            f"{Promedio_Redes:.2f}"
        )

    print()

    # ========================================================================
    # 11: Medios
    # ========================================================================
    print("PASO 11: Procesando medios de prensa...")
    print("-"*70)

    Medios_Prensa = [
        'ambito financiero',
        'prensa obrera',
        'diario universal',
        'popular',
        'izquierda diario',
        'clarin',
        'perfil',
        'pagina 12',
        'infobae',
        'el cronista',
        'la nacion',
        'tiempo argentino',
        'ninguno'
    ]

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        # Normalizar columna Medios_Prensa.
        df['Medios_Prensa'] = df['Medios_Prensa'].apply(
            lambda x: unidecode(str(x)).lower() if pd.notna(x) else x
        )

        # Crear columna binaria para cada medio.
        for Medio in Medios_Prensa:
            Nombre_Columna = f"{Medio}"
            df[Nombre_Columna] = df['Medios_Prensa'].str.contains(
                Medio, case=False, na=False
            ).astype(int)

        # Renombrar columnas a Pascal_Snake_Case.
        Mapeo_Medios = {
            Medio: Medio.title().replace(' ', '_')
            for Medio in Medios_Prensa
        }
        df.rename(columns=Mapeo_Medios, inplace=True)

        # Renombrar especialmente la columna "Ninguno".
        df.rename(
            columns={'Ninguno': 'Medios_Prensa_Ninguno'}, inplace=True
        )

        # Calcular promedio de medios por usuario (sin Ninguno).
        Medios_Pascal = [
            Medio.title().replace(' ', '_') for Medio in Medios_Prensa
            if Medio != 'ninguno'
        ]
        Promedio_Medios = sum(
            df[Medio].sum() for Medio in Medios_Pascal
        ) / len(df)

        print(
            f"  OK {Nombre}: Promedio de medios por usuario="
            f"{Promedio_Medios:.2f}"
        )

    print()

    # ========================================================================
    # 12: Agrupamientos de variables socioeconómicas
    # ========================================================================
    print(
        "PASO 12: Agrupando variables socioeconómicas y demográficas..."
    )
    print("-"*70)

    # Definir todos los mapeos.
    Region = {
        'jujuy': 'Norte',
        'salta': 'Norte',
        'tucumán': 'Norte',
        'catamarca': 'Norte',
        'santiago del estero': 'Norte',
        'formosa': 'Norte',
        'chaco': 'Norte',
        'misiones': 'Norte',
        'corrientes': 'Norte',
        'la rioja': 'Norte',
        'córdoba': 'Centro',
        'santa fe': 'Centro',
        'entre ríos': 'Centro',
        'buenos aires': 'BsAs',
        'ciudad autónoma de buenos aires': 'CABA',
        'mendoza': 'Cuyo',
        'san juan': 'Cuyo',
        'san luis': 'Cuyo',
        'río negro': 'Patagonia',
        'neuquén': 'Patagonia',
        'chubut': 'Patagonia',
        'santa cruz': 'Patagonia',
        'tierra del fuego': 'Patagonia',
        'la pampa': 'Patagonia'
    }

    AID = {
        1: 'A', 2: 'A', 3: 'A',
        4: 'B', 5: 'B', 6: 'B', 7: 'B',
        8: 'C', 9: 'C', 10: 'C'
    }

    ACP = {
        1: 'A', 2: 'A', 3: 'A',
        4: 'B', 5: 'B', 6: 'B', 7: 'B',
        8: 'C', 9: 'C', 10: 'C'
    }

    APA = {
        1: 'A', 2: 'A', 3: 'A',
        4: 'B', 5: 'B', 6: 'B', 7: 'B',
        8: 'C', 9: 'C', 10: 'C'
    }

    Mapeo_Genero = {
        'femenino': 'Femenino',
        'sexo femenino': 'Femenino',
        'masculino': 'Masculino',
        'no binario': 'Otro',
        'fluido': 'Otro',
        'género fluido ': 'Otro',
        'trans femenina': 'Otro',
        'travesti': 'Otro',
        '.': 'Otro',
        'sin especificar ': 'Otro'
    }

    Mapeo_Inmueble = {
        'propio': 'Propio',
        'propio hipotecado': 'Propio',
        'propio con hipoteca ': 'Propio',
        'propio con hipoteca uva': 'Propio',
        'en proceso de propio': 'Propio',
        'hipotecado': 'Propio',
        'herencia en vida': 'Propio',
        'propio en terreno familiar ': 'Propio',
        'alquilo': 'Alquilo',
        'alquilo a un familiar (accesible)': 'Alquilo',
        'cedido': 'Cedido',
        'prestado': 'Prestado',
        'prestado por mis suegros': 'Prestado',
        'prestado de palabra': 'Prestado',
        'familiar': 'Familiar',
        'de mis padres': 'Familiar',
        'de mi pareja': 'Familiar',
        'casa de mis padres': 'Familiar',
        'vivo con mis padres ': 'Familiar',
        'padres': 'Familiar',
        'de mí pareja': 'Familiar',
        'de mi madre': 'Familiar',
        'vivo con mi mamá ': 'Familiar',
        'en la casa de mi madre': 'Familiar',
        'de mi familia': 'Familiar',
        'con padres': 'Familiar',
        'mi madre es propietaria ': 'Familiar',
        'la casa de mis xadres.': 'Familiar',
        'la casa de mis padres': 'Familiar',
        'casa de mis abuelos': 'Familiar',
        'me mude hace poco con mis papás ': 'Familiar',
        'es de mi pareja': 'Familiar',
        'la casa de mis papas': 'Familiar',
        'lo de mí suegro ': 'Familiar',
        'de mi familia ': 'Familiar',
        'con mis padres': 'Familiar',
        'familiar compartido': 'Familiar',
        'casa de padre': 'Familiar',
        'casa de mi pareja': 'Familiar',
        'vivo con mis padres, que son propietarios ': 'Familiar',
        'es de mi madre, vivo con ella': 'Familiar',
        ' compartido con familia': 'Familiar',
        'comoartido con un familiar': 'Familiar',
        'departamento que mi papá me cede': 'Cedido',
        'ocupado': 'Otro',
        '.nb': 'Otro',
        'convivo con una persona que tiene su casa propia hace '
        'algunos meses.': 'Otro',
        'usufructo': 'Otro',
        'voy y vengo entre casa de mí padre y mí pareja': 'Otro',
        'plan de vivienda': 'Otro',
        'conviviente': 'Otro'
    }

    Mapeo_Fuente_Ingreso = {
        'trabajo': 'Trabajo',
        'trabajo informal, y cobro de asignación ': 'Trabajo',
        'panadero fermentacion natural , independiente.': 'Trabajo',
        'profesional de salud independiente ': 'Trabajo',
        'trabajador autónomo / profesional': 'Trabajo',
        'trabajo y sustento familiar. ambos.': 'Trabajo',
        'autónoma': 'Trabajo',
        'docencia ': 'Trabajo',
        'changas': 'Trabajo',
        'trabajo, ayuda familiar, etc': 'Trabajo',
        'sustento familiar y trabajo': 'Trabajo',
        'sustento familiar': 'Sustento_Familiar',
        'sustento familiar y también trabajo': 'Sustento_Familiar',
        'beca institucional': 'Beca',
        'beca estatal': 'Beca',
        'tiene una empresa': 'Empresa',
        'tengo una pequeña empresa de recreación ': 'Empresa',
        'tengo un negocio': 'Empresa',
        'empresa familiar ': 'Empresa',
        'renta': 'Renta',
        'ahorros en plazo fijo ': 'Renta',
        'plazo fijo': 'Renta',
        'inversiones': 'Renta',
        'jubilación ': 'Jubilacion',
        'jubilacion': 'Jubilacion',
        'jubilada': 'Jubilacion',
        'y …jubilada ': 'Jubilacion',
        'vivo de mi profesión y además soy jubilada': 'Jubilacion',
        'jubilada docente universitaris': 'Jubilacion',
        'jubilación docente': 'Jubilacion',
        'soy jubilada': 'Jubilacion',
        'monotributista y jubilada docente': 'Jubilacion',
        'estoy jubilada': 'Jubilacion',
        'jubilación mia y de mi marido': 'Jubilacion',
        'jubilacion y pension': 'Jubilacion',
        'pensión y jubilacion': 'Jubilacion',
        'jubilación': 'Jubilacion',
        'emprendimiento': 'Emprendimiento',
        'emprendedora': 'Emprendimiento',
        'independiente': 'Emprendimiento',
        'propio': 'Emprendimiento',
        'emprendimiento propio': 'Emprendimiento',
        'hasta hace un mes tenía un negocio que se fundio':
        'Emprendimiento',
        'monotributo ': 'Emprendimiento',
        'soy emprendedor ': 'Emprendimiento',
        'emprendedor': 'Emprendimiento',
        'emprendimiento autogestivo': 'Emprendimiento',
        'emprendimiento ': 'Emprendimiento',
        'autogestion (arte y diseño)': 'Emprendimiento',
        'tengo un emprendimiento de 14 años': 'Emprendimiento',
        'microemprendimiento ': 'Emprendimiento',
        'artesana': 'Emprendimiento',
        'ama de casa': 'Ama_De_Casa',
        'soy estudiante y ama de casa ': 'Ama_De_Casa',
        'trabaja mi marido ,yo ama de casa': 'Ama_De_Casa',
        'soy ama de casa': 'Ama_De_Casa',
        'en este momento ama de casa': 'Ama_De_Casa',
        'pensión por discapacidad ': 'Pension',
        'subsidio por enfermedad ': 'Pension',
        'retiro voluntario por discapacidad': 'Pension',
        'auh': 'Asistencia_Social',
        'suaf': 'Asistencia_Social',
        'asignación universal por hijos ': 'Asistencia_Social',
        'chagas auh ': 'Asistencia_Social',
        'programa social ': 'Asistencia_Social',
        'plan social': 'Asistencia_Social',
        'desocupado': 'Desocupado',
        'desempleada ': 'Desocupado',
        'no poseo, me encuentro desempleada': 'Desocupado',
        'no trabajo ': 'Desocupado',
        'ayudo a mi viejo en el trabajo los fines de semana':
        'Otro',
        'un poco de esto un poco de aquello': 'Otro',
        'herencia': 'Otro',
        '.': 'Otro'
    }

    Mapeo_Estrato_Social = {
        'medio': 'Medio',
        'bajo medio': 'Bajo_Medio',
        'medio alto': 'Medio_Alto',
        'bajo': 'Bajo',
        'alto': 'Alto'
    }

    Mapeo_Nivel_Educativo = {
        'universitario': 'Universitario',
        'terciario': 'Terciario',
        'secundario': 'Secundario',
        'posgrado': 'Posgrado',
        'primario': 'Primario'
    }

    Mapeo_Votara_2023 = {
        'si': 'Si',
        'no': 'No',
        'no sabe': 'No_Sabe'
    }

    Mapeo_Voto_2019 = {
        'Alberto Fernandez': 'Alberto_Fernandez',
        'Mauricio Macri': 'Mauricio_Macri',
        'Nicolás Del Caño': 'Nicolas_Del_Caño',
        'no vote': 'No_Vote',
        'Voto en blanco': 'Voto_En_Blanco',
        'Roberto Lavagna': 'Roberto_Lavagna',
        'prefiero no decirlo': 'Prefiero_No_Decirlo',
        'J.L. Espert': 'JL_Espert',
        'J. Gomez Centurión': 'J_Gomez_Centurion'
    }

    Mapeo_Voto_PASO_2023 = {
        'si': 'Si',
        'no': 'No',
        'prefiero no decirlo': 'Prefiero_No_Decirlo'
    }

    Mapeo_Candidato_PASO_2023 = {
        'Gabriel Solano': 'Gabriel_Solano',
        'Guillermo Moreno': 'Guillermo_Moreno',
        'Horacio Rodriguez Larreta': 'Horacio_Rodriguez_Larreta',
        'Javier Milei': 'Javier_Milei',
        'Jesús Escobar': 'Jesus_Escobar',
        'Juan Grabois': 'Juan_Grabois',
        'Juan Schiaretti': 'Juan_Schiaretti',
        'Julio Bárbaro': 'Julio_Barbaro',
        'Manuela Castañeira': 'Manuela_Castañeira',
        'Marcelo Ramal': 'Marcelo_Ramal',
        'Mempo Giardinelli': 'Mempo_Giardinelli',
        'Myriam Bregman': 'Myriam_Bregman',
        'Nazareno Etchepare': 'Nazareno_Etchepare',
        'Patricia Bullrich': 'Patricia_Bullrich',
        'Prefiero no decirlo': 'Prefiero_No_Decirlo',
        'Santiago Cuneo': 'Santiago_Cuneo',
        'Sergio Massa': 'Sergio_Massa',
        'Voto en blanco': 'Blanco',
        'no aplica': 'No_Aplica'
    }

    Mapeo_Categoria_PASO_2023 = {
        'Blank': 'Blank',
        'Centre': 'Centre',
        'Left wing': 'Left_Wing',
        'Moderate right A': 'Moderate_Right_A',
        'Moderate right B': 'Moderate_Right_B',
        'No apply': 'No_Apply',
        'No response': 'No_Response',
        'Other': 'Other',
        'Progressivism': 'Progressivism',
        'Right wing libertarian': 'Right_Wing_Libertarian'
    }

    Mapeo_Afiliacion_Politica = {
        'si': 'Si',
        'no': 'No'
    }

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        # Región geográfica.
        df['Region'] = df['Provincia'].str.strip().str.lower().map(
            Region
        ).fillna('Sin region')

        # Edad agrupada.
        Rangos = [0, 18, 35, 50, 70, float('inf')]
        Etiquetas = ['0_18', '19_35', '36_50', '51_70', '71_100']
        df['Edad'] = pd.to_numeric(df['Edad'], errors='coerce')
        df['Edad_Agrupada'] = pd.cut(
            df['Edad'], bins=Rangos, labels=Etiquetas, right=True
        )

        # Autopercepciones agrupadas.
        df['Autopercepcion_Izq_Der'] = pd.to_numeric(
            df['Autopercepcion_Izq_Der'], errors='coerce'
        )
        df['Autopercepcion_Izq_Der_Agrupada'] = df[
            'Autopercepcion_Izq_Der'
        ].map(AID).fillna('Otro')

        df['Autopercepcion_Con_Pro'] = pd.to_numeric(
            df['Autopercepcion_Con_Pro'], errors='coerce'
        )
        df['Autopercepcion_Con_Pro_Agrupada'] = df[
            'Autopercepcion_Con_Pro'
        ].map(ACP).fillna('Otro')

        df['Autopercepcion_Per_Antiper'] = pd.to_numeric(
            df['Autopercepcion_Per_Antiper'], errors='coerce'
        )
        df['Autopercepcion_Per_Antiper_Agrupada'] = df[
            'Autopercepcion_Per_Antiper'
        ].map(APA).fillna('Otro')

        # Aplicar mapeos de variables.
        if 'Genero' in df.columns:
            df['Genero'] = df['Genero'].map(
                Mapeo_Genero
            ).fillna('Otro')

        if 'Estrato_Social' in df.columns:
            df['Estrato_Social'] = df['Estrato_Social'].map(
                Mapeo_Estrato_Social
            ).fillna('Otro')

        if 'Inmueble_Residencia' in df.columns:
            df['Inmueble_Residencia'] = df['Inmueble_Residencia'].map(
                Mapeo_Inmueble
            ).fillna('Otro')

        if 'Nivel_Educativo' in df.columns:
            df['Nivel_Educativo'] = df['Nivel_Educativo'].map(
                Mapeo_Nivel_Educativo
            ).fillna('Otro')

        if 'Votara_2023' in df.columns:
            df['Votara_2023'] = df['Votara_2023'].map(
                Mapeo_Votara_2023
            ).fillna('Otro')

        if 'Voto_2019' in df.columns:
            df['Voto_2019'] = df['Voto_2019'].map(
                Mapeo_Voto_2019
            ).fillna('Otro')

        if 'Voto_PASO_2023' in df.columns:
            df['Voto_PASO_2023'] = df['Voto_PASO_2023'].map(
                Mapeo_Voto_PASO_2023
            ).fillna('Otro')

        if 'Fuente_Ingreso' in df.columns:
            df['Fuente_Ingreso'] = df['Fuente_Ingreso'].map(
                Mapeo_Fuente_Ingreso
            ).fillna('Otro')

        if 'Candidato_PASO_2023' in df.columns:
            df['Candidato_PASO_2023'] = df['Candidato_PASO_2023'].map(
                Mapeo_Candidato_PASO_2023
            ).fillna('Otro')

        if 'Categoria_PASO_2023' in df.columns:
            df['Categoria_PASO_2023'] = df['Categoria_PASO_2023'].map(
                Mapeo_Categoria_PASO_2023
            ).fillna('Otro')

        if 'Afiliacion_Politica' in df.columns:
            df['Afiliacion_Politica'] = df['Afiliacion_Politica'].map(
                Mapeo_Afiliacion_Politica
            ).fillna('Otro')

        print(f"  OK {Nombre}: Variables agrupadas creadas")

    print()

    # ========================================================================
    # 13: Crear dummies
    # ========================================================================
    print("PASO 13: Creando variables dummy...")
    print("-"*70)

    Variables_A_Dummear = [
        'Genero',
        'Estrato_Social',
        'Nivel_Educativo',
        'Fuente_Ingreso',
        'Inmueble_Residencia',
        'Voto_2019',
        'Voto_PASO_2023',
        'Candidato_PASO_2023',
        'Afiliacion_Politica',
        'Edad_Agrupada',
        'Votara_2023',
        'Autopercepcion_Izq_Der_Agrupada',
        'Autopercepcion_Con_Pro_Agrupada',
        'Autopercepcion_Per_Antiper_Agrupada',
        'Region',
        'Categoria_PASO_2023'
    ]

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        Columnas_Iniciales = len(df.columns)

        # Verificar qué variables existen en este DataFrame.
        Variables_Existentes = [
            var for var in Variables_A_Dummear if var in df.columns
        ]

        # Crear dummies para las variables existentes.
        if Variables_Existentes:
            for Variable in Variables_Existentes:
                # Crear variables dummy.
                Dummies = pd.get_dummies(
                    df[Variable], prefix=Variable, dummy_na=False
                )

                # Agregar las columnas dummy al DataFrame.
                df = pd.concat([df, Dummies], axis=1)

        Columnas_Finales = len(df.columns)
        Dummies_Creadas = Columnas_Finales - Columnas_Iniciales

        print(
            f"  OK {Nombre}: {Dummies_Creadas} variables dummy creadas "
            f"({Columnas_Finales} columnas totales)"
        )

        # Actualizar el DataFrame en la lista.
        Dfs_Finales[i] = df

    print()

    # ========================================================================
    # 14: Ordenamiento de columnas
    # ========================================================================
    print("PASO 14: Ordenando columnas...")
    print("-"*70)

    # Lista explícita de orden de columnas (igual que notebook 14).
    Orden_Columnas = [
        'ID',
        # Demográficos - Edad.
        'Edad', 'Edad_Agrupada', 'Edad_Agrupada_0_18',
        'Edad_Agrupada_19_35', 'Edad_Agrupada_36_50',
        'Edad_Agrupada_51_70', 'Edad_Agrupada_71_100',
        # Género.
        'Genero', 'Genero_Femenino', 'Genero_Masculino', 'Genero_Otro',
        # Estrato social.
        'Estrato_Social', 'Estrato_Social_Alto', 'Estrato_Social_Bajo',
        'Estrato_Social_Bajo_Medio', 'Estrato_Social_Medio',
        'Estrato_Social_Medio_Alto',
        # Nivel educativo.
        'Nivel_Educativo', 'Nivel_Educativo_Primario',
        'Nivel_Educativo_Secundario', 'Nivel_Educativo_Terciario',
        'Nivel_Educativo_Universitario', 'Nivel_Educativo_Posgrado',
        # Fuente de ingreso.
        'Fuente_Ingreso', 'Fuente_Ingreso_Trabajo',
        'Fuente_Ingreso_Ama_De_Casa', 'Fuente_Ingreso_Asistencia_Social',
        'Fuente_Ingreso_Beca', 'Fuente_Ingreso_Desocupado',
        'Fuente_Ingreso_Emprendimiento', 'Fuente_Ingreso_Empresa',
        'Fuente_Ingreso_Jubilacion', 'Fuente_Ingreso_Pension',
        'Fuente_Ingreso_Renta', 'Fuente_Ingreso_Sustento_Familiar',
        'Fuente_Ingreso_Otro',
        # Inmueble de residencia.
        'Inmueble_Residencia', 'Inmueble_Residencia_Alquilo',
        'Inmueble_Residencia_Cedido', 'Inmueble_Residencia_Familiar',
        'Inmueble_Residencia_Prestado', 'Inmueble_Residencia_Propio',
        'Inmueble_Residencia_Otro',
        # Residencia.
        'Provincia', 'Region', 'Region_BsAs', 'Region_CABA',
        'Region_Centro', 'Region_Cuyo', 'Region_Norte', 'Region_Patagonia',
        # Datos demográficos básicos.
        'Nacionalidad',
        # Electorales - Voto 2019.
        'Voto_2019', 'Voto_2019_Alberto_Fernandez', 'Voto_2019_JL_Espert',
        'Voto_2019_J_Gomez_Centurion', 'Voto_2019_Mauricio_Macri',
        'Voto_2019_Nicolas_Del_Caño', 'Voto_2019_Roberto_Lavagna',
        'Voto_2019_Voto_En_Blanco', 'Voto_2019_No_Vote',
        'Voto_2019_Prefiero_No_Decirlo',
        # Voto PASO 2023.
        'Voto_PASO_2023', 'Voto_PASO_2023_Si', 'Voto_PASO_2023_No',
        'Voto_PASO_2023_Prefiero_No_Decirlo',
        # Candidato PASO 2023.
        'Candidato_PASO_2023', 'Candidato_PASO_2023_Sergio_Massa',
        'Candidato_PASO_2023_Javier_Milei', 'Candidato_PASO_2023_Myriam_Bregman',
        'Candidato_PASO_2023_Horacio_Rodriguez_Larreta',
        'Candidato_PASO_2023_Patricia_Bullrich',
        'Candidato_PASO_2023_Juan_Grabois', 'Candidato_PASO_2023_Juan_Schiaretti',
        'Candidato_PASO_2023_Gabriel_Solano', 'Candidato_PASO_2023_Guillermo_Moreno',
        'Candidato_PASO_2023_Jesus_Escobar', 'Candidato_PASO_2023_Julio_Barbaro',
        'Candidato_PASO_2023_Manuela_Castañeira', 'Candidato_PASO_2023_Marcelo_Ramal',
        'Candidato_PASO_2023_Mempo_Giardinelli',
        'Candidato_PASO_2023_Nazareno_Etchepare',
        'Candidato_PASO_2023_Santiago_Cuneo', 'Candidato_PASO_2023_Blanco',
        'Candidato_PASO_2023_Prefiero_No_Decirlo', 'Candidato_PASO_2023_No_Aplica',
        # Categoría PASO 2023.
        'Categoria_PASO_2023', 'Categoria_PASO_2023_Blank',
        'Categoria_PASO_2023_Centre', 'Categoria_PASO_2023_Left_Wing',
        'Categoria_PASO_2023_Moderate_Right_A', 'Categoria_PASO_2023_Moderate_Right_B',
        'Categoria_PASO_2023_Progressivism', 'Categoria_PASO_2023_Right_Wing_Libertarian',
        'Categoria_PASO_2023_Other', 'Categoria_PASO_2023_No_Response',
        'Categoria_PASO_2023_No_Apply',
        # Votará 2023.
        'Votara_2023', 'Votara_2023_Si', 'Votara_2023_No', 'Votara_2023_No_Sabe',
        # Políticos - Afiliación política.
        'Afiliacion_Politica', 'Afiliacion_Politica_Si', 'Afiliacion_Politica_No',
        # Autopercepciones.
        'Autopercepcion_Izq_Der', 'Autopercepcion_Izq_Der_Agrupada',
        'Autopercepcion_Izq_Der_Agrupada_A', 'Autopercepcion_Izq_Der_Agrupada_B',
        'Autopercepcion_Izq_Der_Agrupada_C',
        'Autopercepcion_Con_Pro', 'Autopercepcion_Con_Pro_Agrupada',
        'Autopercepcion_Con_Pro_Agrupada_A', 'Autopercepcion_Con_Pro_Agrupada_B',
        'Autopercepcion_Con_Pro_Agrupada_C',
        'Autopercepcion_Per_Antiper', 'Autopercepcion_Per_Antiper_Agrupada',
        'Autopercepcion_Per_Antiper_Agrupada_A',
        'Autopercepcion_Per_Antiper_Agrupada_B',
        'Autopercepcion_Per_Antiper_Agrupada_C',
        # Cercanía con candidatos.
        'Cercania_Massa', 'Cercania_Bullrich', 'Cercania_Bregman',
        'Cercania_Milei', 'Cercania_Schiaretti',
        # Redes sociales.
        'Influencia_Redes', 'Red_Social',
        'Twitter', 'Facebook', 'Instagram', 'Threads', 'Tiktok',
        'Youtube', 'Whatsapp', 'Telegram',
        # Medios de prensa.
        'Influencia_Prensa', 'Medios_Informacion', 'Medios_Prensa',
        'Ambito_Financiero', 'Prensa_Obrera', 'Diario_Universal', 'Popular',
        'Izquierda_Diario', 'Clarin', 'Perfil', 'Pagina_12', 'Infobae',
        'El_Cronista', 'La_Nacion', 'Tiempo_Argentino', 'Medios_Prensa_Ninguno',
        # Clima electoral - ECE.
        'ECE_Item_1', 'ECE_Item_2', 'ECE_Item_3', 'ECE_Item_4',
        'ECE_Item_5', 'ECE_Item_6', 'ECE_Item_7', 'ECE_Item_5_Negativo',
        # Caracterización de candidatos - CDC.
        'CDC_Massa_0', 'CDC_Massa_1', 'CDC_Massa_Tiempo',
        'CDC_Bullrich_0', 'CDC_Bullrich_1', 'CDC_Bullrich_Tiempo',
        'CDC_Schiaretti_0', 'CDC_Schiaretti_1', 'CDC_Schiaretti_Tiempo',
        'CDC_Milei_0', 'CDC_Milei_1', 'CDC_Milei_Tiempo',
        'CDC_Bregman_0', 'CDC_Bregman_1', 'CDC_Bregman_Tiempo',
        # IP Items.
        'IP_Item_3_Respuesta', 'IP_Item_3_Tiempo',
        'IP_Item_3_Izq_Candidato', 'IP_Item_3_Izq_Respuesta', 'IP_Item_3_Izq_Tiempo',
        'IP_Item_3_Der_Candidato', 'IP_Item_3_Der_Respuesta', 'IP_Item_3_Der_Tiempo',
        'IP_Item_4_Respuesta', 'IP_Item_4_Tiempo',
        'IP_Item_4_Izq_Candidato', 'IP_Item_4_Izq_Respuesta', 'IP_Item_4_Izq_Tiempo',
        'IP_Item_4_Der_Candidato', 'IP_Item_4_Der_Respuesta', 'IP_Item_4_Der_Tiempo',
        'IP_Item_5_Respuesta', 'IP_Item_5_Tiempo',
        'IP_Item_5_Izq_Candidato', 'IP_Item_5_Izq_Respuesta', 'IP_Item_5_Izq_Tiempo',
        'IP_Item_5_Der_Candidato', 'IP_Item_5_Der_Respuesta', 'IP_Item_5_Der_Tiempo',
        'IP_Item_6_Respuesta', 'IP_Item_6_Tiempo',
        'IP_Item_6_Izq_Candidato', 'IP_Item_6_Izq_Respuesta', 'IP_Item_6_Izq_Tiempo',
        'IP_Item_6_Der_Candidato', 'IP_Item_6_Der_Respuesta', 'IP_Item_6_Der_Tiempo',
        'IP_Item_7_Respuesta', 'IP_Item_7_Tiempo',
        'IP_Item_7_Izq_Candidato', 'IP_Item_7_Izq_Respuesta', 'IP_Item_7_Izq_Tiempo',
        'IP_Item_7_Der_Candidato', 'IP_Item_7_Der_Respuesta', 'IP_Item_7_Der_Tiempo',
        'IP_Item_8_Respuesta', 'IP_Item_8_Tiempo',
        'IP_Item_8_Izq_Candidato', 'IP_Item_8_Izq_Respuesta', 'IP_Item_8_Izq_Tiempo',
        'IP_Item_8_Der_Candidato', 'IP_Item_8_Der_Respuesta', 'IP_Item_8_Der_Tiempo',
        'IP_Item_9_Respuesta', 'IP_Item_9_Tiempo',
        'IP_Item_9_Izq_Candidato', 'IP_Item_9_Izq_Respuesta', 'IP_Item_9_Izq_Tiempo',
        'IP_Item_9_Der_Candidato', 'IP_Item_9_Der_Respuesta', 'IP_Item_9_Der_Tiempo',
        'IP_Item_10_Respuesta', 'IP_Item_10_Tiempo',
        'IP_Item_10_Izq_Candidato', 'IP_Item_10_Izq_Respuesta', 'IP_Item_10_Izq_Tiempo',
        'IP_Item_10_Der_Candidato', 'IP_Item_10_Der_Respuesta', 'IP_Item_10_Der_Tiempo',
        'IP_Item_11_Respuesta', 'IP_Item_11_Tiempo',
        'IP_Item_11_Izq_Candidato', 'IP_Item_11_Izq_Respuesta', 'IP_Item_11_Izq_Tiempo',
        'IP_Item_11_Der_Candidato', 'IP_Item_11_Der_Respuesta', 'IP_Item_11_Der_Tiempo',
        'IP_Item_16_Respuesta', 'IP_Item_16_Tiempo',
        'IP_Item_16_Izq_Candidato', 'IP_Item_16_Izq_Respuesta', 'IP_Item_16_Izq_Tiempo',
        'IP_Item_16_Der_Candidato', 'IP_Item_16_Der_Respuesta', 'IP_Item_16_Der_Tiempo',
        'IP_Item_19_Respuesta', 'IP_Item_19_Tiempo',
        'IP_Item_19_Izq_Candidato', 'IP_Item_19_Izq_Respuesta', 'IP_Item_19_Izq_Tiempo',
        'IP_Item_19_Der_Candidato', 'IP_Item_19_Der_Respuesta', 'IP_Item_19_Der_Tiempo',
        'IP_Item_20_Respuesta', 'IP_Item_20_Tiempo',
        'IP_Item_20_Izq_Candidato', 'IP_Item_20_Izq_Respuesta', 'IP_Item_20_Izq_Tiempo',
        'IP_Item_20_Der_Candidato', 'IP_Item_20_Der_Respuesta', 'IP_Item_20_Der_Tiempo',
        'IP_Item_22_Respuesta', 'IP_Item_22_Tiempo',
        'IP_Item_22_Izq_Candidato', 'IP_Item_22_Izq_Respuesta', 'IP_Item_22_Izq_Tiempo',
        'IP_Item_22_Der_Candidato', 'IP_Item_22_Der_Respuesta', 'IP_Item_22_Der_Tiempo',
        'IP_Item_23_Respuesta', 'IP_Item_23_Tiempo',
        'IP_Item_23_Izq_Candidato', 'IP_Item_23_Izq_Respuesta', 'IP_Item_23_Izq_Tiempo',
        'IP_Item_23_Der_Candidato', 'IP_Item_23_Der_Respuesta', 'IP_Item_23_Der_Tiempo',
        'IP_Item_24_Respuesta', 'IP_Item_24_Tiempo',
        'IP_Item_24_Izq_Candidato', 'IP_Item_24_Izq_Respuesta', 'IP_Item_24_Izq_Tiempo',
        'IP_Item_24_Der_Candidato', 'IP_Item_24_Der_Respuesta', 'IP_Item_24_Der_Tiempo',
        'IP_Item_25_Respuesta', 'IP_Item_25_Tiempo',
        'IP_Item_25_Izq_Candidato', 'IP_Item_25_Izq_Respuesta', 'IP_Item_25_Izq_Tiempo',
        'IP_Item_25_Der_Candidato', 'IP_Item_25_Der_Respuesta', 'IP_Item_25_Der_Tiempo',
        'IP_Item_27_Respuesta', 'IP_Item_27_Tiempo',
        'IP_Item_27_Izq_Candidato', 'IP_Item_27_Izq_Respuesta', 'IP_Item_27_Izq_Tiempo',
        'IP_Item_27_Der_Candidato', 'IP_Item_27_Der_Respuesta', 'IP_Item_27_Der_Tiempo',
        'IP_Item_28_Respuesta', 'IP_Item_28_Tiempo',
        'IP_Item_28_Izq_Candidato', 'IP_Item_28_Izq_Respuesta', 'IP_Item_28_Izq_Tiempo',
        'IP_Item_28_Der_Candidato', 'IP_Item_28_Der_Respuesta', 'IP_Item_28_Der_Tiempo',
        'IP_Item_29_Respuesta', 'IP_Item_29_Tiempo',
        'IP_Item_29_Izq_Candidato', 'IP_Item_29_Izq_Respuesta', 'IP_Item_29_Izq_Tiempo',
        'IP_Item_29_Der_Candidato', 'IP_Item_29_Der_Respuesta', 'IP_Item_29_Der_Tiempo',
        'IP_Item_30_Respuesta', 'IP_Item_30_Tiempo',
        'IP_Item_30_Izq_Candidato', 'IP_Item_30_Izq_Respuesta', 'IP_Item_30_Izq_Tiempo',
        'IP_Item_30_Der_Candidato', 'IP_Item_30_Der_Respuesta', 'IP_Item_30_Der_Tiempo',
        # Orden de IP Items.
        'Orden_IP_Items', 'Ultimo_IP_Item', 'Orden_IP_Items_Asociados',
        'Primeros_IP_Items_Asociados', 'Ultimo_IP_Item_Asociado',
        # Variables calculadas.
        'Indice_Positividad', 'Indice_Progresismo', 'Indice_Progresismo_Tiempo',
        'Indice_Conservadurismo', 'Indice_Conservadurismo_Tiempo',
        # Tiempos de respuesta promedio.
        'Tiempos_Respuesta_Promedio_Pro_Izq', 'Tiempos_Respuesta_Promedio_Pro_Der',
        'Tiempos_Respuesta_Promedio_Con_Izq', 'Tiempos_Respuesta_Promedio_Con_Der',
        'Tiempos_Respuesta_Promedio_Pro', 'Tiempos_Respuesta_Promedio_Con',
        # Control.
        'Sentimiento_Fase_Final', 'Evento_Estresante', 'Sabia_Del_Experimento',
        # Uso de sustancias.
        'Usa_Sustancias', 'Tabaco', 'Alcohol', 'Marihuana',
        'Drogas_Recreativas', 'Cual_Droga_Recreativa',
        'Medicamentos', 'Cual_Medicamento', 'Otras_Sustancias',
        # Sueño.
        'Horas_Sueno_Promedio', 'Sueno_Noche_Anterior'
    ]

    # Mapeos para renombrar columnas.
    Mapeo_Redes_Sociales = {
        'Twitter': 'Red_Social_Twitter',
        'Facebook': 'Red_Social_Facebook',
        'Instagram': 'Red_Social_Instagram',
        'Threads': 'Red_Social_Threads',
        'Tiktok': 'Red_Social_Tiktok',
        'Youtube': 'Red_Social_Youtube',
        'Whatsapp': 'Red_Social_Whatsapp',
        'Telegram': 'Red_Social_Telegram'
    }

    Mapeo_Medios_Prensa = {
        'Ambito_Financiero': 'Medios_Prensa_Ambito_Financiero',
        'Prensa_Obrera': 'Medios_Prensa_Prensa_Obrera',
        'Diario_Universal': 'Medios_Prensa_Diario_Universal',
        'Popular': 'Medios_Prensa_Popular',
        'Izquierda_Diario': 'Medios_Prensa_Izquierda_Diario',
        'Clarin': 'Medios_Prensa_Clarin',
        'Perfil': 'Medios_Prensa_Perfil',
        'Pagina_12': 'Medios_Prensa_Pagina_12',
        'Infobae': 'Medios_Prensa_Infobae',
        'El_Cronista': 'Medios_Prensa_El_Cronista',
        'La_Nacion': 'Medios_Prensa_La_Nacion',
        'Tiempo_Argentino': 'Medios_Prensa_Tiempo_Argentino'
    }

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        # Filtrar solo las columnas que existen en este DataFrame.
        Columnas_Existentes = [
            Columna for Columna in Orden_Columnas if Columna in df.columns
        ]

        # Reordenar el DataFrame.
        df = df[Columnas_Existentes]

        # Renombrar redes sociales.
        df.rename(columns=Mapeo_Redes_Sociales, inplace=True)

        # Renombrar medios de prensa.
        df.rename(columns=Mapeo_Medios_Prensa, inplace=True)

        Dfs_Finales[i] = df

        print(
            f"  OK {Nombre}: {len(df.columns)} columnas ordenadas"
        )

    print()

    # ========================================================================
    # 14.5: Analisis de warm-up
    # ========================================================================
    print("PASO 14.5: Analisis de warm-up...")
    print("-"*70)

    dfs_Finales = {
        'Generales': Dfs_Finales[0],
        'Ballotage': Dfs_Finales[1]
    }

    Items_IP = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 19, 20, 22, 23, 24,
        25, 27, 28, 29, 30
    ]

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{Nombre_df}:")
        CO_Calculados = 0

        for Numero_Item in Items_IP:
            Columna_Base = f'IP_Item_{Numero_Item}_Respuesta'
            Columna_Izq = f'IP_Item_{Numero_Item}_Izq_Respuesta'
            Columna_Der = f'IP_Item_{Numero_Item}_Der_Respuesta'

            Nueva_Columna_Izq = f'CO_Item_{Numero_Item}_Izq'
            Nueva_Columna_Der = f'CO_Item_{Numero_Item}_Der'

            if all(Col in df.columns for Col in [
                    Columna_Base, Columna_Izq, Columna_Der
            ]):
                Base_Numerica = pd.to_numeric(
                    df[Columna_Base], errors='coerce'
                )
                Izq_Numerica = pd.to_numeric(
                    df[Columna_Izq], errors='coerce'
                )
                Der_Numerica = pd.to_numeric(
                    df[Columna_Der], errors='coerce'
                )

                dfs_Finales[Nombre_df][Nueva_Columna_Izq] = (
                    Izq_Numerica - Base_Numerica
                )
                dfs_Finales[Nombre_df][Nueva_Columna_Der] = (
                    Der_Numerica - Base_Numerica
                )

                CO_Calculados += 2

        print(f"  CO provisionales calculados: {CO_Calculados} columnas")

    print("\n" + "="*70)

    print("CREACION DE CO_CRONOLOGICOS")
    print("="*70)

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{Nombre_df}:")

        Lista_CO_Cronologicos = []

        for _, Fila in df.iterrows():
            Orden_IP_Items_Asociados = Fila['Orden_IP_Items_Asociados']
            CO_Cronologicos_Participante = []

            if (isinstance(Orden_IP_Items_Asociados, list) and
                    len(Orden_IP_Items_Asociados) > 0):

                Primeros_10_Items = Orden_IP_Items_Asociados[:10]

                for Item in Primeros_10_Items:
                    Nombre_Columna_CO = f'CO_Item_{Item}'

                    if Nombre_Columna_CO in df.columns:
                        Valor_CO = Fila[Nombre_Columna_CO]
                        CO_Cronologicos_Participante.append(Valor_CO)
                    else:
                        CO_Cronologicos_Participante.append(np.nan)

            if len(CO_Cronologicos_Participante) > 0:
                Lista_CO_Cronologicos.append(CO_Cronologicos_Participante)
            else:
                Lista_CO_Cronologicos.append(np.nan)

        dfs_Finales[Nombre_df]['CO_Cronologicos'] = Lista_CO_Cronologicos

        N_Validos = sum(
            1 for x in Lista_CO_Cronologicos
            if not (isinstance(x, float) and np.isnan(x))
        )
        print("  CO_Cronologicos creados: "
              f"{N_Validos}/{len(df)} participantes")

        if N_Validos > 0:
            Ejemplo_Index = next(
                i for i, x in enumerate(Lista_CO_Cronologicos)
                if not (isinstance(x, float) and np.isnan(x))
            )
            print(f"\n  Ejemplo (ID {df.iloc[Ejemplo_Index]['ID']}):")
            print("    Orden_IP_Items_Asociados[:10]: "
                  f"{df.iloc[Ejemplo_Index]['Orden_IP_Items_Asociados'][:10]}")
            print("    CO_Cronologicos (primeros 5): "
                  f"{Lista_CO_Cronologicos[Ejemplo_Index][:5]}")

    print("\n" + "="*70)

    print("ANALISIS DESCRIPTIVO DE CO CRONOLOGICOS")
    print("="*70)

    Resultados_Analisis = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{Nombre_df}:")
        print("-" * 70)

        Matriz_CO = []

        for CO_Lista in df['CO_Cronologicos']:
            if isinstance(CO_Lista, list) and len(CO_Lista) == 10:
                Matriz_CO.append(CO_Lista)

        Matriz_CO = np.array(Matriz_CO, dtype=float)

        print("  Participantes con CO cronologicos completos: "
              f"{len(Matriz_CO)}")
        print("\n  Estadisticas por posicion cronologica:")
        print("  {'Posicion':<12} {'N':<8} {'Media':<10} {'Mediana':<10} {'DE':<10} {'|CO| Media':<12}")
        print("  " + "-"*68)

        Estadisticas_Por_Posicion = []

        for Posicion in range(10):
            CO_Posicion = Matriz_CO[:, Posicion]
            CO_Posicion_Validos = CO_Posicion[~np.isnan(CO_Posicion)]

            if len(CO_Posicion_Validos) > 0:
                Media = np.mean(CO_Posicion_Validos)
                Mediana = np.median(CO_Posicion_Validos)
                DE = np.std(CO_Posicion_Validos, ddof=1)
                Media_Absoluta = np.mean(np.abs(CO_Posicion_Validos))
                N = len(CO_Posicion_Validos)

                Estadisticas_Por_Posicion.append({
                    'Posicion': Posicion + 1,
                    'N': N,
                    'Media': Media,
                    'Mediana': Mediana,
                    'DE': DE,
                    'Media_Absoluta': Media_Absoluta
                })

                print(
                    f"  {Posicion + 1:<12} {N:<8} {Media:<10.3f} "
                    f"{Mediana:<10.3f} {DE:<10.3f} "
                    f"{Media_Absoluta:<12.3f}"
                )
            else:
                print(f"  {Posicion + 1:<12} 0        -          -          -          -")

        Resultados_Analisis[Nombre_df] = Estadisticas_Por_Posicion

    print("\n" + "="*70)

    print("ANALISIS DE VALORES EXTREMOS (|CO| = 3 o 4)")
    print("="*70)

    Resultados_Extremos = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{Nombre_df}:")
        print("-" * 70)

        Matriz_CO = []

        for CO_Lista in df['CO_Cronologicos']:
            if isinstance(CO_Lista, list) and len(CO_Lista) == 10:
                Matriz_CO.append(CO_Lista)

        Matriz_CO = np.array(Matriz_CO, dtype=float)

        print("  Porcentaje de valores extremos por posicion cronologica:")
        print("  {'Posicion':<12} {'N Total':<10} {'|CO|=3':<10} {'|CO|=4':<10} {'|CO|>=3':<10} {'% Extremos':<12}")
        print("  " + "-"*72)

        Estadisticas_Extremos = []

        for Posicion in range(10):
            CO_Posicion = Matriz_CO[:, Posicion]
            CO_Posicion_Validos = CO_Posicion[~np.isnan(CO_Posicion)]

            if len(CO_Posicion_Validos) > 0:
                CO_Absolutos = np.abs(CO_Posicion_Validos)
                N_Total = len(CO_Posicion_Validos)
                N_Igual_3 = np.sum(CO_Absolutos == 3)
                N_Igual_4 = np.sum(CO_Absolutos == 4)
                N_Mayor_Igual_3 = np.sum(CO_Absolutos >= 3)
                Porcentaje_Extremos = (N_Mayor_Igual_3 / N_Total) * 100

                Estadisticas_Extremos.append({
                    'Posicion': Posicion + 1,
                    'N_Total': N_Total,
                    'N_Igual_3': N_Igual_3,
                    'N_Igual_4': N_Igual_4,
                    'N_Mayor_Igual_3': N_Mayor_Igual_3,
                    'Porcentaje_Extremos': Porcentaje_Extremos
                })

                print(
                    f"  {Posicion + 1:<12} {N_Total:<10} {N_Igual_3:<10} "
                    f"{N_Igual_4:<10} {N_Mayor_Igual_3:<10} "
                    f"{Porcentaje_Extremos:<12.2f}%"
                )
            else:
                print(f"  {Posicion + 1:<12} 0          -          -          -          -")

        Resultados_Extremos[Nombre_df] = Estadisticas_Extremos

        if len(Estadisticas_Extremos) >= 10:
            Porcentaje_Primeros_3 = np.mean([
                stat['Porcentaje_Extremos']
                for stat in Estadisticas_Extremos[:3]
            ])
            Porcentaje_Resto = np.mean([
                stat['Porcentaje_Extremos']
                for stat in Estadisticas_Extremos[3:]
            ])

            print("\n  Resumen:")
            print("    Promedio % extremos (Posiciones 1-3): "
                  f"{Porcentaje_Primeros_3:.2f}%")
            print("    Promedio % extremos (Posiciones 4-10): "
                  f"{Porcentaje_Resto:.2f}%")
            print("    Reduccion: "
                  f"{Porcentaje_Primeros_3 - Porcentaje_Resto:.2f} puntos porcentuales")

    print("\n" + "="*70)

    print("GRAFICO DE EVOLUCION DE CO CRONOLOGICOS")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if 'Generales' in Resultados_Analisis and len(Resultados_Analisis['Generales']) > 0:
        Estadisticas = Resultados_Analisis['Generales']
        Posiciones = [stat['Posicion'] for stat in Estadisticas]
        Medias_Abs = [stat['Media_Absoluta'] for stat in Estadisticas]

        axes[0].plot(
            Posiciones, Medias_Abs, marker='o', linewidth=2,
            markersize=8, color='#2E86AB'
        )
        axes[0].set_title('Generales', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Posicion cronologica', fontsize=12)
        axes[0].set_ylabel('CO (Promedio)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(range(1, 11))

        axes[0].axvline(
            x=3.5, color='red', linestyle='--', linewidth=2,
            alpha=0.7, label='Limite (3 items unicos)'
        )
        axes[0].legend()

    if 'Ballotage' in Resultados_Analisis and len(Resultados_Analisis['Ballotage']) > 0:
        Estadisticas = Resultados_Analisis['Ballotage']
        Posiciones = [stat['Posicion'] for stat in Estadisticas]
        Medias_Abs = [stat['Media_Absoluta'] for stat in Estadisticas]

        axes[1].plot(
            Posiciones, Medias_Abs, marker='o', linewidth=2,
            markersize=8, color='#A23B72'
        )
        axes[1].set_title('Ballotage', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Posicion cronologica', fontsize=12)
        axes[1].set_ylabel('CO (Promedio)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(range(1, 11))

        axes[1].axvline(
            x=3.5, color='red', linestyle='--', linewidth=2,
            alpha=0.7, label='Limite (3 items unicos)'
        )
        axes[1].legend()

    plt.suptitle('', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    print('\nGrafico generado exitosamente.')
    print('\n' + '='*70)

    print('LIMPIEZA DE COLUMNAS CO PROVISIONALES')
    print('='*70)

    for Nombre_df, df in dfs_Finales.items():
        Columnas_CO = [
            col for col in df.columns if col.startswith('CO_Item_')
        ]
        Columnas_A_Eliminar = [
            col for col in Columnas_CO if col != 'CO_Cronologicos'
        ]

        if len(Columnas_A_Eliminar) > 0:
            dfs_Finales[Nombre_df] = df.drop(columns=Columnas_A_Eliminar)
            print(f"\n{Nombre_df}: {len(Columnas_A_Eliminar)} columnas CO eliminadas")

    print('\n' + '='*70)

    Ruta_Controles = os.path.join(
        os.path.dirname(Ruta_Datos_Crudos), 'Controles'
    )
    os.makedirs(Ruta_Controles, exist_ok=True)

    for Nombre_df, df in dfs_Finales.items():
        Ruta_Archivo = os.path.join(
            Ruta_Controles,
            f'14.5. Analisis de warm-up ({Nombre_df}).xlsx'
        )
        df.head(50).to_excel(Ruta_Archivo, index=False)

    Dfs_Finales[0] = dfs_Finales['Generales']
    Dfs_Finales[1] = dfs_Finales['Ballotage']

    print()

    # ========================================================================
    # 15: Eliminación de primeros ítems
    # ========================================================================
    print(
        "PASO 15: Eliminando primeros 3 ítems asociados (warm-up)..."
    )
    print("-"*70)

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        df = Eliminar_Primeros_Datos_IP_Items_Asociados(df)
        Dfs_Finales[i] = df

        print(
            f"  OK {Nombre}: Primeros 3 ítems únicos eliminados por "
            f"participante"
        )

    print()

    # ========================================================================
    # 16: Eliminación de outliers por categoría
    # ========================================================================
    print("PASO 16: Eliminando categoría 'Other'...")
    print("-"*70)

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        Filas_Iniciales = len(df)
        df = df[df['Categoria_PASO_2023'] != 'Other'].reset_index(
            drop=True
        )
        Filas_Finales = len(df)
        Eliminados = Filas_Iniciales - Filas_Finales

        print(
            f"  OK {Nombre}: {Eliminados} participantes eliminados "
            f"(quedan {Filas_Finales})"
        )

        Dfs_Finales[i] = df

    print()

    # ========================================================================
    # 16.5: Comparacion Moderate Right A vs B
    # ========================================================================
    print('PASO 16.5: Comparacion Moderate Right A vs B...')
    print('-'*70)

    dfs_Finales = {
        'Generales': Dfs_Finales[0],
        'Ballotage': Dfs_Finales[1]
    }

    Variables_Comparacion = [
        ('ID', 'Izquierda-Derecha'),
        ('CP', 'Conservador-Progresista'),
        ('AP', 'Antiperonista-Peronista'),
        ('Indice_Progresismo', 'Indice de Progresismo'),
        ('Indice_Conservadurismo', 'Indice de Conservadurismo'),
        ('Indice_Positividad', 'Indice de Positividad'),
        ('Cercania_Massa', 'Cercania a Massa'),
        ('Cercania_Milei', 'Cercania a Milei'),
        ('Cercania_Bullrich', 'Cercania a Bullrich'),
        ('Cercania_Bregman', 'Cercania a Bregman'),
        ('Cercania_Schiaretti', 'Cercania a Schiaretti')
    ]

    Resultados_Comparacion = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*80}")
        print(f"{Nombre_df}")
        print(f"{'='*80}")

        df_A = df[df['Categoria_PASO_2023'] == 'Moderate_Right_A']
        df_B = df[df['Categoria_PASO_2023'] == 'Moderate_Right_B']

        print(f"\nN Moderate_Right_A (Larreta): {len(df_A)}")
        print(f"N Moderate_Right_B (Bullrich): {len(df_B)}")

        print(
            f"\n{'Variable':<35} {'Larreta (A)':<20} "
            f"{'Bullrich (B)':<20} {'p-valor':<12} {'Sig'}"
        )
        print(f"{'-'*100}")

        Resultados_Eleccion = []

        for Variable, Nombre_Variable in Variables_Comparacion:
            if Variable not in df.columns:
                continue

            Datos_A = pd.to_numeric(df_A[Variable], errors='coerce').dropna()
            Datos_B = pd.to_numeric(df_B[Variable], errors='coerce').dropna()

            if len(Datos_A) > 0 and len(Datos_B) > 0:
                Media_A = Datos_A.mean()
                DE_A = Datos_A.std()
                Media_B = Datos_B.mean()
                DE_B = Datos_B.std()

                _, p_valor = stats.ttest_ind(Datos_A, Datos_B)

                if p_valor < 0.001:
                    Sig = '***'
                elif p_valor < 0.01:
                    Sig = '**'
                elif p_valor < 0.05:
                    Sig = '*'
                else:
                    Sig = 'ns'

                Str_A = f"M={Media_A:.2f}, DE={DE_A:.2f}"
                Str_B = f"M={Media_B:.2f}, DE={DE_B:.2f}"
                Str_p = f"{p_valor:.4f}" if p_valor >= 0.0001 else '<.0001'

                print(
                    f"{Nombre_Variable:<35} {Str_A:<20} "
                    f"{Str_B:<20} {Str_p:<12} {Sig}"
                )

                Resultados_Eleccion.append({
                    'Variable': Variable,
                    'Nombre': Nombre_Variable,
                    'Media_A': Media_A,
                    'DE_A': DE_A,
                    'N_A': len(Datos_A),
                    'Media_B': Media_B,
                    'DE_B': DE_B,
                    'N_B': len(Datos_B),
                    'p_valor': p_valor,
                    'Sig': Sig
                })

        Resultados_Comparacion[Nombre_df] = Resultados_Eleccion

        print('\nNotas: *** p<.001, ** p<.01, * p<.05, ns = no significativo')

    print(f"\n{'='*80}")

    print('RESUMEN: VARIABLES CON DIFERENCIAS SIGNIFICATIVAS')
    print('='*80)

    for Nombre_df, Resultados in Resultados_Comparacion.items():
        print(f"\n{Nombre_df}:")
        print('-' * 80)

        Significativas = [r for r in Resultados if r['Sig'] != 'ns']

        if len(Significativas) > 0:
            print("\nVariables con diferencias significativas: "
                  f"{len(Significativas)}/{len(Resultados)}\n")

            for r in Significativas:
                Diferencia = r['Media_B'] - r['Media_A']
                Direccion = 'Mayor' if Diferencia > 0 else 'Menor'

                print(f"  {r['Nombre']}:")
                print(f"    Larreta (A):  M={r['Media_A']:.2f}, DE={r['DE_A']:.2f}")
                print(f"    Bullrich (B): M={r['Media_B']:.2f}, DE={r['DE_B']:.2f}")
                print(f"    Diferencia:   {Diferencia:+.2f} ({Direccion} en Bullrich)")
                print(f"    p-valor:      {r['p_valor']:.4f} {r['Sig']}")
                print()
        else:
            print('\n  No se encontraron diferencias significativas.')

    print('='*80)

    print('GRAFICOS DE COMPARACION')
    print('='*80)

    Variables_Graficar = [
        'ID', 'CP', 'AP', 'Indice_Progresismo', 'Indice_Conservadurismo'
    ]

    for Nombre_df, df in dfs_Finales.items():
        print(f"\nGenerando graficos para {Nombre_df}...")

        df_A = df[df['Categoria_PASO_2023'] == 'Moderate_Right_A']
        df_B = df[df['Categoria_PASO_2023'] == 'Moderate_Right_B']

        Variables_Disponibles = [
            v for v in Variables_Graficar if v in df.columns
        ]
        N_Vars = len(Variables_Disponibles)

        if N_Vars == 0:
            print('  No hay variables disponibles para graficar.')
            continue

        Fig, Axes = plt.subplots(1, N_Vars, figsize=(4*N_Vars, 5))

        if N_Vars == 1:
            Axes = [Axes]

        for i, Variable in enumerate(Variables_Disponibles):
            Datos_Plot = pd.DataFrame({
                'Valor': pd.concat([
                    pd.to_numeric(df_A[Variable], errors='coerce'),
                    pd.to_numeric(df_B[Variable], errors='coerce')
                ]),
                'Categoria': [
                    'Larreta (A)'
                ]*len(df_A) + [
                    'Bullrich (B)'
                ]*len(df_B)
            }).dropna()

            sns.boxplot(
                data=Datos_Plot, x='Categoria', y='Valor',
                ax=Axes[i], palette=['#66C2A5', '#FC8D62']
            )
            Axes[i].set_title(Variable, fontweight='bold')
            Axes[i].set_xlabel('')
            Axes[i].set_ylabel('Valor')
            Axes[i].grid(True, alpha=0.3, axis='y')

        plt.suptitle(
            f'Comparacion Moderate_Right_A vs B - {Nombre_df}',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        plt.show()

        print('  Graficos generados exitosamente.')

    print('\n' + '='*80)

    print('EXPORTANDO RESULTADOS')
    print('='*80)

    Ruta_Controles = os.path.join(
        os.path.dirname(Ruta_Datos_Crudos), 'Controles'
    )
    os.makedirs(Ruta_Controles, exist_ok=True)

    for Nombre_df, Resultados in Resultados_Comparacion.items():
        if len(Resultados) > 0:
            df_Resultados = pd.DataFrame(Resultados)
            Ruta_Archivo = os.path.join(
                Ruta_Controles,
                f'16.5. Comparacion Moderate Right A vs B ({Nombre_df}).xlsx'
            )
            df_Resultados.to_excel(Ruta_Archivo, index=False)
            print(f"\n{Nombre_df}: Exportado exitosamente.")

    print('\n' + '='*80)

    Dfs_Finales[0] = dfs_Finales['Generales']
    Dfs_Finales[1] = dfs_Finales['Ballotage']

    print()

    # ========================================================================
    # 17: Eliminación de outliers por tiempos
    # ========================================================================
    print(
        "PASO 17: Eliminando outliers por tiempos de respuesta..."
    )
    print("-"*70)

    # Crear lista de todas las columnas de tiempo.
    Columnas_Tiempo = []

    for Numero in Numeros_IP:
        Columnas_Tiempo.append(f'IP_Item_{Numero}_Tiempo')
        Columnas_Tiempo.append(f'IP_Item_{Numero}_Izq_Tiempo')
        Columnas_Tiempo.append(f'IP_Item_{Numero}_Der_Tiempo')

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        Filas_Iniciales = len(df)
        df = Eliminar_Filas_Por_Desviacion_Estandar(
            df,
            Columnas_Tiempo,
            3
        )
        Filas_Finales = len(df)

        print(
            f"  OK {Nombre}: Muestra final = {Filas_Finales} participantes"
        )

        Dfs_Finales[i] = df

    print()

    # ========================================================================
    # 18: Columnas de CO y CT
    # ========================================================================
    print("PASO 18: Creando columnas de CO y CT por ítem...")
    print("-"*70)

    Diccionario_Dataframes = {
        'Generales': Dfs_Finales[0],
        'Ballotage': Dfs_Finales[1]
    }

    Diccionario_Dataframes = Crear_Columnas_Cambio_Opinion(
        Diccionario_Dataframes
    )
    Diccionario_Dataframes = Crear_Columnas_Cambio_Tiempo(
        Diccionario_Dataframes
    )

    Dfs_Finales[0] = Diccionario_Dataframes['Generales']
    Dfs_Finales[1] = Diccionario_Dataframes['Ballotage']

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"

        Columnas_CO = [
            col for col in df.columns if col.startswith('CO_Item_')
        ]
        Columnas_CT = [
            col for col in df.columns if col.startswith('CT_Item_')
        ]

        print(
            f"  OK {Nombre}: {len(Columnas_CO)} columnas CO, "
            f"{len(Columnas_CT)} columnas CT creadas"
        )

    print()

    # ========================================================================
    # 19: CO y CT agregados
    # ========================================================================
    print("PASO 19: Creando variables agregadas de CO y CT...")
    print("-"*70)

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        # CO agregados.
        Cols_CO_Pro_Izq = [
            f'CO_Item_{Num}_Izq' for Num in Items_Progresistas
        ]
        Cols_CO_Pro_Der = [
            f'CO_Item_{Num}_Der' for Num in Items_Progresistas
        ]
        Cols_CO_Con_Izq = [
            f'CO_Item_{Num}_Izq' for Num in Items_Conservadores
        ]
        Cols_CO_Con_Der = [
            f'CO_Item_{Num}_Der' for Num in Items_Conservadores
        ]

        # CO_Pro y CO_Con se calculan sobre todas las columnas directamente.
        Cols_CO_Pro = Cols_CO_Pro_Izq + Cols_CO_Pro_Der
        Cols_CO_Con = Cols_CO_Con_Izq + Cols_CO_Con_Der

        # Variables de promedio (CO).
        df['CO_Pro'] = df[Cols_CO_Pro].mean(axis=1)
        df['CO_Pro_Izq'] = df[Cols_CO_Pro_Izq].mean(axis=1)
        df['CO_Pro_Der'] = df[Cols_CO_Pro_Der].mean(axis=1)
        df['CO_Con'] = df[Cols_CO_Con].mean(axis=1)
        df['CO_Con_Izq'] = df[Cols_CO_Con_Izq].mean(axis=1)
        df['CO_Con_Der'] = df[Cols_CO_Con_Der].mean(axis=1)

        # Variables de suma (CO) - Notebook 27.
        df['Cambio_Op_Sum_Pro_Izq'] = df[Cols_CO_Pro_Izq].sum(axis=1)
        df['Cambio_Op_Sum_Pro_Der'] = df[Cols_CO_Pro_Der].sum(axis=1)
        df['Cambio_Op_Sum_Con_Izq'] = df[Cols_CO_Con_Izq].sum(axis=1)
        df['Cambio_Op_Sum_Con_Der'] = df[Cols_CO_Con_Der].sum(axis=1)

        # CT agregados.
        Cols_CT_Pro_Izq = [
            f'CT_Item_{Num}_Izq' for Num in Items_Progresistas
        ]
        Cols_CT_Pro_Der = [
            f'CT_Item_{Num}_Der' for Num in Items_Progresistas
        ]
        Cols_CT_Con_Izq = [
            f'CT_Item_{Num}_Izq' for Num in Items_Conservadores
        ]
        Cols_CT_Con_Der = [
            f'CT_Item_{Num}_Der' for Num in Items_Conservadores
        ]

        # CT_Pro y CT_Con se calculan sobre todas las columnas directamente.
        Cols_CT_Pro = Cols_CT_Pro_Izq + Cols_CT_Pro_Der
        Cols_CT_Con = Cols_CT_Con_Izq + Cols_CT_Con_Der

        # Variables de promedio (CT).
        df['CT_Pro'] = df[Cols_CT_Pro].mean(axis=1)
        df['CT_Pro_Izq'] = df[Cols_CT_Pro_Izq].mean(axis=1)
        df['CT_Pro_Der'] = df[Cols_CT_Pro_Der].mean(axis=1)
        df['CT_Con'] = df[Cols_CT_Con].mean(axis=1)
        df['CT_Con_Izq'] = df[Cols_CT_Con_Izq].mean(axis=1)
        df['CT_Con_Der'] = df[Cols_CT_Con_Der].mean(axis=1)

        # Variables de suma (CT) - Notebook 27.
        df['Cambio_Tiempo_Sum_Pro_Izq'] = df[Cols_CT_Pro_Izq].sum(axis=1)
        df['Cambio_Tiempo_Sum_Pro_Der'] = df[Cols_CT_Pro_Der].sum(axis=1)
        df['Cambio_Tiempo_Sum_Con_Izq'] = df[Cols_CT_Con_Izq].sum(axis=1)
        df['Cambio_Tiempo_Sum_Con_Der'] = df[Cols_CT_Con_Der].sum(axis=1)

        # Estadísticas.
        CO_Pro_Media = df['CO_Pro'].mean()
        CO_Con_Media = df['CO_Con'].mean()
        CT_Pro_Media = df['CT_Pro'].mean()
        CT_Con_Media = df['CT_Con'].mean()

        print(f"  OK {Nombre}:")
        print(
            f"    CO_Pro Media={CO_Pro_Media:.3f}, "
            f"CO_Con Media={CO_Con_Media:.3f}"
        )
        print(
            f"    CT_Pro Media={CT_Pro_Media:.2f}s, "
            f"CT_Con Media={CT_Con_Media:.2f}s"
        )

        Dfs_Finales[i] = df

    print()

    # ========================================================================
    # 20: CO y CT Congruente e Incongruente
    # ========================================================================
    print(
        "PASO 20: Creando variables de congruencia ideológica..."
    )
    print("-"*70)

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        print(f"  Procesando {Nombre}...")

        # ====================================================================
        # CO CONGRUENTE
        # ====================================================================
        # Congruente = Progresistas→Izq + Conservadores→Der.

        # Contar ítems progresistas hacia izquierda.
        Cols_CO_Pro_Izq = [
            f'CO_Item_{Num}_Izq' for Num in Items_Progresistas
        ]
        N_Pro_Izq = df[Cols_CO_Pro_Izq].notna().sum(axis=1)

        # Contar ítems conservadores hacia derecha.
        Cols_CO_Con_Der = [
            f'CO_Item_{Num}_Der' for Num in Items_Conservadores
        ]
        N_Con_Der = df[Cols_CO_Con_Der].notna().sum(axis=1)

        # Total de ítems congruentes.
        N_Congruente_CO = N_Pro_Izq + N_Con_Der

        # Suma y promedio.
        Suma_Congruente_CO = (
            df['Cambio_Op_Sum_Pro_Izq'] +
            df['Cambio_Op_Sum_Con_Der']
        )
        df['CO_Congruente'] = np.where(
            N_Congruente_CO > 0,
            Suma_Congruente_CO / N_Congruente_CO,
            np.nan
        )

        # ====================================================================
        # CO INCONGRUENTE
        # ====================================================================
        # Incongruente = Progresistas→Der + Conservadores→Izq.

        # Contar ítems progresistas hacia derecha.
        Cols_CO_Pro_Der = [
            f'CO_Item_{Num}_Der' for Num in Items_Progresistas
        ]
        N_Pro_Der = df[Cols_CO_Pro_Der].notna().sum(axis=1)

        # Contar ítems conservadores hacia izquierda.
        Cols_CO_Con_Izq = [
            f'CO_Item_{Num}_Izq' for Num in Items_Conservadores
        ]
        N_Con_Izq = df[Cols_CO_Con_Izq].notna().sum(axis=1)

        # Total de ítems incongruentes.
        N_Incongruente_CO = N_Pro_Der + N_Con_Izq

        # Suma y promedio.
        Suma_Incongruente_CO = (
            df['Cambio_Op_Sum_Pro_Der'] +
            df['Cambio_Op_Sum_Con_Izq']
        )
        df['CO_Incongruente'] = np.where(
            N_Incongruente_CO > 0,
            Suma_Incongruente_CO / N_Incongruente_CO,
            np.nan
        )

        # ====================================================================
        # CT CONGRUENTE
        # ====================================================================

        # Contar ítems progresistas hacia izquierda (CT).
        Cols_CT_Pro_Izq = [
            f'CT_Item_{Num}_Izq' for Num in Items_Progresistas
        ]
        N_CT_Pro_Izq = df[Cols_CT_Pro_Izq].notna().sum(axis=1)

        # Contar ítems conservadores hacia derecha (CT).
        Cols_CT_Con_Der = [
            f'CT_Item_{Num}_Der' for Num in Items_Conservadores
        ]
        N_CT_Con_Der = df[Cols_CT_Con_Der].notna().sum(axis=1)

        # Total de ítems congruentes CT.
        N_Congruente_CT = N_CT_Pro_Izq + N_CT_Con_Der

        # Suma y promedio.
        Suma_Congruente_CT = (
            df['Cambio_Tiempo_Sum_Pro_Izq'] +
            df['Cambio_Tiempo_Sum_Con_Der']
        )
        df['CT_Congruente'] = np.where(
            N_Congruente_CT > 0,
            Suma_Congruente_CT / N_Congruente_CT,
            np.nan
        )

        # ====================================================================
        # CT INCONGRUENTE
        # ====================================================================

        # Contar ítems progresistas hacia derecha (CT).
        Cols_CT_Pro_Der = [
            f'CT_Item_{Num}_Der' for Num in Items_Progresistas
        ]
        N_CT_Pro_Der = df[Cols_CT_Pro_Der].notna().sum(axis=1)

        # Contar ítems conservadores hacia izquierda (CT).
        Cols_CT_Con_Izq = [
            f'CT_Item_{Num}_Izq' for Num in Items_Conservadores
        ]
        N_CT_Con_Izq = df[Cols_CT_Con_Izq].notna().sum(axis=1)

        # Total de ítems incongruentes CT.
        N_Incongruente_CT = N_CT_Pro_Der + N_CT_Con_Izq

        # Suma y promedio.
        Suma_Incongruente_CT = (
            df['Cambio_Tiempo_Sum_Pro_Der'] +
            df['Cambio_Tiempo_Sum_Con_Izq']
        )
        df['CT_Incongruente'] = np.where(
            N_Incongruente_CT > 0,
            Suma_Incongruente_CT / N_Incongruente_CT,
            np.nan
        )

        # Estadísticas.
        CO_Cong_Media = df['CO_Congruente'].mean()
        CO_Incong_Media = df['CO_Incongruente'].mean()
        CT_Cong_Media = df['CT_Congruente'].mean()
        CT_Incong_Media = df['CT_Incongruente'].mean()

        print(f"  OK {Nombre}:")
        print(
            f"    CO_Congruente Media={CO_Cong_Media:.4f}, "
            f"CO_Incongruente Media={CO_Incong_Media:.4f}"
        )
        print(
            f"    CT_Congruente Media={CT_Cong_Media:.2f}s, "
            f"CT_Incongruente Media={CT_Incong_Media:.2f}s"
        )

        Dfs_Finales[i] = df

    print()

    # ========================================================================
    # 21: Variables del cluster
    # ========================================================================
    print("PASO 21: Agregando variables de cluster...")
    print("-"*70)

    # Ruta al archivo de clusters.
    Ruta_Clusters = os.path.join(
        Ruta_Datos_Crudos,
        'Clusters.xlsx'
    )

    # Verificar si existe el archivo.
    if os.path.exists(Ruta_Clusters):
        df_Clusters = pd.read_excel(Ruta_Clusters)

        # Columnas a transferir.
        Columnas_Cluster = [
            'Conservative_Median',
            'Progressive_Median',
            'Conservative_Cluster',
            'Progressive_Cluster'
        ]

        for i, df in enumerate(Dfs_Finales):
            Nombre = "Generales" if i == 0 else "Ballotage"
            # En Clusters.xlsx el valor es 'General', no 'Generales'.
            Nombre_Election = "General" if i == 0 else "Ballotage"
            print(f"  Procesando {Nombre}...")

            # Filtrar datos correspondientes a este tipo de elección.
            Datos_Filtrados = df_Clusters[
                df_Clusters['Election'] == Nombre_Election
            ].copy()

            # Seleccionar solo ID y las columnas a transferir.
            Datos_A_Mergear = Datos_Filtrados[
                ['ID'] + Columnas_Cluster
            ]

            # Realizar merge.
            df = df.merge(
                Datos_A_Mergear,
                on='ID',
                how='left'
            )

            # Contar valores no nulos agregados.
            N_Con_Cluster = df['Conservative_Cluster'].notna().sum()

            print(
                f"  OK {Nombre}: {N_Con_Cluster} participantes con "
                f"datos de cluster"
            )

            Dfs_Finales[i] = df
    else:
        print(f"  ⚠️  Archivo no encontrado: {Ruta_Clusters}")
        print("  Saltando paso de clusters...")

    print()

    # ========================================================================
    # EXPORTAR A EXCEL
    # ========================================================================
    print("PASO FINAL: Exportando bases de datos a Excel...")
    print("-"*70)

    os.makedirs(Ruta_Salida, exist_ok=True)

    for i, df in enumerate(Dfs_Finales):
        Nombre = "Generales" if i == 0 else "Ballotage"
        Ruta_Excel = os.path.join(
            Ruta_Salida,
            f'{Nombre}.xlsx'
        )

        df.to_excel(Ruta_Excel, index=False, engine='openpyxl')

        print(
            f"  OK Exportado: {Ruta_Excel} "
            f"({len(df)} filas × {len(df.columns)} columnas)"
        )

    print()
    print("="*70)
    print("PROCESAMIENTO COMPLETO FINALIZADO")
    print("="*70)
    print()
    print("Resumen de muestra final:")
    print(f"  - Generales: {len(Dfs_Finales[0])} participantes")
    print(f"  - Ballotage: {len(Dfs_Finales[1])} participantes")
    print()

    return {
        'Generales': Dfs_Finales[0],
        'Ballotage': Dfs_Finales[1]
    }


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":

    # Configurar rutas.
    Ruta_Base = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    Ruta_Datos_Crudos = os.path.join(Ruta_Base, "Datos crudos")
    Ruta_Salida = os.path.join(Ruta_Base, "Datos definitivos")

    # Ejecutar pipeline completo.
    Bases_Procesadas = Procesar_Datos_Completo(
        Ruta_Datos_Crudos,
        Ruta_Salida
    )

    print("Script finalizado exitosamente.")
