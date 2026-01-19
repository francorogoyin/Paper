# -*- coding: utf-8 -*-

"""
Script para comparar las bases de datos generadas por el script de
procesamiento (Paper/Datos definitivos) con las bases originales
(Data/Bases definitivas).

Compara para cada variable:
- Número de filas
- Promedio (media)
- Mediana
- Desvío estándar
- Mínimo y máximo
- Valores únicos
- Valores nulos

"""

import pandas as pd
import numpy as np
import os
import sys


TOLERANCIA_NUMERICA = 0.000001
MAX_MUESTRAS_DIF = 5


def Comparar_Variable(
    Serie_Paper: pd.Series,
    Serie_Data: pd.Series,
    Nombre_Variable: str
) -> dict:

    """
    Compara estadísticas de una variable entre dos Series de pandas.

    Parámetros:
        Serie_Paper: Serie de la base generada por el script.
        Serie_Data: Serie de la base original.
        Nombre_Variable: Nombre de la variable a comparar.

    Retorna:
        Diccionario con las comparaciones y si hay diferencias.

    """

    Resultado = {
        'Variable': Nombre_Variable,
        'Hay_Diferencia': False,
        'Detalles': []
    }

    # Verificar si la variable existe en ambas bases.
    Paper_Existe = Serie_Paper is not None
    Data_Existe = Serie_Data is not None

    if not Paper_Existe and not Data_Existe:
        return None

    if not Paper_Existe:
        Resultado['Hay_Diferencia'] = True
        Resultado['Detalles'].append(
            f"  ⚠️  Solo existe en DATA (no en Paper)"
        )
        return Resultado

    if not Data_Existe:
        Resultado['Hay_Diferencia'] = True
        Resultado['Detalles'].append(
            f"  ⚠️  Solo existe en PAPER (no en Data)"
        )
        return Resultado

    # Comparar cantidad de valores no nulos.
    N_Paper = Serie_Paper.notna().sum()
    N_Data = Serie_Data.notna().sum()

    if N_Paper != N_Data:
        Resultado['Hay_Diferencia'] = True
        Resultado['Detalles'].append(
            f"  N válidos: Paper={N_Paper}, Data={N_Data} "
            f"(Dif={N_Paper - N_Data})"
        )
    else:
        Resultado['Detalles'].append(
            f"  N válidos: {N_Paper} (igual)"
        )

    # Comparar nulos.
    Nulos_Paper = Serie_Paper.isna().sum()
    Nulos_Data = Serie_Data.isna().sum()

    if Nulos_Paper != Nulos_Data:
        Resultado['Hay_Diferencia'] = True
        Resultado['Detalles'].append(
            f"  Nulos: Paper={Nulos_Paper}, Data={Nulos_Data} "
            f"(Dif={Nulos_Paper - Nulos_Data})"
        )

    # Verificar si es numérica (excluyendo booleanos).
    Es_Bool_Paper = pd.api.types.is_bool_dtype(Serie_Paper)
    Es_Bool_Data = pd.api.types.is_bool_dtype(Serie_Data)
    Es_Numerica_Paper = (
        pd.api.types.is_numeric_dtype(Serie_Paper) and not Es_Bool_Paper
    )
    Es_Numerica_Data = (
        pd.api.types.is_numeric_dtype(Serie_Data) and not Es_Bool_Data
    )

    # Alinear series para comparar valores.
    Serie_Paper, Serie_Data = Serie_Paper.align(Serie_Data, join='outer')

    if Es_Numerica_Paper and Es_Numerica_Data:
        # Comparar estadísticas numéricas.

        # Media.
        Media_Paper = Serie_Paper.mean()
        Media_Data = Serie_Data.mean()

        if pd.notna(Media_Paper) and pd.notna(Media_Data):
            Dif_Media = abs(Media_Paper - Media_Data)
            if Dif_Media > 0.0001:
                Resultado['Hay_Diferencia'] = True
                Resultado['Detalles'].append(
                    f"  Media: Paper={Media_Paper:.6f}, "
                    f"Data={Media_Data:.6f} (Dif={Dif_Media:.6f})"
                )
            else:
                Resultado['Detalles'].append(
                    f"  Media: {Media_Paper:.6f} (igual)"
                )

        # Mediana.
        Mediana_Paper = Serie_Paper.median()
        Mediana_Data = Serie_Data.median()

        if pd.notna(Mediana_Paper) and pd.notna(Mediana_Data):
            Dif_Mediana = abs(Mediana_Paper - Mediana_Data)
            if Dif_Mediana > 0.0001:
                Resultado['Hay_Diferencia'] = True
                Resultado['Detalles'].append(
                    f"  Mediana: Paper={Mediana_Paper:.6f}, "
                    f"Data={Mediana_Data:.6f} (Dif={Dif_Mediana:.6f})"
                )
            else:
                Resultado['Detalles'].append(
                    f"  Mediana: {Mediana_Paper:.6f} (igual)"
                )

        # Desvío estándar.
        Std_Paper = Serie_Paper.std()
        Std_Data = Serie_Data.std()

        if pd.notna(Std_Paper) and pd.notna(Std_Data):
            Dif_Std = abs(Std_Paper - Std_Data)
            if Dif_Std > 0.0001:
                Resultado['Hay_Diferencia'] = True
                Resultado['Detalles'].append(
                    f"  Desvío: Paper={Std_Paper:.6f}, "
                    f"Data={Std_Data:.6f} (Dif={Dif_Std:.6f})"
                )
            else:
                Resultado['Detalles'].append(
                    f"  Desvío: {Std_Paper:.6f} (igual)"
                )

        # Mínimo.
        Min_Paper = Serie_Paper.min()
        Min_Data = Serie_Data.min()

        if pd.notna(Min_Paper) and pd.notna(Min_Data):
            Dif_Min = abs(Min_Paper - Min_Data)
            if Dif_Min > 0.0001:
                Resultado['Hay_Diferencia'] = True
                Resultado['Detalles'].append(
                    f"  Mínimo: Paper={Min_Paper:.6f}, "
                    f"Data={Min_Data:.6f} (Dif={Dif_Min:.6f})"
                )
            else:
                Resultado['Detalles'].append(
                    f"  Mínimo: {Min_Paper:.6f} (igual)"
                )

        # Máximo.
        Max_Paper = Serie_Paper.max()
        Max_Data = Serie_Data.max()

        if pd.notna(Max_Paper) and pd.notna(Max_Data):
            Dif_Max = abs(Max_Paper - Max_Data)
            if Dif_Max > 0.0001:
                Resultado['Hay_Diferencia'] = True
                Resultado['Detalles'].append(
                    f"  Máximo: Paper={Max_Paper:.6f}, "
                    f"Data={Max_Data:.6f} (Dif={Dif_Max:.6f})"
                )
            else:
                Resultado['Detalles'].append(
                    f"  Máximo: {Max_Paper:.6f} (igual)"
                )


        # Diferencias valor a valor.
        Serie_Paper_Num = pd.to_numeric(Serie_Paper, errors='coerce')
        Serie_Data_Num = pd.to_numeric(Serie_Data, errors='coerce')
        Diferencias = (Serie_Paper_Num - Serie_Data_Num).abs()
        Diferencias = Diferencias[Diferencias.notna()]

        N_Dif = (Diferencias > TOLERANCIA_NUMERICA).sum()
        if N_Dif > 0:
            Resultado['Hay_Diferencia'] = True
            Resultado['Detalles'].append(
                f"  Diferencias > {TOLERANCIA_NUMERICA}: {N_Dif}"
            )
            Max_Dif = Diferencias.max()
            Resultado['Detalles'].append(
                f"  Max dif: {Max_Dif:.6f}"
            )

            Indices_Top = Diferencias.sort_values(
                ascending=False
            ).head(MAX_MUESTRAS_DIF).index
            for Indice in Indices_Top:
                Val_Paper = Serie_Paper_Num.get(Indice, np.nan)
                Val_Data = Serie_Data_Num.get(Indice, np.nan)
                Resultado['Detalles'].append(
                    f"    ID={Indice}: Paper={Val_Paper} | Data={Val_Data}"
                )
        else:
            Resultado['Detalles'].append(
                f"  Diferencias > {TOLERANCIA_NUMERICA}: 0"
            )

    else:
        # Variable categórica o de texto.
        Unicos_Paper = Serie_Paper.nunique()
        Unicos_Data = Serie_Data.nunique()

        if Unicos_Paper != Unicos_Data:
            Resultado['Hay_Diferencia'] = True
            Resultado['Detalles'].append(
                f"  Valores únicos: Paper={Unicos_Paper}, "
                f"Data={Unicos_Data} (Dif={Unicos_Paper - Unicos_Data})"
            )
        else:
            Resultado['Detalles'].append(
                f"  Valores únicos: {Unicos_Paper} (igual)"
            )

        # Comparar conteos de cada valor.
        Conteo_Paper = Serie_Paper.value_counts(dropna=False).sort_index()
        Conteo_Data = Serie_Data.value_counts(dropna=False).sort_index()

        # Obtener todos los valores únicos de ambas series.
        Todos_Valores = set(Conteo_Paper.index) | set(Conteo_Data.index)

        for Valor in sorted(Todos_Valores, key=lambda x: (x is np.nan, str(x))):
            Cant_Paper = Conteo_Paper.get(Valor, 0)
            Cant_Data = Conteo_Data.get(Valor, 0)

            if pd.isna(Valor):
                Valor_Str = "NaN"
            else:
                Valor_Str = str(Valor)

            if Cant_Paper != Cant_Data:
                Resultado['Hay_Diferencia'] = True
                Resultado['Detalles'].append(
                    f"    '{Valor_Str}': Paper={Cant_Paper}, "
                    f"Data={Cant_Data} (Dif={Cant_Paper - Cant_Data})"
                )
            else:
                Resultado['Detalles'].append(
                    f"    '{Valor_Str}': {Cant_Paper} (igual)"
                )

        # Diferencias valor a valor.
        Serie_Paper_Str = Serie_Paper.astype(str)
        Serie_Data_Str = Serie_Data.astype(str)
        Mask_Dif = Serie_Paper_Str != Serie_Data_Str
        N_Dif = Mask_Dif.sum()
        if N_Dif > 0:
            Resultado['Hay_Diferencia'] = True
            Resultado['Detalles'].append(
                f"  Diferencias valor a valor: {N_Dif}"
            )

            Indices_Top = Serie_Paper_Str[Mask_Dif].head(
                MAX_MUESTRAS_DIF
            ).index
            for Indice in Indices_Top:
                Val_Paper = Serie_Paper_Str.get(Indice, 'NaN')
                Val_Data = Serie_Data_Str.get(Indice, 'NaN')
                Resultado['Detalles'].append(
                    f"    ID={Indice}: Paper='{Val_Paper}' | Data='{Val_Data}'"
                )
        else:
            Resultado['Detalles'].append(
                "  Diferencias valor a valor: 0"
            )

    return Resultado


def Comparar_Bases(
    df_Paper: pd.DataFrame,
    df_Data: pd.DataFrame,
    Nombre_Base: str
) -> None:

    """
    Compara dos DataFrames y muestra las diferencias.

    Parámetros:
        df_Paper: DataFrame generado por el script.
        df_Data: DataFrame original.
        Nombre_Base: Nombre de la base (Generales o Ballotage).

    """

    print("=" * 70)
    print(f"COMPARACIÓN: {Nombre_Base.upper()}")
    print("=" * 70)
    print()

    # Comparar IDs.
    if 'ID' in df_Paper.columns and 'ID' in df_Data.columns:
        print("IDENTIFICADORES:")
        print("-" * 70)
        Ids_Paper = df_Paper['ID']
        Ids_Data = df_Data['ID']

        Duplicados_Paper = Ids_Paper[Ids_Paper.duplicated()].unique()
        Duplicados_Data = Ids_Data[Ids_Data.duplicated()].unique()

        print(f"  IDs Paper: {Ids_Paper.nunique()} ?nicos")
        print(f"  IDs Data: {Ids_Data.nunique()} ?nicos")

        if len(Duplicados_Paper) > 0:
            print("  ADVERTENCIA Duplicados en Paper: ")
            print(f"    {Duplicados_Paper[:MAX_MUESTRAS_DIF]}")
        if len(Duplicados_Data) > 0:
            print("  ADVERTENCIA Duplicados en Data: ")
            print(f"    {Duplicados_Data[:MAX_MUESTRAS_DIF]}")

        Set_Paper = set(Ids_Paper.dropna())
        Set_Data = set(Ids_Data.dropna())
        Solo_Paper_ID = sorted(Set_Paper - Set_Data)
        Solo_Data_ID = sorted(Set_Data - Set_Paper)

        print(f"  IDs solo en Paper: {len(Solo_Paper_ID)}")
        print(f"  IDs solo en Data: {len(Solo_Data_ID)}")

        if Solo_Paper_ID:
            print(f"    Ejemplos: {Solo_Paper_ID[:MAX_MUESTRAS_DIF]}")
        if Solo_Data_ID:
            print(f"    Ejemplos: {Solo_Data_ID[:MAX_MUESTRAS_DIF]}")

        if (len(Duplicados_Paper) == 0 and len(Duplicados_Data) == 0 and
                not Solo_Paper_ID and not Solo_Data_ID):
            df_Paper = df_Paper.sort_values('ID').reset_index(drop=True)
            df_Data = df_Data.sort_values('ID').reset_index(drop=True)

        print()

        # Comparar dimensiones.
    print("DIMENSIONES:")
    print("-" * 70)
    print(f"  Filas: Paper={len(df_Paper)}, Data={len(df_Data)}")

    if len(df_Paper) != len(df_Data):
        print(f"  ⚠️  DIFERENCIA en filas: {len(df_Paper) - len(df_Data)}")

    print(
        f"  Columnas: Paper={len(df_Paper.columns)}, "
        f"Data={len(df_Data.columns)}"
    )

    if len(df_Paper.columns) != len(df_Data.columns):
        print(
            f"  ⚠️  DIFERENCIA en columnas: "
            f"{len(df_Paper.columns) - len(df_Data.columns)}"
        )

    print()

    # Columnas en común y exclusivas.
    Cols_Paper = set(df_Paper.columns)
    Cols_Data = set(df_Data.columns)
    Cols_Comunes = Cols_Paper & Cols_Data
    Solo_Paper = Cols_Paper - Cols_Data
    Solo_Data = Cols_Data - Cols_Paper

    print("COLUMNAS:")
    print("-" * 70)
    print(f"  En común: {len(Cols_Comunes)}")
    print(f"  Solo en Paper: {len(Solo_Paper)}")
    print(f"  Solo en Data: {len(Solo_Data)}")

    if Solo_Paper:
        print()
        print("  Columnas solo en PAPER:")
        for Col in sorted(Solo_Paper):
            print(f"    - {Col}")

    if Solo_Data:
        print()
        print("  Columnas solo en DATA:")
        for Col in sorted(Solo_Data):
            print(f"    - {Col}")

    print()

    # Comparar orden de columnas.
    if list(df_Paper.columns) != list(df_Data.columns):
        print("ORDEN DE COLUMNAS:")
        print("-" * 70)
        print("  ADVERTENCIA Orden de columnas no coincide.")
        Cols_Dif = [
            (i, c1, c2)
            for i, (c1, c2) in enumerate(
                zip(df_Paper.columns, df_Data.columns)
            ) if c1 != c2
        ]
        for i, c1, c2 in Cols_Dif[:MAX_MUESTRAS_DIF]:
            print(f"    Pos {i}: Paper={c1} | Data={c2}")
        print()

        # Comparar variables en común.
    print("COMPARACIÓN DE VARIABLES EN COMÚN:")
    print("-" * 70)
    print()

    Variables_Con_Diferencia = []
    Variables_Iguales = []

    for Col in sorted(Cols_Comunes):
        Serie_Paper = df_Paper[Col]
        Serie_Data = df_Data[Col]

        Resultado = Comparar_Variable(Serie_Paper, Serie_Data, Col)

        if Resultado:
            if Resultado['Hay_Diferencia']:
                Variables_Con_Diferencia.append(Resultado)
            else:
                Variables_Iguales.append(Resultado)

    # Mostrar resumen.
    print(f"Variables iguales: {len(Variables_Iguales)}")
    print(f"Variables con diferencias: {len(Variables_Con_Diferencia)}")
    print()

    # Mostrar detalles de variables con diferencias.
    if Variables_Con_Diferencia:
        print("DETALLE DE DIFERENCIAS:")
        print("-" * 70)

        for Resultado in Variables_Con_Diferencia:
            print()
            print(f"📊 {Resultado['Variable']}:")
            for Detalle in Resultado['Detalles']:
                print(Detalle)

    if Variables_Iguales:
        print("VARIABLES SIN DIFERENCIAS:")
        print("-" * 70)

        for Resultado in Variables_Iguales:
            print()
            print(f"📊 {Resultado['Variable']}:")
            for Detalle in Resultado['Detalles']:
                print(Detalle)

    print()
    print("=" * 70)
    print()


def Main():

    """
    Función principal que ejecuta la comparación de bases.

    """

    Ruta_Base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    Ruta_Salida = os.path.join(
        Ruta_Base, "Controles", "Comparacion_Bases.txt"
    )
    sys.stdout = open(Ruta_Salida, "w", encoding="utf-8")

    print()
    print("=" * 70)
    print("COMPARADOR DE BASES DE DATOS")
    print("Paper vs Data/Bases definitivas")
    print("=" * 70)
    print()

    # Definir rutas.
    Ruta_Paper = os.path.join(Ruta_Base, "Datos definitivos")
    Ruta_Data = os.path.join(Ruta_Base, "Data", "Bases definitivas")
    if not os.path.exists(Ruta_Data):
        Ruta_Data = os.path.join(
            os.path.dirname(os.path.dirname(Ruta_Base)),
            "Downloads",
            "Redaccion",
            "Data",
            "Bases definitivas"
        )

    # Cargar bases de Paper.
    print("Cargando bases de PAPER...")

    Ruta_Paper_Generales = os.path.join(Ruta_Paper, "Generales.xlsx")
    Ruta_Paper_Ballotage = os.path.join(Ruta_Paper, "Ballotage.xlsx")

    if not os.path.exists(Ruta_Paper_Generales):
        print(f"  ⚠️  No existe: {Ruta_Paper_Generales}")
        return

    if not os.path.exists(Ruta_Paper_Ballotage):
        print(f"  ⚠️  No existe: {Ruta_Paper_Ballotage}")
        return

    df_Paper_Generales = pd.read_excel(Ruta_Paper_Generales)
    df_Paper_Ballotage = pd.read_excel(Ruta_Paper_Ballotage)

    print(f"  OK Generales: {len(df_Paper_Generales)} filas")
    print(f"  OK Ballotage: {len(df_Paper_Ballotage)} filas")
    print()

    # Cargar bases de Data.
    print("Cargando bases de DATA...")

    Ruta_Data_Generales = os.path.join(Ruta_Data, "Generales.xlsx")
    Ruta_Data_Ballotage = os.path.join(Ruta_Data, "Ballotage.xlsx")

    if not os.path.exists(Ruta_Data_Generales):
        print(f"  ⚠️  No existe: {Ruta_Data_Generales}")
        return

    if not os.path.exists(Ruta_Data_Ballotage):
        print(f"  ⚠️  No existe: {Ruta_Data_Ballotage}")
        return

    df_Data_Generales = pd.read_excel(Ruta_Data_Generales)
    df_Data_Ballotage = pd.read_excel(Ruta_Data_Ballotage)

    print(f"  OK Generales: {len(df_Data_Generales)} filas")
    print(f"  OK Ballotage: {len(df_Data_Ballotage)} filas")
    print()

    # Comparar bases.
    Comparar_Bases(df_Paper_Generales, df_Data_Generales, "Generales")
    Comparar_Bases(df_Paper_Ballotage, df_Data_Ballotage, "Ballotage")

    print("COMPARACIÓN FINALIZADA")
    print("=" * 70)

    sys.stdout.close()


if __name__ == "__main__":
    Main()
