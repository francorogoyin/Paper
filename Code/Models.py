# -*- coding: utf-8 -*-
"""
Statistical Models.

This script contains functions to fit regression models:
- Robust Linear Model with HC3 errors
- Recursive GLM model (Gaussian)
- Recursive Binary Logistic Regression

Each model uses sequential variable elimination based on
information criteria (AIC) and statistical significance (p < 0.05).

"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import jarque_bera
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# VARIABLE NAME MAPPING SPANISH -> ENGLISH.
# =============================================================================

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
    # Closeness.
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
    Translates the variable name from Spanish to English.

    Parameters:
        Nombre_Variable (str): Name of the variable in Spanish.

    Returns:
        str: Translated variable name or the original if not found.

    """

    return Mapeo_Variables_Es_En.get(Nombre_Variable, Nombre_Variable)


# =============================================================================
# CONSTANTS AND CONFIGURATION.
# =============================================================================

# Dependent variables for robust model (Opinion Change).
# Includes aggregated variables and significant individual items.
Variables_Dependientes_Robusto = {
    'Generales': [
        'CO_Congruente',
        'CO_Incongruente',
        'CT_Congruente',
        'CT_Incongruente',
        'CO_Pro',
        'CO_Con',
        'CO_Pro_Der',
        'CO_Con_Der',
        'CO_Pro_Izq',
        'CO_Con_Izq',
        'CO_Item_5_Izq',
        'CO_Item_25_Izq',
        'CO_Item_9_Izq',
        'CO_Item_24_Izq',
        'CO_Item_8_Izq',
        'CO_Item_30_Izq',
        'CO_Item_7_Izq',
        'CO_Item_3_Izq',
        'CO_Item_10_Izq',
        'CO_Item_30_Der',
        'CO_Item_10_Der'
    ],
    'Ballotage': [
        'CO_Congruente',
        'CO_Incongruente',
        'CT_Congruente',
        'CT_Incongruente',
        'CO_Pro',
        'CO_Con',
        'CO_Pro_Der',
        'CO_Con_Der',
        'CO_Pro_Izq',
        'CO_Con_Izq',
        'CO_Item_6_Izq',
        'CO_Item_25_Izq',
        'CO_Item_25_Der',
        'CO_Item_11_Der',
        'CO_Item_29_Izq',
        'CO_Item_30_Izq',
        'CO_Item_22_Izq',
        'CO_Item_7_Der',
        'CO_Item_22_Der'
    ]
}

# Dependent variables for GLM (Indices and Self-perceptions).
Variables_Dependientes_GLM = [
    'Indice_Progresismo',
    'Indice_Conservadurismo',
    'Autopercepcion_Izq_Der',
    'Autopercepcion_Con_Pro',
    'Autopercepcion_Per_Antiper'
]

# Candidates for closeness logistic regression.
Candidatos_Cercania = [
    'Massa',
    'Milei',
    'Schiaretti',
    'Bullrich',
    'Bregman'
]

# Common independent variables for all models.
Variables_Independientes = [
    'Conservative_Cluster',
    'Progressive_Cluster',
    'Edad',
    'Genero_Masculino',
    'Genero_Otro',
    'Region_CABA',
    'Region_Centro',
    'Region_Cuyo',
    'Region_Norte',
    'Region_Patagonia',
    'Voto_2019_JL_Espert',
    'Voto_2019_J_Gomez_Centurion',
    'Voto_2019_Mauricio_Macri',
    'Voto_2019_Nicolas_Del_Caño',
    'Voto_2019_Roberto_Lavagna',
    'Categoria_PASO_2023_Left_Wing',
    'Categoria_PASO_2023_Moderate_Right_A',
    'Categoria_PASO_2023_Moderate_Right_B',
    'Categoria_PASO_2023_Right_Wing_Libertarian',
    'Categoria_PASO_2023_Centre',
    'Autopercepcion_Izq_Der',
    'Autopercepcion_Con_Pro',
    'Autopercepcion_Per_Antiper',
    'Cercania_Massa',
    'Cercania_Milei',
    'Influencia_Redes',
    'Red_Social_Facebook',
    'Red_Social_Instagram',
    'Red_Social_Threads',
    'Red_Social_Tiktok',
    'Red_Social_Youtube',
    'Red_Social_Whatsapp',
    'Red_Social_Telegram',
    'Influencia_Prensa',
    'Medios_Prensa_Prensa_Obrera',
    'Medios_Prensa_Diario_Universal',
    'Medios_Prensa_Popular',
    'Medios_Prensa_Izquierda_Diario',
    'Medios_Prensa_Clarin',
    'Medios_Prensa_Perfil',
    'Medios_Prensa_Pagina_12',
    'Medios_Prensa_Infobae',
    'Medios_Prensa_El_Cronista',
    'Medios_Prensa_La_Nacion',
    'Medios_Prensa_Tiempo_Argentino',
    'Indice_Positividad',
    'Indice_Progresismo',
    'Indice_Progresismo_Tiempo',
    'Indice_Conservadurismo',
    'Indice_Conservadurismo_Tiempo'
]


def Formatear_P_Valor(P_Valor: float) -> str:

    """
    Formats a p-value in scientific notation with 2 decimals
    only when it is very small (< 0.001). For larger values
    uses standard decimal format.

    Parameters:
        P_Valor (float): The p-value to format.

    Returns:
        str: The formatted p-value as string.
            - If p < 0.001: scientific notation (e.g.: "2.12E-15")
            - If p >= 0.001: decimal format (e.g.: "0.043")

    """

    if pd.isna(P_Valor):
        return "NaN"

    if P_Valor < 0.001:
        return f"{P_Valor:.2E}"
    else:
        return f"{P_Valor:.4f}"


# =============================================================================
# DATA ANALYSIS AND PREPARATION FUNCTIONS.
# =============================================================================

def Analizar_Variable_Dependiente(
    Variable_Y: pd.Series,
    Nombre_Variable: str
) -> dict:

    """
    Exhaustively analyzes the statistical characteristics of a
    dependent variable to determine the most appropriate regression
    model.

    This function examines the distribution, key characteristics and
    statistical properties of the dependent variable, providing
    specific recommendations on which type of model to use.

    Parameters:
        Variable_Y (pd.Series): Series with the values of the
            dependent variable to analyze.
        Nombre_Variable (str): Descriptive name of the variable.

    Returns:
        dict: Dictionary with detected boolean characteristics:
            - 'Es_Ordinal': True if it has ordinal characteristics.
            - 'Tiene_Muchos_Ceros': True if more than 50% are zeros.
            - 'Es_Simetrica': True if the distribution is symmetric.
            - 'Rango_Limitado': True if the total range is <= 10.

    """

    print(f"ANÁLISIS DE {Nombre_Variable}:")
    print("=" * 50)

    # Value distribution.
    print("Distribución de valores:")
    Distribucion = Variable_Y.value_counts().sort_index()
    Porcentajes = Variable_Y.value_counts(normalize=True).sort_index() * 100

    for Valor in sorted(Variable_Y.unique()):
        Frecuencia = Distribucion[Valor]
        Porcentaje = Porcentajes[Valor]
        print(f"  Valor {Valor:2.0f}: {Frecuencia:4d} obs ({Porcentaje:5.1f}%)")

    # Key characteristics.
    Es_Ordinal = (
        len(Variable_Y.unique()) <= 10 and
        Variable_Y.min() >= -10 and
        Variable_Y.max() <= 10
    )
    Tiene_Muchos_Ceros = (Variable_Y == 0).mean() > 0.5
    Es_Simetrica = abs(Variable_Y.skew()) < 0.5
    Rango_Limitado = Variable_Y.max() - Variable_Y.min() <= 10

    print(f"\nCaracterísticas:")
    print(f"  Valores únicos: {Variable_Y.nunique()}")
    print(f"  Rango: [{Variable_Y.min()}, {Variable_Y.max()}]")
    print(f"  Media: {Variable_Y.mean():.3f}")
    print(f"  Mediana: {Variable_Y.median():.3f}")
    print(f"  Desviación estándar: {Variable_Y.std():.3f}")
    print(f"  Asimetría: {Variable_Y.skew():.3f}")
    print(f"  Exceso de ceros: {(Variable_Y == 0).mean()*100:.1f}%")

    # Model recommendation.
    print(f"\nRECOMENDACIÓN DE MODELO:")

    if Es_Ordinal and Rango_Limitado:
        print("  ✅ REGRESIÓN ORDINAL - Variable con categorías ordenadas")

    if Tiene_Muchos_Ceros:
        print("  ✅ MODELO ZERO-INFLATED - Exceso de valores cero")

    if not Es_Simetrica and Variable_Y.min() >= 0:
        print("  ✅ REGRESIÓN GAMMA/LOG-NORMAL - Distribución sesgada")

    if Variable_Y.nunique() == 2:
        print("  ✅ REGRESIÓN LOGÍSTICA - Variable binaria")

    if Variable_Y.min() >= 0 and all(Variable_Y % 1 == 0):
        print("  ✅ REGRESIÓN POISSON - Variable de conteo")

    return {
        'Es_Ordinal': Es_Ordinal,
        'Tiene_Muchos_Ceros': Tiene_Muchos_Ceros,
        'Es_Simetrica': Es_Simetrica,
        'Rango_Limitado': Rango_Limitado
    }


def Limpiar_Y_Estructurar_Datos_Para_Modelado(
    Dataframe: pd.DataFrame,
    Variable_Dependiente: str,
    Variables_Independientes: list
) -> tuple:

    """
    Cleans, structures and prepares a dataset for model training,
    handling missing values and filtering valid observations.

    Performs essential preprocessing including: removal of
    observations without dependent variable, intelligent imputation
    of missing values using median for numeric and mode for
    categorical, and final validation of data integrity.

    Parameters:
        Dataframe (pd.DataFrame): Complete DataFrame with all
            study variables.
        Variable_Dependiente (str): Name of the column representing
            the dependent variable (Y).
        Variables_Independientes (list): List with the names of
            columns to be used as independent variables.

    Returns:
        tuple: (Variables_X, Variable_Y) where Variables_X is a
            DataFrame with clean independent variables and
            Variable_Y is a Series with the dependent variable.

    """

    # Filter observations that have the dependent variable.
    Datos_Modelo = Dataframe.dropna(subset=[Variable_Dependiente]).copy()

    print(f"Variable dependiente: {Variable_Dependiente}")
    print(f"Observaciones disponibles: {len(Datos_Modelo)}")

    # Prepare independent variables.
    Variables_X = Datos_Modelo[Variables_Independientes].copy()
    Variable_Y = Datos_Modelo[Variable_Dependiente].copy()

    # Impute missing values in independent variables.
    print(f"Valores faltantes en X antes de imputación:")
    Faltantes_Antes = Variables_X.isnull().sum()
    Faltantes_Antes = Faltantes_Antes[Faltantes_Antes > 0]

    if len(Faltantes_Antes) > 0:
        for Variable in Faltantes_Antes.index:
            if Variables_X[Variable].dtype in ['int64', 'float64']:
                # Numeric variables: impute with median.
                Variables_X[Variable] = Variables_X[Variable].fillna(
                    Variables_X[Variable].median()
                )
                print(f"  {Variable}: {Faltantes_Antes[Variable]} → "
                      f"imputados con mediana")
            else:
                # Categorical variables: impute with mode.
                Moda = Variables_X[Variable].mode()
                if len(Moda) > 0:
                    Variables_X[Variable] = Variables_X[Variable].fillna(
                        Moda[0]
                    )
                    print(f"  {Variable}: {Faltantes_Antes[Variable]} → "
                          f"imputados con moda")
    else:
        print("  ✅ No hay valores faltantes")

    # Verify that no missing values remain.
    Faltantes_Despues = Variables_X.isnull().sum().sum()
    if Faltantes_Despues > 0:
        print(f"⚠️  Quedan {Faltantes_Despues} valores faltantes")
        # Remove observations with remaining missing values.
        Indices_Completos = Variables_X.dropna().index
        Variables_X = Variables_X.loc[Indices_Completos]
        Variable_Y = Variable_Y.loc[Indices_Completos]
        print(f"Observaciones finales: {len(Variables_X)}")

    return Variables_X, Variable_Y


def Codificar_Variables_Categoricas_A_Numericas(
    Variables_X: pd.DataFrame
) -> pd.DataFrame:

    """
    Transforms categorical variables into numeric representations
    using direct conversion and label encoding as a fallback
    strategy.

    Systematically processes all object-type columns, first
    attempting direct conversion to numeric values. For variables
    that cannot be converted directly, applies Label Encoding.

    Parameters:
        Variables_X (pd.DataFrame): DataFrame containing the
            independent variables, including object-type columns
            that need to be converted.

    Returns:
        pd.DataFrame: DataFrame with all variables converted
            to numeric types.

    """

    from sklearn.preprocessing import LabelEncoder

    Variables_X_Convertidas = Variables_X.copy()

    for Columna in Variables_X.columns:
        if Variables_X[Columna].dtype == 'object':
            print(f"Convirtiendo {Columna} a numérica...")

            # Try to convert to numeric directly.
            Variables_X_Convertidas[Columna] = pd.to_numeric(
                Variables_X[Columna], errors='coerce'
            )

            # If non-numeric values remain, use LabelEncoder.
            if Variables_X_Convertidas[Columna].isnull().any():
                Encoder = LabelEncoder()
                Valores_Validos = Variables_X[Columna].dropna()
                if len(Valores_Validos) > 0:
                    Variables_X_Convertidas[Columna] = \
                        Variables_X_Convertidas[Columna].fillna(-999)
                    Variables_X_Convertidas[Columna] = Encoder.fit_transform(
                        Variables_X_Convertidas[Columna].astype(str)
                    )

    return Variables_X_Convertidas


def Codificar_Variables_Booleanas_A_Numericas(
    Variables_X: pd.DataFrame
) -> pd.DataFrame:

    """
    Transforms boolean variables (True/False) to binary integer
    representation (1/0) to ensure compatibility with statistical
    modeling libraries.

    Automatically identifies all boolean-type columns and converts
    them to binary integers: True -> 1, False -> 0.

    Parameters:
        Variables_X (pd.DataFrame): DataFrame containing the
            independent variables, including possible boolean-type
            columns.

    Returns:
        pd.DataFrame: DataFrame with all boolean variables
            converted to integers (0 and 1).

    """

    Variables_X_Convertidas = Variables_X.copy()

    Variables_Booleanas = []
    for Columna in Variables_X.columns:
        if Variables_X[Columna].dtype == 'bool':
            Variables_Booleanas.append(Columna)
            Variables_X_Convertidas[Columna] = \
                Variables_X[Columna].astype(int)

    print(f"Variables booleanas convertidas a enteros: "
          f"{len(Variables_Booleanas)}")

    return Variables_X_Convertidas


# =============================================================================
# DATA LOADING FUNCTIONS.
# =============================================================================

def Cargar_Bases_Datos(Ruta_Carpeta: str) -> dict:

    """
    Loads the Generales and Ballotage databases from
    Excel files.

    Parameters:
        Ruta_Carpeta (str): Path to the folder containing
            the Excel files.

    Returns:
        dict: Dictionary with the loaded databases.

    """

    dfs_Finales = {}

    for Nombre in ['Generales', 'Ballotage']:
        Ruta_Archivo = os.path.join(Ruta_Carpeta, f'{Nombre}.xlsx')
        if os.path.exists(Ruta_Archivo):
            dfs_Finales[Nombre] = pd.read_excel(Ruta_Archivo)
            print(f"  Cargado: {Nombre} "
                  f"({len(dfs_Finales[Nombre])} registros)")
        else:
            print(f"  Advertencia: No se encontró {Ruta_Archivo}")

    return dfs_Finales


# =============================================================================
# ROBUST LINEAR MODEL (OLS WITH HC3 ERRORS).
# =============================================================================

def Modelo_Lineal_Robusto(
    Variables_X: pd.DataFrame,
    Variable_Y: pd.Series,
    Nombre_Variable: str,
    Criterio_Eliminacion: str = 'aic',
    Umbral_VIF: float = 5.0,
    Alpha_Significancia: float = 0.05,
    Detectar_Outliers: bool = True,
    Iteracion: int = 1,
    Variables_Originales: list = None,
    Historial_Proceso: list = None
) -> dict:

    """
    Implements robust linear regression with HC3 standard errors,
    sequential variable elimination based on AIC and multicollinearity
    detection using VIF.

    Parameters:
        Variables_X (pd.DataFrame): DataFrame with numeric
            independent variables.
        Variable_Y (pd.Series): Numeric dependent variable.
        Nombre_Variable (str): Descriptive name of the dependent
            variable.
        Criterio_Eliminacion (str): Elimination criterion
            ('aic', 'bic', 'pvalue').
        Umbral_VIF (float): Maximum threshold for VIF.
        Alpha_Significancia (float): Significance level.
        Detectar_Outliers (bool): Whether to detect influential outliers.
        Iteracion (int): Current iteration number (internal).
        Variables_Originales (list): Original list of variables
            (internal).
        Historial_Proceso (list): Process history (internal).

    Returns:
        dict: Dictionary with final model results.

    """

    # Initialize parameters in first iteration.
    if Variables_Originales is None:
        Variables_Originales = Variables_X.columns.tolist()

    if Historial_Proceso is None:
        Historial_Proceso = []

    # Verify input data.
    if len(Variables_X.columns) == 0:
        print("  ERROR: No quedan variables independientes")
        return None

    if len(Variables_X) != len(Variable_Y):
        print("  ERROR: Dimensiones inconsistentes")
        return None

    # Detection of influential outliers (first iteration only).
    Indices_Outliers = []
    if Detectar_Outliers and Iteracion == 1:
        Variables_X_Temp = sm.add_constant(Variables_X)
        Modelo_Temp = sm.OLS(Variable_Y, Variables_X_Temp).fit()
        Influence = OLSInfluence(Modelo_Temp)

        Cooks_Distance = Influence.cooks_distance[0]
        Leverage = Influence.hat_matrix_diag
        Residuos_Studentizados = Influence.resid_studentized_external

        N_Obs = len(Variables_X)
        P_Vars = len(Variables_X.columns) + 1
        Umbral_Cooks = 4.0 / N_Obs
        Umbral_Leverage = 2.0 * P_Vars / N_Obs
        Umbral_Residuos = 3.0

        Outliers_Cooks = np.where(Cooks_Distance > Umbral_Cooks)[0]
        Outliers_Leverage = np.where(Leverage > Umbral_Leverage)[0]
        Outliers_Residuos = np.where(
            np.abs(Residuos_Studentizados) > Umbral_Residuos
        )[0]

        Indices_Outliers = np.unique(np.concatenate([
            Outliers_Cooks, Outliers_Leverage, Outliers_Residuos
        ])).tolist()

    # Detection and elimination of multicollinearity with VIF.
    Variables_X_VIF = Variables_X.copy()
    Variables_Eliminadas_VIF = []

    while True:
        if len(Variables_X_VIF.columns) < 2:
            break

        Variables_X_Con_Constante = sm.add_constant(Variables_X_VIF)
        VIF_Valores = {}

        for i, Variable in enumerate(Variables_X_VIF.columns):
            try:
                VIF = variance_inflation_factor(
                    Variables_X_Con_Constante.values, i + 1
                )
                VIF_Valores[Variable] = VIF
            except:
                VIF_Valores[Variable] = float('inf')

        Max_VIF_Variable = max(
            VIF_Valores.keys(),
            key=lambda x: VIF_Valores[x]
        )
        Max_VIF_Valor = VIF_Valores[Max_VIF_Variable]

        if Max_VIF_Valor <= Umbral_VIF:
            break

        Variables_Eliminadas_VIF.append(Max_VIF_Variable)
        Variables_X_VIF = Variables_X_VIF.drop(
            columns=[Max_VIF_Variable]
        )

    # Calculate final VIF.
    VIF_Final = {}
    if len(Variables_X_VIF.columns) >= 2:
        Variables_X_Con_Constante = sm.add_constant(Variables_X_VIF)
        for i, Variable in enumerate(Variables_X_VIF.columns):
            try:
                VIF_Final[Variable] = variance_inflation_factor(
                    Variables_X_Con_Constante.values, i + 1
                )
            except:
                VIF_Final[Variable] = float('nan')

    Variables_X_Finales = Variables_X_VIF.copy()

    if len(Variables_X_Finales.columns) == 0:
        print("  ERROR: Todas las variables eliminadas por VIF")
        return None

    # Fit model with robust HC3 standard errors.
    Variables_X_Con_Constante = sm.add_constant(Variables_X_Finales)
    Modelo = sm.OLS(
        Variable_Y, Variables_X_Con_Constante
    ).fit(cov_type='HC3')

    # Variable evaluation and elimination criteria.
    P_Valores = Modelo.pvalues
    Variables_Candidatas = [
        var for var in P_Valores.index if var != 'const'
    ]

    if len(Variables_Candidatas) == 0:
        return None

    Variable_A_Eliminar = None
    Razon_Eliminacion = None

    if Criterio_Eliminacion == 'pvalue':
        P_Vals_Variables = {
            var: P_Valores[var] for var in Variables_Candidatas
        }
        Peor_Variable = max(
            P_Vals_Variables.keys(),
            key=lambda x: P_Vals_Variables[x]
        )

        if P_Vals_Variables[Peor_Variable] >= Alpha_Significancia:
            Variable_A_Eliminar = Peor_Variable
            Razon_Eliminacion = (
                f"p-valor = {P_Vals_Variables[Peor_Variable]:.6e}"
            )

    elif Criterio_Eliminacion in ['aic', 'bic']:
        AIC_Actual = Modelo.aic if Criterio_Eliminacion == 'aic' \
                     else Modelo.bic
        Mejor_Criterio = AIC_Actual

        for Variable_Test in Variables_Candidatas:
            Variables_Sin_Test = [
                v for v in Variables_Candidatas if v != Variable_Test
            ]

            if len(Variables_Sin_Test) == 0:
                continue

            X_Test = sm.add_constant(
                Variables_X_Finales[Variables_Sin_Test]
            )
            Modelo_Test = sm.OLS(
                Variable_Y, X_Test
            ).fit(cov_type='HC3')

            Criterio_Test = Modelo_Test.aic if Criterio_Eliminacion \
                            == 'aic' else Modelo_Test.bic

            if Criterio_Test < Mejor_Criterio:
                Mejor_Criterio = Criterio_Test
                Variable_A_Eliminar = Variable_Test
                Razon_Eliminacion = (
                    f"{Criterio_Eliminacion.upper()} mejora: "
                    f"{AIC_Actual:.2f} → {Criterio_Test:.2f}"
                )

    elif Criterio_Eliminacion == 'combinado':
        # Combined: AIC + significance.
        P_Vals_Variables = {
            var: P_Valores[var] for var in Variables_Candidatas
        }
        Variables_No_Sig = [
            var for var in Variables_Candidatas
            if P_Vals_Variables[var] >= Alpha_Significancia
        ]

        if Variables_No_Sig:
            # If there are non-significant ones, use AIC among them.
            AIC_Actual = Modelo.aic
            Mejor_AIC = AIC_Actual

            for Variable_Test in Variables_No_Sig:
                Variables_Sin_Test = [
                    v for v in Variables_Candidatas if v != Variable_Test
                ]

                if len(Variables_Sin_Test) == 0:
                    continue

                X_Test = sm.add_constant(
                    Variables_X_Finales[Variables_Sin_Test]
                )
                Modelo_Test = sm.OLS(
                    Variable_Y, X_Test
                ).fit(cov_type='HC3')

                if Modelo_Test.aic < Mejor_AIC:
                    Mejor_AIC = Modelo_Test.aic
                    Variable_A_Eliminar = Variable_Test
                    Razon_Eliminacion = (
                        f"No significativa "
                        f"(p={P_Vals_Variables[Variable_Test]:.4e}) "
                        f"+ AIC mejora"
                    )

    # Identify significant variables.
    Variables_Significativas = [
        var for var in Variables_Candidatas
        if P_Valores[var] < Alpha_Significancia
    ]

    # Record iteration in history.
    Historial_Actual = {
        'Iteracion': Iteracion,
        'Variables_Incluidas': Variables_Candidatas.copy(),
        'Variables_Significativas': Variables_Significativas.copy(),
        'R_Cuadrado_Ajustado': Modelo.rsquared_adj,
        'AIC': Modelo.aic,
        'BIC': Modelo.bic,
        'Variable_Eliminada': Variable_A_Eliminar,
        'Razon_Eliminacion': Razon_Eliminacion
    }
    Historial_Proceso.append(Historial_Actual)

    # Decide next action.
    if Variable_A_Eliminar is None:
        # Final model reached.

        # Diagnostic tests.
        JB_Stat, JB_Pval = jarque_bera(Modelo.resid)
        BP_LM, BP_LM_Pval, BP_F, BP_F_Pval = het_breuschpagan(
            Modelo.resid, Variables_X_Con_Constante
        )
        DW_Stat = durbin_watson(Modelo.resid)
        Num_Condicion = np.linalg.cond(Variables_X_Con_Constante)

        Variables_Eliminadas_Total = [
            var for var in Variables_Originales
            if var not in Variables_Significativas
        ]

        Coeficientes = Modelo.params.to_dict()
        P_Valores_Dict = P_Valores.to_dict()
        Intervalos_Confianza = Modelo.conf_int().to_dict()

        Resultados_Finales = {
            'Variable_Dependiente': Nombre_Variable,
            'N_Observaciones': int(Modelo.nobs),
            'N_Observaciones_Outliers': len(Indices_Outliers),
            'R_Cuadrado': Modelo.rsquared,
            'R_Cuadrado_Ajustado': Modelo.rsquared_adj,
            'AIC': Modelo.aic,
            'BIC': Modelo.bic,
            'F_Estadistico': Modelo.fvalue,
            'P_Valor_Modelo': Modelo.f_pvalue,
            'Parametro_Escala_Robusto': Modelo.scale,
            'Variables_Significativas': Variables_Significativas,
            'Variables_Eliminadas_Total': Variables_Eliminadas_Total,
            'Variables_Eliminadas_VIF': Variables_Eliminadas_VIF,
            'Coeficientes': Coeficientes,
            'P_Valores': P_Valores_Dict,
            'Intervalos_Confianza_95': Intervalos_Confianza,
            'VIF_Final': VIF_Final,
            'Iteraciones_Totales': Iteracion,
            'Criterio_Utilizado': Criterio_Eliminacion,
            'Umbral_VIF_Usado': Umbral_VIF,
            'Alpha_Significancia_Usado': Alpha_Significancia,
            'Outliers_Detectados': Indices_Outliers,
            'Historial_Eliminaciones': Historial_Proceso,
            'Test_Normalidad_JB': (JB_Stat, JB_Pval),
            'Test_Heterocedasticidad_BP': (BP_LM, BP_LM_Pval),
            'Durbin_Watson': DW_Stat,
            'Numero_Condicion': Num_Condicion,
            'Modelo_Objeto': Modelo,
            'Matriz_Covarianza_Robusta': Modelo.cov_HC3
        }

        return Resultados_Finales

    else:
        # Continue eliminating variables.
        Variables_X_Nuevas = Variables_X_Finales.drop(
            columns=[Variable_A_Eliminar]
        )

        return Modelo_Lineal_Robusto(
            Variables_X_Nuevas, Variable_Y, Nombre_Variable,
            Criterio_Eliminacion, Umbral_VIF, Alpha_Significancia,
            Detectar_Outliers, Iteracion + 1, Variables_Originales,
            Historial_Proceso
        )


def Evaluar_Modelo_Robusto(
    Resultados_Modelo: dict,
    Mostrar_Detalles_Tecnicos: bool = True
) -> dict:

    """
    Interprets and evaluates the results of a robust linear
    regression model, providing analysis of methodological validity,
    goodness of fit, predictor significance and recommendations.

    The evaluation includes analysis of:
    - Goodness of fit and explanatory power
    - Validity of fundamental statistical assumptions
    - Significance and interpretation of coefficients
    - Presence of residual multicollinearity
    - Detection of influential observations
    - Adequacy of sample size
    - Numerical stability of the model

    Parameters:
        Resultados_Modelo (dict): Dictionary with complete results
            from the robust model generated by Modelo_Lineal_Robusto().
        Mostrar_Detalles_Tecnicos (bool): If True, includes section
            with detailed technical diagnostics.

    Returns:
        dict: Dictionary with structured model evaluation:
            - 'Validez_General': str
            - 'Calidad_Ajuste': str
            - 'Cumplimiento_Supuestos': dict
            - 'Interpretaciones_Coeficientes': dict
            - 'Recomendaciones_Principales': list
            - 'Limitaciones_Identificadas': list
            - 'Puntuacion_Confianza': float

    """

    if Resultados_Modelo is None:
        return None

    # General validity criteria.
    R2_Ajustado = Resultados_Modelo['R_Cuadrado_Ajustado']
    N_Variables = len(Resultados_Modelo['Variables_Significativas'])
    N_Obs = Resultados_Modelo['N_Observaciones']
    P_Valor_Modelo = Resultados_Modelo['P_Valor_Modelo']

    # Evaluate basic criteria.
    Criterios_Validez = {
        'Variables_Significativas': N_Variables > 0,
        'Modelo_Globalmente_Significativo': P_Valor_Modelo < 0.05,
        'Tamaño_Muestra_Adecuado': N_Obs >= (N_Variables * 10 + 50),
        'R2_Minimo_Aceptable': R2_Ajustado >= 0.10
    }

    Criterios_Cumplidos = sum(Criterios_Validez.values())
    Total_Criterios = len(Criterios_Validez)

    # Determine general validity.
    if Criterios_Cumplidos == Total_Criterios:
        Validez_General = "VALIDO"
    elif Criterios_Cumplidos >= Total_Criterios * 0.75:
        Validez_General = "VALIDO_CON_RESERVAS"
    else:
        Validez_General = "NO_VALIDO"

    # Categorize quality according to adjusted R².
    if R2_Ajustado >= 0.70:
        Calidad_Ajuste = "EXCELENTE"
    elif R2_Ajustado >= 0.50:
        Calidad_Ajuste = "BUENA"
    elif R2_Ajustado >= 0.30:
        Calidad_Ajuste = "MODERADA"
    elif R2_Ajustado >= 0.10:
        Calidad_Ajuste = "BAJA"
    else:
        Calidad_Ajuste = "MUY_BAJA"

    # Evaluate statistical assumptions.
    Supuestos = {}

    # Normality of residuals (Jarque-Bera).
    JB_Stat, JB_Pval = Resultados_Modelo['Test_Normalidad_JB']
    Supuestos['Normalidad'] = JB_Pval >= 0.05

    # Homoscedasticity (Breusch-Pagan).
    BP_Stat, BP_Pval = Resultados_Modelo['Test_Heterocedasticidad_BP']
    Supuestos['Homocedasticidad'] = BP_Pval >= 0.05

    # Independence (Durbin-Watson).
    DW = Resultados_Modelo['Durbin_Watson']
    Supuestos['Independencia'] = 1.5 <= DW <= 2.5

    # Multicollinearity (VIF).
    VIF_Valores = list(Resultados_Modelo['VIF_Final'].values()) \
                  if Resultados_Modelo['VIF_Final'] else []
    Max_VIF = max(VIF_Valores) if VIF_Valores else 1.0
    Supuestos['Sin_Multicolinealidad'] = Max_VIF <= 5.0

    Supuestos_Cumplidos = sum(Supuestos.values())
    Total_Supuestos = len(Supuestos)

    # Interpret coefficients.
    Variables_Sig = Resultados_Modelo['Variables_Significativas']
    Coeficientes = Resultados_Modelo['Coeficientes']
    P_Valores = Resultados_Modelo['P_Valores']
    Intervalos = Resultados_Modelo['Intervalos_Confianza_95']

    Interpretaciones_Variables = {}

    for Variable in Variables_Sig:
        if Variable in Coeficientes:
            Beta = Coeficientes[Variable]
            P_Val = P_Valores[Variable]

            # Determine effect magnitude.
            Magnitud_Abs = abs(Beta)
            if Magnitud_Abs >= 1.0:
                Magnitud_Desc = "GRANDE"
            elif Magnitud_Abs >= 0.5:
                Magnitud_Desc = "MODERADO"
            elif Magnitud_Abs >= 0.2:
                Magnitud_Desc = "PEQUEÑO"
            else:
                Magnitud_Desc = "MUY_PEQUEÑO"

            Direccion = "POSITIVO" if Beta > 0 else "NEGATIVO"

            Interpretaciones_Variables[Variable] = {
                'Beta': Beta,
                'P_Valor': P_Val,
                'Magnitud': Magnitud_Desc,
                'Direccion': Direccion
            }

    # Generate recommendations and limitations.
    Recomendaciones = []
    Limitaciones = []

    if Validez_General == "VALIDO":
        Recomendaciones.append(
            "El modelo es estadísticamente válido para inferencia"
        )
    elif Validez_General == "VALIDO_CON_RESERVAS":
        Recomendaciones.append(
            "El modelo tiene limitaciones pero puede ser útil"
        )
        Limitaciones.append(
            "Interpretar resultados con cautela"
        )
    else:
        Limitaciones.append(
            "Modelo no válido para inferencia estadística"
        )

    if not Supuestos['Normalidad']:
        if JB_Pval < 0.001:
            Limitaciones.append(
                "Violación severa de normalidad"
            )
        else:
            Limitaciones.append(
                "Violación leve de normalidad - HC3 mitiga"
            )

    if not Supuestos['Homocedasticidad']:
        Recomendaciones.append(
            "Heterocedasticidad detectada - HC3 apropiado"
        )

    if not Supuestos['Independencia']:
        Limitaciones.append(
            "Posible autocorrelación en residuos"
        )

    if not Supuestos['Sin_Multicolinealidad']:
        Limitaciones.append(
            f"Multicolinealidad residual (VIF max = {Max_VIF:.2f})"
        )

    if R2_Ajustado < 0.30:
        Limitaciones.append(
            "Bajo poder explicativo"
        )

    # Calculate confidence score (0-100).
    N_Outliers = Resultados_Modelo['N_Observaciones_Outliers']
    Porcentaje_Outliers = (N_Outliers / N_Obs) * 100

    Puntuacion = 0
    Puntuacion += (Criterios_Cumplidos / Total_Criterios) * 40
    Puntuacion += (Supuestos_Cumplidos / Total_Supuestos) * 30

    if R2_Ajustado >= 0.70:
        Puntuacion += 20
    elif R2_Ajustado >= 0.50:
        Puntuacion += 15
    elif R2_Ajustado >= 0.30:
        Puntuacion += 10
    elif R2_Ajustado >= 0.10:
        Puntuacion += 5

    if Porcentaje_Outliers <= 5:
        Puntuacion += 10
    elif Porcentaje_Outliers <= 10:
        Puntuacion += 5

    # Determine confidence level.
    if Puntuacion >= 80:
        Nivel_Confianza = "ALTA"
    elif Puntuacion >= 60:
        Nivel_Confianza = "MODERADA"
    elif Puntuacion >= 40:
        Nivel_Confianza = "BAJA"
    else:
        Nivel_Confianza = "MUY_BAJA"

    Evaluacion_Estructurada = {
        'Validez_General': Validez_General,
        'Calidad_Ajuste': Calidad_Ajuste,
        'Cumplimiento_Supuestos': Supuestos,
        'Interpretaciones_Coeficientes': Interpretaciones_Variables,
        'Recomendaciones_Principales': Recomendaciones,
        'Limitaciones_Identificadas': Limitaciones,
        'Puntuacion_Confianza': Puntuacion,
        'Nivel_Confianza': Nivel_Confianza
    }

    return Evaluacion_Estructurada


def Ajustar_Modelos_Robustos(
    dfs_Finales: dict,
    Variables_Dep: dict = None
) -> dict:

    """
    Fits robust linear models for all dependent variables
    specified in both datasets.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.
        Variables_Dep (dict): Dictionary with dependent variables
            per dataset.

    Returns:
        dict: Model results by dataset and variable.

    """

    if Variables_Dep is None:
        Variables_Dep = Variables_Dependientes_Robusto

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"MODELOS ROBUSTOS - {Nombre_df}")
        print('='*60)

        Resultados[Nombre_df] = {}

        Variables_A_Modelar = Variables_Dep.get(Nombre_df, [])

        for Variable_Dep in Variables_A_Modelar:
            if Variable_Dep not in df.columns:
                print(f"  {Variable_Dep}: No encontrada")
                continue

            print(f"\n  Ajustando: {Variable_Dep}")

            Vars_Indep_Existentes = Variables_Independientes

            # Prepare data with the same logic as the notebooks.
            X, y = Limpiar_Y_Estructurar_Datos_Para_Modelado(
                df, Variable_Dep, Vars_Indep_Existentes
            )
            X = Codificar_Variables_Booleanas_A_Numericas(X)
            X = Codificar_Variables_Categoricas_A_Numericas(X)

            Resultado = Modelo_Lineal_Robusto(
                X,
                y,
                Variable_Dep,
                Criterio_Eliminacion='aic',
                Umbral_VIF=5.0,
                Alpha_Significancia=0.05,
                Detectar_Outliers=True
            )

            if Resultado is not None:
                Evaluacion = Evaluar_Modelo_Robusto(
                    Resultado,
                    Mostrar_Detalles_Tecnicos=False
                )
                Resultado['Evaluacion_Modelo'] = Evaluacion
                Resultados[Nombre_df][Variable_Dep] = Resultado
                print(f"    N={Resultado['N_Observaciones']}, "
                      f"R²={Resultado['R_Cuadrado_Ajustado']:.4f}, "
                      f"Vars={len(Resultado['Variables_Significativas'])}")
            else:
                print(f"    Error en el ajuste")

    return Resultados


# =============================================================================
# RECURSIVE GLM MODEL.
# =============================================================================

def Ajustar_GLM_Recursivo(
    df: pd.DataFrame,
    Variable_Dependiente: str,
    Variables_Indep: list,
    Nombre_Eleccion: str,
    Familia_GLM = None
) -> dict:

    """
    Fits GLM model recursively until convergence.
    Removes non-significant variables (p >= 0.05) in each
    iteration until all remaining variables are significant.

    Parameters:
        df (pd.DataFrame): DataFrame with the data.
        Variable_Dependiente (str): Name of the dependent variable.
        Variables_Indep (list): List of independent variables.
        Nombre_Eleccion (str): Election identifier.
        Familia_GLM: Distribution family for the GLM. If None,
            uses Gaussian with Identity link.

    Returns:
        dict: Dictionary with final model and statistics.

    """

    # Prepare data removing NaN.
    Columnas_Necesarias = [Variable_Dependiente] + Variables_Indep

    df_Limpio = df[Columnas_Necesarias].dropna()

    print(f"    Observaciones válidas: {len(df_Limpio)}")

    # Available independent variables.
    Variables_Actuales = Variables_Indep.copy()

    Iteracion = 0
    Variables_Anteriores = []

    # Default family: Gaussian with Identity link.
    if Familia_GLM is None:
        Familia_GLM = sm.families.Gaussian(sm.families.links.Identity())

    while True:
        Iteracion += 1

        if len(Variables_Actuales) == 0:
            return None

        # Prepare X and y.
        X = df_Limpio[Variables_Actuales].copy()
        y = df_Limpio[Variable_Dependiente]

        # Convert booleans to int.
        for Col in X.columns:
            if X[Col].dtype == bool:
                X[Col] = X[Col].astype(int)

        # Add constant.
        X_Con_Constante = sm.add_constant(X)

        # Fit model.
        Modelo = sm.GLM(y, X_Con_Constante, family=Familia_GLM).fit()

        # Extract p-values.
        P_Valores = Modelo.pvalues

        # Filter significant variables.
        Variables_Significativas = [
            var for var in Variables_Actuales
            if P_Valores[var] < 0.05
        ]

        # Verify convergence.
        if set(Variables_Significativas) == set(Variables_Anteriores):
            break

        if len(Variables_Significativas) == 0:
            break

        # Update for next iteration.
        Variables_Anteriores = Variables_Actuales.copy()
        Variables_Actuales = Variables_Significativas.copy()

    # Calculate final statistics.
    AIC = Modelo.aic
    BIC = Modelo.bic

    Deviance_Nulo = Modelo.null_deviance
    Deviance_Residual = Modelo.deviance
    Deviance_Explicada = 1 - (Deviance_Residual / Deviance_Nulo)

    N = len(df_Limpio)
    K = len(Variables_Actuales)

    if K > 0 and N > K + 1:
        F_Estadistico = (
            ((Deviance_Nulo - Deviance_Residual) / K) /
            (Deviance_Residual / (N - K - 1))
        )
        P_Valor_F = 1 - stats.f.cdf(F_Estadistico, K, N - K - 1)
    else:
        F_Estadistico = np.nan
        P_Valor_F = np.nan

    # Build results DataFrame by variable.
    Resultados_Vars = []

    for Var in Variables_Actuales:
        Coef = Modelo.params[Var]
        P_Val = Modelo.pvalues[Var]
        IC = Modelo.conf_int().loc[Var]

        Resultados_Vars.append({
            'Variable': Var,
            'Coeficiente': Coef,
            'P_Valor': P_Val,
            'IC_Inferior': IC[0],
            'IC_Superior': IC[1]
        })

    df_Resultados = pd.DataFrame(Resultados_Vars)
    if len(df_Resultados) > 0:
        df_Resultados = df_Resultados.sort_values('P_Valor')

    return {
        'Modelo': Modelo,
        'Eleccion': Nombre_Eleccion,
        'Variable_Dependiente': Variable_Dependiente,
        'N_Observaciones': N,
        'N_Variables': K,
        'AIC': AIC,
        'BIC': BIC,
        'F_Estadistico': F_Estadistico,
        'P_Valor_F': P_Valor_F,
        'Deviance_Explicada': Deviance_Explicada,
        'Variables_Significativas': Variables_Actuales,
        'Resultados_Variables': df_Resultados,
        'N_Iteraciones': Iteracion
    }


def Obtener_Familia_GLM(Variable_Dependiente: str):

    """
    Returns the appropriate GLM family according to the dependent variable.

    - Indice_Conservadurismo: Gamma with Log link (skewed distribution).
    - Rest: Gaussian with Identity link.

    Parameters:
        Variable_Dependiente (str): Name of the dependent variable.

    Returns:
        statsmodels.families: Corresponding GLM family.

    """

    if Variable_Dependiente == 'Indice_Conservadurismo':
        # Positively skewed distribution, heavy tails.
        return sm.families.Gamma(sm.families.links.Log())
    else:
        # Approximately normal distribution.
        return sm.families.Gaussian(sm.families.links.Identity())


def Obtener_Variables_Independientes_GLM(Variable_Dependiente: str) -> list:

    """
    Returns the appropriate list of independent variables for each
    GLM dependent variable, excluding the dependent variable but
    keeping its time variable if it exists.

    Parameters:
        Variable_Dependiente (str): Name of the dependent variable.

    Returns:
        list: List of independent variables.

    """

    # Base list of independent variables.
    Variables_Base = [
        'Conservative_Cluster',
        'Progressive_Cluster',
        'Edad',
        'Genero_Masculino',
        'Genero_Otro',
        'Region_CABA',
        'Region_Centro',
        'Region_Cuyo',
        'Region_Norte',
        'Region_Patagonia',
        'Voto_2019_JL_Espert',
        'Voto_2019_J_Gomez_Centurion',
        'Voto_2019_Mauricio_Macri',
        'Voto_2019_Nicolas_Del_Caño',
        'Voto_2019_Roberto_Lavagna',
        'Categoria_PASO_2023_Left_Wing',
        'Categoria_PASO_2023_Moderate_Right_A',
        'Categoria_PASO_2023_Moderate_Right_B',
        'Categoria_PASO_2023_Right_Wing_Libertarian',
        'Categoria_PASO_2023_Centre',
        'Cercania_Massa',
        'Cercania_Milei',
        'Influencia_Redes',
        'Red_Social_Facebook',
        'Red_Social_Instagram',
        'Red_Social_Threads',
        'Red_Social_Tiktok',
        'Red_Social_Youtube',
        'Red_Social_Whatsapp',
        'Red_Social_Telegram',
        'Influencia_Prensa',
        'Medios_Prensa_Prensa_Obrera',
        'Medios_Prensa_Diario_Universal',
        'Medios_Prensa_Popular',
        'Medios_Prensa_Izquierda_Diario',
        'Medios_Prensa_Clarin',
        'Medios_Prensa_Perfil',
        'Medios_Prensa_Pagina_12',
        'Medios_Prensa_Infobae',
        'Medios_Prensa_El_Cronista',
        'Medios_Prensa_La_Nacion',
        'Medios_Prensa_Tiempo_Argentino',
        'Indice_Positividad'
    ]

    # Add variables according to the dependent variable.
    if Variable_Dependiente == 'Indice_Progresismo':
        # Excludes Indice_Progresismo, includes the rest.
        Variables_Base.extend([
            'Autopercepcion_Izq_Der',
            'Autopercepcion_Con_Pro',
            'Autopercepcion_Per_Antiper',
            'Indice_Progresismo_Tiempo',
            'Indice_Conservadurismo',
            'Indice_Conservadurismo_Tiempo'
        ])

    elif Variable_Dependiente == 'Indice_Conservadurismo':
        # Excludes Indice_Conservadurismo, includes the rest.
        Variables_Base.extend([
            'Autopercepcion_Izq_Der',
            'Autopercepcion_Con_Pro',
            'Autopercepcion_Per_Antiper',
            'Indice_Progresismo',
            'Indice_Progresismo_Tiempo',
            'Indice_Conservadurismo_Tiempo'
        ])

    elif Variable_Dependiente == 'Autopercepcion_Izq_Der':
        # Excludes Autopercepcion_Izq_Der, includes the rest.
        Variables_Base.extend([
            'Autopercepcion_Con_Pro',
            'Autopercepcion_Per_Antiper',
            'Indice_Progresismo',
            'Indice_Progresismo_Tiempo',
            'Indice_Conservadurismo',
            'Indice_Conservadurismo_Tiempo'
        ])

    elif Variable_Dependiente == 'Autopercepcion_Con_Pro':
        # Excludes Autopercepcion_Con_Pro, includes the rest.
        Variables_Base.extend([
            'Autopercepcion_Izq_Der',
            'Autopercepcion_Per_Antiper',
            'Indice_Progresismo',
            'Indice_Progresismo_Tiempo',
            'Indice_Conservadurismo',
            'Indice_Conservadurismo_Tiempo'
        ])

    elif Variable_Dependiente == 'Autopercepcion_Per_Antiper':
        # Excludes Autopercepcion_Per_Antiper, includes the rest.
        Variables_Base.extend([
            'Autopercepcion_Izq_Der',
            'Autopercepcion_Con_Pro',
            'Indice_Progresismo',
            'Indice_Progresismo_Tiempo',
            'Indice_Conservadurismo',
            'Indice_Conservadurismo_Tiempo'
        ])

    return Variables_Base


def Ajustar_Modelos_GLM(dfs_Finales: dict) -> dict:

    """
    Fits GLM models for indices and self-perceptions in both
    datasets.

    Uses Gamma family with Log link for Indice_Conservadurismo
    (skewed distribution) and Gaussian family with Identity link
    for the rest.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.

    Returns:
        dict: GLM model results.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"MODELOS GLM - {Nombre_df}")
        print('='*60)

        Resultados[Nombre_df] = {}

        for Variable_Dep in Variables_Dependientes_GLM:
            if Variable_Dep not in df.columns:
                print(f"  {Variable_Dep}: No encontrada")
                continue

            print(f"\n  Ajustando: {Variable_Dep}")

            # Get specific independent variables.
            Vars_Indep = Obtener_Variables_Independientes_GLM(Variable_Dep)

            # Get appropriate GLM family.
            Familia = Obtener_Familia_GLM(Variable_Dep)

            Resultado = Ajustar_GLM_Recursivo(
                df, Variable_Dep, Vars_Indep, Nombre_df, Familia
            )

            if Resultado is not None:
                Resultados[Nombre_df][Variable_Dep] = Resultado
                print(f"    N={Resultado['N_Observaciones']}, "
                      f"R²={Resultado['Deviance_Explicada']:.4f}, "
                      f"Vars={Resultado['N_Variables']}")
            else:
                print(f"    Error en el ajuste")

    return Resultados


# =============================================================================
# BINARY LOGISTIC REGRESSION.
# =============================================================================

def Ajustar_Regresion_Logistica_Recursiva(
    df: pd.DataFrame,
    Variable_Dependiente: str,
    Variables_Indep: list,
    Nombre_Eleccion: str
) -> dict:

    """
    Fits binary logistic regression model recursively until
    convergence.

    The dependent variable must be binary (0/1).
    Removes non-significant variables (p >= 0.05) in each
    iteration.

    Parameters:
        df (pd.DataFrame): DataFrame with the data.
        Variable_Dependiente (str): Name of binary dependent
            variable.
        Variables_Indep (list): List of independent variables.
        Nombre_Eleccion (str): Election identifier.

    Returns:
        dict: Dictionary with final model and statistics.

    """

    # Prepare data removing NaN.
    Columnas_Necesarias = [Variable_Dependiente] + Variables_Indep

    df_Limpio = df[Columnas_Necesarias].dropna()

    print(f"    Observaciones válidas: {len(df_Limpio)}")

    # Verify variability in dependent variable.
    if df_Limpio[Variable_Dependiente].nunique() < 2:
        return None

    # Available independent variables.
    Variables_Actuales = Variables_Indep.copy()

    Iteracion = 0
    Variables_Anteriores = []

    while True:
        Iteracion += 1

        if len(Variables_Actuales) == 0:
            return None

        # Prepare X and y.
        X = df_Limpio[Variables_Actuales].copy()
        y = df_Limpio[Variable_Dependiente]

        # Convert booleans to int.
        for Col in X.columns:
            if X[Col].dtype == bool:
                X[Col] = X[Col].astype(int)

        # Add constant.
        X_Con_Constante = sm.add_constant(X)

        # Fit logistic model.
        Modelo = sm.GLM(
            y, X_Con_Constante,
            family=sm.families.Binomial()
        ).fit()

        # Extract p-values.
        P_Valores = Modelo.pvalues

        # Filter significant variables.
        Variables_Significativas = [
            var for var in Variables_Actuales
            if P_Valores[var] < 0.05
        ]

        # Verify convergence.
        if set(Variables_Significativas) == set(Variables_Anteriores):
            break

        if len(Variables_Significativas) == 0:
            break

        # Update for next iteration.
        Variables_Anteriores = Variables_Actuales.copy()
        Variables_Actuales = Variables_Significativas.copy()

    # Calculate final statistics.
    AIC = Modelo.aic
    BIC = Modelo.bic

    Deviance_Nulo = Modelo.null_deviance
    Deviance_Residual = Modelo.deviance
    Pseudo_R2 = 1 - (Deviance_Residual / Deviance_Nulo)

    N = len(df_Limpio)
    K = len(Variables_Actuales)

    Chi2_Estadistico = Deviance_Nulo - Deviance_Residual
    P_Valor_Chi2 = 1 - stats.chi2.cdf(Chi2_Estadistico, K) if K > 0 \
                   else np.nan

    # Build results DataFrame with Odds Ratios.
    Resultados_Vars = []

    for Var in Variables_Actuales:
        Coef = Modelo.params[Var]
        P_Val = Modelo.pvalues[Var]
        IC = Modelo.conf_int().loc[Var]

        OR = np.exp(Coef)
        OR_IC_Inf = np.exp(IC[0])
        OR_IC_Sup = np.exp(IC[1])

        Resultados_Vars.append({
            'Variable': Var,
            'Coeficiente': Coef,
            'OR': OR,
            'P_Valor': P_Val,
            'OR_IC_Inferior': OR_IC_Inf,
            'OR_IC_Superior': OR_IC_Sup
        })

    df_Resultados = pd.DataFrame(Resultados_Vars)
    if len(df_Resultados) > 0:
        df_Resultados = df_Resultados.sort_values('P_Valor')

    return {
        'Modelo': Modelo,
        'Eleccion': Nombre_Eleccion,
        'Variable_Dependiente': Variable_Dependiente,
        'N_Observaciones': N,
        'N_Variables': K,
        'AIC': AIC,
        'BIC': BIC,
        'Deviance_Nulo': Deviance_Nulo,
        'Deviance_Residual': Deviance_Residual,
        'Pseudo_R2_McFadden': Pseudo_R2,
        'Chi2_Estadistico': Chi2_Estadistico,
        'P_Valor_Chi2': P_Valor_Chi2,
        'Variables_Significativas': Variables_Actuales,
        'Resultados_Variables': df_Resultados,
        'N_Iteraciones': Iteracion
    }


def Ajustar_Modelos_Logisticos(dfs_Finales: dict) -> dict:

    """
    Fits binary logistic regression models for closeness to each
    candidate in both datasets.

    The dependent variable is binarized: Close (1) if >= 4,
    Not close (0) if <= 3.

    Parameters:
        dfs_Finales (dict): Dictionary with DataFrames.

    Returns:
        dict: Logistic model results.

    """

    Resultados = {}

    for Nombre_df, df in dfs_Finales.items():
        print(f"\n{'='*60}")
        print(f"MODELOS LOGÍSTICOS (CERCANÍA) - {Nombre_df}")
        print('='*60)

        Resultados[Nombre_df] = {}

        for Candidato in Candidatos_Cercania:
            Variable_Cercania = f'Cercania_{Candidato}'

            # Follow notebooks: these candidates only in Generales.
            if (Nombre_df == 'Ballotage' and
                Candidato in ['Schiaretti', 'Bullrich', 'Bregman']):
                print(f"  Cercania_{Candidato}: Omitido en Ballotage")
                continue

            if Variable_Cercania not in df.columns:
                print(f"  {Variable_Cercania}: No encontrada")
                continue

            print(f"\n  Ajustando: Cercanía a {Candidato}")

            # Binarize variable: 1 if close (>= 4), 0 if not.
            Nombre_Binaria = f'Cercano_{Candidato}'
            df[Nombre_Binaria] = (df[Variable_Cercania] >= 4).astype(int)

            Proporcion = df[Nombre_Binaria].mean()
            print(f"    Proporción cercanos: {Proporcion:.2%}")

            # Exclude closeness to candidates from independent variables.
            Vars_Indep = [
                v for v in Variables_Independientes
                if not v.startswith('Cercania_')
            ]

            Resultado = Ajustar_Regresion_Logistica_Recursiva(
                df, Nombre_Binaria, Vars_Indep, Nombre_df
            )

            if Resultado is not None:
                Resultados[Nombre_df][Candidato] = Resultado
                print(f"    N={Resultado['N_Observaciones']}, "
                      f"Pseudo R²={Resultado['Pseudo_R2_McFadden']:.4f}, "
                      f"Vars={Resultado['N_Variables']}")
            else:
                print(f"    Error en el ajuste")

    return Resultados


# =============================================================================
# MAIN EXECUTION.
# =============================================================================

def Ejecutar_Todos_Los_Modelos(Ruta_Datos: str) -> dict:

    """
    Executes all statistical models and returns the results.

    Execution order:
    1. Robust Models (OLS with HC3) for CO/CT
    2. GLM Models for Indices and Self-perceptions
    3. Logistic Models for Closeness to Candidates

    Parameters:
        Ruta_Datos (str): Path to the folder with the data.

    Returns:
        dict: Dictionary with all results.

    """

    print("="*70)
    print("MODELOS ESTADÍSTICOS COMPLETOS")
    print("="*70)

    # Load data.
    print("\nCargando datos...")
    dfs_Finales = Cargar_Bases_Datos(Ruta_Datos)

    if not dfs_Finales:
        print("Error: No se pudieron cargar los datos.")
        return {}

    Todos_Resultados = {}

    # =========================================================================
    # SECTION 1: ROBUST MODELS (OLS WITH HC3).
    # =========================================================================

    print("\n" + "#"*70)
    print("# SECCIÓN 1: MODELOS ROBUSTOS (OLS CON ERRORES HC3)")
    print("#"*70)

    Todos_Resultados['Modelos_Robustos'] = Ajustar_Modelos_Robustos(
        dfs_Finales
    )

    # Additional summaries as in notebooks.
    Evaluaciones_Robustos = {}
    Variables_Que_Explican = {}
    for Nombre_df, Resultados_df in Todos_Resultados['Modelos_Robustos'].items():
        Evaluaciones_Robustos[Nombre_df] = {}
        Variables_Que_Explican[Nombre_df] = {}
        for Variable_Dep, Resultado in Resultados_df.items():
            Evaluaciones_Robustos[Nombre_df][Variable_Dep] = (
                Resultado.get('Evaluacion_Modelo')
            )
            Variables_Que_Explican[Nombre_df][Variable_Dep] = (
                Resultado.get('Variables_Significativas') or ['-']
            )

    Todos_Resultados['Evaluaciones_Modelos_Robustos'] = (
        Evaluaciones_Robustos
    )
    Todos_Resultados['Variables_Que_Explican_Robustos'] = (
        Variables_Que_Explican
    )

    # =========================================================================
    # SECTION 2: GLM MODELS (INDICES AND SELF-PERCEPTIONS).
    # =========================================================================

    print("\n" + "#"*70)
    print("# SECCIÓN 2: MODELOS GLM (ÍNDICES Y AUTOPERCEPCIONES)")
    print("#"*70)

    Todos_Resultados['Modelos_GLM'] = Ajustar_Modelos_GLM(dfs_Finales)

    # =========================================================================
    # SECTION 3: LOGISTIC MODELS (CLOSENESS TO CANDIDATES).
    # =========================================================================

    print("\n" + "#"*70)
    print("# SECCIÓN 3: MODELOS LOGÍSTICOS (CERCANÍA A CANDIDATOS)")
    print("#"*70)

    Todos_Resultados['Modelos_Logisticos'] = Ajustar_Modelos_Logisticos(
        dfs_Finales
    )

    print("\n" + "="*70)
    print("MODELOS COMPLETADOS")
    print("="*70)

    return Todos_Resultados, dfs_Finales


def Generar_Reporte_Proceso_TXT(
    Todos_Resultados: dict,
    Ruta_Salida: str
) -> None:

    """
    Generates a TXT file with the detail of the model fitting
    process: iterations, variable elimination, convergence, etc.

    Parameters:
        Todos_Resultados (dict): Dictionary with all results.
        Ruta_Salida (str): Path to the output TXT file.

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
    Agregar("MODEL FITTING PROCESS - RECURSION DETAILS")
    Agregar("Proyecto de Tesis - Experimento Electoral Argentina 2023")
    Separador("=")
    Agregar("")

    # =========================================================================
    # SECTION 1: ROBUST MODELS - PROCESS.
    # =========================================================================
    Separador("#")
    Agregar("# SECCIÓN 1: MODELOS ROBUSTOS - PROCESO DE AJUSTE")
    Separador("#")
    Agregar("")

    Modelos_Robustos = Todos_Resultados.get('Modelos_Robustos', {})

    for Nombre_df in ['Generales', 'Ballotage']:
        Resultados_df = Modelos_Robustos.get(Nombre_df, {})

        if not Resultados_df:
            continue

        Separador("=")
        Agregar(f"MODELOS ROBUSTOS - {Nombre_df}")
        Separador("=")
        Agregar("")

        for Variable_Dep, Resultado in Resultados_df.items():
            Separador("-")
            Agregar(f"Variable Dependiente: {Variable_Dep}")
            Separador("-")

            # Process information.
            Agregar(f"Variables iniciales: "
                    f"{len(Resultado.get('Variables_Originales', []))}")
            Agregar(f"Iteraciones totales: "
                    f"{Resultado['Iteraciones_Totales']}")
            Agregar(f"Criterio de eliminación: AIC")
            Agregar(f"Umbral VIF: 5.0")
            Agregar(f"Alpha significancia: 0.05")
            Agregar("")

            # Process history if available.
            Historial = Resultado.get('Historial_Proceso', [])
            if Historial:
                Agregar("Historial de iteraciones:")
                for i, Paso in enumerate(Historial, 1):
                    Agregar(f"  Iteración {i}:")
                    if 'Variable_Eliminada' in Paso:
                        Agregar(f"    Variable eliminada: "
                                f"{Paso['Variable_Eliminada']}")
                    if 'Razon' in Paso:
                        Agregar(f"    Razón: {Paso['Razon']}")
                    if 'VIF_Max' in Paso:
                        Agregar(f"    VIF máximo: {Paso['VIF_Max']:.2f}")
                    if 'AIC' in Paso:
                        Agregar(f"    AIC: {Paso['AIC']:.2f}")
                Agregar("")

            # Final variables.
            Variables_Sig = Resultado['Variables_Significativas']
            Variables_Orig = Resultado.get('Variables_Originales', [])
            Variables_Elim = set(Variables_Orig) - set(Variables_Sig)

            Agregar(f"Variables eliminadas: {len(Variables_Elim)}")
            if Variables_Elim:
                for Var in sorted(Variables_Elim):
                    Agregar(f"  - {Traducir_Variable(Var)}")

            Agregar("")
            Agregar(f"Variables finales (significativas): "
                    f"{len(Variables_Sig)}")
            for Var in Variables_Sig:
                Agregar(f"  + {Traducir_Variable(Var)}")

            Agregar("")
            Agregar(f"Convergencia: R² ajustado = "
                    f"{Resultado['R_Cuadrado_Ajustado']:.4f}")
            Agregar("")

    # =========================================================================
    # SECTION 2: GLM MODELS - PROCESS.
    # =========================================================================
    Separador("#")
    Agregar("# SECCIÓN 2: MODELOS GLM - PROCESO DE AJUSTE")
    Separador("#")
    Agregar("")

    Modelos_GLM = Todos_Resultados.get('Modelos_GLM', {})

    for Nombre_df in ['Generales', 'Ballotage']:
        Resultados_df = Modelos_GLM.get(Nombre_df, {})

        if not Resultados_df:
            continue

        Separador("=")
        Agregar(f"MODELOS GLM - {Nombre_df}")
        Separador("=")
        Agregar("")

        for Variable_Dep, Resultado in Resultados_df.items():
            Separador("-")
            Agregar(f"Variable Dependiente: {Variable_Dep}")
            Separador("-")

            # GLM family used.
            if Variable_Dep == 'Indice_Conservadurismo':
                Familia = "Gamma con link Log"
            else:
                Familia = "Gaussiana con link Identity"

            Agregar(f"Familia GLM: {Familia}")
            Agregar(f"Iteraciones hasta convergencia: "
                    f"{Resultado['N_Iteraciones']}")
            Agregar(f"Criterio de convergencia: p < 0.05")
            Agregar("")

            # Final variables.
            df_Vars = Resultado['Resultados_Variables']
            Agregar(f"Variables en modelo final: {len(df_Vars)}")
            for _, Fila in df_Vars.iterrows():
                Agregar(f"  + {Fila['Variable']}")

            Agregar("")
            Agregar(f"Convergencia: Deviance explicada = "
                    f"{Resultado['Deviance_Explicada']:.4f}")
            Agregar("")

    # =========================================================================
    # SECTION 3: LOGISTIC MODELS - PROCESS.
    # =========================================================================
    Separador("#")
    Agregar("# SECCIÓN 3: MODELOS LOGÍSTICOS - PROCESO DE AJUSTE")
    Separador("#")
    Agregar("")

    Modelos_Log = Todos_Resultados.get('Modelos_Logisticos', {})

    for Nombre_df in ['Generales', 'Ballotage']:
        Resultados_df = Modelos_Log.get(Nombre_df, {})

        if not Resultados_df:
            continue

        Separador("=")
        Agregar(f"MODELOS LOGÍSTICOS - {Nombre_df}")
        Separador("=")
        Agregar("")

        for Candidato, Resultado in Resultados_df.items():
            Separador("-")
            Agregar(f"Cercanía a: {Candidato}")
            Separador("-")

            Agregar(f"Variable binarizada: Cercano_{Candidato}")
            Agregar(f"Criterio binarización: >= 4 es cercano (1), "
                    f"<= 3 no cercano (0)")
            Agregar(f"Iteraciones hasta convergencia: "
                    f"{Resultado['N_Iteraciones']}")
            Agregar(f"Criterio de convergencia: p < 0.05")
            Agregar("")

            # Final variables.
            df_Vars = Resultado['Resultados_Variables']
            Agregar(f"Variables en modelo final: {len(df_Vars)}")
            for _, Fila in df_Vars.iterrows():
                Agregar(f"  + {Fila['Variable']}")

            Agregar("")
            Agregar(f"Convergencia: Pseudo R² = "
                    f"{Resultado['Pseudo_R2_McFadden']:.4f}")
            Agregar("")

    Separador("=")
    Agregar("FIN DEL REPORTE DE PROCESO")
    Separador("=")

    # Write file.
    with open(Ruta_Salida, 'w', encoding='utf-8') as Archivo:
        Archivo.write('\n'.join(Lineas))

    print(f"\nReporte de proceso guardado en: {Ruta_Salida}")


def Generar_Reporte_Resultados_TXT(
    Todos_Resultados: dict,
    Ruta_Salida: str
) -> None:

    """
    Generates a TXT file with the final results of the models:
    coefficients, statistics, Odds Ratios, etc.

    Parameters:
        Todos_Resultados (dict): Dictionary with all results.
        Ruta_Salida (str): Path to the output TXT file.

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
    Agregar("STATISTICAL MODEL RESULTS")
    Agregar("Proyecto de Tesis - Experimento Electoral Argentina 2023")
    Separador("=")
    Agregar("")

    # =========================================================================
    # SECTION 1: ROBUST MODELS.
    # =========================================================================
    Separador("#")
    Agregar("# SECCIÓN 1: MODELOS ROBUSTOS (OLS CON ERRORES HC3)")
    Separador("#")
    Agregar("")
    Agregar("Regresión lineal con errores estándar robustos (HC3)")
    Agregar("Eliminación secuencial basada en AIC y VIF")
    Agregar("")

    Modelos_Robustos = Todos_Resultados.get('Modelos_Robustos', {})

    for Nombre_df in ['Generales', 'Ballotage']:
        Resultados_df = Modelos_Robustos.get(Nombre_df, {})

        if not Resultados_df:
            continue

        Separador("=")
        Agregar(f"MODELOS ROBUSTOS - {Nombre_df}")
        Separador("=")
        Agregar("")

        for Variable_Dep, Resultado in Resultados_df.items():
            Separador("-")
            Agregar(f"Variable Dependiente: {Variable_Dep}")
            Separador("-")

            Agregar(f"N observaciones: {Resultado['N_Observaciones']}")
            Agregar(f"R² ajustado: {Resultado['R_Cuadrado_Ajustado']:.4f}")
            Agregar(f"AIC: {Resultado['AIC']:.2f}")
            Agregar(f"BIC: {Resultado['BIC']:.2f}")
            Agregar(f"F-estadístico: {Resultado['F_Estadistico']:.4f}")
            Agregar(f"p-valor modelo: "
                    f"{Formatear_P_Valor(Resultado['P_Valor_Modelo'])}")
            Agregar(f"Parámetro escala robusto: "
                    f"{Resultado['Parametro_Escala_Robusto']:.6f}")
            Agregar(f"Iteraciones: {Resultado['Iteraciones_Totales']}")
            Agregar("")

            # Significant variables.
            Variables_Sig = Resultado['Variables_Significativas']
            Agregar(f"Variables significativas: {len(Variables_Sig)}")

            if Variables_Sig:
                Agregar("")
                for Var in Variables_Sig:
                    Beta = Resultado['Coeficientes'].get(Var, np.nan)
                    P_Val = Resultado['P_Valores'].get(Var, np.nan)
                    Var_Traducida = Traducir_Variable(Var)

                    Agregar(f"  {Var_Traducida:40s} "
                            f"β = {Beta:8.4f}  "
                            f"p = {Formatear_P_Valor(P_Val)}")

            # Diagnostics.
            Agregar("")
            Agregar("Diagnósticos:")
            JB_Stat, JB_Pval = Resultado['Test_Normalidad_JB']
            BP_Stat, BP_Pval = Resultado['Test_Heterocedasticidad_BP']
            DW = Resultado['Durbin_Watson']

            Agregar(f"  Jarque-Bera (normalidad): "
                    f"p = {Formatear_P_Valor(JB_Pval)}")
            Agregar(f"  Breusch-Pagan (heterocedasticidad): "
                    f"p = {Formatear_P_Valor(BP_Pval)}")
            Agregar(f"  Durbin-Watson: {DW:.3f}")
            Agregar("")

    # =========================================================================
    # SECTION 2: GLM MODELS.
    # =========================================================================
    Separador("#")
    Agregar("# SECCIÓN 2: MODELOS GLM (ÍNDICES Y AUTOPERCEPCIONES)")
    Separador("#")
    Agregar("")
    Agregar("GLM con eliminación recursiva hasta convergencia (p < 0.05)")
    Agregar("- Indice_Conservadurismo: Familia Gamma con link Log")
    Agregar("- Resto: Familia Gaussiana con link Identity")
    Agregar("")

    Modelos_GLM = Todos_Resultados.get('Modelos_GLM', {})

    for Nombre_df in ['Generales', 'Ballotage']:
        Resultados_df = Modelos_GLM.get(Nombre_df, {})

        if not Resultados_df:
            continue

        Separador("=")
        Agregar(f"MODELOS GLM - {Nombre_df}")
        Separador("=")
        Agregar("")

        for Variable_Dep, Resultado in Resultados_df.items():
            Separador("-")
            Agregar(f"Variable Dependiente: {Variable_Dep}")
            Separador("-")

            Agregar(f"N observaciones: {Resultado['N_Observaciones']}")
            Agregar(f"Variables en modelo final: {Resultado['N_Variables']}")
            Agregar(f"Deviance explicada (R²): "
                    f"{Resultado['Deviance_Explicada']:.4f} "
                    f"({Resultado['Deviance_Explicada']*100:.2f}%)")
            Agregar(f"AIC: {Resultado['AIC']:.2f}")
            Agregar(f"BIC: {Resultado['BIC']:.2f}")
            Agregar(f"F-estadístico: {Resultado['F_Estadistico']:.4f}")
            Agregar(f"p-valor (F): "
                    f"{Formatear_P_Valor(Resultado['P_Valor_F'])}")
            Agregar(f"Iteraciones: {Resultado['N_Iteraciones']}")
            Agregar("")

            # Significant variables.
            df_Vars = Resultado['Resultados_Variables']

            if len(df_Vars) > 0:
                Agregar("Variables significativas:")
                Agregar("")

                for _, Fila in df_Vars.iterrows():
                    Var_Traducida = Traducir_Variable(Fila['Variable'])
                    Agregar(f"  {Var_Traducida:40s} "
                            f"β = {Fila['Coeficiente']:8.4f}  "
                            f"p = {Formatear_P_Valor(Fila['P_Valor'])}  "
                            f"IC 95% = [{Fila['IC_Inferior']:7.3f}, "
                            f"{Fila['IC_Superior']:7.3f}]")

            Agregar("")

    # =========================================================================
    # SECTION 3: LOGISTIC MODELS.
    # =========================================================================
    Separador("#")
    Agregar("# SECCIÓN 3: MODELOS LOGÍSTICOS (CERCANÍA A CANDIDATOS)")
    Separador("#")
    Agregar("")
    Agregar("Regresión Logística Binaria")
    Agregar("Variable binarizada: Cercano (1) si >= 4, No cercano (0) si <= 3")
    Agregar("Eliminación recursiva hasta convergencia (p < 0.05)")
    Agregar("")

    Modelos_Log = Todos_Resultados.get('Modelos_Logisticos', {})

    for Nombre_df in ['Generales', 'Ballotage']:
        Resultados_df = Modelos_Log.get(Nombre_df, {})

        if not Resultados_df:
            continue

        Separador("=")
        Agregar(f"MODELOS LOGÍSTICOS - {Nombre_df}")
        Separador("=")
        Agregar("")

        for Candidato, Resultado in Resultados_df.items():
            Separador("-")
            Agregar(f"Cercanía a: {Candidato}")
            Separador("-")

            Agregar(f"N observaciones: {Resultado['N_Observaciones']}")
            Agregar(f"Variables en modelo final: {Resultado['N_Variables']}")
            Agregar(f"Pseudo R² (McFadden): "
                    f"{Resultado['Pseudo_R2_McFadden']:.4f}")
            Agregar(f"AIC: {Resultado['AIC']:.2f}")
            Agregar(f"BIC: {Resultado['BIC']:.2f}")
            Agregar(f"Deviance nulo: {Resultado['Deviance_Nulo']:.2f}")
            Agregar(f"Deviance residual: {Resultado['Deviance_Residual']:.2f}")
            Agregar(f"χ² estadístico: {Resultado['Chi2_Estadistico']:.4f}")
            Agregar(f"p-valor (χ²): "
                    f"{Formatear_P_Valor(Resultado['P_Valor_Chi2'])}")
            Agregar(f"Iteraciones: {Resultado['N_Iteraciones']}")
            Agregar("")

            # Significant variables with Odds Ratios.
            df_Vars = Resultado['Resultados_Variables']

            if len(df_Vars) > 0:
                Agregar("Variables significativas (con Odds Ratios):")
                Agregar("")

                for _, Fila in df_Vars.iterrows():
                    Var_Traducida = Traducir_Variable(Fila['Variable'])
                    Agregar(f"  {Var_Traducida:40s} "
                            f"OR = {Fila['OR']:7.3f}  "
                            f"p = {Formatear_P_Valor(Fila['P_Valor'])}  "
                            f"IC 95% = [{Fila['OR_IC_Inferior']:6.3f}, "
                            f"{Fila['OR_IC_Superior']:6.3f}]")

            Agregar("")

    # =========================================================================
    # FINAL SUMMARY.
    # =========================================================================
    Separador("=")
    Agregar("RESUMEN DE MODELOS")
    Separador("=")
    Agregar("")

    for Nombre_df in ['Generales', 'Ballotage']:
        Agregar(f"{Nombre_df}:")
        Separador("-", 40)

        # Robust models.
        Robustos = Modelos_Robustos.get(Nombre_df, {})
        if Robustos:
            Agregar(f"  Modelos Robustos: {len(Robustos)} ajustados")
            for Var, Res in Robustos.items():
                Agregar(f"    {Var}: R²={Res['R_Cuadrado_Ajustado']:.4f}, "
                        f"Vars={len(Res['Variables_Significativas'])}")

        # GLM models.
        GLM = Modelos_GLM.get(Nombre_df, {})
        if GLM:
            Agregar(f"  Modelos GLM: {len(GLM)} ajustados")
            for Var, Res in GLM.items():
                Agregar(f"    {Var}: R²={Res['Deviance_Explicada']:.4f}, "
                        f"Vars={Res['N_Variables']}")

        # Logistic models.
        Log = Modelos_Log.get(Nombre_df, {})
        if Log:
            Agregar(f"  Modelos Logísticos: {len(Log)} ajustados")
            for Cand, Res in Log.items():
                Agregar(f"    {Cand}: Pseudo R²="
                        f"{Res['Pseudo_R2_McFadden']:.4f}, "
                        f"Vars={Res['N_Variables']}")

        Agregar("")

    Separador("=")
    Agregar("FIN DEL REPORTE")
    Separador("=")

    # Write file.
    with open(Ruta_Salida, 'w', encoding='utf-8') as Archivo:
        Archivo.write('\n'.join(Lineas))

    print(f"\nReporte de resultados guardado en: {Ruta_Salida}")


if __name__ == '__main__':
    # Path to the folder with the final data.
    Ruta_Base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    Ruta_Datos = os.path.join(Ruta_Base, 'Processed_Data')

    # Execute all models.
    Resultados, dfs = Ejecutar_Todos_Los_Modelos(Ruta_Datos)

    # Create Results/Reports folder if it doesn't exist.
    Ruta_Reportes = os.path.join(Ruta_Base, 'Results', 'Reports')
    if not os.path.exists(Ruta_Reportes):
        os.makedirs(Ruta_Reportes)

    # Generate PROCESS report (recursions, iterations).
    Ruta_Proceso = os.path.join(
        Ruta_Reportes,
        'Report_Models_Process.txt'
    )
    Generar_Reporte_Proceso_TXT(Resultados, Ruta_Proceso)

    # Generate RESULTS report (coefficients, statistics).
    Ruta_Resultados_Archivo = os.path.join(
        Ruta_Reportes,
        'Report_Models_Results.txt'
    )
    Generar_Reporte_Resultados_TXT(Resultados, Ruta_Resultados_Archivo)

    print("\nResultados guardados en el diccionario 'Resultados'.")
    print(f"Archivos generados:")
    print(f"  - {Ruta_Proceso}")
    print(f"  - {Ruta_Resultados_Archivo}")
