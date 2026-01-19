from pathlib import Path

import matplotlib.pyplot as Matplotlib_Pyplot
import pandas as Pandas
import re as Expresiones_Regulares
import seaborn as Seaborn

Colores_Categorias_Paso = {
    "Progressivism": "#0078bf",
    "Moderate_Right_A": "#f7d117",
    "Moderate_Right_B": "#f7d117",
    "Centre": "#009cdd",
    "Left_Wing": "#f65058",
    "Right_Wing_Libertarian": "#753bbd",
}

Categorias_Paso = [
    "Left_Wing",
    "Progressivism",
    "Centre",
    "Moderate_Right_A",
    "Moderate_Right_B",
    "Right_Wing_Libertarian",
]

Items_Progresistas = [
    5,
    6,
    9,
    11,
    16,
    20,
    24,
    25,
    27,
    28,
]

Items_Conservadores = [
    3,
    4,
    7,
    8,
    10,
    19,
    22,
    23,
    29,
    30,
]


def Configurar_Estilo_Grafico():

    """
    Configurar estilo general para graficos.

    Retorna:
    - Ninguno.

    Ejemplos:
    - Configurar_Estilo_Grafico()
    """

    Matplotlib_Pyplot.style.use("seaborn-v0_8")
    Seaborn.set_palette("husl")
    Seaborn.set_context("notebook")


def Obtener_Ruta_Proyecto():

    """
    Obtener la ruta base del proyecto Paper.

    Retorna:
    - Ruta_Proyecto: Ruta base del proyecto.

    Ejemplos:
    - Ruta_Proyecto = Obtener_Ruta_Proyecto()
    """

    return Path(__file__).resolve().parents[1]


def Obtener_Ruta_Base_Datos(Nombre_Base):

    """
    Obtener la ruta del archivo de datos por base.

    Parametros:
    - Nombre_Base: "Generales" o "Ballotage".

    Retorna:
    - Ruta_Base: Ruta del archivo de datos.

    Ejemplos:
    - Ruta_Base = Obtener_Ruta_Base_Datos("Generales")
    """

    Ruta_Proyecto = Obtener_Ruta_Proyecto()
    Nombre_Base_Limpio = Nombre_Base.strip().lower()

    Rutas_Bases = {
        "generales": (
            Ruta_Proyecto
            / "Datos definitivos"
            / "Generales.xlsx"
        ),
        "ballotage": (
            Ruta_Proyecto
            / "Datos definitivos"
            / "Ballotage.xlsx"
        ),
    }

    Ruta_Base = Rutas_Bases.get(Nombre_Base_Limpio)
    if Ruta_Base is None:
        raise ValueError(
            "Nombre_Base invalido. Usar Generales o Ballotage."
        )

    return Ruta_Base


def Obtener_Ruta_Carpeta_Graficos():

    """
    Crear y devolver la carpeta de salida de graficos.

    Retorna:
    - Ruta_Carpeta: Ruta de salida para SVG.

    Ejemplos:
    - Ruta_Carpeta = Obtener_Ruta_Carpeta_Graficos()
    """

    Ruta_Proyecto = Obtener_Ruta_Proyecto()
    Ruta_Carpeta = Ruta_Proyecto / "Resultados" / "Graficos_Cleveland"
    Ruta_Carpeta.mkdir(parents = True, exist_ok = True)
    return Ruta_Carpeta


def Cargar_Base_Datos(Nombre_Base):

    """
    Cargar base de datos por nombre.

    Parametros:
    - Nombre_Base: "Generales" o "Ballotage".

    Retorna:
    - Datos: DataFrame con los datos.

    Ejemplos:
    - Datos = Cargar_Base_Datos("Generales")
    """

    Ruta_Base = Obtener_Ruta_Base_Datos(Nombre_Base)
    return Pandas.read_excel(Ruta_Base)


def Obtener_Items_Co(Datos):

    """
    Obtener lista de numeros de items CO disponibles.

    Parametros:
    - Datos: DataFrame con columnas CO_Item.

    Retorna:
    - Items: Lista ordenada de numeros de item.

    Ejemplos:
    - Items = Obtener_Items_Co(Datos)
    """

    Columnas = [
        Columna
        for Columna in Datos.columns
        if Columna.startswith("CO_Item_")
    ]

    Items = []
    for Columna in Columnas:
        Partes = Columna.split("_")
        if len(Partes) >= 3:
            try:
                Numero_Item = int(Partes[2])
                Items.append(Numero_Item)
            except ValueError:
                continue

    return sorted(set(Items))


def Determinar_Asteriscos(P_Valor):

    """
    Convertir un valor p a asteriscos de significancia.

    Parametros:
    - P_Valor: Valor p numerico.

    Retorna:
    - Asteriscos: Cadena de asteriscos.

    Ejemplos:
    - Asteriscos = Determinar_Asteriscos(0.03)
    """

    if P_Valor < 0.001:
        return "***"
    if P_Valor < 0.01:
        return "**"
    if P_Valor < 0.05:
        return "*"
    return ""


# Diccionario de mapeo para nombres de categorias formateados.
Mapeo_Nombres_Categorias = {
    "Left_Wing": "Left-wing",
    "Progressivism": "Progressivism",
    "Centre": "Centre",
    "Moderate_Right_A": "Moderate Right (a)",
    "Moderate_Right_B": "Moderate Right (b)",
    "Right_Wing_Libertarian": "Libertarian Right-wing",
}


def Formatear_Nombre_Categoria(Nombre_Categoria):

    """
    Formatear nombre de categoria con espacios legibles.

    Parametros:
    - Nombre_Categoria: Nombre original de la categoria.

    Retorna:
    - Nombre_Formateado: Nombre con formato especifico.

    Ejemplos:
    - Formatear_Nombre_Categoria("Left_Wing") -> "Left-wing"
    - Formatear_Nombre_Categoria("Moderate_Right_A") -> "Moderate Right (a)"

    """

    # Usar mapeo especifico si existe.
    if Nombre_Categoria in Mapeo_Nombres_Categorias:
        return Mapeo_Nombres_Categorias[Nombre_Categoria]

    # Fallback: reemplazar guiones bajos por espacios.
    Nombre_Formateado = Nombre_Categoria.replace("_", " ")
    Nombre_Formateado = Expresiones_Regulares.sub(
        r"(?<!^)([A-Z])",
        r" \1",
        Nombre_Formateado,
    )
    return Nombre_Formateado.strip()


def Formatear_Nombre_Categoria_Mathtext(Nombre_Categoria):

    """
    Formatear nombre de categoria para mathtext.

    Parametros:
    - Nombre_Categoria: Nombre original de la categoria.

    Retorna:
    - Nombre_Formateado: Nombre con espacios escapados.

    Ejemplos:
    - Formatear_Nombre_Categoria_Mathtext("LeftWing")
    - Formatear_Nombre_Categoria_Mathtext("Left_Wing")
    """

    Nombre_Formateado = Formatear_Nombre_Categoria(
        Nombre_Categoria
    )
    return Nombre_Formateado.replace(" ", r"\ ")


def Calcular_Estadisticas_Co_Item(
    Datos,
    Numero_Item,
    Categorias_Paso,
    Minimo_Muestra = 5,
    Usar_Prueba = True,
):

    """
    Calcular estadisticas para un item CO por categoria.

    Parametros:
    - Datos: DataFrame con columnas CO_Item.
    - Numero_Item: Numero del item.
    - Categorias_Paso: Lista de categorias PASO.
    - Minimo_Muestra: Minimo de casos por categoria.
    - Usar_Prueba: Ejecutar Wilcoxon pareado.

    Retorna:
    - Estadisticas: Lista de diccionarios por categoria.

    Ejemplos:
    - Estadisticas = Calcular_Estadisticas_Co_Item(
      Datos,
      Numero_Item = 3,
      Categorias_Paso = Categorias_Paso,
    )
    """

    Variable_Izquierda = f"CO_Item_{Numero_Item}_Izq"
    Variable_Derecha = f"CO_Item_{Numero_Item}_Der"

    Datos_Filtrados = Datos[
        Datos["Categoria_PASO_2023"].isin(Categorias_Paso)
    ].copy()

    Estadisticas = []
    for Categoria in Categorias_Paso:
        Datos_Categoria = Datos_Filtrados[
            Datos_Filtrados["Categoria_PASO_2023"] == Categoria
        ]

        Datos_Validos = Datos_Categoria.dropna(
            subset = [Variable_Izquierda, Variable_Derecha]
        )

        if len(Datos_Validos) < Minimo_Muestra:
            continue

        Media_Izquierda = Datos_Validos[Variable_Izquierda].mean()
        Media_Derecha = Datos_Validos[Variable_Derecha].mean()
        N_Muestra = len(Datos_Validos)
        Diferencia_Media = Media_Izquierda - Media_Derecha

        P_Valor = 1.0
        if Usar_Prueba:
            from scipy.stats import wilcoxon as Wilcoxon

            try:
                Diferencias = (
                    Datos_Validos[Variable_Derecha]
                    - Datos_Validos[Variable_Izquierda]
                )
                if (Diferencias != 0).sum() >= 3:
                    _, P_Valor = Wilcoxon(
                        Datos_Validos[Variable_Derecha],
                        Datos_Validos[Variable_Izquierda],
                        alternative = "two-sided",
                    )
            except Exception:
                P_Valor = 1.0

        Asteriscos = Determinar_Asteriscos(P_Valor)

        Estadisticas.append(
            {
                "Categoria": Categoria,
                "N_Muestra": N_Muestra,
                "Media_Izquierda": Media_Izquierda,
                "Media_Derecha": Media_Derecha,
                "Diferencia_Media": Diferencia_Media,
                "P_Valor": P_Valor,
                "Asteriscos": Asteriscos,
            }
        )

    return Estadisticas


def Calcular_Estadisticas_Dos_Variables(
    Datos,
    Columna_Izquierda,
    Columna_Derecha,
    Variable_Categoria,
    Orden_Categorias,
    Minimo_Muestra = 5,
    Usar_Prueba = True,
):

    """
    Calcular estadisticas para dos columnas por categoria.

    Parametros:
    - Datos: DataFrame con columnas numericas.
    - Columna_Izquierda: Nombre de columna izquierda.
    - Columna_Derecha: Nombre de columna derecha.
    - Variable_Categoria: Columna de categorias.
    - Orden_Categorias: Lista ordenada de categorias.
    - Minimo_Muestra: Minimo de casos por categoria.
    - Usar_Prueba: Ejecutar Wilcoxon pareado.

    Retorna:
    - Estadisticas: Lista de diccionarios por categoria.

    Ejemplos:
    - Estadisticas = Calcular_Estadisticas_Dos_Variables(
      Datos,
      Columna_Izquierda = "CO_Item_3_Izq",
      Columna_Derecha = "CO_Item_3_Der",
      Variable_Categoria = "Categoria_PASO_2023",
      Orden_Categorias = Categorias_Paso,
    )
    """

    Datos_Filtrados = Datos[
        Datos[Variable_Categoria].isin(Orden_Categorias)
    ].copy()

    Estadisticas = []
    for Categoria in Orden_Categorias:
        Datos_Categoria = Datos_Filtrados[
            Datos_Filtrados[Variable_Categoria] == Categoria
        ]

        Datos_Validos = Datos_Categoria.dropna(
            subset = [Columna_Izquierda, Columna_Derecha]
        )

        if len(Datos_Validos) < Minimo_Muestra:
            continue

        Media_Izquierda = Datos_Validos[Columna_Izquierda].mean()
        Media_Derecha = Datos_Validos[Columna_Derecha].mean()
        N_Muestra = len(Datos_Validos)
        Diferencia_Media = Media_Izquierda - Media_Derecha

        P_Valor = 1.0
        if Usar_Prueba:
            from scipy.stats import wilcoxon as Wilcoxon

            try:
                Diferencias = (
                    Datos_Validos[Columna_Derecha]
                    - Datos_Validos[Columna_Izquierda]
                )
                if (Diferencias != 0).sum() >= 3:
                    _, P_Valor = Wilcoxon(
                        Datos_Validos[Columna_Derecha],
                        Datos_Validos[Columna_Izquierda],
                        alternative = "two-sided",
                    )
            except Exception:
                P_Valor = 1.0

        Asteriscos = Determinar_Asteriscos(P_Valor)

        Estadisticas.append(
            {
                "Categoria": Categoria,
                "N_Muestra": N_Muestra,
                "Media_Izquierda": Media_Izquierda,
                "Media_Derecha": Media_Derecha,
                "Diferencia_Media": Diferencia_Media,
                "P_Valor": P_Valor,
                "Asteriscos": Asteriscos,
            }
        )

    return Estadisticas


def Crear_Grafico_Cleveland_Dos_Variables(
    Datos,
    Columna_Izquierda,
    Columna_Derecha,
    Variable_Categoria,
    Orden_Categorias = None,
    Colores_Categorias = None,
    Titulo = None,
    Nombre_Archivo = None,
    Etiqueta_Eje_X = "Cambio de Opinion",
    Etiqueta_Eje_Y = "",
    Limites_X = (-1, 1),
    Minimo_Muestra = 5,
    Guardar_Archivo = True,
    Mostrar_Grafico = True,
    Cerrar_Figura = True,
    Ajustar_Diseno = True,
    Figura = None,
    Eje = None,
    Carpeta_Salida = None,
):

    """
    Crear grafico Cleveland para dos columnas por categoria.

    Parametros:
    - Datos: DataFrame con columnas numericas.
    - Columna_Izquierda: Nombre de columna izquierda.
    - Columna_Derecha: Nombre de columna derecha.
    - Variable_Categoria: Columna de categorias.
    - Orden_Categorias: Lista ordenada de categorias.
    - Colores_Categorias: Diccionario de colores.
    - Titulo: Titulo del grafico.
    - Nombre_Archivo: Nombre del SVG de salida.
    - Etiqueta_Eje_X: Texto del eje X.
    - Etiqueta_Eje_Y: Texto del eje Y.
    - Limites_X: Tupla con limites del eje X.
    - Minimo_Muestra: Minimo de casos por categoria.
    - Guardar_Archivo: Guardar el SVG en carpeta de salida.
    - Mostrar_Grafico: Mostrar figura en notebook.
    - Cerrar_Figura: Cerrar figura al final.
    - Ajustar_Diseno: Ajustar layout con tight_layout.
    - Figura: Figura opcional para paneles.
    - Eje: Eje opcional para paneles.
    - Carpeta_Salida: Ruta opcional de salida.

    Retorna:
    - Estadisticas: Lista de estadisticas por categoria.
    - Figura: Objeto Figure.
    - Eje: Objeto Axes.

    Ejemplos:
    - Estadisticas, Figura, Eje = (
      Crear_Grafico_Cleveland_Dos_Variables(
      Datos,
      Columna_Izquierda = "CO_Item_3_Izq",
      Columna_Derecha = "CO_Item_3_Der",
      Variable_Categoria = "Categoria_PASO_2023",
      Orden_Categorias = Categorias_Paso,
      Titulo = "Item 3",
      Nombre_Archivo = "Cleveland_Item_3.svg",
    )
    """

    Configurar_Estilo_Grafico()

    if Orden_Categorias is None:
        Orden_Categorias = list(Categorias_Paso)

    if Colores_Categorias is None:
        Colores_Categorias = Colores_Categorias_Paso

    Estadisticas = Calcular_Estadisticas_Dos_Variables(
        Datos,
        Columna_Izquierda = Columna_Izquierda,
        Columna_Derecha = Columna_Derecha,
        Variable_Categoria = Variable_Categoria,
        Orden_Categorias = Orden_Categorias,
        Minimo_Muestra = Minimo_Muestra,
        Usar_Prueba = True,
    )

    if len(Estadisticas) == 0:
        raise ValueError("No hay datos suficientes para graficar.")

    Es_Nueva_Figura = False
    if Figura is None or Eje is None:
        Figura, Eje = Matplotlib_Pyplot.subplots(figsize = (12, 8))
        Es_Nueva_Figura = True

    Eje.set_facecolor("#D3D3D3")
    if Es_Nueva_Figura:
        Figura.patch.set_facecolor("white")

    Posiciones_Y = list(reversed(range(len(Estadisticas))))

    for Indice, Estadistica in enumerate(Estadisticas):
        Posicion_Y = Posiciones_Y[Indice]
        Color_Categoria = Colores_Categorias.get(
            Estadistica["Categoria"],
            "#808080",
        )

        # Linea de conexion tipo sombra: gris clara y ancha.
        Eje.plot(
            [
                Estadistica["Media_Izquierda"],
                Estadistica["Media_Derecha"],
            ],
            [Posicion_Y, Posicion_Y],
            color = "#D3D3D3",
            alpha = 0.9,
            linewidth = 8,
            solid_capstyle = "round",
            zorder = 1,
        )

        # Tamaño reducido de los marcadores.
        Tamano_Punto = min(
            50 + Estadistica["N_Muestra"] * 0.3,
            150,
        )

        Eje.scatter(
            Estadistica["Media_Izquierda"],
            Posicion_Y,
            color = Color_Categoria,
            s = Tamano_Punto,
            alpha = 0.9,
            marker = "o",
            edgecolors = "black",
            linewidth = 1.5,
            zorder = 3,
        )

        Eje.scatter(
            Estadistica["Media_Derecha"],
            Posicion_Y,
            color = Color_Categoria,
            s = Tamano_Punto,
            alpha = 0.9,
            marker = "D",
            edgecolors = "black",
            linewidth = 1.5,
            zorder = 3,
        )

    Eje.axvline(
        x = 0,
        color = "black",
        linestyle = ":",
        alpha = 0.7,
        linewidth = 2,
    )

    Eje.set_xlim(Limites_X)
    Eje.set_ylim(-0.5, len(Estadisticas) - 0.5)
    Eje.set_yticks(Posiciones_Y)

    Etiquetas_Y = []
    for Estadistica in Estadisticas:
        Nombre_Legible = Formatear_Nombre_Categoria(
            Estadistica["Categoria"]
        )
        Nombre_Mathtext = Formatear_Nombre_Categoria_Mathtext(
            Estadistica["Categoria"]
        )
        if Estadistica["Asteriscos"]:
            Nombre_Partido = (
                f"$\\mathbf{{{Nombre_Mathtext}}}$ "
                f"{Estadistica['Asteriscos']}"
            )
        else:
            Nombre_Partido = f"$\\mathbf{{{Nombre_Mathtext}}}$"

        Info_Linea = (
            f"n = {Estadistica['N_Muestra']}, "
            f"Δ = {Estadistica['Diferencia_Media']:.2f}"
        )
        Etiqueta_Completa = f"{Nombre_Partido}\n{Info_Linea}"
        Etiquetas_Y.append(Etiqueta_Completa)

    Eje.set_yticklabels(Etiquetas_Y, fontsize = 10)

    if Titulo is None:
        Titulo = "Grafico Cleveland"
    Eje.set_title(
        Titulo,
        fontsize = 16,
        fontweight = "bold",
        loc = "left",
        pad = 15,
    )

    Eje.set_xlabel(
        Etiqueta_Eje_X,
        fontsize = 12,
        fontweight = "bold",
    )
    Eje.set_ylabel(
        Etiqueta_Eje_Y,
        fontsize = 12,
        fontweight = "bold",
    )

    Eje.grid(True, color = "white", linestyle = ":", linewidth = 1.0)
    Eje.spines["top"].set_visible(False)
    Eje.spines["right"].set_visible(False)

    if Ajustar_Diseno:
        Figura.tight_layout()

    if Guardar_Archivo and Es_Nueva_Figura:
        if Carpeta_Salida is None:
            Ruta_Salida = Obtener_Ruta_Carpeta_Graficos()
        else:
            Ruta_Salida = Path(Carpeta_Salida)
            Ruta_Salida.mkdir(parents = True, exist_ok = True)

        if Nombre_Archivo is None:
            Nombre_Archivo = "Cleveland.svg"
        Ruta_Archivo = Ruta_Salida / Nombre_Archivo
        Figura.savefig(
            Ruta_Archivo,
            format = "svg",
            dpi = 300,
            bbox_inches = "tight",
            pad_inches = 0.1,
            facecolor = "white",
        )

    if Mostrar_Grafico and Es_Nueva_Figura:
        Matplotlib_Pyplot.show()
    if Cerrar_Figura and Es_Nueva_Figura:
        Matplotlib_Pyplot.close(Figura)

    return Estadisticas, Figura, Eje


def Crear_Grafico_Cleveland_Co_Item(
    Datos,
    Numero_Item,
    Nombre_Base,
    Limites_X = (-1, 1),
    Minimo_Muestra = 5,
    Mostrar_Grafico = True,
    Cerrar_Figura = True,
    Carpeta_Salida = None,
):

    """
    Crear grafico Cleveland para un item CO por categoria.

    Parametros:
    - Datos: DataFrame con columnas CO_Item.
    - Numero_Item: Numero del item.
    - Nombre_Base: Nombre de la base.
    - Limites_X: Tupla con limites del eje X.
    - Minimo_Muestra: Minimo de casos por categoria.
    - Mostrar_Grafico: Mostrar figura en notebook.
    - Cerrar_Figura: Cerrar figura al final.
    - Carpeta_Salida: Ruta opcional de salida.

    Retorna:
    - Estadisticas: Lista de estadisticas por categoria.
    - Figura: Objeto Figure.
    - Eje: Objeto Axes.

    Ejemplos:
    - Estadisticas, Figura, Eje = Crear_Grafico_Cleveland_Co_Item(
      Datos,
      Numero_Item = 3,
      Nombre_Base = "Generales",
    )
    """

    if Numero_Item in Items_Progresistas:
        Tipo_Item = "Progresista"
    else:
        Tipo_Item = "Conservador"
    Titulo = f"Item {Numero_Item} ({Tipo_Item})"
    Columna_Izquierda = f"CO_Item_{Numero_Item}_Izq"
    Columna_Derecha = f"CO_Item_{Numero_Item}_Der"
    Nombre_Archivo = (
        f"Cleveland_CO_Item_{Numero_Item}_{Nombre_Base}.svg"
    )

    return Crear_Grafico_Cleveland_Dos_Variables(
        Datos,
        Columna_Izquierda = Columna_Izquierda,
        Columna_Derecha = Columna_Derecha,
        Variable_Categoria = "Categoria_PASO_2023",
        Orden_Categorias = Categorias_Paso,
        Colores_Categorias = Colores_Categorias_Paso,
        Titulo = Titulo,
        Nombre_Archivo = Nombre_Archivo,
        Limites_X = Limites_X,
        Minimo_Muestra = Minimo_Muestra,
        Mostrar_Grafico = Mostrar_Grafico,
        Cerrar_Figura = Cerrar_Figura,
        Carpeta_Salida = Carpeta_Salida,
    )
