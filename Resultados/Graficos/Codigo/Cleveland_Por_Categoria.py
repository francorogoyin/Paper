from pathlib import Path
import sys as Sistema

import matplotlib.pyplot as Matplotlib_Pyplot
from matplotlib import transforms as Transformaciones
import pandas as Pandas

Ruta_Archivo = Path(__file__).resolve()
# Subir tres niveles (Codigo -> Graficos -> Resultados -> Paper) y a Codigo.
Ruta_Proyecto = Ruta_Archivo.parents[3]
Ruta_Codigo = Ruta_Proyecto / "Codigo"
Sistema.path.append(str(Ruta_Codigo))

from Graficos import (
    Cargar_Base_Datos,
    Calcular_Estadisticas_Dos_Variables,
    Crear_Grafico_Cleveland_Dos_Variables,
    Determinar_Asteriscos,
    Items_Progresistas,
    Items_Conservadores,
    Categorias_Paso,
    Colores_Categorias_Paso,
    Obtener_Ruta_Proyecto,
)


def Obtener_Ruta_Tablas():

    """
    Obtener la carpeta Resultados/Tablas del proyecto.

    Retorna:
    - Ruta_Tablas: Ruta a la carpeta de tablas.

    Ejemplos:
    - Ruta_Tablas = Obtener_Ruta_Tablas()
    """

    Ruta_Proyecto = Obtener_Ruta_Proyecto()
    Ruta_Tablas = Ruta_Proyecto / "Resultados" / "Tablas"
    Ruta_Tablas.mkdir(parents = True, exist_ok = True)
    return Ruta_Tablas


def Obtener_Ruta_Graficos_Cleveland():

    """
    Obtener la carpeta de graficos Cleveland por categoria.

    Retorna:
    - Ruta_Graficos: Ruta a la carpeta de graficos.

    Ejemplos:
    - Ruta_Graficos = Obtener_Ruta_Graficos_Cleveland()
    """

    Ruta_Proyecto = Obtener_Ruta_Proyecto()
    Ruta_Graficos = (
        Ruta_Proyecto
        / "Resultados"
        / "Graficos"
        / "Cleveland_Por_Categoria"
    )
    Ruta_Graficos.mkdir(parents = True, exist_ok = True)
    return Ruta_Graficos


def Cargar_Tabla_Dunn_CO(Ruta_Tablas = None):

    """
    Cargar tabla Dunn para CO por categoria.

    Parametros:
    - Ruta_Tablas: Ruta opcional a la carpeta de tablas.

    Retorna:
    - Tabla: DataFrame con comparaciones Dunn.

    Ejemplos:
    - Tabla = Cargar_Tabla_Dunn_CO()
    """

    if Ruta_Tablas is None:
        Ruta_Tablas = Obtener_Ruta_Tablas()
    else:
        Ruta_Tablas = Path(Ruta_Tablas)

    Ruta_Archivo = (
        Ruta_Tablas / "Dunn_CO_Items.xlsx"
    )
    if not Ruta_Archivo.exists():
        return Pandas.DataFrame()

    return Pandas.read_excel(Ruta_Archivo)


def Filtrar_Comparaciones_Dunn(
    Tabla_Dunn,
    Nombre_Base,
    Variable,
    Solo_Significativas = True,
):

    """
    Filtrar comparaciones Dunn por dataset y variable.

    Parametros:
    - Tabla_Dunn: DataFrame con comparaciones Dunn.
    - Nombre_Base: Nombre del dataset.
    - Variable: Nombre de la variable.
    - Solo_Significativas: Filtrar p < 0.05.

    Retorna:
    - Comparaciones: Lista de comparaciones.

    Ejemplos:
    - Comparaciones = Filtrar_Comparaciones_Dunn(
      Tabla_Dunn,
      Nombre_Base = "Generales",
      Variable = "CO_Item_3_Izq",
    )
    """

    if Tabla_Dunn is None or len(Tabla_Dunn) == 0:
        return []

    Subset = Tabla_Dunn[
        (Tabla_Dunn["Dataset"] == Nombre_Base)
        & (Tabla_Dunn["Variable"] == Variable)
    ].copy()

    if Solo_Significativas:
        Subset = Subset[Subset["Valor_p"] < 0.05]

    return Subset.to_dict("records")


def Dibujar_Comparaciones_Post_Hoc(
    Eje,
    Estadisticas,
    Comparaciones,
    Marcador_Extremo = "o",
    Desplazamiento_X = 0.02,
    Separacion_X = 0.04,
):

    """
    Dibujar lineas post-hoc en el borde derecho del grafico.

    Parametros:
    - Eje: Objeto Axes.
    - Estadisticas: Lista de estadisticas por categoria.
    - Comparaciones: Lista de comparaciones con p-valor.
    - Marcador_Extremo: Marcador en extremos.
    - Desplazamiento_X: Corrimiento del primer marcador.
    - Separacion_X: Distancia entre marcadores.

    Retorna:
    - Ninguno.

    Ejemplos:
    - Dibujar_Comparaciones_Post_Hoc(Eje, Estadisticas, [])
    """

    if not Comparaciones:
        return

    Posiciones_Y = list(reversed(range(len(Estadisticas))))
    Mapa_Posiciones = {
        Estadistica["Categoria"]: Posiciones_Y[i]
        for i, Estadistica in enumerate(Estadisticas)
    }

    Transformacion = Transformaciones.blended_transform_factory(
        Eje.transAxes,
        Eje.transData,
    )

    for Indice, Comparacion in enumerate(Comparaciones):
        Categoria_1 = Comparacion.get("Categoria_1")
        Categoria_2 = Comparacion.get("Categoria_2")

        if Categoria_1 not in Mapa_Posiciones:
            continue
        if Categoria_2 not in Mapa_Posiciones:
            continue

        Y_1 = Mapa_Posiciones[Categoria_1]
        Y_2 = Mapa_Posiciones[Categoria_2]
        X_Pos = 1.02 + Desplazamiento_X + (Indice * Separacion_X)

        Asteriscos = Determinar_Asteriscos(
            Comparacion.get("Valor_p", 1.0)
        )
        if not Asteriscos:
            continue

        Eje.plot(
            [X_Pos, X_Pos],
            [Y_1, Y_2],
            transform = Transformacion,
            color = "black",
            linewidth = 1.5,
            clip_on = False,
        )
        Eje.scatter(
            [X_Pos, X_Pos],
            [Y_1, Y_2],
            transform = Transformacion,
            marker = Marcador_Extremo,
            s = 60,
            facecolors = "white",
            edgecolors = "black",
            linewidths = 1.5,
            clip_on = False,
            zorder = 5,
        )

        Y_Texto = (Y_1 + Y_2) / 2
        Eje.text(
            X_Pos + 0.01,
            Y_Texto,
            Asteriscos,
            transform = Transformacion,
            ha = "left",
            va = "center",
            fontsize = 10,
            fontweight = "bold",
            clip_on = False,
        )


def Guardar_Tabla_Estadisticas_Cleveland(
    Filas,
    Nombre_Base,
    Ruta_Tablas = None,
):

    """
    Guardar tabla de estadisticas Cleveland en xlsx.

    Parametros:
    - Filas: Lista de filas con estadisticas.
    - Nombre_Base: Nombre del dataset.
    - Ruta_Tablas: Ruta opcional a la carpeta de tablas.

    Retorna:
    - Ruta_Archivo: Ruta del xlsx generado.

    Ejemplos:
    - Guardar_Tabla_Estadisticas_Cleveland(Filas, "Generales")
    """

    if not Filas:
        return None

    if Ruta_Tablas is None:
        Ruta_Tablas = Obtener_Ruta_Tablas()
    else:
        Ruta_Tablas = Path(Ruta_Tablas)

    Tabla = Pandas.DataFrame(Filas)
    Tabla = Tabla.sort_values(
        ["Item", "Categoria"],
    ).reset_index(drop = True)

    Ruta_Archivo = (
        Ruta_Tablas
        / f"Cleveland_Por_Categoria_{Nombre_Base}.xlsx"
    )
    Tabla.to_excel(Ruta_Archivo, index = False)
    return Ruta_Archivo


def Generar_Graficos_Cleveland_Por_Items(
    Nombre_Base,
    Items,
    Patron_Izquierda = "CO_Item_{Item}_Izq",
    Patron_Derecha = "CO_Item_{Item}_Der",
    Variable_Categoria = "Categoria_PASO_2023",
    Etiqueta_Eje_X = "Change of Opinion Media",
    Etiqueta_Eje_Y = "",
    Usar_Post_Hoc = True,
    Solo_Significativas = True,
    Mostrar_Grafico = True,
    Cerrar_Figura = False,
    Guardar_Tabla = True,
):

    """
    Generar graficos Cleveland por item con post-hoc opcional.

    Parametros:
    - Nombre_Base: Nombre del dataset.
    - Items: Lista de numeros de item.
    - Patron_Izquierda: Patron para columna izquierda.
    - Patron_Derecha: Patron para columna derecha.
    - Variable_Categoria: Columna de categoria.
    - Etiqueta_Eje_X: Texto del eje X.
    - Etiqueta_Eje_Y: Texto del eje Y.
    - Usar_Post_Hoc: Dibujar comparaciones Dunn.
    - Solo_Significativas: Filtrar p < 0.05.
    - Mostrar_Grafico: Mostrar figura en notebook.
    - Cerrar_Figura: Cerrar figura al final.
    - Guardar_Tabla: Guardar tabla xlsx de estadisticas.

    Retorna:
    - Filas: Lista con estadisticas generadas.

    Ejemplos:
    - Generar_Graficos_Cleveland_Por_Items(
      "Generales",
      Items = [3, 5, 7],
    )
    """

    Datos = Cargar_Base_Datos(Nombre_Base)
    Ruta_Graficos = Obtener_Ruta_Graficos_Cleveland()
    Tabla_Dunn = Cargar_Tabla_Dunn_CO()

    Filas = []

    for Numero_Item in Items:
        Columna_Izquierda = Patron_Izquierda.format(
            Item = Numero_Item
        )
        Columna_Derecha = Patron_Derecha.format(Item = Numero_Item)

        if Numero_Item in Items_Progresistas:
            Tipo_Item = "Progressive"
            Tipo_Item_Esp = "Progresista"
        else:
            Tipo_Item = "Conservative"
            Tipo_Item_Esp = "Conservador"

        Titulo = f"Item {Numero_Item} ({Tipo_Item})"
        Nombre_Archivo = f"Cleveland_Item_{Numero_Item}.svg"

        Estadisticas, Figura, Eje = (
            Crear_Grafico_Cleveland_Dos_Variables(
                Datos,
                Columna_Izquierda = Columna_Izquierda,
                Columna_Derecha = Columna_Derecha,
                Variable_Categoria = Variable_Categoria,
                Orden_Categorias = Categorias_Paso,
                Colores_Categorias = Colores_Categorias_Paso,
                Titulo = Titulo,
                Nombre_Archivo = Nombre_Archivo,
                Etiqueta_Eje_X = Etiqueta_Eje_X,
                Etiqueta_Eje_Y = Etiqueta_Eje_Y,
                Guardar_Archivo = False,
                Mostrar_Grafico = False,
                Cerrar_Figura = False,
                Carpeta_Salida = Ruta_Graficos,
            )
        )

        if Usar_Post_Hoc:
            Comparaciones_Izq = Filtrar_Comparaciones_Dunn(
                Tabla_Dunn,
                Nombre_Base,
                Columna_Izquierda,
                Solo_Significativas = Solo_Significativas,
            )
            Dibujar_Comparaciones_Post_Hoc(
                Eje,
                Estadisticas,
                Comparaciones_Izq,
                Marcador_Extremo = "o",
                Desplazamiento_X = 0.02,
            )

            Comparaciones_Der = Filtrar_Comparaciones_Dunn(
                Tabla_Dunn,
                Nombre_Base,
                Columna_Derecha,
                Solo_Significativas = Solo_Significativas,
            )
            Dibujar_Comparaciones_Post_Hoc(
                Eje,
                Estadisticas,
                Comparaciones_Der,
                Marcador_Extremo = "s",
                Desplazamiento_X = 0.08,
            )

        Ruta_Salida = Ruta_Graficos / Nombre_Archivo
        Figura.savefig(
            Ruta_Salida,
            format = "svg",
            dpi = 300,
            bbox_inches = "tight",
            pad_inches = 0.1,
            facecolor = "white",
        )

        if Mostrar_Grafico:
            Matplotlib_Pyplot.show()
        if Cerrar_Figura:
            Matplotlib_Pyplot.close(Figura)

        for Estadistica in Estadisticas:
            Filas.append({
                "Dataset": Nombre_Base,
                "Item": Numero_Item,
                "Tipo_Item": Tipo_Item_Esp,
                "Categoria": Estadistica["Categoria"],
                "N_Muestra": Estadistica["N_Muestra"],
                "Media_Izquierda": Estadistica["Media_Izquierda"],
                "Media_Derecha": Estadistica["Media_Derecha"],
                "Diferencia_Media": Estadistica["Diferencia_Media"],
                "P_Valor": Estadistica["P_Valor"],
                "Asteriscos": Estadistica["Asteriscos"],
                "Columna_Izquierda": Columna_Izquierda,
                "Columna_Derecha": Columna_Derecha,
            })

    if Guardar_Tabla:
        Guardar_Tabla_Estadisticas_Cleveland(
            Filas,
            Nombre_Base = Nombre_Base,
        )

    return Filas


def Generar_Paneles_Cleveland_Por_Items(
    Nombre_Base,
    Items,
    Columnas_Panel = 4,
    Patron_Izquierda = "CO_Item_{Item}_Izq",
    Patron_Derecha = "CO_Item_{Item}_Der",
    Variable_Categoria = "Categoria_PASO_2023",
    Etiqueta_Eje_X = "Opinion Change Media",
    Etiqueta_Eje_Y = "",
    Titulo_General = None,
    Usar_Post_Hoc = True,
    Solo_Significativas = True,
    Mostrar_Grafico = True,
    Guardar_Archivo = False,
    Nombre_Archivo = None,
    Carpeta_Salida = None,
):

    """
    Generar paneles Cleveland en una sola figura.

    Parametros:
    - Nombre_Base: Nombre del dataset.
    - Items: Lista de numeros de item.
    - Columnas_Panel: Cantidad de columnas de panel.
    - Patron_Izquierda: Patron para columna izquierda.
    - Patron_Derecha: Patron para columna derecha.
    - Variable_Categoria: Columna de categoria.
    - Etiqueta_Eje_X: Texto del eje X.
    - Etiqueta_Eje_Y: Texto del eje Y.
    - Titulo_General: Titulo general de la figura.
    - Usar_Post_Hoc: Dibujar comparaciones Dunn.
    - Solo_Significativas: Filtrar p < 0.05.
    - Mostrar_Grafico: Mostrar figura en notebook.

    Retorna:
    - Figura: Figura final.
    - Ejes: Lista de ejes.

    Ejemplos:
    - Figura, Ejes = Generar_Paneles_Cleveland_Por_Items(
      "Generales",
      Items = [3, 5, 7, 8],
    )
    """

    Datos = Cargar_Base_Datos(Nombre_Base)
    Tabla_Dunn = Cargar_Tabla_Dunn_CO()

    Total_Items = len(Items)
    Filas_Panel = (Total_Items + Columnas_Panel - 1) // Columnas_Panel

    # Altura extra para la leyenda en pulgadas.
    Altura_Leyenda_Pulgadas = 2.0
    Altura_Total = (6 * Filas_Panel) + Altura_Leyenda_Pulgadas

    Figura, Ejes = Matplotlib_Pyplot.subplots(
        Filas_Panel,
        Columnas_Panel,
        figsize = (11 * Columnas_Panel, Altura_Total),
    )

    if hasattr(Ejes, "ravel"):
        Ejes = list(Ejes.ravel())
    elif isinstance(Ejes, (list, tuple)):
        Ejes = list(Ejes)
    else:
        Ejes = [Ejes]

    for Indice, Numero_Item in enumerate(Items):
        Eje = Ejes[Indice]
        Columna_Izquierda = Patron_Izquierda.format(
            Item = Numero_Item
        )
        Columna_Derecha = Patron_Derecha.format(Item = Numero_Item)

        if Numero_Item in Items_Progresistas:
            Tipo_Item = "Progressive"
        else:
            Tipo_Item = "Conservative"

        Titulo = f"Item {Numero_Item} ({Tipo_Item})"

        Estadisticas, _, _ = Crear_Grafico_Cleveland_Dos_Variables(
            Datos,
            Columna_Izquierda = Columna_Izquierda,
            Columna_Derecha = Columna_Derecha,
            Variable_Categoria = Variable_Categoria,
            Orden_Categorias = Categorias_Paso,
            Colores_Categorias = Colores_Categorias_Paso,
            Titulo = Titulo,
            Etiqueta_Eje_X = Etiqueta_Eje_X,
            Etiqueta_Eje_Y = Etiqueta_Eje_Y,
            Guardar_Archivo = False,
            Mostrar_Grafico = False,
            Cerrar_Figura = False,
            Ajustar_Diseno = False,
            Figura = Figura,
            Eje = Eje,
        )

        if Usar_Post_Hoc:
            Comparaciones_Izq = Filtrar_Comparaciones_Dunn(
                Tabla_Dunn,
                Nombre_Base,
                Columna_Izquierda,
                Solo_Significativas = Solo_Significativas,
            )
            Dibujar_Comparaciones_Post_Hoc(
                Eje,
                Estadisticas,
                Comparaciones_Izq,
                Marcador_Extremo = "o",
                Desplazamiento_X = 0.02,
            )

            Comparaciones_Der = Filtrar_Comparaciones_Dunn(
                Tabla_Dunn,
                Nombre_Base,
                Columna_Derecha,
                Solo_Significativas = Solo_Significativas,
            )
            Dibujar_Comparaciones_Post_Hoc(
                Eje,
                Estadisticas,
                Comparaciones_Der,
                Marcador_Extremo = "s",
                Desplazamiento_X = 0.08,
            )

    for Eje in Ejes[Total_Items:]:
        Eje.axis("off")

    # Agregar titulo general si se especifica, alineado a la izquierda.
    if Titulo_General:
        Figura.suptitle(
            Titulo_General,
            fontsize = 18,
            fontweight = "bold",
            x = 0.01,
            y = 1.02,
            ha = "left",
        )

    # Ajustar espaciado entre paneles.
    Figura.tight_layout(pad = 3.0, h_pad = 4.0, w_pad = 3.0)

    # Calcular fraccion de la leyenda respecto a la altura total.
    Altura_Figura_Pulgadas = Figura.get_size_inches()[1]
    Fraccion_Leyenda = Altura_Leyenda_Pulgadas / Altura_Figura_Pulgadas

    # Ajustar bottom para dejar espacio fijo para la leyenda.
    Figura.subplots_adjust(bottom = Fraccion_Leyenda + 0.02)

    # Agregar leyenda centrada debajo de todos los paneles.
    # Crear un eje invisible para la leyenda con altura fija.
    Eje_Leyenda = Figura.add_axes(
        [0, 0.01, 1, Fraccion_Leyenda - 0.02],
        frameon = False,
    )
    Eje_Leyenda.set_xlim(0, 1)
    Eje_Leyenda.set_ylim(0, 1)
    Eje_Leyenda.axis("off")

    # Posiciones X para cada elemento de la leyenda.
    Posiciones_X = [0.05, 0.28, 0.52, 0.78]
    Posicion_Y_Simbolo = 0.5
    Posicion_Y_Texto = 0.5

    # Elemento 1: Rombo - Associated with a Right-wing Candidate.
    Eje_Leyenda.scatter(
        [Posiciones_X[0]], [Posicion_Y_Simbolo],
        marker = "D",
        s = 80,
        facecolors = "white",
        edgecolors = "black",
        linewidths = 1.5,
        zorder = 5,
    )
    Eje_Leyenda.text(
        Posiciones_X[0] + 0.02, Posicion_Y_Texto,
        "Associated with a Right-wing\nCandidate",
        fontsize = 9,
        ha = "left",
        va = "center",
    )

    # Elemento 2: Circulo - Associated with a Left-wing Candidate.
    Eje_Leyenda.scatter(
        [Posiciones_X[1]], [Posicion_Y_Simbolo],
        marker = "o",
        s = 80,
        facecolors = "white",
        edgecolors = "black",
        linewidths = 1.5,
        zorder = 5,
    )
    Eje_Leyenda.text(
        Posiciones_X[1] + 0.02, Posicion_Y_Texto,
        "Associated with a Left-wing\nCandidate",
        fontsize = 9,
        ha = "left",
        va = "center",
    )

    # Elemento 3: Linea con cuadrados - Statistical comparison Right-wing.
    Ancho_Linea = 0.03
    Eje_Leyenda.plot(
        [Posiciones_X[2], Posiciones_X[2] + Ancho_Linea],
        [Posicion_Y_Simbolo, Posicion_Y_Simbolo],
        color = "black",
        linewidth = 1.5,
    )
    Eje_Leyenda.scatter(
        [Posiciones_X[2], Posiciones_X[2] + Ancho_Linea],
        [Posicion_Y_Simbolo, Posicion_Y_Simbolo],
        marker = "s",
        s = 50,
        facecolors = "white",
        edgecolors = "black",
        linewidths = 1,
        zorder = 5,
    )
    Eje_Leyenda.text(
        Posiciones_X[2] + Ancho_Linea + 0.015, Posicion_Y_Texto,
        "Statistical comparison when associated\nwith a Right-wing candidate",
        fontsize = 9,
        ha = "left",
        va = "center",
    )

    # Elemento 4: Linea con circulos - Statistical comparison Left-wing.
    Eje_Leyenda.plot(
        [Posiciones_X[3], Posiciones_X[3] + Ancho_Linea],
        [Posicion_Y_Simbolo, Posicion_Y_Simbolo],
        color = "black",
        linewidth = 1.5,
    )
    Eje_Leyenda.scatter(
        [Posiciones_X[3], Posiciones_X[3] + Ancho_Linea],
        [Posicion_Y_Simbolo, Posicion_Y_Simbolo],
        marker = "o",
        s = 50,
        facecolors = "white",
        edgecolors = "black",
        linewidths = 1,
        zorder = 5,
    )
    Eje_Leyenda.text(
        Posiciones_X[3] + Ancho_Linea + 0.015, Posicion_Y_Texto,
        "Statistical comparison when associated\nwith a Left-wing candidate",
        fontsize = 9,
        ha = "left",
        va = "center",
    )

    # Guardar el archivo SVG si se especifica.
    if Guardar_Archivo:
        if Carpeta_Salida is None:
            Carpeta_Salida = Obtener_Ruta_Graficos_Cleveland()
        else:
            Carpeta_Salida = Path(Carpeta_Salida)
            Carpeta_Salida.mkdir(parents = True, exist_ok = True)

        if Nombre_Archivo is None:
            Nombre_Archivo = f"Cleveland_Paneles_{Nombre_Base}.svg"

        Ruta_Archivo = Carpeta_Salida / Nombre_Archivo
        Figura.savefig(
            Ruta_Archivo,
            format = "svg",
            dpi = 300,
            bbox_inches = "tight",
            pad_inches = 0.2,
            facecolor = "white",
        )

    if Mostrar_Grafico:
        Matplotlib_Pyplot.show()

    return Figura, Ejes
