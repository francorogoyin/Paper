# -*- coding: utf-8 -*-

"""
Script principal para ejecutar todo el pipeline del proyecto.

Orden de ejecucion:
1) Procesamiento de bases.
2) Clustering de participantes.
3) Analisis descriptivo.
4) Analisis estadisticos.
5) Analisis de correlaciones.
6) Analisis de poblaciones segmentadas.
7) Modelos.

"""

import os
import sys
import subprocess


def Ejecutar_Script(Ruta_Script: str) -> None:

    """
    Ejecuta un script con el mismo interprete de Python.

    Parametros:
        Ruta_Script (str): Ruta al archivo .py a ejecutar.

    Retorna:
        None.

    """

    if not os.path.exists(Ruta_Script):
        raise FileNotFoundError(f"No existe: {Ruta_Script}")

    print("\n" + "=" * 80)
    print(f"Ejecutando: {Ruta_Script}")
    print("=" * 80)

    Resultado = subprocess.run(
        [sys.executable, Ruta_Script],
        cwd=os.path.dirname(Ruta_Script),
        check=False
    )

    if Resultado.returncode != 0:
        raise RuntimeError(
            f"Fallo al ejecutar {Ruta_Script} "
            f"(codigo {Resultado.returncode})"
        )


def Main() -> None:

    """
    Ejecuta todos los scripts en el orden definido.

    """

    Ruta_Base = os.path.dirname(os.path.abspath(__file__))

    Scripts = [
        os.path.join(Ruta_Base, "Code", "Data_Processing.py"),
        os.path.join(Ruta_Base, "Code", "Clustering.py"),
        os.path.join(Ruta_Base, "Code", "Descriptive_Analysis.py"),
        os.path.join(Ruta_Base, "Code", "Statistical_Analysis.py"),
        os.path.join(Ruta_Base, "Code", "Correlation_Analysis.py"),
        os.path.join(
            Ruta_Base,
            "Code",
            "Segmented_Population_Analysis.py"
        ),
        os.path.join(Ruta_Base, "Code", "Models.py")
    ]

    for Ruta_Script in Scripts:
        Ejecutar_Script(Ruta_Script)

    print("\n" + "=" * 80)
    print("Pipeline completado.")
    print("=" * 80)


if __name__ == "__main__":
    Main()
