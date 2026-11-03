import numpy as np
import pandas as pd
from functools import reduce

# ==========================================
# 1. Definición de Funciones de Pertenencia
# ==========================================

def mf_triangular(x, a, b, c):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac1 = np.where(b != a, (x - a) / (b - a), 1.0)
        frac2 = np.where(c != b, (c - x) / (c - b), 1.0)
    return np.round(np.maximum(np.minimum(frac1, frac2), 0.0), 4)

def mf_trapezoidal(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac1 = np.where(b != a, (x - a) / (b - a), 1.0)
        frac2 = np.where(d != c, (d - x) / (d - c), 1.0)
    return np.round(np.maximum(np.minimum(np.minimum(frac1, 1.0), frac2), 0.0), 4)

def mf_gaussiana(x, c, sigma):
    x = np.asarray(x, dtype=float)
    if sigma == 0:
        raise ValueError("El parámetro sigma no puede ser 0.")
    return np.round(np.exp(-0.5 * ((x - c) / sigma) ** 2), 4)

def mf_bell(x, a, b, c):
    x = np.asarray(x, dtype=float)
    if a == 0:
        raise ValueError("El parámetro 'a' no puede ser 0.")
    denominador = 1.0 + np.abs((x - c) / a) ** (2.0 * b)
    return np.round(1.0 / denominador, 4)

# ==========================================
# 2. Definición de T-Normas y T-Conormas
# ==========================================

# --- T-Normas (Intersección / AND) ---
def tnorm_minimo(arrays):
    """Mínimo estándar de Zadeh: min(A, B, ...)"""
    return reduce(np.minimum, arrays)

def tnorm_producto(arrays):
    """Producto algebraico: A * B * ..."""
    return reduce(lambda x, y: x * y, arrays)

def tnorm_lukasiewicz(arrays):
    """Diferencia acotada (Lukasiewicz): max(0, A + B - 1) generalizado"""
    return reduce(lambda x, y: np.maximum(0.0, x + y - 1.0), arrays)

# --- T-Conormas / S-Normas (Unión / OR) ---
def tconorm_maximo(arrays):
    """Máximo estándar de Zadeh: max(A, B, ...)"""
    return reduce(np.maximum, arrays)

def tconorm_suma_algebraica(arrays):
    """Suma algebraica (probabilística): A + B - (A * B) generalizado"""
    return reduce(lambda x, y: x + y - (x * y), arrays)

def tconorm_lukasiewicz(arrays):
    """Suma acotada (Lukasiewicz): min(1, A + B) generalizado"""
    return reduce(lambda x, y: np.minimum(1.0, x + y), arrays)

# ==========================================
# 3. Flujo Principal
# ==========================================

def main():
    print("=" * 60)
    print("      CALCULADORA DIFUSA: PERTENENCIA, T-NORMS Y T-CONORMS")
    print("=" * 60)

    # 1. Ingreso de datos
    entrada_valores = input("Ingresa los valores a evaluar separados por comas (ej. 5, 9, 8, 3, 10):\n> ")
    valores = np.array([float(v.strip()) for v in entrada_valores.split(",")])

    # 2. Selección de función de pertenencia
    print("\nSelecciona la función de pertenencia a utilizar:")
    print("  1. Triangular")
    print("  2. Trapezoidal")
    print("  3. Gaussiana")
    print("  4. Campana Generalizada (Bell)")
    opcion = input("Elige una opción (1-4): ").strip()
    while opcion not in ["1", "2", "3", "4"]:
        opcion = input("Opción inválida. Elige del 1 al 4: ").strip()

    # 3. Definición de categorías
    n_categorias = int(input("\n¿En cuántas categorías deseas dividir el conjunto?: "))
    categorias = {}

    for i in range(n_categorias):
        nombre = input(f"\nNombre de la categoría {i + 1} (ej. Bajo, Medio, Alto): ").strip()
        if opcion == "1":
            a = float(input("  Límite inferior (a): "))
            b = float(input("  Pico central (b): "))
            c = float(input("  Límite superior (c): "))
            categorias[nombre] = (a, b, c)
        elif opcion == "2":
            a = float(input("  Inicio pendiente ascendente (a): "))
            b = float(input("  Inicio meseta (b): "))
            c = float(input("  Fin meseta (c): "))
            d = float(input("  Fin pendiente descendente (d): "))
            categorias[nombre] = (a, b, c, d)
        elif opcion == "3":
            c = float(input("  Centro / media (c): "))
            sigma = float(input("  Dispersión / ancho (sigma > 0): "))
            categorias[nombre] = (c, sigma)
        elif opcion == "4":
            a = float(input("  Semiancho (a > 0): "))
            b = float(input("  Pendiente / forma (b > 0): "))
            c = float(input("  Centro / pico (c): "))
            categorias[nombre] = (a, b, c)

    # 4. Cálculo de pertenencia básica
    funciones_mf = {
        "1": mf_triangular,
        "2": mf_trapezoidal,
        "3": mf_gaussiana,
        "4": mf_bell
    }
    
    resultado_dict = {'Dato': valores}
    listado_pertenencias = []

    for nombre, params in categorias.items():
        grados = funciones_mf[opcion](valores, *params)
        resultado_dict[nombre] = grados
        listado_pertenencias.append(grados)

    # 5. Preguntar por T-Normas
    print("\n" + "-" * 60)
    usar_tnorms = input("¿Quieres usar T-Normas (intersección / AND)? (s/n): ").strip().lower()
    if usar_tnorms == 's':
        print("\nLas 3 T-Normas disponibles son:")
        print("  1. Mínimo estándar: min(A, B)")
        print("  2. Producto algebraico: A * B")
        print("  3. Diferencia acotada (Lukasiewicz): max(0, A + B - 1)")
        
        opc_tnorm = input("Elige la T-Norma a aplicar (1-3): ").strip()
        while opc_tnorm not in ["1", "2", "3"]:
            opc_tnorm = input("Opción inválida. Elige 1, 2 o 3: ").strip()

        if opc_tnorm == "1":
            resultado_dict['T-Norm (Mínimo)'] = np.round(tnorm_minimo(listado_pertenencias), 4)
        elif opc_tnorm == "2":
            resultado_dict['T-Norm (Producto)'] = np.round(tnorm_producto(listado_pertenencias), 4)
        elif opc_tnorm == "3":
            resultado_dict['T-Norm (Lukasiewicz)'] = np.round(tnorm_lukasiewicz(listado_pertenencias), 4)

    # 6. Preguntar por T-Conormas
    print("\n" + "-" * 60)
    usar_tconorms = input("¿Quieres usar T-Conormas / S-Normas (unión / OR)? (s/n): ").strip().lower()
    if usar_tconorms == 's':
        print("\nLas 3 T-Conormas disponibles son:")
        print("  1. Máximo estándar: max(A, B)")
        print("  2. Suma algebraica / probabilística: A + B - (A * B)")
        print("  3. Suma acotada (Lukasiewicz): min(1, A + B)")
        
        opc_tconorm = input("Elige la T-Conorma a aplicar (1-3): ").strip()
        while opc_tconorm not in ["1", "2", "3"]:
            opc_tconorm = input("Opción inválida. Elige 1, 2 o 3: ").strip()

        if opc_tconorm == "1":
            resultado_dict['T-Conorm (Máximo)'] = np.round(tconorm_maximo(listado_pertenencias), 4)
        elif opc_tconorm == "2":
            resultado_dict['T-Conorm (Suma Alg.)'] = np.round(tconorm_suma_algebraica(listado_pertenencias), 4)
        elif opc_tconorm == "3":
            resultado_dict['T-Conorm (Lukasiewicz)'] = np.round(tconorm_lukasiewicz(listado_pertenencias), 4)

    # 7. Despliegue de resultados finales en DataFrame
    df_resultado = pd.DataFrame(resultado_dict)
    print("\n" + "=" * 60)
    print("               TABLA FINAL DE RESULTADOS")
    print("=" * 60)
    print(df_resultado.to_string(index=False))

if __name__ == "__main__":
    main()