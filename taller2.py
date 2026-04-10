"""
Análisis Avanzado de Datos – Taller 2
Regresión no lineal sobre el dataset Auto (ISLR2)
Autores: Sara Castillejo, Stefany Mojica, Juan Rodríguez

Nota: Este script implementa los métodos vistos en la Sesión 3 del curso
("Más allá de la linealidad"), usando el dataset Auto con la relación
entre horsepower (HP) y mpg.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings("ignore")

# Se fija la semilla para reproducibilidad
SEMILLA = 26
np.random.seed(SEMILLA)

# =============================================================================
# CARGA DE DATOS
# =============================================================================

def cargar_auto():
    """
    Se cargan los datos del dataset Auto desde la librería ISLR2.
    Se eliminan filas con valores faltantes en las columnas relevantes.
    """
    try:
        # Se intenta cargar desde statsmodels (disponible sin instalar ISLR2)
        from statsmodels.datasets import get_rdataset
        auto = get_rdataset("Auto", "ISLR2").data
    except Exception:
        # Si no está disponible, se descarga desde URL pública
        url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Auto.csv"
        auto = pd.read_csv(url, na_values="?").dropna()

    # Se retiran las filas con valores faltantes
    auto = auto.dropna(subset=["horsepower", "mpg"])
    auto = auto.reset_index(drop=True)
    return auto


# =============================================================================
# PUNTO 1: SEPARACIÓN ENTRENAMIENTO / PRUEBA (90% / 10%)
# =============================================================================

def separar_datos(auto, verbose=True):
    """
    Se separa aleatoriamente el conjunto de datos en entrenamiento (90%)
    y prueba (10%), fijando la semilla para reproducibilidad.
    """
    n = len(auto)
    indices = np.random.permutation(n)
    n_train = int(0.9 * n)

    idx_train = indices[:n_train]
    idx_test  = indices[n_train:]

    train = auto.iloc[idx_train].reset_index(drop=True)
    test  = auto.iloc[idx_test].reset_index(drop=True)

    if verbose:
        print(f"[Punto 1] Tamaño entrenamiento: {len(train)} | Prueba: {len(test)}")
    return train, test


# =============================================================================
# AUXILIARES: B-SPLINE MANUAL (equivalente a bs() de R)
# =============================================================================

def base_bspline(x, knots, grado=3):
    """
    Se construye la matriz de diseño de un B-spline cúbico usando
    polinomios truncados, equivalente a splines::bs() en R.

    Un spline cúbico con K knots tiene df = K + 3 + 1
    parámetros. Se usa la base truncada vista en clase:

        phi_j(x) = (x - k_j)^3_+  para cada knot k_j

    combinada con el polinomio global 1, x, x^2, x^3.
    """
    # Se construye la parte polinomial global (grado d)
    X_poly = np.column_stack([x**j for j in range(grado + 1)])

    # Se añade una columna truncada por cada knot interior
    truncadas = []
    for k in knots:
        col = np.where(x >= k, (x - k)**grado, 0.0)
        truncadas.append(col)

    if truncadas:
        X_knots = np.column_stack(truncadas)
        return np.hstack([X_poly, X_knots])
    return X_poly


class RegresionSpline(BaseEstimator, RegressorMixin):
    """
    Se implementa la regresión spline usando la base de polinomios truncados,
    tal como se presentó en clase. El estimador es MCO
    aplicado sobre la base de funciones {phi_j}.
    """

    def __init__(self, knots, grado=3):
        self.knots = knots
        self.grado = grado

    def fit(self, X, y):
        """Se ajusta el modelo usando MCO sobre la base de funciones."""
        x = X.ravel()
        Phi = base_bspline(x, self.knots, self.grado)
        self.reg_ = LinearRegression(fit_intercept=False).fit(Phi, y)
        return self

    def predict(self, X):
        """Se predice usando los coeficientes estimados."""
        x = X.ravel()
        Phi = base_bspline(x, self.knots, self.grado)
        return self.reg_.predict(Phi)


# =============================================================================
# PUNTO 2: SELECCIÓN DEL NÚMERO ÓPTIMO DE KNOTS (CV 10-FOLDS)
# =============================================================================

def punto2_seleccion_knots(train, verbose=True):
    """
    Se determina el número óptimo de knots (1 a 10) para el regression spline
    mediante validación cruzada en 10 folds.

    Los knots se ubican igualmente espaciados en el rango de horsepower,
    como se sugiere en el enunciado del taller.

    df = K + d + 1, con d=3 (spline cúbico).
    """
    X_train = train[["horsepower"]].values
    y_train = train["mpg"].values

    hp_min = X_train.min()
    hp_max = X_train.max()

    # Se define la validación cruzada con 10 folds
    kf = KFold(n_splits=10, shuffle=True, random_state=SEMILLA)

    ecm_por_knots = {}

    for K in range(1, 11):
        # Se ubican K knots igualmente espaciados en el rango de HP
        knots_k = np.linspace(hp_min, hp_max, K + 2)[1:-1]  # Excluye extremos

        errores_fold = []
        for idx_tr, idx_val in kf.split(X_train):
            X_tr, y_tr = X_train[idx_tr], y_train[idx_tr]
            X_val, y_val = X_train[idx_val], y_train[idx_val]

            modelo = RegresionSpline(knots=knots_k)
            modelo.fit(X_tr, y_tr)
            pred = modelo.predict(X_val)
            errores_fold.append(mean_squared_error(y_val, pred))

        ecm_por_knots[K] = np.mean(errores_fold)

    # Se identifica el número de knots con menor ECM de validación
    K_optimo = min(ecm_por_knots, key=ecm_por_knots.get)
    ecm_optimo = ecm_por_knots[K_optimo]

    if verbose:
        print(f"\n[Punto 2] ECM por número de knots (CV 10-folds):")
        for K, ecm in ecm_por_knots.items():
            marca = " <-- ÓPTIMO" if K == K_optimo else ""
            print(f"  K={K:2d}: ECM = {ecm:.4f}{marca}")
        print(f"\n  Número óptimo de knots: K = {K_optimo} (ECM = {ecm_optimo:.4f})")

    return K_optimo, ecm_por_knots


# =============================================================================
# PUNTO 3: COMPARACIÓN DE MODELOS BASADOS EN BASE DE FUNCIONES (CV 10-FOLDS)
# =============================================================================

def suavizamiento_spline_cv(X_tr, y_tr, X_val):
    x_tr = X_tr.ravel()
    
    # Se ordenan los datos de entrenamiento: UnivariateSpline lo requiere
    idx_orden = np.argsort(x_tr)
    x_sorted  = x_tr[idx_orden]
    y_sorted  = y_tr[idx_orden]
    
    # Se eliminan duplicados en x para evitar inestabilidad numérica
    x_uniq, idx_uniq = np.unique(x_sorted, return_index=True)
    y_uniq = y_sorted[idx_uniq]
    
    spl  = UnivariateSpline(x_uniq, y_uniq, s=None, k=3)
    pred = spl(X_val.ravel())
    
    # Se reemplazan NaN residuales por la media de y_tr (predicción conservadora)
    nan_mask = np.isnan(pred)
    if nan_mask.any():
        pred[nan_mask] = np.mean(y_tr)
    
    return pred


def punto3_comparacion_modelos(train, K_optimo, verbose=True):
    """
    Se comparan tres modelos basados en base de funciones usando CV 10-folds:

    1. Polinomio global grado 2: yi = b0 + b1*xi + b2*xi^2
       (base funcional: 1, x, x^2)

    2. Smoothing spline: minimiza SCES = sum(yi - f(xi))^2 + lambda * integral(f''(x)^2 dx)
       (Sesión 3: equivalente a spline natural con n knots y penalización)

    3. Regression spline óptimo: spline cúbico con K_optimo knots en
       posiciones igualmente espaciadas.
    """
    X_train = train[["horsepower"]].values
    y_train = train["mpg"].values
    hp_min, hp_max = X_train.min(), X_train.max()

    kf = KFold(n_splits=10, shuffle=True, random_state=SEMILLA)

    ecm_polinomio = []
    ecm_smooth    = []
    ecm_spline    = []

    knots_opt = np.linspace(hp_min, hp_max, K_optimo + 2)[1:-1]

    for idx_tr, idx_val in kf.split(X_train):
        X_tr, y_tr = X_train[idx_tr], y_train[idx_tr]
        X_val, y_val = X_train[idx_val], y_train[idx_val]

        # --- Modelo 1: Polinomio global grado 2 ---
        pip_poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        pip_poly.fit(X_tr, y_tr)
        pred_poly = pip_poly.predict(X_val)
        ecm_polinomio.append(mean_squared_error(y_val, pred_poly))

        # --- Modelo 2: Smoothing spline ---
        pred_ss = suavizamiento_spline_cv(X_tr, y_tr, X_val)
        ecm_smooth.append(mean_squared_error(y_val, pred_ss))

        # --- Modelo 3: Regression spline óptimo (K_optimo knots) ---
        mod_rs = RegresionSpline(knots=knots_opt)
        mod_rs.fit(X_tr, y_tr)
        pred_rs = mod_rs.predict(X_val)
        ecm_spline.append(mean_squared_error(y_val, pred_rs))

    resultados = {
        "Polinomio grado 2": np.mean(ecm_polinomio),
        "Smoothing Spline":  np.mean(ecm_smooth),
        f"Reg. Spline (K={K_optimo})": np.mean(ecm_spline),
    }

    if verbose:
        print(f"\n[Punto 3] Comparación de modelos basados en base de funciones (CV 10-folds):")
        for nombre, ecm in resultados.items():
            print(f"  {nombre}: ECM = {ecm:.4f}")

    mejor = min(resultados, key=resultados.get)
    if verbose:
        print(f"\n  Modelo seleccionado: {mejor} (ECM = {resultados[mejor]:.4f})")

    return resultados, mejor, knots_opt


# =============================================================================
# PUNTO 4: REGRESIÓN LOCAL (LOESS / NADARAYA-WATSON)
# =============================================================================

class RegresionLocal(BaseEstimator, RegressorMixin):
    """
    Se implementa la regresión polinomial local con kernel gaussiano,
    equivalente al estimador de Nadaraya-Watson con polinomios locales
    mencionado en el enunciado (ksmooth en R).

    Para cada punto de predicción x0 se resuelve una regresión ponderada:
        min sum_i K_h(xi - x0) * (yi - f(xi))^2
    donde K_h es un kernel gaussiano con ancho h.

    Este método es más flexible que el spline en el sentido de que la
    suavidad varía localmente.
    """

    def __init__(self, bandwidth, grado=1):
        """
        bandwidth : ancho de banda del kernel gaussiano (equivale a span en loess)
        grado     : grado del polinomio local (1 ó 2, según el punto 4)
        """
        self.bandwidth = bandwidth
        self.grado = grado

    def fit(self, X, y):
        """Se almacenan los datos de entrenamiento (ajuste diferido)."""
        self.X_train_ = X.ravel()
        self.y_train_ = y
        return self

    def _kernel_gaussiano(self, x0):
        """Se calculan los pesos del kernel gaussiano centrado en x0."""
        dist = (self.X_train_ - x0) / self.bandwidth
        return np.exp(-0.5 * dist**2)

    def _predecir_punto(self, x0):
        """
        Se ajusta un polinomio local de grado self.grado alrededor de x0
        usando regresión ponderada por el kernel gaussiano.
        """
        w = self._kernel_gaussiano(x0)
        # Se evitan pesos despreciables para estabilidad numérica
        mask = w > 1e-10
        if mask.sum() < (self.grado + 1):
            return np.nan
        X_loc = np.column_stack([self.X_train_[mask]**j for j in range(self.grado + 1)])
        W_loc = np.diag(w[mask])
        y_loc = self.y_train_[mask]
        # Se resuelve el sistema de ecuaciones normales ponderadas: (X'WX)b = X'Wy
        try:
            coef = np.linalg.lstsq(X_loc.T @ W_loc @ X_loc,
                                   X_loc.T @ W_loc @ y_loc,
                                   rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.nan
        x0_vec = np.array([x0**j for j in range(self.grado + 1)])
        return x0_vec @ coef

    def predict(self, X):
        """Se predicen los valores aplicando el estimador local a cada punto."""
        return np.array([self._predecir_punto(x0) for x0 in X.ravel()])


def seleccionar_bandwidth(X_tr, y_tr):
    """
    Se selecciona el ancho de banda óptimo para la regresión local usando
    la regla de Silverman (análogo al bandwidth por defecto de loess en R).

    Regla de Silverman: h = 1.06 * std(x) * n^(-1/5)
    """
    n = len(X_tr)
    std_x = np.std(X_tr.ravel())
    h_optimo = 1.06 * std_x * (n ** (-1/5))
    return h_optimo


def punto4_regresion_local(train, verbose=True):
    """
    Se determina el mejor modelo de regresión local (grado 1 ó 2)
    usando CV 10-folds. El ancho de banda se selecciona automáticamente
    con la regla de Silverman (equivalente al default de loess en R).
    """
    X_train = train[["horsepower"]].values
    y_train = train["mpg"].values

    kf = KFold(n_splits=10, shuffle=True, random_state=SEMILLA)

    ecm_grado = {1: [], 2: []}

    for idx_tr, idx_val in kf.split(X_train):
        X_tr, y_tr = X_train[idx_tr], y_train[idx_tr]
        X_val, y_val = X_train[idx_val], y_train[idx_val]

        # Se selecciona h con la regla de Silverman sobre el fold de entrenamiento
        h = seleccionar_bandwidth(X_tr, y_tr)

        for g in [1, 2]:
            mod = RegresionLocal(bandwidth=h, grado=g)
            mod.fit(X_tr, y_tr)
            pred = mod.predict(X_val)
            # Se omiten predicciones NaN por pesos insuficientes
            mask_valid = ~np.isnan(pred)
            if mask_valid.sum() > 0:
                ecm_grado[g].append(mean_squared_error(y_val[mask_valid], pred[mask_valid]))

    resultados = {f"Local grado {g}": np.mean(v) for g, v in ecm_grado.items()}

    if verbose:
        print(f"\n[Punto 4] Regresión local (CV 10-folds):")
        for nombre, ecm in resultados.items():
            print(f"  {nombre}: ECM = {ecm:.4f}")

    mejor_grado = min(ecm_grado, key=lambda g: np.mean(ecm_grado[g]))
    if verbose:
        print(f"\n  Grado óptimo: {mejor_grado}")

    return mejor_grado, resultados


# =============================================================================
# PUNTO 5: COMPARACIÓN FINAL SOBRE DATOS DE PRUEBA
# =============================================================================

def punto5_ecm_prueba(train, test, K_optimo, mejor_grado_local, verbose=True):
    """
    Se ajustan los tres mejores modelos sobre TODO el conjunto de entrenamiento
    y se evalúa el ECM sobre el conjunto de prueba externo (10% de los datos).

    Los tres paradigmas comparados son:
      1. Mejor modelo basado en base de funciones (polinomio grado 2 vs spline)
      2. Mejor modelo de regresión local (grado 1 ó 2)
      3. Polinomio global grado 2 (referencia de clase)

    Según la Tabla de la Sesión 3, el polinomio grado 2 logra ECM ≈ 16.38.
    """
    X_train = train[["horsepower"]].values
    y_train = train["mpg"].values
    X_test  = test[["horsepower"]].values
    y_test  = test["mpg"].values

    hp_min, hp_max = X_train.min(), X_train.max()
    knots_opt = np.linspace(hp_min, hp_max, K_optimo + 2)[1:-1]

    # --- Modelo A: Polinomio global grado 2 (referencia y mejor basado en base) ---
    pip_poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    pip_poly.fit(X_train, y_train)
    ecm_poly = mean_squared_error(y_test, pip_poly.predict(X_test))

    # --- Modelo B: Regression spline óptimo ---
    mod_rs = RegresionSpline(knots=knots_opt)
    mod_rs.fit(X_train, y_train)
    ecm_rs = mean_squared_error(y_test, mod_rs.predict(X_test))

    # --- Modelo C: Smoothing spline (ajustado sobre todo el entrenamiento) ---
    x_tr = X_train.ravel()
    idx_orden = np.argsort(x_tr)
    x_sorted  = x_tr[idx_orden]
    y_sorted  = y_train[idx_orden]

    # Se eliminan duplicados antes de ajustar el spline
    x_uniq, idx_uniq = np.unique(x_sorted, return_index=True)
    y_uniq = y_sorted[idx_uniq]

    spl      = UnivariateSpline(x_uniq, y_uniq, s=None, k=3)
    pred_spl = spl(X_test.ravel())

    # Se imputan NaN residuales con la media de entrenamiento
    nan_mask = np.isnan(pred_spl)
    if nan_mask.any():
        pred_spl[nan_mask] = np.mean(y_train)

    ecm_ss = mean_squared_error(y_test, pred_spl)

    # --- Modelo D: Mejor regresión local ---
    h = seleccionar_bandwidth(X_train, y_train)
    mod_local = RegresionLocal(bandwidth=h, grado=mejor_grado_local)
    mod_local.fit(X_train, y_train)
    pred_local = mod_local.predict(X_test)
    mask_valid = ~np.isnan(pred_local)
    if mask_valid.sum() > 0:
        ecm_local = mean_squared_error(y_test[mask_valid], pred_local[mask_valid])
    else:
        ecm_local = np.nan

    resultados = {
        "Polinomio grado 2":              ecm_poly,
        f"Reg. Spline (K={K_optimo})":    ecm_rs,
        "Smoothing Spline":               ecm_ss,
        f"Local grado {mejor_grado_local}": ecm_local,
    }

    if verbose:
        print(f"\n[Punto 5] ECM de prueba (sobre el 10% externo):")
        for nombre, ecm in resultados.items():
            print(f"  {nombre}: ECM = {ecm:.4f}")

    mejor = min(resultados, key=resultados.get)
    if verbose:
        print(f"\n  Mejor modelo: {mejor} (ECM = {resultados[mejor]:.4f})")

    return resultados, pip_poly, mod_rs, spl, mod_local, mejor_grado_local


# =============================================================================
# VISUALIZACIÓN GENERAL DE RESULTADOS
# =============================================================================

def graficar_resultados(train, test, K_optimo, ecm_knots,
                        resultados_p3, resultados_p5,
                        mod_poly, mod_rs, mod_ss, mod_local, mejor_grado_local):
    """
    Se visualizan los resultados de los puntos 2 al 5 en una figura con
    cuatro paneles, siguiendo el estilo de los gráficos de la Sesión 3.
    """
    # Se usa el rango interno de HP (sin extrapolar) para evitar
    # inestabilidad numérica del smoothing spline fuera de su dominio
    hp_range = np.linspace(train["horsepower"].min(),
                           train["horsepower"].max(), 400)
    X_range  = hp_range.reshape(-1, 1)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # --- Panel 1: ECM vs número de knots (Punto 2) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ks  = list(ecm_knots.keys())
    ecms = list(ecm_knots.values())
    ax1.plot(ks, ecms, marker="o", color="steelblue", linewidth=2)
    ax1.axvline(K_optimo, color="red", linestyle="--", alpha=0.7, label=f"K óptimo = {K_optimo}")
    ax1.set_xlabel("Número de knots (K)")
    ax1.set_ylabel("ECM (CV 10-folds)")
    ax1.set_title("Punto 2: Selección de knots")
    ax1.legend(fontsize=9)
    ax1.set_xticks(ks)

    # --- Panel 2: Modelos base de funciones ajustados (Punto 3) ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(train["horsepower"], train["mpg"],
                alpha=0.4, color="grey", s=20, label="Datos entrenamiento", marker="^")

    # Polinomio grado 2
    pred_poly_range = mod_poly.predict(X_range)
    ax2.plot(hp_range, pred_poly_range, color="orange", linewidth=2,
             label=f"Polin. g2 ECM={resultados_p5.get('Polinomio grado 2', 0):.2f}")

    # Regression spline óptimo
    pred_rs_range = mod_rs.predict(X_range)
    ax2.plot(hp_range, pred_rs_range, color="dodgerblue", linewidth=2, linestyle="--",
             label=f"Reg. Spline K={K_optimo}")

    # Smoothing spline (limitado al rango de datos para evitar explosión)
    pred_ss_range = mod_ss(hp_range)
    y_margin = train["mpg"].max() - train["mpg"].min()
    y_lo = train["mpg"].min() - 0.2 * y_margin
    y_hi = train["mpg"].max() + 0.2 * y_margin
    pred_ss_clipped = np.clip(pred_ss_range, y_lo, y_hi)
    ax2.plot(hp_range, pred_ss_clipped, color="purple", linewidth=2, linestyle=":",
             label="Smooth Spline")

    ax2.set_xlabel("HP")
    ax2.set_ylabel("mpg")
    ax2.set_title("Punto 3: Base de funciones")
    ax2.legend(fontsize=7)

    # --- Panel 3: Regresión local (Punto 4) ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(train["horsepower"], train["mpg"],
                alpha=0.4, color="grey", s=20, marker="^")

    pred_local_range = mod_local.predict(X_range)
    ax3.plot(hp_range, pred_local_range, color="green", linewidth=2,
             label=f"Local grado {mejor_grado_local}")
    ax3.plot(hp_range, pred_poly_range, color="orange", linewidth=2,
             linestyle="--", alpha=0.7, label="Polin. g2 (ref.)")
    ax3.set_xlabel("HP")
    ax3.set_ylabel("mpg")
    ax3.set_title("Punto 4: Regresión local")
    ax3.legend(fontsize=9)

    # --- Panel 4: ECM de prueba comparado (Punto 5) ---
    ax4 = fig.add_subplot(gs[1, 1])
    nombres  = list(resultados_p5.keys())
    ecms_p5  = list(resultados_p5.values())
    colores  = ["orange", "dodgerblue", "purple", "green"][:len(nombres)]
    bars = ax4.barh(nombres, ecms_p5, color=colores, alpha=0.85)

    ecm_min = min(ecms_p5)
    ax4.axvline(ecm_min, color="red", linestyle="--", linewidth=1.5, label="Mejor ECM")
    for bar, v in zip(bars, ecms_p5):
        ax4.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{v:.2f}", va="center", fontsize=9)
    ax4.set_xlabel("ECM de prueba")
    ax4.set_title("Punto 5: Comparación final")
    ax4.legend(fontsize=9)
    ax4.set_xlim(0, max(ecms_p5) * 1.2)

    plt.suptitle("Taller 2 – Análisis Avanzado de Datos\nPuntos 1–5: Más allá de la linealidad",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.savefig("resultados_taller2.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n[Gráfica] Figura guardada como 'resultados_taller2.png'")


# =============================================================================
# PUNTO 6: SIMULACIÓN DE 10 REPETICIONES
# =============================================================================

def punto6_simulacion_completa(auto, n_iter=10):
    """
    Se repiten los pasos (1) a (5) un total de 10 veces, generando una nueva
    muestra de validación en cada iteración. Se obtienen los ECM de prueba
    para cada uno de los tres paradigmas de modelamiento.
    """
    print("\n" + "="*60)
    print(f" PUNTO 6: Simulando {n_iter} repeticiones del proceso...")
    print("="*60)

    ecm_base_funciones = []
    ecm_regresion_local = []
    ecm_polinomio_global = []

    for i in range(n_iter):
        # Se cambia la semilla en cada iteración para variar la partición
        semilla_iter = SEMILLA + i
        np.random.seed(semilla_iter)
        
        # (1) Nueva muestra (90% entrenamiento, 10% prueba)
        train_iter, test_iter = separar_datos(auto, verbose=False)
        
        # (2) Knots óptimos (enfoque silencioso)
        K_optimo_i, _ = punto2_seleccion_knots(train_iter, verbose=False)
        
        # (3) Mejor modelo base de funciones (CV 10-folds)
        res_p3_i, mejor_p3_i, _ = punto3_comparacion_modelos(train_iter, K_optimo_i, verbose=False)
        
        # (4) Mejor regresión local
        mejor_grado_local_i, _ = punto4_regresion_local(train_iter, verbose=False)
        
        # (5) ECM de prueba para cada paradigma
        res_p5_i, _, _, _, _, _ = punto5_ecm_prueba(
            train_iter, test_iter, K_optimo_i, mejor_grado_local_i, verbose=False
        )
        
        # Registro de resultados por paradigma
        # Paradigma A: Polinomio Global Grado 2
        ecm_polinomio_global.append(res_p5_i["Polinomio grado 2"])
        
        # Paradigma B: Regresión Local (mejor grado)
        ecm_regresion_local.append(res_p5_i[f"Local grado {mejor_grado_local_i}"])
        
        # Paradigma C: Base de funciones (el mejor entre los splines)
        # El Punto 5 ya evalúa los splines. Seleccionamos el mejor de ellos.
        ecm_splines = [res_p5_i[f"Reg. Spline (K={K_optimo_i})"], res_p5_i["Smoothing Spline"]]
        ecm_base_funciones.append(min(ecm_splines))
        
        if (i + 1) % 2 == 0:
            print(f"  > Iteración {i+1}/{n_iter} completada.")

    # --- Visualización de Distribuciones (Boxplot) ---
    resultados_sim = {
        "Base de Funciones": ecm_base_funciones,
        "Regresión Local": ecm_regresion_local,
        "Polinomio Global": ecm_polinomio_global
    }
    
    df_sim = pd.DataFrame(resultados_sim)
    
    plt.figure(figsize=(10, 6))
    boxplot = plt.boxplot([df_sim[col] for col in df_sim.columns], 
                          labels=df_sim.columns, patch_artist=True,
                          medianprops={'color': 'black', 'linewidth': 1.5})
    
    colores = ["dodgerblue", "green", "orange"]
    for patch, color in zip(boxplot['boxes'], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.ylabel("ECM de prueba")
    plt.title("Punto 6: Distribución del ECM de prueba por Paradigma (10 Repeticiones)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    filename = "comparacion_paradigmas_p6.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[Gráfica] Distribuciones guardadas como '{filename}'")

    # --- Resumen Estadístico ---
    print("\nResumen de ECM de prueba (10 iteraciones):")
    resumen = df_sim.describe().loc[['mean', 'std', 'min', 'max']]
    print(resumen)
    
    mejor_paradigma = resumen.loc['mean'].idxmin()
    print(f"\nConclusión: Basado en el ECM promedio, el acercamiento '{mejor_paradigma}' "
          "resulta ser el más efectivo para predecir mpg en este conjunto de datos.")

    return df_sim


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" TALLER 2 – Análisis Avanzado de Datos")
    print(" Puntos 1 al 6: Regresión no lineal (dataset Auto)")
    print("=" * 60)

    # Se carga el dataset
    auto = cargar_auto()
    print(f"\nDataset cargado: {len(auto)} autos")

    # Ejecución inicial (Puntos 1-5) para visualización detallada
    print("\n--- Ejecución inicial detallada ---")
    train, test = separar_datos(auto)
    K_optimo, ecm_knots = punto2_seleccion_knots(train)
    resultados_p3, mejor_p3, knots_opt = punto3_comparacion_modelos(train, K_optimo)
    mejor_grado_local, resultados_p4 = punto4_regresion_local(train)
    (resultados_p5, mod_poly, mod_rs, mod_ss,
     mod_local, mejor_grado_local) = punto5_ecm_prueba(
        train, test, K_optimo, mejor_grado_local
    )

    # Visualización Puntos 1-5
    graficar_resultados(
        train, test, K_optimo, ecm_knots,
        resultados_p3, resultados_p5,
        mod_poly, mod_rs, mod_ss, mod_local, mejor_grado_local
    )

    # Punto 6: Simulación y comparación de paradigmas
    punto6_simulacion_completa(auto, n_iter=10)

    print("\n" + "="*60)
    print(" [FINAL] Ejecución completada exitosamente.")
    print("="*60)
