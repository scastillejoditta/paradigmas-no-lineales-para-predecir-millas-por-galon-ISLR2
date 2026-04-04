# Taller 2 – Análisis Avanzado de Datos
## Más allá de la linealidad: Splines y Regresión Local

**Curso:** Análisis Avanzado de Datos
**Profesor:** Nicolás López
**Estudiantes:** Sara Castillejo, Stefany Mojica y Juan Rodríguez
**Maestría:** Matemáticas Aplicadas y Ciencias de la Computación  
**Semestre:** 2026/1  
**Universidad:** Universidad del Rosario  

---

## Descripción

El problema central es que la regresión lineal asume que la relación entre dos variables es una línea recta, pero en la realidad casi nunca lo es.

En este taller trabajamos con el dataset `Auto` (ISLR2), donde queremos predecir el rendimiento de un carro (mpg, millas por galón) a partir de su potencia (horsepower). Al graficar esos datos, vemos que la relación tiene forma de curva: los carros con poca potencia son muy eficientes y, a medida que sube la potencia, el rendimiento cae, pero no de forma pareja — cae rápido al principio y luego se estabiliza.

Una línea recta no puede capturar eso bien, y por eso su error de predicción es alto (ECM ≈ 22). Entonces la pregunta que intentamos responder es: **¿cuál es la mejor manera de ajustar una curva a esos datos, de forma que prediga bien en datos nuevos que el modelo nunca vio?**
Para responderla exploramos tres familias de herramientas:

1. Base de funciones (polinomios y splines): se construye una curva matemática flexible dividiendo el rango de HP en segmentos y ajustando polinomios que empalman suavemente entre sí.
2. Regresión local: en lugar de ajustar una curva global, se ajusta un modelo pequeño en cada punto usando solo los datos vecinos.
3. Polinomio global: la referencia más simple, que extiende la línea recta a una curva con un solo polinomio para todos los datos.

No basta con que el modelo se ajuste bien a los datos que ya conoce: Lo importante es que prediga bien datos nuevos. Por eso usamos validación cruzada y un conjunto de prueba separado para medir el error real de generalización. Al final, entregamos nuestro concepto sobre cuál paradigma resuelve mejor este problema. 

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `taller2_puntos1_5.py` | Script Python con la solución completa de los puntos 1 al 5 |
| `taller2_notebook.Rmd` | Notebook RMarkdown con explicaciones conceptuales y contexto teórico |
| `requirements.txt` | Dependencias de Python |

## Métodos implementados

- **Punto 1:** Separación entrenamiento/prueba (90/10) con semilla fija
- **Punto 2:** Selección de knots óptimos (K=1..10) por CV 10-folds — Regression Spline
- **Punto 3:** Comparación por CV 10-folds: Polinomio grado 2, Smoothing Spline, Regression Spline
- **Punto 4:** Regresión local (Nadaraya-Watson) con kernel gaussiano, grado 1 vs 2
- **Punto 5:** Evaluación final sobre datos de prueba externos
- **Punto 6:** Obtener 10 ECM de prueba para cada paradigma de modelamiento y elegir el mejor.

## Cómo ejecutar

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
python taller2.py
```

## Referencias

- James et al. *An Introduction to Statistical Learning with R*, 2a ed., Cap. 7
- Hastie et al. *The Elements of Statistical Learning*, Cap. 5 y 9
- López, N. *Sesión 3: Más allá de la linealidad*, AAD 2026
