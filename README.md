# Predicción de alertas de desnutrición infantil

**Universidad Yachay Tech — Maestría en Ciencia de Datos**

**Asignatura:** Fundamentos de Inteligencia Artificial

**Autor:** Mat. Héctor Alexander Román Potosí

**Docente:** Ph.D. Manuel Eugenio Morocho Cayamcela

**Periodo académico:** IISEM2026

## Descripción

Proyecto académico de clasificación binaria sobre un conjunto
de datos sintético de alertas de desnutrición infantil en Ecuador.

Se compararon algoritmos de aprendizaje automático y modelos
probabilísticos, priorizando accuracy como métrica oficial.

## Resultado oficial de Kaggle

- Competencia: Ecuador Infant Malnutrition Competition (IISEM2026 MCD)
- Accuracy público: 0.911254675280 (91.1255 %)
- Referencia del envío: 57815647
- Modelo enviado: Random Forest
- Predicciones: 30.000
- Estado: COMPLETE

**Importante:** todas las predicciones enviadas fueron clase cero.
Por tanto, el rendimiento coincide con el baseline de clase mayoritaria.

## Modelos evaluados

- DummyClassifier
- Regresión logística
- Random Forest
- Extra Trees
- HistGradientBoosting
- LightGBM
- CatBoost
- XGBoost
- Red neuronal MLP
- Modelo beta-binomial
- Modelo estocástico Probit

## Resultados principales

| Modelo | Accuracy de validación |
|---|---:|
| DummyClassifier | 0.89987 |
| Random Forest | 0.89987 |
| HistGradientBoosting | 0.89987 |
| XGBoost (5 folds) | 0.89987 |

Los modelos evaluados no demostraron una mejora consistente
del accuracy frente al baseline.

## Estructura del repositorio

- informe/: informe académico PDF y fuente LaTeX.
- midterm_ia.ipynb: notebook reproducible.
- midterm_ia_ejecutado.ipynb: notebook con resultados.
- scripts/: experimentos complementarios.
- outputs/: métricas, comparaciones y predicciones.
- evidencia/: resultados de la competencia.
- requirements.txt: dependencias de Python.

## Reproducción

### 1. Crear el entorno virtual

~~~powershell
python -m venv .venv
~~~

### 2. Instalar las dependencias

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt kaggle
~~~

### 3. Obtener el dataset oficial

Es necesario tener acceso a la competencia en Kaggle
y haber aceptado sus reglas.

~~~powershell
New-Item -ItemType Directory -Force data\ii_sem2026 | Out-Null

.\.venv\Scripts\kaggle.exe competitions download -c ecuador-infant-malnutrition-competition-iisem-2026-mcd -p data\ii_sem2026 --unzip
~~~

### 4. Ejecutar el notebook

~~~powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute midterm_ia.ipynb --output midterm_verificado.ipynb
~~~

### 5. Consultar los resultados

El notebook permite reproducir la preparación de datos,
la evaluación de los modelos y la generación de submission.csv.

Los scripts complementarios documentan los experimentos
de aprendizaje automático e inferencia probabilística.

## Metodología

1. Análisis exploratorio de datos.
2. Identificación de valores faltantes y anomalías.
3. Preparación y codificación de variables.
4. Construcción del baseline.
5. Entrenamiento y comparación de clasificadores.
6. Validación cruzada estratificada.
7. Análisis de umbrales y accuracy.
8. Modelado estocástico y probabilístico.
9. Predicciones para Kaggle.
10. Documentación y auditoría reproducible.

## Interpretación matemática

Con costos de error iguales, la regla bayesiana
de clasificación predice la clase de mayor
probabilidad condicional.

Los experimentos mostraron que los clasificadores
no lograron identificar de manera consistente
observaciones para las que predecir la clase positiva
aumentara el accuracy.

El modelo Probit ofreció una explicación probabilística
compatible con ese resultado, bajo sus supuestos.

## Limitaciones

- El dataset utilizado es sintético.
- El accuracy público no es el resultado privado definitivo.
- Un recall de cero no es adecuado para un sistema clínico.
- No se utilizaron etiquetas ocultas para ajustar los modelos.
- Los datos originales no se redistribuyen en este repositorio.

## Referencias

- Breiman (2001). Random Forests.
- Grinsztajn, Oyallon y Varoquaux (2022).
- McElfresh y colaboradores (2023).
- Repositorio académico del docente:
  https://github.com/eugeniomorocho/ArtificialIntelligence

## Autor

**Mat. Héctor Alexander Román Potosí**

Maestría en Ciencia de Datos

Universidad Yachay Tech — Ecuador

Octubre de 2026
