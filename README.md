# 2. Diseño detallado de la solución utilizando el paradigma de Aspectos (AOP)

En este apartado se presenta el **diseño arquitectónico** para una solución de **regresión lineal** basada en el **paradigma de programación orientada a aspectos (AOP)**. El objetivo es identificar de manera explícita los **puntos de corte (pointcuts)**, los **aspectos transversales** y las **responsabilidades centrales (Core)** que intervienen en el proceso de entrenamiento de un modelo de regresión lineal.

Este diseño NO incluye implementación; se enfoca en la estructura, responsabilidades y comportamiento del sistema.

---

## 2.1 Objetivo del diseño
El objetivo de la arquitectura AOP es **separar las preocupaciones principales** (entrenar un modelo de regresión lineal) de las **preocupaciones transversales** (logging, medición de tiempo, validación, sincronización, etc.).

Esto permite:
- Un diseño más modular y extensible.
- Aislar los elementos transversales.
- Permitir instrumentación sin modificar el código central.
- Facilitar futuras optimizaciones o extensiones del sistema.

---

## 2.2 Componentes del sistema
El sistema se divide en dos grandes grupos: **Core** y **Aspects**.

### **A. Módulos Core**
Estos representan la lógica esencial de la regresión lineal.

### **1. LinearModel**
Encargado de mantener los parámetros del modelo y realizar predicciones.
- **Atributos:**
  - `w: float` (pendiente)
  - `b: float` (intercepto)
- **Funciones:**
  - `predict(x)` → calcula `w * x + b`

### **2. Dataset**
Provee los datos necesarios para el entrenamiento.
- **Atributos:** `X`, `y`
- **Funciones:**
  - `next_batch(n)` → devuelve subconjuntos de entrenamiento

### **3. Optimizer**
Se encarga del cálculo de gradientes y actualización de parámetros.
- **Funciones:**
  - `compute_gradients(model, Xb, yb)`
  - `update(model, dw, db)`

### **4. Trainer**
Controla el ciclo completo de entrenamiento.
- **Atributos:** `epochs`, `learning_rate`, `batch_size`
- **Funciones:**
  - `train()` (bucle principal)
  - `startEpoch()` (join point)
  - `endEpoch()` (join point)
  - `validateModel()` (join point)

---

### **B. Aspectos transversales (AOP)**
Estos módulos interceptan la ejecución en los **join points** definidos por el `Trainer`.

#### **1. LoggingAspect**
Responsable de registrar métricas durante el entrenamiento.
- `pointcut endEpoch()`
- `after endEpoch()`: imprime MSE, w, b, gradientes

#### **2. TimingAspect**
Mide el tiempo que toma cada época.
- `pointcut startEpoch()`
- `around startEpoch()`: inicia y detiene temporizador

#### **3. ValidationAspect**
Realiza validación periódica.
- `pointcut validateModel()`
- `after validateModel()`: calcula MSE en datos de validación

Estos aspectos funcionan sin modificar el código del `Trainer` gracias a la técnica de *weaving*.

---

## 2.3 Flujo general de ejecución
1. `Trainer.train()` inicia el proceso.
2. Se activa `startEpoch()` → interceptado por **TimingAspect**.
3. Se obtiene un batch desde `Dataset`.
4. `Optimizer` calcula gradientes.
5. `Optimizer.update()` modifica los parámetros.
6. Se activa `endEpoch()` → interceptado por **LoggingAspect**.
7. Cada N épocas se activa `validateModel()` → interceptado por **ValidationAspect**.
8. Repite hasta alcanzar `epochs`.

---

## 2.4 Correspondencia con el diagrama UML
El diseño descrito corresponde directamente al diagrama PlantUML que se generará en la siguiente sección, integrando:
- Paquete **Core** con sus relaciones.
- Paquete **Aspects** con todos los pointcuts.
- Conexiones mediante líneas discontinuas que indican el *weaving*.

---
# 3. Implementación en Rust de Regresión Lineal y Comparación de Desempeño con Python

## 3.1 Implementación de Regresión Lineal en Rust

La siguiente implementación realiza el entrenamiento de un modelo de regresión lineal mediante **descenso de gradiente**. Esta versión no utiliza librerías externas, siguiendo una aproximación de bajo nivel para resaltar las capacidades del lenguaje Rust en operaciones numéricas:

```rust
fn main() {
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let mut w: f64 = 0.0;
    let mut b: f64 = 0.0;

    let learning_rate = 0.01;
    let epochs = 1000;

    let m = x.len() as f64;

    for epoch in 0..epochs {
        let y_pred: Vec<f64> = x.iter().map(|xi| w * xi + b).collect();
        let error: Vec<f64> = y_pred.iter().zip(y.iter()).map(|(yp, yt)| yp - yt).collect();

        let dw = (2.0 / m) * error.iter().zip(x.iter()).map(|(e, xi)| e * xi).sum::<f64>();
        let db = (2.0 / m) * error.iter().sum::<f64>();

        w -= learning_rate * dw;
        b -= learning_rate * db;
    }

    println!("w = {:.4}, b = {:.4}", w, b);
}
```

Este programa implementa de manera explícita todos los pasos del cálculo: predicciones, error, gradientes y actualización de parámetros.

---

## 3.2 Comparación de Desempeño: Python vs Rust

### **Python**
- Python utiliza librerías altamente optimizadas (NumPy, BLAS, MKL), pero sigue operando sobre un intérprete.
- Fácil de escribir y leer, ideal para prototipado rápido.
- En bucles intensivos sin NumPy, Python es significativamente más lento debido a la sobrecarga del intérprete.

### **Rust**
- Rust compila a código máquina, lo que elimina sobrecarga de ejecución.
- Control total de memoria y optimización: excelente para cálculos numéricos intensivos.
- A pesar de no usar librerías optimizadas en este ejemplo, Rust suele superar a Python en **entre 10x y 50x** en operaciones repetitivas basadas en bucles.

### **Resultado Esperado de Benchmarks (aproximado)**
> *Estos valores pueden cambiar dependiendo del hardware y optimizaciones específicas.*

| Lenguaje | Implementación | Tiempo Aproximado |
|----------|----------------|------------------|
| Python (NumPy) | Vectorizada | **1–3 ms** |
| Rust (con `ndarray`) | Vectorizada | **1–3 ms** |

### **Conclusión de Desempeño**
- Para ejecuciones matemáticas de bajo nivel Rust suele ser **notablemente más rápido** que Python.
- Python sigue siendo preferible para prototipado rápido y tareas de alto nivel gracias a NumPy.
- Rust es ideal cuando se necesita:
  - Alto rendimiento.
  - Bajo consumo de memoria.
  - Seguridad en el manejo de datos.
  - Sistemas de producción donde la consistencia y velocidad son críticas.
