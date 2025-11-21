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

| Lenguaje | Implementación | Tiempo Aproximado | Comentario |
|----------|----------------|------------------|-------------|
| Python (NumPy) | Vectorizada | **1–3 ms** | Muy rápida gracias a librerías C/Fortran |
| Python (sin NumPy) | Puro Python | **60–120 ms** | Mucho más lenta por sobrecarga del intérprete |
| Rust (sin crates) | Bucle manual | **2–8 ms** | Similar o más rápido que NumPy en dataset pequeño |
| Rust (con `ndarray`) | Vectorizada | **1–3 ms** | Equiparable o superior a NumPy |

### **Conclusión de Desempeño**
- Para ejecuciones matemáticas de bajo nivel Rust suele ser **notablemente más rápido** que Python.
- Python sigue siendo preferible para prototipado rápido y tareas de alto nivel gracias a NumPy.
- Rust es ideal cuando se necesita:
  - Alto rendimiento.
  - Bajo consumo de memoria.
  - Seguridad en el manejo de datos.
  - Sistemas de producción donde la consistencia y velocidad son críticas.

---

## (Espacio reservado para los Puntos 1 y 2)

> *Aquí va la introducción teórica general (punto 1) y la explicación de regresión lineal (punto 2). No se incluyen en este documento debido a que el usuario indicó que este corresponde al punto 3.*

