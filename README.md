# VitaSense

**Monitoreo experimental de signos vitales sin contacto mediante visión por computadora y procesamiento de señales**

> \*\*Estado:\*\* prototipo de investigación para hackathon. \*\*No es un dispositivo médico\*\* ni está validado para diagnóstico, vigilancia clínica o decisiones terapéuticas.

## ¿Qué es VitaSense?

VitaSense es un prototipo desarrollado en Python que utiliza una cámara convencional para investigar la estimación **sin contacto** de la **frecuencia cardíaca (FC)** y la **frecuencia respiratoria (FR)**. Combina detección facial, fotopletismografía remota (*remote photoplethysmography*, **rPPG**), análisis de movimiento y procesamiento digital de señales. También incorpora una estructura de investigación para explorar la **presión arterial (PA)** mediante aprendizaje automático, pero **no dispone de un modelo de PA listo para medir de forma fiable**.



**Problema que aborda:** muchos sistemas de monitoreo requieren sensores de contacto. Remote Vital explora si una webcam de bajo costo puede ofrecer información fisiológica preliminar, con indicadores explícitos de calidad y sin confundir una estimación experimental con una medición clínica.



### Estado actual de las funciones

|Función|Estado|Qué muestra|
|-|-|-|
|FC por rPPG facial|Implementada, experimental|Estimaciones en lpm, señal de pulso, evolución temporal, SNR y calidad|
|FR por fusión de señales|Implementada, experimental|Estimación en respiraciones/min cuando hay datos y señal suficientes; puede no devolver valor|
|Sesiones de 15, 30 y 60 segundos|Implementadas en v0.7|Selección por teclado, cronómetro y exportación de resultados|
|PA por aprendizaje automático|**Investigación / no validada**|Estructura para recopilar datos, entrenar modelos y aplicar calibración; no hay estimador clínico listo|
|SpO₂|Exploratoria y opcional|Razones cromáticas o estimaciones sujetas a calibración; no equivale a un pulsioxímetro|

## Requisitos

* **Python 3.11** (versión utilizada durante el desarrollo).
* Windows 10/11, Linux o macOS con cámara compatible con OpenCV; los ejemplos siguientes están orientados a **Windows y PowerShell**.
* Webcam integrada o USB; una cámara **720p a 30 FPS** permite realizar pruebas exploratorias. La frecuencia de captura efectiva puede ser menor.
* Iluminación frontal, uniforme y estable; rostro visible y, para la fuente torácica de FR, hombros/parte superior del tórax dentro del encuadre.
* Conexión a Internet **en la instalación inicial** si es necesario descargar dependencias o el modelo facial de MediaPipe.

### Instalación

1. Descarga o clona el proyecto: https://github.com/braytwing/VitaSense y abre una terminal en la carpeta que contiene `run\_live.py` y `requirements.txt`.
2. Crea un entorno virtual e instala las dependencias:

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

El archivo `requirements.txt` de la versión revisada declara **NumPy, SciPy, OpenCV y MediaPipe**. El proyecto usa principalmente estas bibliotecas para cálculo numérico, filtrado, cámara y detección facial. El módulo opcional de PA contempla un adaptador PyTorch, que requiere instalación adicional y un modelo compatible; **no es necesario para la demo de FC/FR**.

Si PowerShell bloquea la activación del entorno, puedes ejecutar directamente `\\.venv\\Scripts\\python.exe` en lugar de `python` en los comandos siguientes.



### Ejecutar la demostración

**Prueba de 60 segundos con FC y FR:**

```powershell
python run\_live.py --camera 0 --duration 60 --rr --width 640 --height 480
```

**Prueba de 30 segundos:**

```powershell
python run\_live.py --camera 0 --duration 30 --rr
```

**Monitor continuo con FC y FR:**

```powershell
python run\_live.py --camera 0 --rr
```

**Probar otro algoritmo de rPPG:**

```powershell
python run\_live.py --camera 0 --duration 60 --rr --algo chrom
```

Si utilizas una webcam USB que aparece como segundo dispositivo, prueba `--camera 1`. En Windows, `--backend dshow` permite solicitar DirectShow. El parámetro `--fps 30` solicita la frecuencia de captura, pero no garantiza los FPS reales.

**Controles en la ventana:** `1` = 15 s; `2` = 30 s; `3` = 60 s; `ESPACIO` = iniciar/repetir; `R` = reiniciar; `Q` = salir. Para que el estimador respiratorio se construya desde el inicio, **arranca con `--rr` o `--duration`**; cambiar la duración durante una sesión iniciada en modo continuo no incorpora por sí solo módulos que no se cargaron.

Al finalizar una sesión temporizada, el programa crea archivos **CSV, TXT y PNG** en la carpeta `measurements/` (relativa al directorio desde el que ejecutas el programa). El CSV registra resultados por ventana, no una medición clínica continua de referencia.



## Arquitectura




Cámara / archivo de video
         |
         v
   sources.py  ----> Fotogramas y marcas de tiempo
         |
         v
      roi.py   ----> Detección facial y regiones de piel
         |            (frente y mejillas; RGB promedio)
         v
     traces.py ----> Series temporales RGB
         |
         v
 algorithms.py ----> POS / CHROM / GREEN
         |
         v
      dsp.py   ----> Filtrado y análisis espectral
         |
         v
   pipeline.py ----> Orquestación de los estimadores
       /    |   \\
      /     |    \\
 estimators.py  respiration.py   bp.py
    FC             FR           PA experimental
      \\             |            /
       \\            |           /
        quality.py + viz.py + run\_live.py
           |       |        |
         calidad  gráficas  interfaz/exportación


```

### Principales módulos

|Archivo|Responsabilidad|
|-|-|
|`run\_live.py`|Configuración, inicio de cámara, modos de tiempo, interfaz y exportación|
|`rppg/sources.py`|Entrada desde webcam o vídeo|
|`rppg/roi.py`|Detección del rostro y selección de regiones de interés (ROI)|
|`rppg/traces.py`|Almacenamiento de señales RGB y tiempos|
|`rppg/algorithms.py`|Algoritmos rPPG POS, CHROM y GREEN|
|`rppg/dsp.py`|Herramientas de procesamiento digital|
|`rppg/estimators.py`|Estimación de FC y seguimiento temporal|
|`rppg/respiration.py`|Estimación de FR mediante distintas fuentes y fusión espectral|
|`rppg/chest.py`|Movimiento torácico mediante flujo óptico|
|`rppg/quality.py`|Métricas y alertas heurísticas de calidad|
|`rppg/pipeline.py`|Coordinación del procesamiento por fotograma|
|`rppg/viz.py`|Onda de pulso, tendencias y paneles de calidad|
|`rppg/bp.py` / `train\_bp.py`|Infraestructura experimental de PA y entrenamiento con datos etiquetados|
|`tests/`|Pruebas automatizadas de componentes y señales sintéticas|

## ¿Cómo estima la frecuencia cardíaca?

1. **Captura:** la webcam obtiene fotogramas sucesivos del rostro.
2. **Regiones de interés:** se identifican áreas de piel, principalmente **frente y mejillas**, para evitar en lo posible ojos, labios y fondo.
3. **Señales RGB:** se calcula la intensidad media de los canales rojo, verde y azul en cada fotograma. Los pequeños cambios de color pueden contener variaciones asociadas al volumen sanguíneo pulsátil.
4. **Extracción rPPG:** se aplican métodos como **POS** (*Plane-Orthogonal-to-Skin*), **CHROM** o una aproximación basada en el canal verde.
5. **Procesamiento:** se filtra la señal y se busca su periodicidad en una banda fisiológicamente plausible mediante análisis espectral.
6. **Estimación:** la frecuencia dominante se transforma de Hz a latidos por minuto: **FC (lpm) = frecuencia (Hz) × 60**.
7. **Control de calidad:** se consideran SNR, movimiento, iluminación, FPS y otras comprobaciones. Una ventana puede contener una frecuencia calculada y, aun así, ser **rechazada por baja calidad**.

**Importante:** el pulso rPPG no es un electrocardiograma. Los picos visibles y las métricas derivadas de ellos no deben interpretarse como ECG ni como variabilidad cardíaca clínicamente validada.



## ¿Cómo estima la frecuencia respiratoria?

La FR no se obtiene contando directamente los latidos. El módulo `respiration.py` explora **varias manifestaciones de la respiración en el vídeo**:

* **Movimiento torácico:** desplazamientos del tórax/hombros estimados mediante flujo óptico (`chest.py`), siempre que estén visibles.
* **Modulación de amplitud del pulso (AM):** cambios respiratorios en la amplitud de la señal rPPG.
* **Modulación de frecuencia (FM):** oscilaciones en el intervalo entre pulsos compatibles con modulación respiratoria.
* **Deriva cromática de línea base:** variaciones lentas de una combinación de canales de color.
* **Micromovimientos faciales:** pequeñas oscilaciones asociadas a la respiración.
* **Aleteo nasal:** fuente **opcional y altamente experimental** (`--alar`); no debe tratarse como método validado para adultos en reposo.

Cada fuente se analiza en la banda respiratoria y sus espectros se combinan ponderando su calidad. Si distintas fuentes coinciden en una periodicidad, aumenta la evidencia para una estimación. El resultado se expresa en **respiraciones por minuto (resp/min)**.

**Tiempo de observación:** el estimador de FR de esta versión utiliza por defecto una ventana de **30 segundos**, con un mínimo de **20 segundos** para intentar la primera estimación. Por ello, una sesión de **15 segundos normalmente no entrega FR**. Las sesiones de **30 y 60 segundos** son más adecuadas; aun así, la salida puede ser `--` si la señal no es suficiente.



## ¿Cómo se espera que funcione la presión arterial?

> \*\*Módulo en investigación. Actualmente NO mide presión arterial de forma validada.\*\*

La hipótesis experimental consiste en estudiar si características de la onda de pulso rPPG, combinadas con datos de referencia, pueden ayudar a **predecir presión arterial sistólica (PAS) y diastólica (PAD)**.

**Flujo de investigación propuesto:**

1. Capturar una señal rPPG facial con calidad suficiente.
2. Extraer características de la señal, como FC, intervalos entre pulsos y descriptores morfológicos de la onda.
3. Asociar las ventanas a lecturas de **tensiómetro de referencia** registradas mediante un protocolo definido.
4. Entrenar o adaptar un modelo de regresión o aprendizaje profundo usando pares señal–PA.
5. Estudiar la calibración individual y comprobar su estabilidad en distintas sesiones.
6. Evaluar en **personas y sesiones no utilizadas durante el entrenamiento**, comparando con una predicción basal sencilla (por ejemplo, la media de PA del conjunto de entrenamiento).
7. Reportar errores separados para PAS/PAD, incluyendo MAE, sesgo, dispersión, cobertura y fallos de calidad.

El código actual incluye infraestructura para recopilar ventanas, un modelo de regresión *ridge* entrenable con datos propios, una interfaz para modelos PyTorch y calibración individual. **No incluye un modelo preentrenado de PA verificado y listo para funcionar con esta webcam.** El estimador está diseñado para no presentar cifras sin modelo y calibración.

**Limitación fundamental:** la rPPG de una webcam a 30 FPS no equivale a una señal PPG de contacto adquirida a 125 Hz. Remuestrear digitalmente la señal no recupera la información fisiológica fina que la cámara no captó. Los resultados de modelos entrenados con PPG de contacto **no pueden extrapolarse directamente** a rPPG facial.



## Calidad de señal y resultados

La interfaz puede mostrar:

* Onda de pulso rPPG en **unidades relativas**.
* Evolución temporal de FC y, cuando está habilitada y es estimable, FR.
* Indicadores heurísticos de calidad y relación señal/ruido (**SNR**).
* Alertas como `MOVIMIENTO`, `POCA\_LUZ`, `SOBREEXPUESTO`, `FPS\_BAJO` o `SNR\_BAJO`.
* Sesiones temporizadas y archivos exportados para análisis posterior.

**Reconocemos que tener una cifra cercana al valor de un oxímetro no demuestra precisión.** Para evaluar el sistema es necesario comparar mediciones simultáneas, registrar los fallos y repetir pruebas bajo distintas condiciones.



### Recomendaciones para una demo

1. Colocar la cámara estable frente al participante, con **luz difusa y constante**.
2. Evitar hablar, mover la cabeza o cambiar la iluminación durante la adquisición.
3. Mantener **rostro, hombros y parte superior del tórax** visibles si se quiere probar FR por movimiento.
4. Ejecutar una sesión de **60 segundos** y mostrar las curvas junto con los indicadores de calidad.
5. Si se dispone de un oxímetro y un conteo respiratorio de referencia, registrar los resultados **sin asumir concordancia automática**.
6. En caso de que aparezca el siguiente menaje: `--` significa que el sistema no dispone de una estimación aceptada, no necesariamente que la cámara dejó de funcionar.
7. 

## Validación pendiente y hoja de ruta

* **FC:** comparar con pulsioxímetro o ECG de referencia según disponibilidad, en reposo y durante recuperación postejercicio, registrando tiempos sincronizados y ventanas rechazadas.
* **FR:** comparar con conteo respiratorio independiente o sensor de referencia, en distintos ritmos y encuadres.
* **Robustez:** evaluar iluminación, tonos de piel, movimiento, FPS reales, distancia a cámara, dispositivos y sujetos distintos; analizar posibles diferencias de desempeño entre grupos.
* **Calidad:** calibrar umbrales y distinguir estimación experimental, estimación aceptada y ausencia de señal.
* **PA:** reunir datos con consentimiento, definir protocolo y evaluar modelos con separación estricta por participante antes de plantear cualquier aplicación clínica.
* **Ingeniería:** pruebas de integración de la interfaz y exportación, manejo de interrupciones de cámara y documentación reproducible del entorno.

## Privacidad y uso responsable

La arquitectura revisada procesa vídeo localmente y exporta principalmente **señales y métricas**. Los CSV y otros archivos no contienen información fisiológica sensible. No se sube grabaciones faciales, identificadores ni lecturas individuales a repositorios públicos sin autorización y consentimiento. Los datos recopilados en pruebas primarias han sido expresamente experimentales donde se explicó a los voluntarios y no se guardó algún dato de ellos.

## Alcance y advertencia

**Remote Vital es una prueba de concepto, no un dispositivo médico certificado.** Las estimaciones pueden fallar por iluminación, movimiento, oclusiones, compresión de vídeo, frecuencia de muestreo, características de la piel o baja relación señal/ruido. No utilizar para diagnosticar arritmias, apnea, hipertensión, hipoxemia u otras enfermedades; tampoco para tomar decisiones de atención médica.

## Tecnologías

**Python · OpenCV · MediaPipe · NumPy · SciPy · rPPG · POS/CHROM · procesamiento digital de señales · análisis espectral · flujo óptico · aprendizaje automático experimental.**

\---

*VitaSense explora el monitoreo fisiológico sin contacto con hardware accesible. Su contribución actual es una arquitectura modular que estima FC y FR de forma experimental, visualiza la calidad de señal y prepara un camino reproducible para investigar PA, sin presentar como clínicamente validadas funciones que todavía no lo están.*
