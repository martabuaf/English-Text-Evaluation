<h1 style="text-align:center;">Evaluación de textos en inglés</h1>

<h3>Resumen:</h3>
<p>El objetivo de este concurso es evaluar la calidad lingüística de textos en inglés. Utilizaremos un conjunto de datos de ensayos escritos por estudiantes para desarrollar una herramienta que sirva de apoyo a cualquier persona que pretenda mejorar su competencia en lengua inglesa.</p>
<p>El conjunto de datos comprende ensayos argumentativos escritos por estudiantes de inglés. Los ensayos se han puntuado según seis medidas analíticas: cohesión, sintaxis, vocabulario, fraseología, gramática y convenciones. Cada medida representa un componente de la competencia en la redacción de ensayos, y las puntuaciones más altas corresponden a una mayor competencia en esa medida. Las puntuaciones van de 1,0 a 5,0 en incrementos de 0,5. Utilizando estos datos, entrenaremos un modelo que predecirá la puntuación de cada una de las seis medidas para los textos introducidos en la herramienta. Para ello divideremos los datos en datos de entrenamiento y datos de prueba.</p>
<p>Los datos con los que hemos trabajado los encontramos <a href = "https://www.kaggle.com/competitions/feedback-prize-english-language-learning/data">aquí</a>.</p>

<h3>Paso 1: Carga y procesamiento de los datos</h3> 
<p>Procesaremos los textos de manera que resulte más fácil su interpretación y, al mismo tiempo, crearemos nuevos atributos que midan la cantidad de errores, ampliando así la cantidad de variables totales del modelo.</p>
<h3>Paso 2: Procesamiento del lenguaje natural (NLP)</h3> 
<p>Una vez hemos limpiado y corregido el texto, nos interesa evaluar la cohesión y la intencionalidad. Añadiremos estos atributos a los datos.</p>
<h3>Paso 3: Clasificación</h3> 
<p>Llevaremos a cabo la clasificación supervisada de los datos. Para ello utilizamos previamente los datos de entrenamiento para entrenar el modelo, que luego aplicaremos sobre los datos de prueba. Estudiaremos diferentes métodos de clasificación y buscaremos los valores óptimos de los parámetros.</p>
<h3>Paso 4: Evaluación de los resultados</h3> 
<p>Tras la evaluación de los distintos métodos, nos centraremos en el que mejor resultados nos aporta para nuestro fin. Representaremos los datos resultantes de la clasificación para los datos de prueba.</p>
<h3>Paso 6: Integración</h3> 
<p>Una vez que tenemos el modelo listo, lo integraremos con Streamlit y lo probaremos sobre nuevos datos.</p>
<p>Puedes utilizar nuestra app de corrector de textos en el siguiente <a href = "[https://www.kaggle.com/competitions/feedback-prize-english-language-learning/data](https://martabuaf-english-text-evaluation-streamlit-app-qnuz0s.streamlit.app/)">enlace</a>.</p>
