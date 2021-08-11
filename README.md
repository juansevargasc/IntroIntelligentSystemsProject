# Proyecto Trip Advisor Hotel Reviews

   Introducción a los Sistemas Inteligentes
   Proyecto Final 
   2021 - I


REPORTE

*Investigación de Datos: Reseñas de Tripadvisor*

Juan Sebastián Vargas Castañeda - **Ingeniería de Sistemas y Computación** 

![Copia de logosimbolo_central_2c](https://user-images.githubusercontent.com/30605037/129082498-383d5a2e-54f8-4e21-8b2e-ca1a7df8e9c1.png)


## INTRODUCCIÓN


El presente proyecto es una pequeña investigación sobre los datos proporcionados por Tripadvisor ® , un sitio web de reseñas sobre viajes y foros de viajeros, el cual tiene servicio gratuito y la mayoría de su contenido es proporcionado por usuarios. El propósito de este análisis es relacionar el contenido de las reseñas escritas por los usuarios (párrafos de texto donde ellos expresan su opinión sobre un lugar, servicio o producto) y la calificación numérica que le otorgan debido a su experiencia. Si trasladamos esto a Machine Learning podríamos plantear un modelo matemático adecuado que aprenda sobre el conjunto de datos, relacionando un conjunto de palabras -que pasan por un tratamiento- y una calificación de 0 a 5. Esta es la esencia del proyecto; sin embargo en el trayecto surgen algunos retos como: el tratamiento de palabras, vectorización, hipótesis, selección de un buen modelo, etc. Esto se construirá a lo largo del documento, y se adoptará una metodología que propenda por tener un buen desarrollo. Esta metodología se conoce como CRISP-DM.






























 ## COMPRENSIÓN DEL NEGOCIO

En la actualidad la industria hotelera está atravesando un capítulo particular en nuestra era moderna: la pandemia. Es en realidad uno de los sectores más afectados y a esto hay que sumarle un fenómeno que provenía de hace varios años, las startups tecnológicas que cambiaron el paradigma. Es decir Airbnb y compañía; esto último no obstante no ha sido necesariamente un factor negativo, pues así como surgieron plataformas para cambiar el modelo de negocio, surgieron plataformas para amplificar el modelo tradicional. Las agencias de viaje operan de manera muy diferente y están muy involucradas con plataformas de reserva de viaje como Booking ® o Tripadvisor ®, y las de reserva de vuelos como Kayak ® , entre otras. Y de no ser por la pandemia sería una industria en crecimiento y en momento de éxito de negocios, dado el comportamiento presentado en 2019 [1].

![reporte - proyecto final](https://user-images.githubusercontent.com/30605037/129083215-7d3979dd-06bd-45c1-8547-947531e269c1.png)


Infografía. Datos del año 2018 [4]

Las plataformas de reserva de hoteles por lo general cuentan con un sistema de calificación que es un punto de referencia muy fuerte para los nuevos huéspedes. En la actualidad es muy probable que una persona que quiera hacer una excursión o un descanso total de vacaciones se remita a Internet. En esa medida es muy importante para el negocio mantener buenas reseñas y obtener información valiosa de ellas. Según un estudio, mencionado en el periódico The Guardian [2], en la plataforma digital un incremento de una estrella en la calificación de un hotel representa un incremento de ingresos entre un 5% a un 9%. 

Con todo lo expuesto, en este contexto de una situación de pre-pandemia y de pandemia -actual-, se va a proponer un objetivo de minería de datos, es decir un objetivo de investigación de datos, que se alinee con las necesidades de la industria y que ayude a solventarlas. Estas necesidades son las que definen los objetivos comerciales, los cuales constituyen la base de los esfuerzos de la industria.

Objetivos y métricas



Objetivos Comerciales
Incrementar el volumen de huéspedes como medida de reactivación.
Aumentar la calificación en el sistema de reserva de hotel.

Métricas
Incrementar el promedio de calificación en un rango de 0,3 a 0,7. El rango actual es 3,5 (p.ej)
Presentar un incremento de la demanda entre un 8% a 15%.




Objetivos de investigación de datos
Determinar el aspecto del hotel que más valoran los huéspedes.
Determinar el aspecto más influyente por el cual se recibe menos 3 (incluido) como calificación.
Determinar el aspecto más influyente para obtener 4 o 5 como calificación.
Métricas
Generar un modelo que tenga un nivel de ‘Accuracy’ mínimo del 80%.











## COMPRENSIÓN DE LOS DATOS

Es importante en el proceso de investigación la fase de comprensión de datos. Esta es la etapa del proyecto en la que se analiza con qué se va a trabajar; en el contexto de un proyecto es una etapa en la que se recopila la información, se analizan las fuentes de la misma, las características, la relevancia, la cantidad, entre otros. En este informe se detalla cual es la característica del dataset que se investigará:

Recopilación de datos iniciales

Fuente
Kaggle
Enlace público
https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews
Nombre y descripción
Trip Advisor Hotel Reviews. 20k Hotel reviews extracted from Tripadvisor.



Referencia
Alam, M. H., Ryu, W.-J., Lee, S., 2016. Joint multi-grain topic sentiment: modeling semantic aspects for online reviews. Information Sciences 339, 206–223. 
Número de muestras
20491



Columnas

Número
Nombre
Tipo 
0
 Review
str
1
Rating
numpy.int64













Características y exploración de datos

El siguiente informe gráfico detalla la exploración de datos en cuanto a los siguientes parámetros:
Calificación de hoteles - Barra de frecuencias.
Calificación de hoteles - Porcentajes, Pie chart.
Palabras más repetidas en cada calificación.



Figura 1. Barra de frecuencias del Rating.





Figura 2. Pie chart de la distribución de porcentajes del Rating.








Figura 3. Pie chart de la distribución de porcentajes del Rating.


## PREPARACIÓN DE LOS DATOS

En el análisis de los datos de fuente se evidencia que no contienen entradas vacías, entradas nulas o alguna característica que pueda afectar su uso para minería de datos. En esencia este set de datos consiste en una columna de tipo string donde se encuentra la reseña, en la mayoría de entradas es un párrafo completo. Y en una columna de calificación de tipo int64. 

En ese sentido se puede considerar que el  el set datos está en buenas condiciones para hacer minería de datos. Seguido a esto lo se debe pensar es en cómo trasladar las palabras a un lenguaje que la máquina pueda entender.  En particular este es el reto de un proyecto que analiza lenguaje natural; aunque no es muy complicado, hay diversas maneras de aplicarlo y algunas se ajustan al modelo de minería de datos que se haga. Si es un modelo de Machine Learning como regresión lineal o si es un modelo más específico de Deep Learning como una red neuronal.

Proceso de construcción del diccionario

Para el tratamiento de los datos de tipo numérico es necesario generar un diccionario. A cada palabra hay que asignarle un identificador. Una vez se tiene, se tiene una base sobre la cual trasladar una reseña a un vector o lista de números, cada uno correspondiendo a una palabra o unidad sintáctica. Para ello se usó una función de Scikit Learn llamada TfidfVectorizer().

Ejemplo

Primera entrada del set de datos dt[0]. Párrafo, una lista de objetos tipo string.

['nice hotel expensive parking got good deal stay hotel anniversary arrived late evening took advice previous reviews did valet parking check quick easy little disappointed nonexistent view room room clean nice size bed comfortable woke stiff neck high pillows not soundproof like heard music room night morning loud bangs doors opening closing hear people talking hallway maybe just noisy neighbors aveda bath products nice did not goldfish stay nice touch taken advantage staying longer location great walking distance shopping overall nice experience having pay 40 parking night  ']

Diccionario generado con la función CountVectorizer(). Para ver completamente veáse el Notebook (ipnyb) del proyecto.

{'nice': 9324,
 'hotel': 6767,
 'expensive': 5030,
 'parking': 10201,
 'got': 6091,
 'good': 6059,
 'deal': 3648,
 'stay': 15444,
 'anniversary': 552,
 'arrived': 779,
 'late': 7852,
 'evening': 4867,
 'took': 16441,
 'advice': 218,
 'previous': 11918,
 'reviews': 13378,
 'valet': 17885,
 'check': 2462,
 'quick': 12613,
 'easy': 4503,

Muestra de la entrada procesada

[0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.13494653 0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.12313344 0.         0.         0…]


Tamaño de la muestra

(20000,)






Muestra de la matriz Tfid Vectorizer.

  (0, 10424)	0.09099795176156944
  (0, 5039)	0.07898460426812665
  (0, 9877)	0.08686502690881456
  (0, 14421)	0.09649080763416126
  (0, 4137)	0.09352881713090914
  (0, 18477)	0.08085640696960425
  (0, 6163)	0.042408901749221954
  (0, 8164)	0.0531377340295593
  (0, 8206)	0.13105562823049124
  (0, 15450)	0.0849839601799433
  (0, 201)	0.13494652624539
  (0, 15963)	0.11192034040115667
  (0, 16486)	0.11813650070642641
  (0, 6048)	0.19877638715080057
  (0, 12150)	0.15161058826590362
  (0, 1209)	0.11280940652640904
  (0, 979)	0.18053016544671838
  (0, 9274)	0.1603801939157931
  (0, 9409)	0.10762766910005123
  (0, 8622)	0.11124902483984275
  (0, 6333)	0.1464481124444372
  (0, 15976)	0.1363827224105086
  (0, 10582)	0.06584532306202767
  (0, 6472)	0.10518635204063914
  (0, 2716)	0.1622839437426362

Nótese que en la muestra de la entrada procesada se encuentran valores de 0 a 1, mientras que en el diccionario de palabras todas las llaves son mayores a 1, casi todos los valores en miles. El procesamiento de la función de Scikit-Learn transforma las llaves del diccionario en llaves únicas e irrepetibles con un nivel de precisión alto, y esto se puede ver en la matriz que hace la correspondencia, la matriz Tfid. Para ver más detalle véase el notebook y la documentación.

Asimismo la muestra es de tamaño 20.000 dado que la función ha sido limitada a este número de palabras con el fin de acotar el espacio muestral o universo de palabras. Acotar  es una técnica bastante usada en minería de datos, dado que es más fácil clasificar una cantidad limitada de categorías que una cantidad muy dispersa.










## MODELADO

Como bien se expuso en la introducción al machine learning del curso Introducción a los Sistemas Inteligentes es necesario hacer un proceso riguroso de procesamiento, exploración, modelamiento, y evaluación. Las dos primeras etapas se acaban de pasar en los apartados anteriores, ahora seguirá el modelamiento y la evaluación de sus resultados.


Figura 4. Proceso de la minería de datos.


Logistic Regression

La regresión logística es un buen modelo de Machine Learning dado que es adecuada para el tipo de modelo que busca clasificar, ya sea de manera binaria o multinomial. Si lo comparamos con una regresi{on lineal, vemos como esta última no está acotada, mientras que la regresión logística se guía por la siguiente función, la función sigmoide [5]:


Figura 5. Función sigmoide









Red neuronal 

De igual manera se plantea una red neuronal para experimentar con Deep Learning. Se plantean dos capas usando activación relu que en esencia se deshace de valores negativos y una función sigmoide final de activación que funciona de la misma manera que la regresión logística, clasifica de manera multinomial.


Red neuronal mejorada

El último modelo planteado es la misma red neuronal solo que se le aplica una técnica muy usada en el área de redes neuronales. Dropout, son capas intermedias que se aplican en medio de las capas de la red neuronal, para que de manera aleatorio se apaguen. Esto se hace con el fin de disminuir la tendencia de este modelo a hacer overfitting o sobreajuste. A pesar de que su exactitud sea alta con los datos de entrenamiento, si no puede generalizar lo suficiente, con nuevos datos no tendrá mayor exactitud. 

Planteamiento de la red neuronal



Figura 6. Planteamiento de la red neuronal

