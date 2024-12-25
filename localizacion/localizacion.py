#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional 
# autor: Stephan Brommer Gutiérrez
# correo: alu0101493497@ull.edu.es
# fecha: 07/01/2025
# Grado en Ingeniería Informática (Cuarto)
# Práctica 5:
#     Simulación de robots móviles holonómicos y no holonómicos.

#localizacion.py

import sys
from math import *
from robot import robot
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ******************************************************************************
# Declaración de funciones

# Distancia entre dos puntos (admite poses)
def distancia(a,b):
  return np.linalg.norm(np.subtract(a[:2],b[:2]))

# Diferencia angular entre una pose y un punto objetivo 'p'
def angulo_rel(pose,p):
  w = atan2(p[1]-pose[1],p[0]-pose[0])-pose[2]
  while w >  pi: w -= 2*pi
  while w < -pi: w += 2*pi
  return w

# Calcular los límites de la región (bordes)
def calcular_bordes(objetivos, tray_ideal, tray_real):
  objT   = np.array(objetivos).T.tolist()
  tray_idealT  = np.array(tray_ideal).T.tolist()
  tray_realT  = np.array(tray_real).T.tolist()
  return [
    min(min(tray_realT[0]), min(objT[0]), min(tray_idealT[0])),  # Mínimo de las coordenadas X
    max(max(tray_realT[0]), max(objT[0]), max(tray_idealT[0])),  # Máximo de las coordenadas X
    min(min(tray_realT[1]), min(objT[1]), min(tray_idealT[1])),  # Mínimo de las coordenadas Y
    max(max(tray_realT[1]), max(objT[1]), max(tray_idealT[1]))   # Máximo de las coordenadas Y
  ]

# Calcular el centro de la gráfica
def calcular_centro(bordes):
  return [(bordes[0]+bordes[1])/2.,(bordes[2]+bordes[3])/2.]

# Calcular el radio de la gráfica
def calcular_radio(bordes):  
  return max(bordes[1]-bordes[0],bordes[3]-bordes[2])*.75

# Mostrar objetivos y trayectoria
def mostrar(objetivos,tray_ideal,tray_real):
  # Fijar los bordes del gráfico
  objT   = np.array(objetivos).T.tolist()
  idealT = np.array(tray_ideal).T.tolist()
  bordes = calcular_bordes(objetivos, tray_ideal, tray_real)
  centro = calcular_centro(bordes)
  radio  = calcular_radio(bordes)
  plt.xlim(centro[0]-radio,centro[0]+radio)
  plt.ylim(centro[1]-radio,centro[1]+radio)
  # Representar objetivos y trayectoria
  plt.plot(idealT[0],idealT[1],'-g')
  plt.plot(tray_real[0][0],tray_real[0][1],'or')
  r = radio * .1
  for p in tray_real:
    plt.plot([p[0],p[0]+r*cos(p[2])],[p[1],p[1]+r*sin(p[2])],'-r')
  objT = np.array(objetivos).T.tolist()
  plt.plot(objT[0],objT[1],'-.o')
  plt.show()
  input()
  plt.clf()

# Optimiza una función utilizando un enfoque de búsqueda local en un vecindario
def optimizar(funcion, rango, precision, dimensiones=1):
  rangos = np.array(rango)
  paso = (rangos[:, 1] - rangos[:, 0]) / 2 # Paso inicial: el tamaño de cada paso es la mitad del rango en cada dimensión
  centro = (rangos[:, 0] + rangos[:, 1]) / 2 # Calculamos el centro del espacio de búsqueda (punto medio del rango)
  # Continuamos refinando la búsqueda hasta que el paso sea menor o igual a la precisión
  while np.any(paso > precision):
    candidatos = []  # Lista para almacenar puntos candidatos
    valores = []  # Lista para almacenar los valores de la función en los puntos candidatos
    # Exploramos los puntos vecinos en un vecindario de 3 puntos por dimensión (optimización local)
    for desplazamiento in np.ndindex(*(3,) * dimensiones):  # Generamos todas las combinaciones posibles de desplazamientos en 3 dimensiones
      punto = centro + (np.array(desplazamiento) - 1) * paso  # Calculamos el punto desplazado respecto al centro
      # Aseguramos que el punto esté dentro del rango especificado
      if np.all(punto >= rangos[:, 0]) and np.all(punto <= rangos[:, 1]):
        candidatos.append(punto)  # Añadimos el punto a los candidatos
        valores.append(funcion(punto))  # Añadimos el valor de la función a la lista de valores
    # Encontramos el punto con el mejor (mínimo) valor de la función
    mejores_indices = np.argmin(valores)
    mejor_punto = candidatos[mejores_indices]
    mejor_valor = valores[mejores_indices]
    # Refinamos la región alrededor del mejor punto encontrado
    centro = mejor_punto  # Actualizamos el centro con el mejor punto
    paso /= 2  # Reducimos el tamaño del paso para hacer una búsqueda más precisa
  # Retornamos el mejor valor encontrado y el punto correspondiente
  return mejor_valor, mejor_punto

# Localización del robot mediante optimización jerárquica basada en mediciones de balizas
def localizacion(balizas, real, ideal, centro, radio, mostrar=0, niveles=4):
  mejor_posicion = ideal.pose()
  imagen = []
  for nivel in range(niveles):
    def evaluar(posicion):
      x, y, theta = posicion
      ideal.set(x, y, theta)
      return ideal.measurement_prob(real.sense(balizas), balizas)    
    # Precisión para cada nivel
    precision_espacial = radio / (2 ** (nivel + 1))
    precision_angular = np.pi / (8 * (2 ** nivel))   
    # Aplicar optimización combinada
    rango_espacial = [(centro[0] - radio, centro[0] + radio), (centro[1] - radio, centro[1] + radio), (-np.pi, np.pi)] # Rango (x,y,orientación)
    _, mejor_punto = optimizar(evaluar, rango_espacial, [precision_espacial, precision_espacial, precision_angular], dimensiones=3)
    if mostrar:
      resolucion = int(4 * radio)
      imagen = np.zeros((resolucion, resolucion)) 
      # Llenar la imagen
      x_range = np.linspace(rango_espacial[0][0], rango_espacial[0][1], resolucion)
      y_range = np.linspace(rango_espacial[1][0], rango_espacial[1][1], resolucion)
      for j, y in enumerate(y_range):
        for i, x in enumerate(x_range):
          imagen[j, i] = evaluar([x, y, mejor_punto[2]])  # Usar theta del mejor punto
    mejor_posicion = mejor_punto
    centro = mejor_posicion[:2]
    radio /= 2
    if mostrar:
      print(f"Nivel {nivel + 1}: Mejor posición [{mejor_posicion[0]}, {mejor_posicion[1]}, {mejor_posicion[2]}], Error {evaluar(mejor_punto)}")
      MIN_X, MAX_X, MIN_Y, MAX_Y = rango_espacial[0][0], rango_espacial[0][1], rango_espacial[1][0], rango_espacial[1][1] 
      # Configuración de la visualización
      plt.xlim(MIN_X, MAX_X)
      plt.ylim(MIN_Y, MAX_Y)
      imagen = np.flipud(imagen) # Esto invierte las filas de la imagen, corrigiendo la inversión
      # Dibujar la imagen con los valores evaluados durante la optimización
      plt.imshow(imagen, extent=[MIN_X, MAX_X, MIN_Y, MAX_Y])
      # Dibujar el recuadro de la zona de búsqueda (ajustado al radio actual)
      rect = plt.Rectangle(
        (centro[0] - radio, centro[1] - radio), 2 * radio, 2 * radio,
        linewidth=2, edgecolor='red', facecolor='none', label=f'Nivel {nivel + 1}'
      )
      plt.gca().add_patch(rect)
      # Dibujar las balizas y las posiciones (ideal y real)
      balT = np.array(balizas).T.tolist()
      plt.plot(balT[0], balT[1], 'or', ms=10)  # Dibujar las balizas en color rojo
      plt.plot(ideal.x, ideal.y, 'D', c='#ff00ff', ms=10, mew=2, label='Ideal')
      plt.plot(real.x, real.y, 'D', c='#00ff00', ms=10, mew=2, label='Real')
      # Dibujar la orientación con flechas (arrows)
      # Calcular las dimensiones de la imagen
      ancho = MAX_X - MIN_X
      alto = MAX_Y - MIN_Y
      # Escalar las flechas de orientación proporcionalmente al tamaño de la imagen
      escala_flechas = min(ancho, alto) * 0.05      
      # Dibujar la orientación con flechas (arrows)
      dx_ideal = cos(ideal.orientation) * escala_flechas
      dy_ideal = sin(ideal.orientation) * escala_flechas
      plt.arrow(ideal.x, ideal.y, dx_ideal, dy_ideal, head_width=escala_flechas * 0.2, head_length=escala_flechas * 0.2, fc='#ff00ff', ec='#ff00ff')
      dx_real = cos(real.orientation) * escala_flechas
      dy_real = sin(real.orientation) * escala_flechas
      plt.arrow(real.x, real.y, dx_real, dy_real, head_width=escala_flechas * 0.2, head_length=escala_flechas * 0.2, fc='#00ff00', ec='#00ff00')
      plt.legend()
      plt.draw()
      plt.pause(2) # Pausar la ejecución durante 2 segundos
      plt.close() # Cerrar la ventana de la figura después de la pausa
  # Configurar el robot ideal en la mejor posición encontrada
  ideal.set(*mejor_posicion)
  if mostrar:
    print("")

# ******************************************************************************

# Definición del robot:
P_INICIAL = [0.,4.,0.] # Pose inicial (posición y orientacion)
V_LINEAL  = .7         # Velocidad lineal    (m/s)
V_ANGULAR = 140.       # Velocidad angular   (º/s)
FPS       = 10.        # Resolución temporal (fps)
HOLONOMICO = 1
GIROPARADO = 0
LONGITUD   = .2

# Definición de trayectorias:
trayectorias = [
    [[1,3]],
    [[0,2],[4,2]],
    [[2,4],[4,0],[0,0]],
    [[2,4],[2,0],[0,2],[4,2]],
    [[2+2*sin(.8*pi*i),2+2*cos(.8*pi*i)] for i in range(5)]
]

# Definición de los puntos objetivo:
if len(sys.argv)<2 or int(sys.argv[1])<0 or int(sys.argv[1])>=len(trayectorias):
  sys.exit(sys.argv[0]+" <indice entre 0 y "+str(len(trayectorias)-1)+">")
objetivos = trayectorias[int(sys.argv[1])]

# Definición de constantes:
EPSILON = .1                # Umbral de distancia
THRESHOLD = .01             # Umbral de separación entre ambos robots
V = V_LINEAL/FPS            # Metros por fotograma
W = V_ANGULAR*pi/(180*FPS)  # Radianes por fotograma

ideal = robot()
ideal.set_noise(0,0,.1)   # Ruido lineal / radial / de sensado
ideal.set(*P_INICIAL)     # operador 'splat'

real = robot()
real.set_noise(.01,.01,.1)  # Ruido lineal / radial / de sensado
real.set(*P_INICIAL)

tray_ideal = [ideal.pose()]  # Trayectoria percibida
tray_real = [real.pose()]    # Trayectoria seguida

tiempo  = 0.
espacio = 0.
random.seed(int(datetime.now().timestamp()))

BORDES = calcular_bordes(objetivos, tray_ideal, tray_real)
CENTRO_GLOBAL = calcular_centro(BORDES)
RADIO_GLOBAL = ceil(calcular_radio(BORDES))
DISTANCIA_ROBOT_REAL_BALIZAS = real.sense(objetivos)

# Localizar Inicialemente al robot
localizacion(objetivos,real,ideal,centro=CENTRO_GLOBAL,radio=RADIO_GLOBAL,mostrar=1) # Buscar en toda la región
for punto in objetivos:
  while distancia(tray_ideal[-1],punto) > EPSILON and len(tray_ideal) <= 1000:
    pose = ideal.pose()
    w = angulo_rel(pose,punto)
    if w > W:  w =  W
    if w < -W: w = -W
    v = distancia(pose,punto)
    if (v > V): v = V
    if (v < 0): v = 0
    if HOLONOMICO:
      if GIROPARADO and abs(w) > .01:
        v = 0
      ideal.move(w,v)
      real.move(w,v)
    else:
      ideal.move_triciclo(w,v,LONGITUD)
      real.move_triciclo(w,v,LONGITUD)
    tray_ideal.append(ideal.pose())
    tray_real.append(real.pose())
    # Si las medidas de real e ideal son similares, realizamos localización
    diff_ideal_real = ideal.measurement_prob(DISTANCIA_ROBOT_REAL_BALIZAS, objetivos)
    if diff_ideal_real >= THRESHOLD:
      # Buscar en una región centrada en la posisicón del robot ideal (con radio = 2 * measurement_prob)
      localizacion(objetivos,real,ideal,centro=[ideal.x, ideal.y],radio=ceil(2 * diff_ideal_real))
    espacio += v
    tiempo  += 1

if len(tray_ideal) > 1000:
  print ("<!> Trayectoria muy larga - puede que no se haya alcanzado la posicion final.")
print ("Recorrido: "+str(round(espacio,3))+"m / "+str(tiempo/FPS)+"s")
print ("Distancia real al objetivo: "+\
  str(round(distancia(tray_real[-1],objetivos[-1]),3))+"m")
mostrar(objetivos,tray_ideal,tray_real)  # Representación gráfica