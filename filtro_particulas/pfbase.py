#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional -
# autor: Stephan Brommer Gutiérrez
# correo: alu0101493497@ull.edu.es
# fecha: 07/01/2025
# Grado en Ingeniería Informática (Cuarto)
# Práctica opcional: Filtros de particulas

from math import *
from robot import *
import random
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import sys
import select
from datetime import datetime
# ******************************************************************************
# Declaración de funciones

# Función que calcula la distancia entre dos puntos a y b
# a y b son listas o tuplas que contienen las coordenadas (x, y) de los puntos.
# Devuelve la distancia euclidiana entre los dos puntos.
def distancia(a, b):
  return np.linalg.norm(np.subtract(a[:2], b[:2]))

# Función que calcula el ángulo relativo entre la pose del robot y un punto objetivo
# La pose es una lista [x, y, theta], donde theta es la orientación.
# p es una lista o tupla que contiene las coordenadas (x, y) del punto objetivo.
# Devuelve el ángulo relativo entre la orientación del robot y la dirección hacia el objetivo.
def angulo_rel(pose, p):
  w = atan2(p[1] - pose[1], p[0] - pose[0]) - pose[2]
  while w > pi: w -= 2 * pi
  while w < -pi: w += 2 * pi
  return w

# Función que dibuja una secuencia de puntos en una gráfica usando Matplotlib
# secuencia es una lista de puntos [x, y] a trazar.
# args son los parámetros de formato para la gráfica (por ejemplo, tipo de línea o color).
def pinta(secuencia, args):
  t = np.array(secuencia).T.tolist()
  plt.plot(t[0], t[1], args)

# Función para mostrar en una gráfica el estado de las partículas del filtro,
# la trayectoria estimada, la trayectoria real y el mejor estimado
# objetivos es una lista de puntos (x, y) que representan las posiciones de los objetivos.
# trayectoria es la trayectoria estimada por el filtro.
# trayectreal es la trayectoria real del robot.
# filtro es el conjunto de partículas que representan el filtro de partículas.
# mejor_pose es la mejor estimación de la posición del robot.
def mostrar(objetivos, trayectoria, trayectreal, filtro, mejor_pose):
  plt.ion()
  plt.clf()
  plt.axis('equal')
  objT = np.array(objetivos).T.tolist()
  bordes = [min(objT[0]), max(objT[0]), min(objT[1]), max(objT[1])]
  centro = [(bordes[0] + bordes[1]) / 2., (bordes[2] + bordes[3]) / 2.]
  radio = max(bordes[1] - bordes[0], bordes[3] - bordes[2])
  plt.xlim(centro[0] - radio, centro[0] + radio)
  plt.ylim(centro[1] - radio, centro[1] + radio)
  for p in filtro:
    dx = cos(p.orientation) * .05
    dy = sin(p.orientation) * .05
    plt.arrow(p.x, p.y, dx, dy, head_width=.05, head_length=.05, color='k')
  pinta(trayectoria, '--g')
  pinta(trayectreal, '-r')
  pinta(objetivos, '-.ob')
  dx = cos(mejor_pose[2]) * .05
  dy = sin(mejor_pose[2]) * .05
  plt.arrow(mejor_pose[0], mejor_pose[1], dx, dy, head_width=.075, head_length=.075, color='m')
  plt.show()
  plt.pause(0.1)

# Función que mueve todas las partículas en el filtro de partículas
# según las velocidades lineales y angulares especificadas.
# filtro es la lista de partículas.
# w es la velocidad angular (giro).
# v es la velocidad lineal (traslación).
# HOLONOMICO es un booleano que indica si el movimiento es holonómico (para robots con movimiento libre como los robots móviles).
# LONGITUD es la longitud del robot, usada solo si el movimiento es no-holonómico (para robots tipo triciclo).
def mover_particulas(filtro, w, v, HOLONOMICO, LONGITUD):
  for particula in filtro:
    if HOLONOMICO:
      particula.move(w , v)
    else:
      particula.move_triciclo(w, v, LONGITUD)

# Función que recalcula los pesos de todas las partículas del filtro de partículas
# según la probabilidad de las medidas observadas, comparadas con las balizas reales.
# filtro es la lista de partículas.
# distancias_reales es la lista de las distancias medidas a las balizas.
# objetivos es la lista de las posiciones de las balizas.
def recalcular_pesos(filtro, distancias_reales, objetivos):
  for particula in filtro:
    particula.weight = particula.measurement_prob(distancias_reales, objetivos)
  suma_pesos = sum(p.weight for p in filtro)
  for particula in filtro:
    particula.weight /= suma_pesos

# Función que genera el filtro de partículas inicializado con un número dado de partículas
# distribuidas aleatoriamente dentro de un área dada.
# num_particulas es el número de partículas en el filtro.
# balizas es la lista de posiciones de las balizas en el entorno.
# real es el robot real que proporciona las medidas.
# centro es el centro del área donde se generan las partículas.
# lado es el tamaño del área cuadrada donde se distribuyen las partículas.
def genera_filtro(num_particulas, balizas, real, centro=[2, 2], lado=3):
  particulas = []
  for _ in range(num_particulas):
    x = random.uniform(centro[0] - lado / 2, centro[0] + lado / 2)
    y = random.uniform(centro[1] - lado / 2, centro[1] + lado / 2)
    orientacion = random.uniform(-pi, pi)
    ideal = robot()
    ideal.set(x, y, orientacion)
    ideal.set_noise(0.05,0.05,.5)
    peso = ideal.measurement_prob(real.sense(balizas), balizas)
    ideal.weight = peso
    particulas.append(ideal)
  return particulas

# Función que calcula la dispersión espacial del filtro de partículas.
# La dispersión se refiere al rango (mínimo y máximo) de las posiciones (x, y) de las partículas en el filtro.
# Devuelve una lista con los valores [mínimo_x, máximo_x, mínimo_y, máximo_y], representando el rango
# en el cual las partículas están distribuidas en el espacio 2D.
def dispersion(filtro):
  return [
    min(particula.x for particula in filtro),
    max(particula.x for particula in filtro),
    min(particula.y for particula in filtro),
    max(particula.y for particula in filtro)
  ]

# Función que calcula el peso medio normalizado del filtro de partículas.
# El peso medio es la media ponderada de las partículas, basada en sus probabilidades de medición.
# Normaliza el peso dividiendo cada peso por la suma total de los pesos para asegurar que la suma
# de los pesos sea 1. Esto ayuda a mantener la coherencia en el cálculo de la probabilidad.
# Devuelve el peso medio normalizado de todas las partículas en el filtro.
def peso_medio(filtro):
  return mean(particula.weight for particula in filtro)

# Ajusta el número de partículas en un filtro de partículas basado en
# la dispersión y el peso promedio de las partículas.
def ajustar_numero_particulas(dispersion_total, peso_promedio, n_partic):
  # Evaluar la dispersión para determinar la incertidumbre
  if dispersion_total > 0.4 or peso_promedio <= 0.01:  # Dispersión alta, y peso bajo; gran incertidumbre
    n_partic = min(100, n_partic + 10)  # Aumentar número de partículas
  elif dispersion_total <= 0.4 or peso_promedio > 0.01:  # Dispersión baja y peso alto; poca incertidumbre 
    n_partic = max(20, n_partic - 10)  # Reducir número de partículas
  return n_partic

# Parámetros iniciales
P_INICIAL_REAL = [0, 0, 0]
P_INICIAL_IDEAL = [0.5, -0.25, 0]
V_LINEAL = .7
V_ANGULAR = 140.
FPS = 10.
HOLONOMICO = 0
GIROPARADO = 0
LONGITUD = .1
n_partic = 50
N_INICIAL = 2000
EPSILON = .1
V = V_LINEAL / FPS
W = V_ANGULAR * pi / (180 * FPS)

# Definición de trayectorias y objetivos
trayectorias = [
  [[0,2],[4,2]],
  [[2,4],[4,0],[0,0]],
  [[2,4],[2,0],[0,2],[4,2]],
  [[2+2*sin(.4*pi*i),2+2*cos(.4*pi*i)] for i in range(5)],
  [[2+2*sin(.8*pi*i),2+2*cos(.8*pi*i)] for i in range(5)],
  [[2+2*sin(1.2*pi*i),2+2*cos(1.2*pi*i)] for i in range(5)],
  [[2*(i+1),4*(1+cos(pi*i))] for i in range(6)],
  [[2+.2*(22-i)*sin(.1*pi*i),2+.2*(22-i)*cos(.1*pi*i)] for i in range(20)],
  [[2+(22-i)/5*sin(.1*pi*i),2+(22-i)/5*cos(.1*pi*i)] for i in range(20)]
]
if len(sys.argv) < 2 or int(sys.argv[1]) < 0 or int(sys.argv[1]) >= len(trayectorias):
  sys.exit(sys.argv[0] + " <indice entre 0 y " + str(len(trayectorias) - 1) + ">")
objetivos = trayectorias[int(sys.argv[1])]

real = robot()
real.set_noise(.01, .01, .01)
real.set(*P_INICIAL_REAL)
trayect_ideal = [P_INICIAL_IDEAL]
trayect_real = [real.pose()]

random.seed(int(datetime.now().timestamp()))
filtro = genera_filtro(N_INICIAL, real.sense(objetivos), real, centro=[0, 0], lado=2)
tiempo = 0.
espacio = 0.

# Simulación
mostrar(objetivos, trayect_ideal, trayect_real, filtro, P_INICIAL_IDEAL) # Mostrar resultados
for punto in objetivos:
  while distancia(trayect_ideal[-1], punto) > EPSILON and len(trayect_ideal) <= 1000:
    mejor_pose = hipotesis(filtro) # Obtener mejor pose
    # Calcular movimientos lineales y angulares
    w = angulo_rel(mejor_pose, punto)
    if w > W: w = W
    if w < -W: w = -W
    v = distancia(mejor_pose, punto)
    if v > V: v = V
    if v < 0: v = 0
    # Mover robot real
    if HOLONOMICO:
      if GIROPARADO and abs(w) > .01: v = 0
      real.move(w, v)
    else:
      real.move_triciclo(w, v, LONGITUD)
    # Actualizar trayectorias
    trayect_real.append(real.pose())
    trayect_ideal.append(mejor_pose)    
    distancias_reales = real.sense(objetivos) # Obtener distancias reales del robot
    mostrar(objetivos, trayect_ideal, trayect_real, filtro, mejor_pose) # Mostrar resultados
    # Evaluar dispersión y peso medio
    disp = dispersion(filtro)
    peso_promedio = peso_medio(filtro)
    # Ajustar dinámicamente el número de partículas basado en dispersión y peso medio
    rango_x = disp[1] - disp[0]
    rango_y = disp[3] - disp[2]
    dispersion_total = max(rango_x, rango_y)
    n_partic = ajustar_numero_particulas(dispersion_total, peso_promedio, n_partic)
    filtro = resample(filtro, n_partic) # Resamplear partículas
    mover_particulas(filtro, w, v, HOLONOMICO, LONGITUD) # Mover las partículas y actualizar sus posiciones
    recalcular_pesos(filtro, distancias_reales, objetivos) # Recalcular los pesos de las partículas      
    espacio += v
    tiempo += 1

if len(trayect_ideal) > 1000:
  print ("<< ! >> Puede que no se haya alcanzado la posicion final.")
print ("Recorrido: "+str(round(espacio,3))+"m / "+str(tiempo/FPS)+"s" )
print ("Error medio de la trayectoria: "+str(round(sum(\
    [distancia(trayect_ideal[i],trayect_real[i])\
    for i in range(len(trayect_ideal))])/tiempo,3))+"m" )
input()