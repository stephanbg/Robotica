#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional - 
# autor: Stephan Brommer Gutiérrez
# correo: alu0101493497@ull.edu.es
# fecha: 28/11/2024
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Resolución de la cinemática inversa mediante CCD
#           (Cyclic Coordinate Descent).

import sys
import os
from math import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys as cs
import time
import json

# ****************************************************************************** 
# Declaración de funciones

# Muestra los orígenes de coordenadas para cada articulación
# 'O' es la lista de coordenadas de las articulaciones, 'final' es una bandera para mostrar el estado final
def muestra_origenes(O, final=0):
  print('Origenes de coordenadas:')
  for i in range(len(O)):
    print('(O' + str(i) + ')0\t= ' + str([round(j, 3) for j in O[i]]))
  if final:
    print('E.Final = ' + str([round(j, 3) for j in final]))

# Función para calcular los márgenes y límites de visualización
def calcular_limites_cuadricula(L, obj, margen = 0.05):
  min_x = min(-L, obj[0])
  max_x = max(L, obj[0])
  min_y = min(-L, obj[1])
  max_y = max(L, obj[1])
  rango_x = max_x - min_x
  rango_y = max_y - min_y
  margen_x = rango_x * margen
  margen_y = rango_y * margen
  return min_x - margen_x, max_x + margen_x, min_y - margen_y, max_y + margen_y

# Muestra el robot gráficamente
# 'O' contiene las posiciones de las articulaciones, 'obj' es la posición del objetivo y 'L' es el largo del brazo
def muestra_robot(O, obj, L):
  plt.figure()
  min_limite_x, max_limite_x, min_limite_y, max_limite_y = calcular_limites_cuadricula(L, obj)
  plt.xlim(min_limite_x, max_limite_x)
  plt.ylim(min_limite_y, max_limite_y)
  T = [np.array(o).T.tolist() for o in O]
  for i in range(len(T)):
    plt.plot(T[i][0], T[i][1], '-o', color=cs.hsv_to_rgb(i / float(len(T)), 1, 1))
  plt.plot(obj[0], obj[1], '*')
  plt.pause(0.0001)
  plt.draw() # Actualiza la figura sin cerrarla
  time.sleep(2) # Espera 2 seg
  plt.close() # Cierra la ventana después emergente

# Genera la matriz de transformación homogénea para una articulación
def matriz_T(d, th, a, al):
  return [[cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th)],
          [sin(th),  cos(th)*cos(al), -sin(al)*cos(th), a*sin(th)],
          [0,          sin(al),          cos(al),         d],
          [0,                0,                0,         1]]

# Normaliza el ángulo 'th_actual' para que esté en el rango [-180, 180]
def normalizar_angulo(th_actual):
  media_vuelta = np.pi
  # Realizo esto para que la fórmula general no devuelva -180 (Llegaría al mismo destino pero girando en el sentido inverso)
  if th_actual == media_vuelta: 
    return media_vuelta
  vuelta_completa = 2 * np.pi
  return (th_actual + media_vuelta) % vuelta_completa - media_vuelta

# Asegurarse de que el dato de entrada esté dentro del rango [min, max]
def aplicar_limites(dato, min, max):
  if dato < min:
    dato = min
  elif dato > max:
    dato = max
  return dato

# ****************************************************************************** 
# Cinemática directa

# Realiza el cálculo de la cinemática directa para un robot
# 'th' es el vector de thetas, 'a' es el vector de longitudes
def cin_dir(th, a):
  T = np.identity(4)
  o = [[0, 0]]
  for i in range(len(th)):
    T = np.dot(T, matriz_T(0, th[i], a[i], 0))
    tmp = np.dot(T, [0, 0, 0, 1])
    o.append([tmp[0], tmp[1]])
  return o

# ****************************************************************************** 
# Cinemática inversa revolución

# Calcula el ángulo de rotación necesario para alinear dos vectores en el caso de una articulación de revolución
def cin_inversa_revolucion(punto_articulacion, punto_a_alinear, objetivo, th_actual):
  punto_articulacion = np.array(punto_articulacion)
  punto_a_alinear = np.array(punto_a_alinear)
  objetivo = np.array(objetivo)
  v1 = punto_a_alinear - punto_articulacion
  v2 = objetivo - punto_articulacion
  cosAlfa = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
  cosAlfa = np.clip(cosAlfa, -1.0, 1.0)
  alfa = np.arccos(cosAlfa)
  producto_cruzado = v1[0] * v2[1] - v1[1] * v2[0]
  if producto_cruzado < 0:
    alfa = -alfa
  th_actual += alfa   
  return normalizar_angulo(th_actual)

# ****************************************************************************** 
# Cinemática inversa prismática

# Calcula la suma acumulada de los ángulos 'th' hasta el índice 'indice'
def calcular_omega(th, indice):
  return sum(th[i] for i in range(indice + 1))

# Calcula la distancia entre el objetivo y el punto de alineación, proyectada en la dirección de 'omega'
def calcular_distancia(omega, objetivo, punto_a_alinear):  
  objetivo = np.array(objetivo)
  punto_a_alinear = np.array(punto_a_alinear)
  direccion = np.array([np.cos(omega), np.sin(omega)]) # Vector unitario en la dirección de omega
  resta = objetivo - punto_a_alinear # Diferencia de vectores
  return np.dot(resta, direccion) # Producto escalar entre la diferencia y la dirección

# Calcula la cinemática inversa prismática, es decir, ajusta el valor de 'a' para alcanzar el objetivo
def cin_inversa_prismatica(a, th, indice, objetivo, punto_a_alinear):
  return a[indice] + calcular_distancia(calcular_omega(th, indice), objetivo, punto_a_alinear)

# ****************************************************************************** 
# Buscar y cargar el fichero JSON en todo el sistema

# Busca recursivamente el archivo 'config_cin.json' en el sistema de archivos
def buscar_config_json():
  for ruta, _, ficheros in os.walk('/'):
    if 'config_cin.json' in ficheros:
      return os.path.join(ruta, 'config_cin.json')
  return None

# Carga la configuración desde el archivo JSON
def cargar_config():
  ruta_config = buscar_config_json()
  if ruta_config:
    with open(ruta_config, 'r') as f:
      config = json.load(f)
    return config
  else:
    raise FileNotFoundError("No se encontró el archivo config_cin.json en el sistema.")

# ****************************************************************************** 
# Función main

def main():
  try:
    config = cargar_config()  # Cargar la configuración desde el archivo JSON
    print("Configuración cargada con éxito:")
    # Rellenar variables de las articulaciones
    articulaciones = config['articulaciones']
    for indice, _ in enumerate(articulaciones):
      if articulaciones[indice]['tipo'] == 'revolucion':
        articulaciones[indice]['th'] = radians(articulaciones[indice]['th'])
        articulaciones[indice]['min_th'] = radians(articulaciones[indice]['min_th'])
        articulaciones[indice]['max_th'] = radians(articulaciones[indice]['max_th'])  
    # Aplicando los límites directamente en el proceso de asignación
    th = [
      aplicar_limites(artic['th'], artic['min_th'], artic['max_th']) 
      if artic['tipo'] == 'revolucion' else 0.0 
      for artic in articulaciones
    ]
    a = [
      aplicar_limites(artic['a'], artic['min_a'], artic['max_a']) 
      if artic['tipo'] == 'prismatica' else artic['a']
      for artic in articulaciones
    ]
    L = sum(a)
    EPSILON = 0.01
    # Introducción del punto para la cinemática inversa
    if len(sys.argv) != 3:
      sys.exit("python " + sys.argv[0] + " x y")
    objetivo = [float(i) for i in sys.argv[1:]]  # Objetivo que queremos alcanzar
    O = cin_dir(th, a)
    print("- Posicion inicial:")
    muestra_origenes(O) # Mostrar la posición inicial
    dist = float("inf")
    prev = 0.
    iteracion = 1
    # Ciclo de cinemática inversa
    while dist > EPSILON and abs(prev - dist) > EPSILON / 100.:
      prev = dist
      # Añadir la nueva posición calculada de la cinemática directa
      O = [cin_dir(th, a)]
      # Para cada combinación de articulaciones:
      for i in range(len(th)):
        indice = len(th) - i - 1
        punto_a_alinear = O[i][-1]
        punto_articulacion = O[i][indice]
        # Actualizar las posiciones de las articulaciones
        if articulaciones[indice]['tipo'] == 'revolucion':
          th[indice] = cin_inversa_revolucion(punto_articulacion, punto_a_alinear, objetivo, th[indice])
          th[indice] = aplicar_limites(th[indice], articulaciones[indice]['min_th'], articulaciones[indice]['max_th'])      
        elif articulaciones[indice]['tipo'] == 'prismatica':
          a[indice] = cin_inversa_prismatica(a, th, indice, objetivo, punto_a_alinear)
          a[indice] = aplicar_limites(a[indice], articulaciones[indice]['min_a'], articulaciones[indice]['max_a'])
        O.append(cin_dir(th, a))  # Recalcular la nueva posición con el nuevo ángulo
      dist = np.linalg.norm(np.subtract(objetivo, O[-1][-1])) # Calcular la distancia al objetivo
      print("\n- Iteracion " + str(iteracion) + ':')
      muestra_origenes(O[-1]) # Mostrar la nueva posición
      muestra_robot(O, objetivo, L) # Mostrar el robot graficamente
      print("Distancia al objetivo = " + str(round(dist, 5)))
      iteracion += 1
      O[0] = O[-1]
    if dist <= EPSILON:
      print("\n" + str(iteracion) + " iteraciones para converger.")
    else:
      print("\nNo hay convergencia tras " + str(iteracion) + " iteraciones.")
    print("- Umbral de convergencia epsilon: " + str(EPSILON))
    print("- Distancia al objetivo:          " + str(round(dist, 5)))
    print("- Valores finales de las articulaciones:")
    for i in range(len(th)):
      print("  theta" + str(i + 1) + " = " + str(round(th[i], 3)))
    for i in range(len(th)):
      print("  L" + str(i + 1) + "     = " + str(round(a[i], 3)))  
  except FileNotFoundError as e:
    print(e)
    
# Llamada al main
main()