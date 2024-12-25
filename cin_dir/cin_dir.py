#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional
# autor: Stephan Brommer Gutiérrez
# correo: alu0101493497@ull.edu.es
# fecha: 07/01/2025
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Resolución de la cinemática directa mediante Denavit-Hartenberg.

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os

# ******************************************************************************
# Declaración de funciones

# Función para procesar las secuencias de movimientos y calcular las transformaciones correspondientes.
# Toma una secuencia y un conjunto de transformaciones, y devuelve el resultado de aplicar cada transformación 
# a los elementos de la secuencia.
def ramal(I,prev=[],base=0):
  # Convierte el robot a una secuencia de puntos para representar
  O = []
  if I:
    if isinstance(I[0][0],list):
      for j in range(len(I[0])):
        O.extend(ramal(I[0][j], prev, base or j < len(I[0])-1))
    else:
      O = [I[0]]
      O.extend(ramal(I[1:],I[0],base))
      if base:
        O.append(prev)
  return O

# Función que visualiza el robot en 3D utilizando los orígenes y puntos generados por las transformaciones
# Calcula un gráfico 3D con los puntos representando el robot en las diferentes posiciones de sus articulaciones.
def muestra_robot(O,ef=[]):
  # Pinta en 3D
  OR = ramal(O)
  OT = np.array(OR).T
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  # Bounding box cúbico para simular el ratio de aspecto correcto
  max_range = np.array([OT[0].max()-OT[0].min()
                       ,OT[1].max()-OT[1].min()
                       ,OT[2].max()-OT[2].min()
                       ]).max()
  Xb = (0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten()
     + 0.5*(OT[0].max()+OT[0].min()))
  Yb = (0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten()
     + 0.5*(OT[1].max()+OT[1].min()))
  Zb = (0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten()
     + 0.5*(OT[2].max()+OT[2].min()))
  for xb, yb, zb in zip(Xb, Yb, Zb):
     ax.plot([xb], [yb], [zb], 'w')
  ax.plot3D(OT[0],OT[1],OT[2],marker='s')
  ax.plot3D([0],[0],[0],marker='o',color='k',ms=10)
  if not ef:
    ef = OR[-1]
  ax.plot3D([ef[0]],[ef[1]],[ef[2]],marker='s',color='r')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.show()
  return

# Función auxiliar que da formato a los orígenes de coordenadas del robot y los muestra por pantalla.
# Itera recursivamente sobre la estructura del robot, mostrando los valores de los orígenes de cada articulación.
def arbol_origenes(O,base=0,sufijo=''):
  # Da formato a los origenes de coordenadas para mostrarlos por pantalla
  if isinstance(O[0],list):
    for i in range(len(O)):
      if isinstance(O[i][0],list):
        for j in range(len(O[i])):
          arbol_origenes(O[i][j],i+base,sufijo+str(j+1))
      else:
        print('(O'+str(i+base)+sufijo+')0\t= '+str([round(j,3) for j in O[i]]))
  else:
    print('(O'+str(base)+sufijo+')0\t= '+str([round(j,3) for j in O]))

# Función que muestra los orígenes de coordenadas para cada articulación, llamando a `arbol_origenes` para 
# presentar la estructura del robot de manera legible. Si se recibe un `final`, lo muestra también.
def muestra_origenes(O,final=0):
  # Muestra los orígenes de coordenadas para cada articulación
  print('Orígenes de coordenadas:')
  arbol_origenes(O)
  if final:
    print('E.Final = '+str([round(j,3) for j in final]))

# Función que calcula la matriz de transformación homogénea utilizando los parámetros Denavit-Hartenberg
# Esta función retorna una matriz 4x4 de la transformación de un eslabón basado en sus parámetros DH.
def matriz_T(d,theta,a,alpha):
  # Calcula la matriz T (ángulos de entrada en grados)
  th=theta*pi/180;
  al=alpha*pi/180;
  return [[cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th)]
         ,[sin(th),  cos(th)*cos(al), -sin(al)*cos(th), a*sin(th)]
         ,[      0,          sin(al),          cos(al),         d]
         ,[      0,                0,                0,         1]
         ]
# ******************************************************************************

# Función para buscar un archivo recursivamente desde el directorio raíz
def buscar_archivo_recursivamente(nombre_archivo, ruta_inicio="/"):
  for ruta, _, ficheros in os.walk(ruta_inicio):
    if nombre_archivo in ficheros:
      return os.path.join(ruta, nombre_archivo)
  return None  # Si no se encuentra el archivo, retornamos None

# Función para leer los parámetros desde un archivo JSON
def leer_datos_desde_json(nombre_archivo):
  # Buscar el archivo recursivamente desde la raíz
  archivo_encontrado = buscar_archivo_recursivamente(nombre_archivo)
  if archivo_encontrado is None:
    raise FileNotFoundError(f"No se encontró el archivo {nombre_archivo} en la raíz o en los subdirectorios.")
  try:
    # Leer el archivo JSON encontrado
    with open(archivo_encontrado, 'r') as f:
      data = json.load(f)
    # Obtener las listas d, th, a, al, orden_T del JSON
    d = data['d']
    th = data['th']
    a = data['a']
    al = data['al']
    orden_T = data['orden_T']
    EF = data.get('EF', None)
    # Comprobar si todas las listas tienen la misma longitud
    if not (len(d) == len(th) == len(a) == len(al)):
      raise ValueError("Las listas d, th, a y al deben tener la misma longitud.")
    # Validar el orden_T (debe ser una lista de listas)
    if not isinstance(orden_T, list) or not all(isinstance(sublist, list) for sublist in orden_T):
      raise ValueError("El campo 'orden_T' debe ser una lista de listas.")    
    # Validar el efector final (EF)
    if EF is not None:
      # Aplanar la estructura de orden_T para obtener todos los puntos válidos
      def flatten(lst):
        for item in lst:
          if isinstance(item, list):
            yield from flatten(item)
          else:
            yield item
      puntos_validos = set(flatten(orden_T))
      if EF not in puntos_validos:
        raise ValueError("El punto medio (EF) debe coincidir con los puntos definidos en 'orden_T'.")    
    # El número de articulaciones es el tamaño de cualquiera de las listas
    num_articulaciones = len(d)
    return d, th, a, al, orden_T, EF, num_articulaciones
  except json.JSONDecodeError:
    raise ValueError("El archivo JSON no está correctamente formateado.")
  except Exception as e:
    raise Exception(f"Error al leer los datos: {e}")

# Función que genera una lista de orígenes para las articulaciones del robot, con un origen para cada articulación.
# El número de articulaciones es determinado por el parámetro `num_articulaciones`.
def obtener_origenes_articulaciones(num_articulaciones):
  origenes = [] 
  for _ in range(num_articulaciones):
    origenes.append([0,0,0,1])
  return origenes

# Función que calcula las matrices de transformación de cada articulación y realiza las multiplicaciones de acuerdo 
# al orden de las secuencias especificadas en `orden_T`. Al final, retorna el resultado de aplicar todas las 
# transformaciones.
def calculo_matrices_T(d, th, a, al, orden_T):
  T = {} # Diccionario para almacenar las matrices de transformación
  T['origen-0'] = matriz_T(d[0], th[0], a[0], al[0])
  # Calcular las matrices de transformación individuales y almacenarlas
  for i in range(1, len(d)):
    T[f'{i}'] = matriz_T(d[i], th[i], a[i], al[i])
  # Realizar las multiplicaciones según el orden especificado en orden_T
  resultado = {}  # Variable para almacenar el resultado final de las multiplicaciones
  resultado['origen-0'] = T['origen-0']
  for secuencia in orden_T:
    for i in range(1, len(secuencia)):
      # bifurcación
      if isinstance(secuencia[i], list): 
        resultado[f'origen-{secuencia[i][0]}'] = np.dot(resultado[f'origen-{secuencia[0]}'], T[f'{secuencia[i][0]}'])
        for j in range(1, len(secuencia[i])):
          resultado[f'origen-{secuencia[i][j]}'] = np.dot(resultado[f'origen-{secuencia[i][j-1]}'], T[f'{secuencia[i][j]}'])
      else:
        resultado[f'origen-{secuencia[i]}'] = np.dot(resultado[f'origen-{secuencia[i-1]}'], T[f'{secuencia[i]}'])
  return resultado

# Función que aplica las transformaciones previamente calculadas a los orígenes de las articulaciones y
# genera los resultados de cada multiplicación de las matrices de transformación.
def aplicar_transformaciones(T, origenes):
  # Lista donde almacenaremos los resultados de las multiplicaciones
  resultados = []
  for i in range(0, len(origenes) - 1):
    resultados.append(np.dot(T[f'origen-{i}'], origenes[i + 1]).tolist())
  return resultados

# Función que procesa una secuencia de transformaciones y las aplica a la lista de puntos especificada.
# Esta función toma una secuencia y un conjunto de transformaciones y devuelve el resultado de aplicar cada 
# transformación a los elementos de la secuencia.
def procesar_secuencia(secuencia, transf):
  secuencia_resultado = []
  for i in secuencia:
    secuencia_resultado.append(transf[i])
  return secuencia_resultado

# Función que construye los parámetros para el cálculo de la cinemática directa, generando la lista de
# transformaciones en función de las secuencias de movimientos y de las bifurcaciones.
# Acomoda el resultado de las bifurcaciones al final de cada secuencia.
def construir_parametros(orden_T, origen, transf, EF):
  parametros = [origen]  # Añadimos el origen al principio
  inicio = 0
  for secuencia in orden_T:
    bifurcacion = []  # Acumulamos el resultado de las bifurcaciones
    secuencia = secuencia[inicio:]
    for elemento in secuencia:
      if isinstance(elemento, list):
        if not (elemento[0] == EF and len(elemento) == 1):
          bifurcacion.append(procesar_secuencia(elemento, transf))
      elif elemento != EF:
          parametros.append(transf[elemento])
    if bifurcacion: # Solo empaquetamos las bifurcaciones al final de la secuencia
      parametros.append(bifurcacion)
    inicio = 1
  return parametros

# Función principal
def main():
  try:
    # Verificar que el número de parámetros sea el correcto
    if len(sys.argv) != 2:
      raise Exception("Error en el número de parámetros, tienen que ser 2")
    nombre_archivo = sys.argv[1] # El nombre del archivo es el primer argumento
    d, th, a, al, orden_T, EF, num_articulaciones = leer_datos_desde_json(nombre_archivo) # Parámetros de JSON
    origenes = obtener_origenes_articulaciones(num_articulaciones + 1) # Orígenes para cada articulación más raíz
    T = calculo_matrices_T(d, th, a, al, orden_T)
    transf = aplicar_transformaciones(T, origenes)
    parametros = construir_parametros(orden_T, origenes[0], transf, EF)

    # Mostrar resultado de la cinemática directa
    if EF is not None:
      muestra_origenes(parametros, transf[EF])
      muestra_robot(parametros, transf[EF])
    else:
      muestra_origenes(parametros)
      muestra_robot(parametros)  
    input()
  except Exception as e:
    # Captura el error y muestra solo un mensaje en el main
    print("Ocurrió un error: " + str(e))

main()
# PARA QUE QUEDE BIEN EJECUTAR CON
# python3 Descargas/cin_dir.py parametrosDH.json
