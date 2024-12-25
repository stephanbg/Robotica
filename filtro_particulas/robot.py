#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional 
# Grado en Ingeniería Informática (Cuarto)
# Clase robot

from math import *
import random
import numpy as np
import copy

# Clase que representa un robot, incluyendo su posición, orientación,
# ruido en sus movimientos, y funciones para simular movimientos y sensado.
class robot:
  # Inicializa el robot con posición (x, y), orientación y parámetros de ruido.
  # También inicializa los pesos y el tamaño del robot. 
  def __init__(self):
    self.x             = 0.
    self.y             = 0.
    self.orientation   = 0.
    self.forward_noise = 0.
    self.turn_noise    = 0.
    self.sense_noise   = 0.
    self.weight        = 1.
    self.old_weight    = 1.
    self.size          = 1.

  # Constructor de copia
  def copy(self):
    return copy.deepcopy(self)
  
  # Modificar la pose
  def set(self, new_x, new_y, new_orientation):
    self.x = float(new_x)
    self.y = float(new_y)
    self.orientation = float(new_orientation)
    while self.orientation >  pi: self.orientation -= 2*pi
    while self.orientation < -pi: self.orientation += 2*pi

  # Modificar los parámetros de ruido
  def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
    self.forward_noise = float(new_f_noise);
    self.turn_noise    = float(new_t_noise);
    self.sense_noise   = float(new_s_noise);

  # Obtener pose actual
  def pose(self):
    return [self.x, self.y, self.orientation]

  # Calcular la distancia a una de las balizas
  def sense1(self,landmark,noise):
    return np.linalg.norm(np.subtract([self.x,self.y],landmark)) \
                                        + random.gauss(0.,noise)

  # Calcular las distancias a cada una de las balizas
  def sense(self, landmarks):
    d = [self.sense1(l,self.sense_noise) for l in landmarks]
    d.append(self.orientation + random.gauss(0.,self.sense_noise))
    return d

  # Modificar pose del robot (holonómico)
  def move(self, turn, forward):
    self.orientation += float(turn) + random.gauss(0., self.turn_noise)
    while self.orientation >  pi: self.orientation -= 2*pi
    while self.orientation < -pi: self.orientation += 2*pi
    dist = float(forward) + random.gauss(0., self.forward_noise)
    self.x += cos(self.orientation) * dist
    self.y += sin(self.orientation) * dist

  # Modificar pose del robot (Ackermann)
  def move_triciclo(self, turn, forward, largo):
    dist = float(forward) + random.gauss(0., self.forward_noise)
    self.orientation += dist * tan(float(turn)) / largo\
              + random.gauss(0.0, self.turn_noise)
    while self.orientation >  pi: self.orientation -= 2*pi
    while self.orientation < -pi: self.orientation += 2*pi
    self.x += cos(self.orientation) * dist
    self.y += sin(self.orientation) * dist

  # Calcular la probabilidad de 'x' para una distribución normal
  # de media 'mu' y desviación típica 'sigma'
  def Gaussian(self, mu, sigma, x):
    if sigma:
      return exp(-(((mu-x)/sigma)**2)/2)/(sigma*sqrt(2*pi))
    else:
      return 0

  # Calcular la probabilidad de una medida.
  def measurement_prob(self, measurements, landmarks):
    self.weight = 0.
    for i in range(len(measurements)-1):
      valor = abs(self.sense1(landmarks[i],0) -measurements[i])
      self.weight +=valor
    distance = ((self.weight))/len(measurements)
    diff = self.orientation - measurements[-1]
    while diff >  pi: diff -= 2*pi
    while diff < -pi: diff += 2*pi
    self.weight = (distance + abs(diff)) 
    self.weight = 1.0/self.weight
    return self.weight

  # Representación de la clase robot
  def __repr__(self):
    return '[x=%.6s y=%.6s orient=%.6s]' % \
            (str(self.x), str(self.y), str(self.orientation))

# Pose de la partícula de mayor peso del filtro de partículas
def hipotesis(pf):
  return max(pf,key=lambda r:r.weight).pose()

# Remuestreo
def resample(pf_in, particulas):
  histograma_acumulativo = [0]
  maximo=0
  i = 0
  for robot in pf_in:
    i = i +1
    maximo = maximo + robot.weight
    histograma_acumulativo.append(maximo)
  if not maximo:
    return pf_in[:particulas]
  histograma_acumulativo = [h/maximo for h in histograma_acumulativo[1:]]
  pf_out = []
  for i in range(particulas):
    r = random.random()
    j = 0
    while r > histograma_acumulativo[j]: j += 1
    pf_out.append(pf_in[j].copy())
    pf_out[-1].old_weight = pf_out[-1].weight
    pf_out[-1].weight = 1.
  return pf_out