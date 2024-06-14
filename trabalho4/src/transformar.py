import argparse
import numpy as np
import skimage
import time

from matplotlib import pyplot as plt

from enum import Enum

import sys

class ScalingMethod(Enum):
  NEAREST_NEIGHBOR = 1
  BILINEAR = 2
  BICUBIC = 3
  LAGRANGE = 4

class Image:
  def __init__(self, farray = None) -> None:
    self.image = farray
    self.rows = 0
    self.cols = 0
    
    if farray is not None:
      self.__get_dimensions()
    
  def __dx(self, x_line):
    return x_line - np.floor(x_line)
    
  def __dy(self, y_line):
    return y_line - np.floor(y_line)
  
  def __near_point(self, x_line, y_line):
    x = round(x_line)
    y = round(y_line)
    
    return x, y
    
  def __get_dimensions(self):
    shape = self.image.shape
    self.rows = shape[0]
    self.cols = shape[1]
  
  def read(self, image_path):
    print("Iniciando leitura da imagem...", end=" ")
    
    self.image = skimage.io.imread(fname=image_path)
    
    self.__get_dimensions()
    
    print("Processo Finalizado")
  
  def padding(self, pad):
    return np.pad(self.image, pad, mode="edge")
  
  def __scale_with_nearest_neighbour(self, factor, dimension):
    new_dimension = dimension
    
    if len(self.image.shape) > 2:
      new_dimension = (dimension[0], dimension[1], self.image.shape[2])
    
    scaled_image = np.zeros(new_dimension, dtype="uint8")
    
    for r in range(new_dimension[0]):
      for c in range(new_dimension[1]):
        x_line = r / factor
        y_line = c / factor
        
        try:
          x, y = self.__near_point(x_line, y_line)
          scaled_image[r, c] = self.image[x, y]
        except:
          scaled_image[r, c] = 0
    
    return scaled_image
  
  def __scale_with_bilinear(self, f_img, factor, dimension):
    new_dimension = dimension
    
    if len(self.image.shape) > 2:
      new_dimension = (dimension[0], dimension[1], self.image.shape[2])
    
    scaled_image = np.zeros(new_dimension, dtype="uint8")
    
    for r in range(new_dimension[0]):
      for c in range(new_dimension[1]):
        x_line = (r + 1) / factor
        y_line = (c + 1) / factor
        
        dx = self.__dx(x_line)
        dy = self.__dy(y_line)
        x, y = self.__near_point(x_line, y_line)
        
        try:
          scaled_image[r, c] = (1 - dx) * (1 - dy) * f_img[x, y] + dx * (1 - dy) * f_img[x + 1, y] + (1 - dx) * dy * f_img[x, y + 1] + dx * dy * f_img[x + 1, y + 1]
        except:
          scaled_image[r, c] = 0
        
    return scaled_image
  
  def __P(self, t):
    return t if t > 0 else 0
    
  def __R(self, s):
    return (1 / 6) * (self.__P(s + 2) ** 3 - 4 * self.__P(s + 1) ** 3 + 6 * self.__P(s) ** 3 - 4 * self.__P(s - 1) ** 3)
  
  def __scale_with_bicubic(self, f_img, factor, dimension):
    new_dimension = dimension
    
    if len(self.image.shape) > 2:
      new_dimension = (dimension[0], dimension[1], self.image.shape[2])
    
    scaled_image = np.zeros(new_dimension, dtype="uint8")
    
    for r in range(new_dimension[0]):
      for c in range(new_dimension[1]):
        x_line = (r + 2) / factor
        y_line = (c + 2) / factor
        
        dx = self.__dx(x_line)
        dy = self.__dy(y_line)
        x, y = self.__near_point(x_line, y_line)
        
        try:
          for m in range(-1, 3):
            for n in range(-1, 3):
              scaled_image[r, c] += f_img[x + m, y + n] * self.__R(m - dx) * self.__R(dy - n)
        except:
          scaled_image[r, c] = 0
          
    return scaled_image
  
  def __L(self, f_img, dx, x, y, n):
    return (-dx * (dx - 1) * (dx - 2) * f_img[x - 1, y + n - 2]) / 6 + ((dx + 1) * (dx - 1) * (dx - 2) * f_img[x, y + n - 2]) / 2 + (-dx * (dx + 1) * (dx - 2) * f_img[x + 1, y + n - 2]) / 2 + (dx * (dx + 1) * (dx - 1) * f_img[x + 2, y + n - 2]) / 6
  
  def __scale_with_lagrange(self, f_img, factor, dimension):
    new_dimension = dimension
    
    if len(self.image.shape) > 2:
      new_dimension = (dimension[0], dimension[1], self.image.shape[2])
    
    scaled_image = np.zeros(new_dimension, dtype="uint8")
    
    for r in range(new_dimension[0]):
      for c in range(new_dimension[1]):
        x_line = (r + 2) / factor
        y_line = (c + 2) / factor
        
        dx = self.__dx(x_line)
        dy = self.__dy(y_line)
        x, y = self.__near_point(x_line, y_line)
        
        try:
          scaled_image[r, c] = (-dy * (dy - 1) * (dy - 2) * self.__L(f_img, dx, x, y, 1)) / 6 + ((dy + 1) * (dy - 1) * (dy - 2) * self.__L(f_img, dx, x, y, 2)) / 2 + (-dy * (dy + 1) * (dy - 2) * self.__L(f_img, dx, x, y, 3)) / 2 + (dy * (dy + 1) * (dy - 1) * self.__L(f_img, dx, x, y, 4)) / 6
        except:
          scaled_image[r, c] = 0
          
    return scaled_image
  
  def scale(self, factor, method, dimension):
    if factor == 0: raise ZeroDivisionError("Fator de escala não pode ser 0")
    
    print(f"Realizando escala da imagem por {method.name}...", end=" ")
    
    init_time = time.time()
    
    scaled_image = None
    
    if method == ScalingMethod.NEAREST_NEIGHBOR:
      scaled_image = self.__scale_with_nearest_neighbour(factor=factor, dimension=dimension)
    elif method == ScalingMethod.BILINEAR:
      image_padding = self.padding(pad=1)
      scaled_image = self.__scale_with_bilinear(f_img=image_padding, factor=factor, dimension=dimension)
    elif method == ScalingMethod.BICUBIC:
      image_padding = self.padding(pad=2)
      scaled_image = self.__scale_with_bicubic(f_img=image_padding, factor=factor, dimension=dimension)
    elif method == ScalingMethod.LAGRANGE:
      image_padding = self.padding(pad=2)
      scaled_image = self.__scale_with_lagrange(f_img=image_padding, factor=factor, dimension=dimension)
    else:
      raise ValueError("Opção de método inválido")
        
    end_time = time.time()
    
    print(f"Escala finalizada. Feito em {((end_time - init_time) * 1000) // 1}ms")  
      
    return Image(farray=scaled_image)
  
  def rotate(self, angle):
    print(f"Realizando rotação de {angle} graus na imagem...", end=" ")
    
    init_time = time.time()
    
    angle_rad = np.deg2rad(angle)
    
    rotate_matrix = np.array([
        [np.cos(angle_rad), np.sin(angle_rad)],
        [-np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    center_row, center_col = self.rows / 2, self.cols / 2
    
    rotated_image = np.zeros_like(self.image)
    
    for r in range(self.rows):
      for c in range(self.cols):
        trans_coords = np.array([r - center_row, c - center_col])
        
        original_coords = np.dot(rotate_matrix, trans_coords)
        
        orig_r, orig_c = original_coords[0] + center_row, original_coords[1] + center_col
        
        if 0 <= orig_r < self.rows and 0 <= orig_c < self.cols:
            rotated_image[r, c] = self.image[int(orig_r), int(orig_c)]
    
    end_time = time.time()
    
    print(f"Rotação finalizada. Feito em {((end_time - init_time) * 1000) // 1}ms")  
      
    return Image(rotated_image)
  
  def show(self, title="Imagem"):
    skimage.io.imshow(self.image)
    plt.title(title)  
    plt.show()
  
  def save(self, save_path):
    skimage.io.imsave(fname=save_path, arr=self.image)

def select_scale_option(method):
  if method == "nearest":
    return ScalingMethod.NEAREST_NEIGHBOR
  if method == "bilinear":
    return ScalingMethod.BILINEAR
  if method == "bicubic":
    return ScalingMethod.BICUBIC
  if method == "lagrange":
    return ScalingMethod.LAGRANGE
  
  raise ValueError("Opção inválida como método de escala")
  
def main():
  parser = argparse.ArgumentParser(description='Transformações geométricas em uma imagem PNG.')
  parser.add_argument('-a', '--angulo', type=float, help='Ângulo de rotação em graus no sentido anti-horário')
  parser.add_argument('-e', '--escala', type=float, help='Fator de escala')
  parser.add_argument('-d', '--dimensao', type=int, nargs=2, metavar=('LARGURA', 'ALTURA'), help='Dimensões da imagem de saída em pixels')
  parser.add_argument('-m', '--metodo', type=str, choices=['nearest', 'bilinear', 'bicubic', 'lagrange'], help='Método de interpolação utilizado')
  parser.add_argument('-i', '--input', type=str, required=True, help='Imagem de entrada no formato PNG')
  parser.add_argument('-o', '--output', type=str, required=True, help='Imagem de saída no formato PNG após a transformação geométrica')
  
  args = parser.parse_args()
  
  if args.metodo:
    try:
      scale_option = select_scale_option(args.metodo)
    except:
      print(f"Metodo de escala inválido: {args.metodo}")
      sys.exit(1)
  
  image = Image()
  
  try:
    image.read(args.input)
  except:
    print(f"Não foi possível abrir a imagem de entrada: {args.input}")
    sys.exit(1)
  
  image.show("Imagem de entrada")
  
  new_rows = image.rows
  new_cols = image.cols
  
  if args.dimensao:
    new_rows = args.dimensao[0]
    new_cols = args.dimensao[1]
  
  if args.angulo:
    image = image.rotate(args.angulo)
    
    image.show(f"Imagem após rotação de {args.angulo} graus")
    
  if args.escala:
    image = image.scale(args.escala, scale_option, (new_rows, new_cols))
    
    image.show(f"Imagem após escala de fator {args.escala}")
  
  image.save(args.output)
  
main()