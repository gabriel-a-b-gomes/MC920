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
  
  def __near_point(self, x_line, y_line):
    x = np.round(x_line)
    y = np.round(y_line)
    
    return x, y
  
  def __scale_with_nearest_neighbour(self, row, col, factor_row, factor_col):
    x_line = row / factor_row
    y_line = col / factor_col
    
    try:
      x, y = self.__near_point(x_line, y_line)
      return self.image[x, y]
    except:
      return 0
  
  def __dx(self, x_line):
    return x_line - np.floor(x_line)
    
  def __dy(self, y_line):
    return y_line - np.floor(y_line)
  
  def __scale_with_bilinear(self, row, col, factor_row, factor_col):
    x_line = row / factor_row
    y_line = col / factor_col
    
    dx = self.__dx(x_line)
    dy = self.__dy(y_line)
    x, y = self.__near_point(x_line, y_line)
    
    try:
      return (1 - dx) * (1 - dy) * self.image[x, y] + dx * (1 - dy) * self.image[x + 1, y] + (1 - dx) * dy * self.image[x, y + 1] + dx * dy * self.image[x + 1, y + 1]
    except:
      return 0
  
  def __P(self, t):
    return t if t > 0 else 0
    
  def __R(self, s):
    return (1 / 6) * (self.__P(s + 2) ** 3 - 4 * self.__P(s + 1) ** 3 + 6 * self.__P(s) ** 3 - 4 * self.__P(s - 1) ** 3)
  
  def __scale_with_bicubic(self, row, col, factor_row, factor_col):
    x_line = row / factor_row
    y_line = col / factor_col
    
    dx = self.__dx(x_line)
    dy = self.__dy(y_line)
    x, y = self.__near_point(x_line, y_line)
    
    point = 0
    
    try:
      for m in range(-1, 3):
        for n in range(-1, 3):
          point += self.image[x + m, y + n] * self.__R(m - dx) * self.__R(dy - n)
      
      return point
    except:
      return 0
  
  def __L(self, f_img, dx, x, y, n):
    l1 = (-dx * (dx - 1) * (dx - 2) * f_img[x - 1, y + n - 2]) / 6
    l2 = ((dx + 1) * (dx - 1) * (dx - 2) * f_img[x, y + n - 2]) / 2
    l3 = (-dx * (dx + 1) * (dx - 2) * f_img[x + 1, y + n - 2]) / 2
    l4 = (dx * (dx + 1) * (dx - 1) * f_img[x + 2, y + n - 2]) / 6
    
    return l1 + l2 + l3 + l4
  
  def __scale_with_lagrange(self, row, col, factor_row, factor_col):
    x_line = row / factor_row
    y_line = col / factor_col
    
    dx = self.__dx(x_line)
    dy = self.__dy(y_line)
    x, y = self.__near_point(x_line, y_line)
    
    try:
      a1 = (-dy * (dy - 1) * (dy - 2) * self.__L(self.image, dx, x, y, 1)) / 6
      a2 = ((dy + 1) * (dy - 1) * (dy - 2) * self.__L(self.image, dx, x, y, 2)) / 2
      a3 = (-dy * (dy + 1) * (dy - 2) * self.__L(self.image, dx, x, y, 3)) / 2
      a4 = (dy * (dy + 1) * (dy - 1) * self.__L(self.image, dx, x, y, 4)) / 6
      
      return a1 + a2 + a3 + a4
    except:
      return 0
  
  def scale(self, factor_row, factor_col, method):
    if factor_col == 0 or factor_row == 0: raise ZeroDivisionError("Fator de escala não pode ser 0")
    
    print(f"Realizando escala da imagem por {method.name}...", end=" ")
    
    init_time = time.time()
    
    scaled_image = None
    
    new_rows = round(self.rows * factor_row)
    new_cols = round(self.cols * factor_col)
    
    scaled_image = np.zeros((new_rows, new_cols), dtype=np.uint8)
    
    scale_method = None
    if method == ScalingMethod.NEAREST_NEIGHBOR:
      scale_method = self.__scale_with_nearest_neighbour
    elif method == ScalingMethod.BILINEAR:
      scale_method = self.__scale_with_bilinear
    elif method == ScalingMethod.BICUBIC:
      scale_method = self.__scale_with_bicubic
    elif method == ScalingMethod.LAGRANGE:
      scale_method = self.__scale_with_lagrange
    else:
      raise ValueError("Opção de método inválido")
    
    for r in range(new_rows):
      for c in range(new_cols):
        scaled_image[r, c] = scale_method(row=r, col=c, factor_row=factor_row, factor_col=factor_col)
        
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
    image = image.scale(args.escala, args.escala, scale_option)
    
    image.show(f"Imagem após escala de fator {args.escala}")
    
  if args.dimensao:
    new_rows = args.dimensao[0]
    new_cols = args.dimensao[1]
    
    image = image.scale(factor_row=(new_rows/image.rows), factor_col=(new_cols/image.cols), method=scale_option)
    
    image.show(f"Imagem após redimensionamento {new_rows}x{new_cols}")
  
  image.save(args.output)
  
main()