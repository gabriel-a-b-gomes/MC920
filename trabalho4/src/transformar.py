import numpy as np
import skimage

from matplotlib import pyplot as plt

from enum import Enum

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
    
  def __dx(self, x_line):
    return x_line - np.floor(x_line)
    
  def __dy(self, y_line):
    return y_line - np.floor(y_line)
  
  def __near_point(self, x_line, y_line):
    x = round(x_line)
    y = round(y_line)
    
    return x, y
    
  
  def read(self, image_path):
    print("Iniciando leitura da imagem...", end=" ")
    
    self.image = skimage.io.imread(fname=image_path)
    self.rows, self.cols = self.image.shape
    
    print("Processo Finalizado")
  
  def padding(self, pad):
    return np.pad(self.image, pad, mode="edge")
  
  def __scale_with_nearest_neighbour(self, factor):
    scaled_image = np.zeros(self.image.shape, dtype="uint8")
    
    for r in range(self.rows):
      for c in range(self.cols):
        x_line = r / factor
        y_line = c / factor
        
        if x_line > self.rows or y_line > self.cols:
          scaled_image[r, c] = 0
        else:
          x, y = self.__near_point(x_line, y_line)
          scaled_image[r, c] = self.image[x, y]
    
    return scaled_image
  
  def __scale_with_bilinear(self, f_img, factor):
    scaled_image = np.zeros(self.image.shape, dtype="uint8")
    
    for r in range(self.rows):
      for c in range(self.cols):
        x_line = (r + 1) / factor
        y_line = (c + 1) / factor
        
        if x_line > self.rows or y_line > self.cols:
          scaled_image[r, c] = 0
        else:
          dx = self.__dx(x_line)
          dy = self.__dy(y_line)
          x, y = self.__near_point(x_line, y_line)
          scaled_image[r, c] = (1 - dx) * (1 - dy) * f_img[x, y] + dx * (1 - dy) * f_img[x + 1, y] + (1 - dx) * dy * f_img[x, y + 1] + dx * dy * f_img[x + 1, y + 1]
    
    return scaled_image
  
  def __P(self, t):
    return t if t > 0 else 0
    
  def __R(self, s):
    return (1 / 6) * (self.__P(s + 2) ** 3 - 4 * self.__P(s + 1) ** 3 + 6 * self.__P(s) ** 3 - 4 * self.__P(s - 1) ** 3)
  
  def __scale_with_bicubic(self, f_img, factor):
    scaled_image = np.zeros(self.image.shape, dtype="uint8")
    
    for r in range(self.rows):
      for c in range(self.cols):
        x_line = (r + 2) / factor
        y_line = (c + 2) / factor
        
        if x_line > self.rows or y_line > self.cols:
          scaled_image[r, c] = 0
        else:
          dx = self.__dx(x_line)
          dy = self.__dy(y_line)
          x, y = self.__near_point(x_line, y_line)
          
          for m in range(-1, 3):
            for n in range(-1, 3):
              scaled_image[r, c] += f_img[x + m, y + n] * self.__R(m - dx) * self.__R(dy - n)
          
    return scaled_image
  
  def __L(self, f_img, dx, x, y, n):
    return (-dx * (dx - 1) * (dx - 2) * f_img[x - 1, y + n - 2]) / 6 + ((dx + 1) * (dx - 1) * (dx - 2) * f_img[x, y + n - 2]) / 2 + (-dx * (dx + 1) * (dx - 2) * f_img[x + 1, y + n - 2]) / 2 + (dx * (dx + 1) * (dx - 1) * f_img[x + 2, y + n - 2]) / 6
  
  def __scale_with_lagrange(self, f_img, factor):
    scaled_image = np.zeros(self.image.shape, dtype="uint8")
    
    for r in range(self.rows):
      for c in range(self.cols):
        x_line = (r + 2) / factor
        y_line = (c + 2) / factor
        
        if x_line > self.rows or y_line > self.cols:
          scaled_image[r, c] = 0
        else:
          dx = self.__dx(x_line)
          dy = self.__dy(y_line)
          x, y = self.__near_point(x_line, y_line)
          
          scaled_image[r, c] = (-dy * (dy - 1) * (dy - 2) * self.__L(f_img, dx, x, y, 1)) / 6 + ((dy + 1) * (dy - 1) * (dy - 2) * self.__L(f_img, dx, x, y, 2)) / 2 + (-dy * (dy + 1) * (dy - 2) * self.__L(f_img, dx, x, y, 3)) / 2 + (dy * (dy + 1) * (dy - 1) * self.__L(f_img, dx, x, y, 4)) / 6
          
    return scaled_image
  
  def scale(self, factor, method):
    if factor == 0: raise ZeroDivisionError("Fator de escala não pode ser 0")
    
    scaled_image = None
    
    if method == ScalingMethod.NEAREST_NEIGHBOR:
      scaled_image = self.__scale_with_nearest_neighbour(factor)
    elif method == ScalingMethod.BILINEAR:
      image_padding = self.padding(pad=1)
      scaled_image = self.__scale_with_bilinear(f_img=image_padding, factor=factor)
    elif method == ScalingMethod.BICUBIC:
      image_padding = self.padding(pad=2)
      scaled_image = self.__scale_with_bicubic(f_img=image_padding, factor=factor)
    elif method == ScalingMethod.LAGRANGE:
      image_padding = self.padding(pad=2)
      scaled_image = self.__scale_with_lagrange(f_img=image_padding, factor=factor)
    else:
      raise ValueError("Opção de método inválido")
        
    return Image(farray=scaled_image)
  
  def rotate(self, angle):
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
        
    return Image(rotated_image)
  
  def show(self):
    skimage.io.imshow(self.image)  
    plt.show()
  
  def save(self, save_path):
    skimage.io.imsave(fname=save_path, arr=self.image)

def main():
  image = Image()
  
  image.read("./files/inputs/baboon.png")
  
  image.show()
  
  scale_image = image.scale(2, ScalingMethod.LAGRANGE)
  
  scale_image.show()
  
  rotate_image = image.rotate(-20)
  
  rotate_image.show()
  
  scale_rotate_image = rotate_image.scale(1.5, ScalingMethod.NEAREST_NEIGHBOR)
  
  scale_rotate_image.show()
  
main()