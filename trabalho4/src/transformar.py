import numpy as np
import skimage

from matplotlib import pyplot as plt

from enum import Enum

class ScalingMethod(Enum):
  NEAREST_NEIGHBOR = 1
  BILINEAR = 2
  BICUBIC = 3

class Image:
  def __init__(self, farray = None) -> None:
    self.image = farray
    self.rows = 0
    self.cols = 0
    
  def read(self, image_path):
    print("Iniciando leitura da imagem...", end=" ")
    
    self.image = skimage.io.imread(fname=image_path)
    self.rows, self.cols = self.image.shape
    
    print("Processo Finalizado")
  
  def __scale_with_nearest_neighbour(self, factor):
    scaled_image = np.zeros(self.image.shape, dtype="uint8")
    
    for r in range(self.rows):
      for c in range(self.cols):
        if r / factor > self.rows or c / factor > self.cols:
          scaled_image[r, c] = 0
        else:
          scaled_image[r, c] = self.image[round(r / factor), round(c / factor)]
    
    return scaled_image
  
  def __scale_with_bilinear(self, factor):
    factor = factor
  
  def scale(self, factor, method):
    if factor == 0: raise ZeroDivisionError("Fator de escala não pode ser 0")
    
    scaled_image = None
    
    if method == ScalingMethod.NEAREST_NEIGHBOR:
      scaled_image = self.__scale_with_nearest_neighbour(factor)
    else:
      raise ValueError("Operação de escala inválida.")
        
    return Image(farray=scaled_image)
  
  def show(self):
    skimage.io.imshow(self.image)  
    plt.show()
  
  def save(self, save_path):
    skimage.io.imsave(fname=save_path, arr=self.image)

def main():
  image = Image()
  
  image.read("./files/inputs/baboon.png")
  
  image.show()
  
  scale_image = image.scale(4, ScalingMethod.NEAREST_NEIGHBOR)
  
  scale_image.show()
  
main()