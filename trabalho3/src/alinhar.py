import numpy as np
import skimage
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks, rotate

from matplotlib import pyplot as plt

import sys

import time

class Image:
  def __init__(self) -> None:
    self.image = None
    self.rows = 0
    self.cols = 0
    
  def read(self, image_path):
    print("Iniciando leitura da imagem...", end=" ")
    
    self.image = skimage.io.imread(fname=image_path, as_gray=True)
    self.rows, self.cols = self.image.shape
    
    print("Processo Finalizado")
    
  def rotate_image(self, angle):
    return rotate(self.image, angle, resize=True, mode="constant", cval=0)
  
  def normalize_rotate_angle(self, angle):
    new_angle = angle
    
    if abs(angle) > 90:
      new_angle = 180 + angle if angle < 0 else angle - 180
      
    return new_angle
  
  def show(self):
    skimage.io.imshow(self.image)  
    plt.show()
  
  def save(self, save_path):
    ubyte_image = skimage.img_as_ubyte(self.image)

    skimage.io.imsave(fname=save_path, arr=ubyte_image)
    
class Horizontal:
  def __init__(self, F: Image, theta1, theta2, theta_step) -> None:
    self.F = F
    self.n = F.image.shape[0]
    self.profile = []
    self.values = {}
    
    self.theta1 = theta1
    self.theta2 = theta2
    self.theta_step = theta_step
    
  def detect_inclination(self):  
    try:  
      print("Iniciando decteção de inclinação da imagem com o método horizontal...", end=" ") 
      
      theta = self.theta1
      
      while theta <= self.theta2:
        self.values[theta] = self.aim_func(theta)
        theta += self.theta_step
      
      angle = max(self.values, key=self.values.get)
      
      print("Processo Finalizado")
      
      return angle
    except Exception as error:
      print("ERRO! Mensagem: " + error) 
      return 0

  def rotate(self, theta):
    rotate = self.F.rotate_image(theta)
    
    return np.sum(rotate, axis=1)
  
  def calc_square_diff(self, profile):
    diff = np.diff(profile)  # Compute the difference between neighbors
    squared_diff = diff ** 2  # Square the differences
    return np.sum(squared_diff)  # Sum the squared differences
    
  def aim_func(self, theta):
    rotate_profile = self.rotate(theta)
    return self.calc_square_diff(rotate_profile)
  
class Hough:
  def __init__(self, F: Image, threshold) -> None:
    self.F = F
    self.threshold = threshold
  
  def detect_inclination(self):
    try:
      print("Iniciando decteção de inclinação da imagem com o método de Hough...", end=" ") 
      
      edges = canny(self.F.image, sigma=2)
      
      h, theta, d = hough_line(edges)
      
      _, angles, _ = hough_line_peaks(h, theta, d, threshold=self.threshold)
      
      angles_deg = np.rad2deg(angles)
      
      median_angle = round(np.median(angles_deg) - 90)
      
      print("Processo Finalizado")
      
      return median_angle
    except Exception as error:
      print(f"ERRO! Mensagem: {error}") 
      return 0

def main():
  if len(sys.argv) != 4:
    print("Usage: python alinhar.py <imagem_entraga.png> <modo> <imagem_saida.png>")
  else:
    init_time = time.time()
    
    angle = 0
    
    input_path = sys.argv[1]
    mode = sys.argv[2].lower()
    output_path = sys.argv[3]
    
    image = Image()
    
    image.read(input_path)
    
    if mode == "hl" or mode == "horizontal":
      
      horizontal = Horizontal(F=image, theta1=-180, theta2=180, theta_step=1)
      
      angle = horizontal.detect_inclination()
      
    elif mode == "hg" or mode == "hough":
      
      hough = Hough(F=image, threshold=(image.cols * 0.3))
      
      angle = hough.detect_inclination()
      
    else:
      print("Operação desconhecida. As operações válidas são horizontal ou hl ou hough ou hg")
      
      return
    
    angle = image.normalize_rotate_angle(angle)
    
    print(f"Resultado obtido: Ângulo de correção de {angle} graus")
    
    image.image = image.rotate_image(angle)
    
    end_time = time.time()
    
    image.show()
    
    image.save(output_path)
    
    print(f"Feito em {((end_time - init_time) * 1000) // 1} ms.")
    
main()