import sys

import numpy as np
import cv2
import skimage
from matplotlib import pyplot as plt

class Image:
  def __init__(self) -> None:
    self.image = None
    
  def read(self, image_path):
    self.image = skimage.io.imread(fname=image_path, as_gray=True)
    
  def rotate_image(self, angle):
    return skimage.transform.rotate(self.image, angle, mode="edge", cval=1)
  
  def show(self):
    skimage.io.imshow(self.image)  
  
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
    self.profile = np.sum(self.F.image, axis=1)
    
    for theta in range(self.theta1, self.theta2 + 1, self.theta_step):
      self.values[theta] = self.aim_func(theta)
      
    thetaM = max(self.values, key=self.values.get)
    
    return thetaM
  
  def rotate(self, theta):
    rotate = self.F.rotate_image(theta)
    
    return np.sum(rotate, axis=1)
  
  def calc_variance(self, profile):
    return np.std(profile) ** 2
    
  def aim_func(self, theta):
    rotate_profile = self.rotate(theta)
    
    return self.calc_variance(rotate_profile)

def main():
  # if len(sys.argv) != 4:
  #   print("Usage: python alinhar.py <imagem_entraga.png> <modo> <imagem_saida.png>")
  # else:
  
  image = Image()
  
  image.read('./src/files/input/sample2.png')
  
  horizontal = Horizontal(F=image, theta1=-180, theta2=180, theta_step=1)
  
  angle = horizontal.detect_inclination()
  
  print(f"Ângulo de correção: {angle}")
  
  image.image = image.rotate_image(angle)
  
  image.show()
  
  image.save('./src/files/output/sample2.png')
    
    
main()