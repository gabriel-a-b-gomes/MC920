import sys

import numpy as np
import cv2
from matplotlib import pyplot as plt

class Image:
  def __init__(self) -> None:
    self.image = None
    
  def read(self, image_path):
    self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
  def save(self, save_path):
    cv2.imwrite(save_path, self.image)
    
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
    
    for theta in range(self.theta1, self.theta2, self.theta_step):
      self.values[theta] = self.aim_func(theta)
      
    thetaM = max(self.values, key=self.values.get)
    
    return thetaM
  
  def rotate(self, theta):
    theta_rad = np.deg2rad(theta)
    
    original_index = np.arange(self.n)
    
    rotate_index = np.round(original_index * np.cos(theta_rad)).astype(int)
    
    rotate_index = np.clip(rotate_index, 0, self.n - 1)
    
    rotate_profile = self.profile[rotate_index]
    
    return rotate_profile
  
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
  
  image.read('./src/files/input/neg_4.png')
  
  horizontal = Horizontal(F=image, theta1=0, theta2=360, theta_step=1)
  
  print(horizontal.detect_inclination())
  
  image.save('./src/files/output/neg_4.png')
    
    
main()