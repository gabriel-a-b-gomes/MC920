import numpy as np
import cv2
from matplotlib import pyplot as plt

import sys

class Image:
  def __init__(self) -> None:
    self.image = None
    
  def read_image(self, path):
    self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  
  def save_image(self, path):
    cv2.imwrite(path, self.image)
    
class LinkingImages:
  def __init__(self, img1: Image, img2: Image) -> None:
    self.img1 = img1
    self.img2 = img2
    
  def __find_descriptors(self, img: Image, method="SURF"):
    detector = cv2.xfeatures2d.SIFT_create()
    
    return detector.detectAndCompute(img.image, None)
    
  def compute_descriptors(self, method):
    self.keypoint1, self.desc1 = self.__find_descriptors(self.img1)
    self.keypoint2, self.desc2 = self.__find_descriptors(self.img2)

  def choose_best_match(self):
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    
    matches = bf.match(self.desc1, self.desc2)
    
    self.matches = sorted(matches, key=lambda x: x.distance)
  
  def find_homography(self, MIN_MATCH_COUNT, reject_threshold):
      
    if len(self.matches) > MIN_MATCH_COUNT:
      src_pts = np.float32([ self.keypoint1[m.queryIdx].pt for m in self.matches ]).reshape(-1,1,2)
      dst_pts = np.float32([ self.keypoint2[m.trainIdx].pt for m in self.matches ]).reshape(-1,1,2)

      self.H, self.status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=reject_threshold)

  def warp_perspective(self, image, shape):
    return cv2.warpPerspective(image, self.H, shape)  
  
  def draw_matches(self):
    return cv2.drawMatches(self.img1.image, self.keypoint1, self.img2.image, self.keypoint2, self.matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # # Display the result (or save the result image as needed)
    # cv2.imshow('Matches', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def main():
  img1 = Image()
  img1.read_image("./files/inputs/foto5A.jpg")
  
  img2 = Image()
  img2.read_image("./files/inputs/foto5B.jpg")
  
  linkImage = LinkingImages(img1, img2)
  
  linkImage.compute_descriptors(method="SURF")
  
  linkImage.choose_best_match()
  
  linkImage.find_homography(MIN_MATCH_COUNT=10, reject_threshold=5.0)
  
  width = img1.image.shape[1] + img2.image.shape[1]  # Width of the combined image
  height = max(img1.image.shape[0], img2.image.shape[0])  # Height of the combined image
  
  result = linkImage.warp_perspective(image=img1.image, shape=(width, height))
  
  r2, c2 = img2.image.shape
  
  result[0:r2, 0:c2] = img2.image
  
  matched_image = linkImage.draw_matches()
  
  plt.figure(figsize=(15, 10))
  plt.imshow(matched_image)
  plt.title('Matched Keypoints')
  plt.axis('off')
  plt.show()
  
  plt.figure(figsize=(15, 10))
  plt.imshow(result)
  plt.title('Panorama')
  plt.axis('off')
  plt.show()
  
main()