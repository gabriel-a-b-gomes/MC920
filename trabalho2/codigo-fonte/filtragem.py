import numpy as np
import cv2
from matplotlib import pyplot as plt

import sys

class Image:
  def __init__(self, image) -> None:
    self.image = image
    
  def generate_histogram(self):
    return np.histogram(self.image.ravel(), 256, [0, 255])
  
  def get_mask_compress_image(self, threshold):
    normalized = cv2.normalize(self.image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return normalized > threshold
    
  def save_image(self, path):
    normalized = cv2.normalize(self.image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    cv2.imwrite(path, normalized)
    

class DFT:
  def __init__(self, folder) -> None:
    self.base_img = np.array([])
    self.dft_shift = np.array([])
    self.magnitude_spectrum = np.array([])
    self.magnitude_image = None
    self.rows = 0
    self.cols = 0
    self.center = (0,0)
    
    self.folder = folder
    
    self.mask_compress_low = np.array([])
    self.mask_compress_middle_low = np.array([])
    self.mask_compress_middle_high = np.array([])
    self.mask_compress_high = np.array([])
    
    self.mask_filter_high = np.array([])
    self.mask_filter_low = np.array([])
    self.mask_filter_pass_range = np.array([])
    self.mask_filter_reject_range = np.array([])
    
    
  def make_fourier_transform(self, path):
    self.base_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    dft = cv2.dft(np.float32(self.base_img), flags=cv2.DFT_COMPLEX_OUTPUT)

    self.dft_shift = np.fft.fftshift(dft)

    self.rows, self.cols = self.base_img.shape
    
    self.center = (self.rows // 2, self.cols // 2)
    
    
  def present_magnitude(self):
    try:
      self.magnitude_spectrum = np.log(cv2.magnitude(self.dft_shift[:,:,0],self.dft_shift[:,:,1]))
      
      self.magnitude_image = Image(self.magnitude_spectrum)
      self.magnitude_image.save_image(f'{self.folder}magnitude.png')
    
      plt.subplot(121),plt.imshow(self.base_img, cmap = 'gray')
      plt.title('Input Image'), plt.xticks([]), plt.yticks([])
      plt.subplot(122),plt.imshow(self.magnitude_spectrum, cmap = 'gray')
      plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
      
      plt.show()
      
      plt.clf()
    except:
      print("Ocorreu um erro ao apresentar os resultados de magnitude")
    
    
  def create_masks(self, lower_radius, bigger_radius):
    try:
      self.mask_filter_high = np.ones(self.dft_shift.shape)

      x, y = np.ogrid[:self.rows, :self.cols]

      circle_area = (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2 <= bigger_radius * bigger_radius

      self.mask_filter_high[circle_area] = 0

      self.mask_filter_low = 1 - self.mask_filter_high

      mid_circle_area = (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2 <= lower_radius * lower_radius

      self.mask_filter_pass_range = np.copy(self.mask_filter_low)

      self.mask_filter_pass_range[mid_circle_area] = 0

      self.mask_filter_reject_range = 1 - self.mask_filter_pass_range
    except:
      print("Ocorreu ao criar as máscaras")
    
    
  def present_mask_magnitude(self):
    try:      
      filtered_magnitude_high = Image(self.magnitude_spectrum * cv2.magnitude(self.mask_filter_high[:,:,0],self.mask_filter_high[:,:,1]))
      filtered_magnitude_high.save_image(f'{self.folder}filter_high.png')

      filtered_magnitude_low = Image(self.magnitude_spectrum * cv2.magnitude(self.mask_filter_low[:,:,0], self.mask_filter_low[:,:,1]))
      filtered_magnitude_low.save_image(f'{self.folder}filter_low.png')

      filtered_magnitude_reject_range = Image(self.magnitude_spectrum * cv2.magnitude(self.mask_filter_reject_range[:,:,0],self.mask_filter_reject_range[:,:,1]))
      filtered_magnitude_reject_range.save_image(f'{self.folder}filter_reject_range.png')

      filtered_magnitude_pass_range = Image(self.magnitude_spectrum * cv2.magnitude(self.mask_filter_pass_range[:,:,0],self.mask_filter_pass_range[:,:,1]))
      filtered_magnitude_pass_range.save_image(f'{self.folder}filter_pass_range.png')
      
      plt.subplot(231)
      plt.imshow(self.magnitude_spectrum, cmap='gray')
      plt.title('Fourier Transformation')
      plt.xticks([]), plt.yticks([])

      plt.subplot(232)
      plt.imshow(filtered_magnitude_high.image, cmap='gray')
      plt.title('Filter Pass High')
      plt.xticks([]), plt.yticks([])

      plt.subplot(233)
      plt.imshow(filtered_magnitude_low.image, cmap='gray')
      plt.title('Filter Pass Low')
      plt.xticks([]), plt.yticks([])

      plt.subplot(234)
      plt.imshow(filtered_magnitude_reject_range.image, cmap='gray')
      plt.title('Filter Reject Range')
      plt.xticks([]), plt.yticks([])

      plt.subplot(235)
      plt.imshow(filtered_magnitude_pass_range.image, cmap='gray')
      plt.title('Filter Pass Range')
      plt.xticks([]), plt.yticks([])

      plt.show()
      
      plt.clf()
    except:
      print("Ocorreu um erro ao apresentar os resultados da mascaras com magnitude")

  def make_inverse(self, fshift):
    try:
      f_ishift = np.fft.ifftshift(fshift)
        
      img_back = cv2.idft(f_ishift, flags=cv2.DFT_INVERSE)
      
      img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
      
      return Image(cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    except:
      return None

  def make_inverse_and_compress(self, mask = None, threshold = 0):
    try:
      fshift_no_compress = np.copy(self.dft_shift)
      fshift_compress = None
      
      if mask is not None:
        fshift_no_compress *= mask
      
      if threshold > 0:
        fshift_compress = np.copy(self.dft_shift)
        
        compress_mask = self.magnitude_image.get_mask_compress_image(threshold)
        
        fshift_compress[:,:,0] *= compress_mask
        fshift_compress[:,:,1] *= compress_mask
      
        if mask is not None:
          fshift_compress *= mask
        
      return self.make_inverse(fshift_no_compress), self.make_inverse(fshift_compress)        
    except Exception as error:
      print(f"Ocorreu um erro ao realizar a inversa com a máscara")
      
      return None, None


  def save_show_resullts(self, img: Image, path_to_save, compress_image: Image = None, path_to_save_compress = None, threshold = 0):
    img.save_image(f"{path_to_save}.png")
    
    # Plot the original image and its histogram
    plt.subplot(2, 2, 1)
    plt.imshow(img.image, cmap='gray')
    plt.title(f'Imagem {path_to_save}')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 2)
    values, counts = img.generate_histogram()
    plt.bar(counts[:-1], values)
    plt.title(f'Histograma {path_to_save}')

    if compress_image != None:
      compress_image.save_image(f"{path_to_save_compress}.png")
      
      plt.subplot(2, 2, 3)
      plt.imshow(compress_image.image, cmap='gray')
      plt.title(f'Comprimida (Limiar: {threshold})')
      plt.xticks([]), plt.yticks([])

      plt.subplot(2, 2, 4)
      values, counts = compress_image.generate_histogram()
      plt.bar(counts[:-1], values)
      plt.title(f'Histograma {path_to_save_compress}')

    plt.show()
 
def main():
  if len(sys.argv) != 5:
    print("Usage: python filtragem.py <imagem_entraga.png> <pasta_arquivo_saida> <raio_menor_filtro> <raio_maior_filtro>")
  else:
    input_path = sys.argv[1]
    output_folder = sys.argv[2]
    radius_lower = int(sys.argv[3])
    radius_bigger = int(sys.argv[4])
    
    image_dft = DFT(output_folder)
    
    image_dft.make_fourier_transform(input_path)
    
    image_dft.present_magnitude()
    
    image_dft.create_masks(radius_lower, radius_bigger)
    
    image_dft.present_mask_magnitude()
    
    image_high_filter, _ = image_dft.make_inverse_and_compress(image_dft.mask_filter_high)
    image_low_filter, _ = image_dft.make_inverse_and_compress(image_dft.mask_filter_low)
    image_pass_range_filter, _ = image_dft.make_inverse_and_compress(image_dft.mask_filter_pass_range)
    image_reject_range_filter, _ = image_dft.make_inverse_and_compress(image_dft.mask_filter_reject_range)
    
    image_dft.save_show_resullts(image_high_filter, f"{output_folder}high_filter")
    image_dft.save_show_resullts(image_low_filter, f"{output_folder}low_filter")
    image_dft.save_show_resullts(image_pass_range_filter, f"{output_folder}pass_range_filter")
    image_dft.save_show_resullts(image_reject_range_filter, f"{output_folder}reject_range_filter")
    
    for threshold in [100, 140, 170, 200]:
      image_no_filter, image_no_filter_compressed = image_dft.make_inverse_and_compress(threshold=threshold)
      image_dft.save_show_resullts(image_no_filter, f"{output_folder}no_filter", image_no_filter_compressed, f"{output_folder}no_filter_compressed_t{threshold}", threshold=threshold)
    
  
main()
 
