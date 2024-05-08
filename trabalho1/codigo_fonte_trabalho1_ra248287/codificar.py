import sys
import cv2
import numpy as np

import time

def text_to_binary(file_address):
  try:
    with open(file_address) as file:
      entry = file.read()
    
    entry += "@@@"
    
    byte_array = np.frombuffer(entry.encode(), dtype=np.uint8)

    bit_array = np.unpackbits(byte_array)
    
    return bit_array
  except Exception as error:
    print("Arquivo de texto de entrada não encontrado!")
    return np.array([])
 

def select_bit_to_clean(bit_plan):
  return 255 - 2 ** bit_plan


def encode_image(height, width, image, bits_to_hide, bit_plan):
  try:
    mod_image_flat = image.reshape(-1)
    
    max_bits_to_hide = len(bits_to_hide)
    
    mod_image_flat[:max_bits_to_hide] = np.bitwise_and(mod_image_flat[:max_bits_to_hide], select_bit_to_clean(bit_plan))

    mod_image_flat[:max_bits_to_hide] = np.bitwise_or(mod_image_flat[:max_bits_to_hide], bits_to_hide << bit_plan)

    return mod_image_flat.reshape(height, width, 3)
  except Exception as error:
    print(f"Ocorreu um erro ao encripitar a mensagem no plano de bit: {bit_plan}")
    return image


def main():
  init_time = time.time()
  
  if len(sys.argv) != 5:
    print("Usage: python codificar.py <imagem_entrada.png> <texto_entrada.txt> <plano_bits> <imagem_saida.png>")
  else:
    input_image_path = sys.argv[1]
    input_text_path = sys.argv[2]
    bit_plans = sys.argv[3]
    output_image_path = sys.argv[4]
  
    image = cv2.imread(input_image_path)
    
    if image is None:
      print("Imagem não encontrada!")
      return
    
    height, width, _ = image.shape+
    
    max_bits = 3 * height * width
    
    bin_to_hide = text_to_binary(input_text_path)
    
    if len(bin_to_hide) > len(bit_plans) * max_bits: 
      print("O texto não cabe dentro destes planos de bits. É preciso aumentar o número de plano de bits, ou escolher outra imagem!")
      print("Iremos encriptar a mensagem, sem a garantia que ela irá caber totalmente!")
    
    # Acomoda os bits decodificados do texto de acordo com o plano de bits passado
    for i in range(len(bit_plans)):
      bit_plan = int(bit_plans[i])
      
      image = encode_image(height, width, image, bin_to_hide[i * max_bits:(i + 1) * max_bits], bit_plan)
    
    if image is not None:
      cv2.imwrite(output_image_path, image)
      
      end_time = time.time()
      print(f"Texto codificado com sucesso para o arquivo: {output_image_path}. Feito em {((end_time - init_time) * 1000) // 1} ms.")
      
      cv2.imshow('Image', image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
      

main()