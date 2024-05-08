import sys
import cv2
import numpy as np

import time

def unpackbits_to_array(image, bit_plan):    
    pure_bits = np.bitwise_and(image, 2 ** bit_plan) >> bit_plan
    
    pure_bits = pure_bits.reshape(-1)
    
    return pure_bits

def main():
  if len(sys.argv) != 4:
    print("Usage: python decodificar.py <imagem_saida.png> <plano_bits> <texto_saida.txt>")
  else:
    init_time = time.time()
      
    output_image_path = sys.argv[1]
    bit_plans = sys.argv[2]
    output_text_path = sys.argv[3]
    
    image = cv2.imread(output_image_path)
    
    bits_array = np.array([], dtype=np.uint8)
    
    for bit in bit_plans:
        bits_array = np.append(bits_array, unpackbits_to_array(image, int(bit)))
    
    # Convert bits to bytes
    bytes_array = np.packbits(bits_array)

    # Convert bytes to string
    string_from_bits = bytes_array.tobytes().decode(errors="ignore")
    
    string_from_bits = string_from_bits.split("@@@")[0]
    
    with open(output_text_path, 'w') as file:
        try:
            file.write(string_from_bits)
            
            end_time = time.time()
            
            print(f"Texto decodificado com sucesso. Escrito em {output_text_path}. Feito em {((end_time - init_time) * 1000) // 1} ms.")
        except Exception as error:
            print("Ocorreu um erro ao decodificar o texto!")
            
main()