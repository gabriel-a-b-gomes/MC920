import sys
import cv2
import numpy as np

import time

def get_bit_plane(image, plane_number):
    plane = cv2.bitwise_and(image, 2 ** plane_number) >> plane_number
    
    return plane


def save_bit_plans_by_color(image, base_name):
    try:
        blue, green, red = cv2.split(image)

        # Get bit planes for bits 0, 1, 2, and 7
        bit_plane_blue_0 = get_bit_plane(blue, 0)
        bit_plane_blue_1 = get_bit_plane(blue, 1)
        bit_plane_blue_2 = get_bit_plane(blue, 2)
        bit_plane_blue_7 = get_bit_plane(blue, 7)

        cv2.imwrite(f'{base_name}_bit_plan_blue_0.png', bit_plane_blue_0 * 255)
        cv2.imwrite(f'{base_name}_bit_plan_blue_1.png', bit_plane_blue_1 * 255)
        cv2.imwrite(f'{base_name}_bit_plan_blue_2.png', bit_plane_blue_2 * 255)
        cv2.imwrite(f'{base_name}_bit_plan_blue_7.png', bit_plane_blue_7 * 255)

        bit_plane_green_0 = get_bit_plane(green, 0)
        bit_plane_green_1 = get_bit_plane(green, 1)
        bit_plane_green_2 = get_bit_plane(green, 2)
        bit_plane_green_7 = get_bit_plane(green, 7)

        cv2.imwrite(f'{base_name}_bit_plan_green_0.png', bit_plane_green_0 * 255)
        cv2.imwrite(f'{base_name}_bit_plan_green_1.png', bit_plane_green_1 * 255)
        cv2.imwrite(f'{base_name}_bit_plan_green_2.png', bit_plane_green_2 * 255)
        cv2.imwrite(f'{base_name}_bit_plan_green_7.png', bit_plane_green_7 * 255)

        bit_plane_red_0 = get_bit_plane(red, 0)
        bit_plane_red_1 = get_bit_plane(red, 1)
        bit_plane_red_2 = get_bit_plane(red, 2)
        bit_plane_red_7 = get_bit_plane(red, 7)

        cv2.imwrite(f'{base_name}_bit_plan_red_0.png', bit_plane_red_0 * 255)
        cv2.imwrite(f'{base_name}_bit_plan_red_1.png', bit_plane_red_1 * 255)
        cv2.imwrite(f'{base_name}_bit_plan_red_2.png', bit_plane_red_2 * 255)
        cv2.imwrite(f'{base_name}_bit_plan_red_7.png', bit_plane_red_7 * 255)
        
        print(f"Plano de bits salvos com sucesso na pasta {base_name}")
    except Exception as error:
        print(f"Ocorreu um erro ao salvar os planos de bits na pasta {base_name}")

def main():
  if len(sys.argv) != 5:
    print("Usage: python plano_bits.py <imagem_entrada.png> <imagem_saida.png> <pasta_plano_bits_entrada> <pasta_plano_bits_saida>")
  else:
    init_time = time.time()
      
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    save_input_plans = sys.argv[3]
    save_output_plans = sys.argv[4]
    
    input_image = cv2.imread(input_image_path)
            
    save_bit_plans_by_color(input_image, f'{save_input_plans}_input')
    
    output_image = cv2.imread(output_image_path)
            
    save_bit_plans_by_color(output_image, f'{save_output_plans}_output')

main()