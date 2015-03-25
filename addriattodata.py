#!/usr/bin/env python


INPUT_FILE = 'realinput.csv'

OUTPUT_FILE = 'mofifiedinput.csv'

import numpy as np

def main():
    input_array = np.genfromtxt(INPUT_FILE, delimiter=",")
    avg_iat = input_array[:, 0]
    min_iat = input_array[:, 1]
    max_iat = input_array[:, 2]
    std_div_iat = input_array[:, 3]
    avg_riat = avg_iat/min_iat
    max_riat = max_iat/min_iat
    std_div_riat = std_div_iat/min_iat

    input_array = np.insert(input_array, 4, avg_riat, axis=1)
    input_array = np.insert(input_array, 5, max_riat, axis=1)
    input_array = np.insert(input_array, 6, std_div_riat, axis=1)
    np.savetxt(OUTPUT_FILE, input_array, delimiter = ",")

if __name__ == '__main__':
    main()