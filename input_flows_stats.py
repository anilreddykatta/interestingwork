__author__ = 'KattaAnil'

import numpy as np

PROTO_DICT = {'http': 0.0, 'gnutella': 1.0, "edonkey": 2.0, "bittorrent": 3.0, "skype":4.0}


def get_numpy_array_from_file(file_name):
    return np.genfromtxt(file_name, delimiter=",")


def main():
    input_data = get_numpy_array_from_file("mofifiedinput.csv")
    http_data = input_data[input_data[:, 17] == PROTO_DICT['http']]
    gnutella_data = input_data[input_data[:, 17] == PROTO_DICT['gnutella']]
    edonkey_data = input_data[input_data[:, 17] == PROTO_DICT['edonkey']]
    bittorrent_data = input_data[input_data[:, 17] == PROTO_DICT['bittorrent']]
    unknown_data = input_data[input_data[:, 17] == PROTO_DICT['skype']]
    print(http_data.shape)
    print(gnutella_data.shape)
    print(edonkey_data.shape)
    print(bittorrent_data.shape)
    print(unknown_data.shape)


if __name__ == '__main__':
    main()