#!/usr/bin/env python

class ProbabilityDistribution(object):
    def __init__(self):
        self.http_g1 = None
        self.http_g2 = None
        self.http_g3 = None
        self.gnutella_g1 = None
        self.gnutella_g2 = None
        self.gnutella_g3 = None
        self.edonkey_g1 = None
        self.edonkey_g2 = None
        self.edonkey_g3 = None
        self.bittorrent_g1 = None
        self.bittorrent_g2 = None
        self.bittorrent_g3 = None

    def get_final_probability(self):
        bittorrent_final = 1.0 - (1.0 - self.bittorrent_g1)*(1.0 - self.bittorrent_g2)*(1.0 - self.bittorrent_g3)
        edonkey_final = 1.0 - (1.0 - self.edonkey_g1)*(1.0 - self.edonkey_g2)*(1.0 - self.edonkey_g3)
        gnutella_final = 1.0 - (1.0 - self.gnutella_g1)*(1.0 - self.gnutella_g2)*(1.0 - self.gnutella_g3)
        http_final = 1.0 - (1.0 - self.http_g1)*(1.0 - self.http_g2)*(1.0 - self.http_g3)

        bittorrent_final = float(self.bittorrent_g1+self.bittorrent_g2+self.bittorrent_g3)/3
        edonkey_final = float(self.edonkey_g1+self.edonkey_g2+self.edonkey_g3)/3
        gnutella_final = float(self.gnutella_g1+self.gnutella_g2+self.gnutella_g3)/3
        http_final = float(self.http_g1+self.http_g2+self.http_g3)/3

        final_probability = max(http_final, gnutella_final, edonkey_final, bittorrent_final)
        if final_probability == http_final:
            return (final_probability, 'http')
        elif final_probability == gnutella_final:
            return (final_probability, 'gnutella')
        elif final_probability == edonkey_final:
            return (final_probability, 'edonkey')
        else:
            return (final_probability, 'bittorrent')

    def __str__(self):
        bit_prob = 1.0 - (1.0 - self.bittorrent_g1)*(1.0 - self.bittorrent_g2)*(1.0 - self.bittorrent_g3)
        http_prob = 1.0 - (1.0 - self.http_g1)*(1.0 - self.http_g2)*(1.0 - self.http_g3)
        gnu_prob = 1.0 - (1.0 - self.gnutella_g1)*(1.0 - self.gnutella_g2)*(1.0 - self.gnutella_g3)
        edo_prob = 1.0 - (1.0 - self.edonkey_g1)*(1.0 - self.edonkey_g2)*(1.0 - self.edonkey_g3)

        bit_prob = float(self.bittorrent_g1+self.bittorrent_g2+self.bittorrent_g3)/3
        http_prob = float(self.http_g1+self.http_g2+self.http_g3)/3
        gnu_prob = float(self.gnutella_g1+self.gnutella_g2+self.gnutella_g3)/3
        edo_prob = float(self.edonkey_g1+self.edonkey_g2+self.edonkey_g3)/3

        header_st = '{0:15}{1:10}{2:10}{3:10}{4:10}{5:10}\n'
        st = '{0:15}{1:1.3f}{2:10.3f}{3:10.3f}{4:10.3f}{5:10.3f}\n'
        header = header_st.format('Application', 'Group 1', 'Group 2', 'Group 3', 'FinalProb', 'MaxProb')
        http_row = st.format('Http', self.http_g1, self.http_g2, self.http_g3, http_prob, self.get_final_probability()[0])
        gnutella_row = st.format('Gnutella', self.gnutella_g1, self.gnutella_g2, self.gnutella_g3, gnu_prob, self.get_final_probability()[0])
        edonkey_row = st.format('Edonkey', self.edonkey_g1, self.edonkey_g2, self.edonkey_g3, edo_prob, self.get_final_probability()[0])
        bittorrent_row = st.format('Bittorrent', self.bittorrent_g1, self.bittorrent_g2, self.bittorrent_g3, bit_prob, self.get_final_probability()[0])
        return header + http_row + gnutella_row + edonkey_row + bittorrent_row 

        