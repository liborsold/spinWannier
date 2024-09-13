import numpy as np
import pickle
import sys
from wannier_utils import files_wann90_to_dict_pickle, eigenval_dict, load_dict
from os.path import exists
import copy


class WannierTBmodel():
    def __init__(self, others):
        """Initialize and load the model."""
        # load the model files and convert to pickle dictionaries
        disentanglement = exists('wannier90_u_dis.mat')
        files_wann90_to_dict_pickle(disentanglement=disentanglement)
        eig_dict = eigenval_dict(eigenval_file="wannier90.eig",  win_file="wannier90.win")
        u_dict = load_dict(fin="u_dict.pickle")
        if disentanglement is True:
            u_dis_dict = load_dict(fin="u_dis_dict.pickle")
        else:
            u_dis_dict = copy.copy(u_dict)
            NW = int(u_dict[list(u_dict.keys())[0]].shape[0])
            for key in u_dis_dict.keys():
                u_dis_dict[key] = np.eye(NW)
        spn_dict = load_dict(fin="spn_dict.pickle")

        # store the model
        self.eig_dict = eig_dict
        self.u_dict = u_dict
        self.u_dis_dict = u_dis_dict
        self.spn_dict = spn_dict
        self.NW = NW

    def interpolate1D():
        raise NotImplementedError
    
    def interpolate2D():
        raise NotImplementedError
    
    def plot1D_bands():
        raise NotImplementedError
    
    def plot2D_spin_texture():
        raise NotImplementedError
    
    def Wannier_quality():
        raise NotImplementedError