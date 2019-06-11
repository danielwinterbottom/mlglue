#usual imports
import numpy as np
import xgboost as xg
import pickle
import pandas as pd
import ROOT as r
from root_numpy import tree2array, testdata, list_branches, fill_hist
from os import system, path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help= 'Input XGBoost model')
args = parser.parse_args()
modelName = args.input

#define the different sets of variables used

Vars = ['Egamma1_tau', 'Egamma2_tau', 'Epi_tau', 'rho_dEta_tau', 'rho_dphi_tau',
 'gammas_dEta_tau', 'gammas_dR_tau', 'DeltaR2WRTtau_tau', 'tau_decay_mode',
 'eta', 'pt', 'Epi0', 'Epi', 'rho_dEta', 'rho_dphi', 'gammas_dEta', 'Mrho', 'Mpi0',
 'DeltaR2WRTtau', 'Mpi0_TwoHighGammas', 'Mrho_OneHighGammas',
 'Mrho_TwoHighGammas', 'Mrho_subleadingGamma', 'strip_pt']

#load models
Model = xg.Booster()
#modelName = 'inclusive_model.model'
xmlName = modelName.split('/')[-1].replace('.model','.xml')
Model.load_model(modelName)
print 'Loaded model called %s'%modelName.split('/')[-1]
weightDir = './'

#if not path.isdir(weightDir): 
#  system('mkdir -p %s'%weightDir)

#convert!
from mlglue.tree import tree_to_tmva, BDTxgboost, BDTsklearn
target_names = ['0','1','2','3']
bdt = BDTxgboost(Model, Vars, target_names, kind='multiclass', max_depth=5, learning_rate=0.05)
bdt.to_tmva('%s/%s'%(weightDir,xmlName))
print 'Created xml called %s'%xmlName
