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

Vars = ['E1_overEa1', 'E2_overEa1', 'E1_overEtau', 'E2_overEtau', 'E3_overEtau',
 'a1_pi0_dEta_timesEtau', 'a1_pi0_dphi_timesEtau', 'h1_h2_dphi_timesE12',
 'h1_h2_dEta_timesE12', 'h1_h3_dphi_timesE13', 'h1_h3_dEta_timesE13',
 'h2_h3_dphi_timesE23', 'h2_h3_dEta_timesE23', 'gammas_dEta_timesEtau',
 'gammas_dR_timesEtau', 'tau_decay_mode', 'mass0', 'mass1', 'mass2', 'E1', 'E2',
 'E3', 'strip_E', 'a1_pi0_dEta', 'a1_pi0_dphi', 'strip_pt', 'pt', 'eta', 'E',
 'h1_h2_dphi', 'h1_h3_dphi', 'h2_h3_dphi', 'h1_h2_dEta', 'h1_h3_dEta',
 'h2_h3_dEta', 'Egamma1', 'Egamma2', 'gammas_dEta', 'Mpi0',
 'Mpi0_TwoHighGammas']

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
target_names = ['0','1','2']
bdt = BDTxgboost(Model, Vars, target_names, kind='multiclass', max_depth=5, learning_rate=0.05)
bdt.to_tmva('%s/%s'%(weightDir,xmlName))
print 'Created xml called %s'%xmlName
