#usual imports
import numpy as np
import xgboost as xg
import pickle
import pandas as pd
import ROOT as r
from root_numpy import tree2array, testdata, list_branches, fill_hist
nClasses=9
#define the different sets of variables used
diphoVars  = ['leadmva','subleadmva','leadptom','subleadptom',
              'leadeta','subleadeta',
              'CosPhi','vtxprob','sigmarv','sigmawv']
classVars  = ['n_rec_jets','dijet_Mjj','diphopt',
              'dijet_leadEta','dijet_subleadEta','dijet_subsubleadEta',
              #'dijet_LeadJPt','dijet_SubJPt','dijet_SubsubJPt',
              'dijet_LeadJPt','dijet_SubJPt',
              'dijet_leadPUMVA','dijet_subleadPUMVA','dijet_subsubleadPUMVA',
              'dijet_leadDeltaPhi','dijet_subleadDeltaPhi','dijet_subsubleadDeltaPhi',
              'dijet_leadDeltaEta','dijet_subleadDeltaEta','dijet_subsubleadDeltaEta']
#load models
altDiphoModel = xg.Booster()
altDiphoModel.load_model('../../altDiphoModel.model')
altClassModel = xg.Booster()
altClassModel.load_model('../../../../Pass3/TwoStep/altDiphoModel.model')
#convert!
from mlglue.tree import tree_to_tmva, BDTxgboost, BDTsklearn
target_names = ['bkg','sig']
bdt = BDTxgboost(altDiphoModel, diphoVars, target_names, kind='binary', max_depth=6, learning_rate=0.3)
bdt.to_tmva('altDiphoModel.xml')
from mlglue.tree import tree_to_tmva, BDTxgboost, BDTsklearn
target_names = [ str(i) for i in range(0,nClasses) ]
bdt = BDTxgboost(altClassModel, classVars, target_names, kind='multiclass', max_depth=6, learning_rate=0.3)
bdt.to_tmva('altClassModel.py')
