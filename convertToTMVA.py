#usual imports
import numpy as np
import xgboost as xg
import pickle
import pandas as pd
import ROOT as r
from root_numpy import tree2array, testdata, list_branches, fill_hist
from os import system, path

#define the different sets of variables used
diphoVars  = ['leadmva','subleadmva','leadptom','subleadptom',
              'leadeta','subleadeta',
              'CosPhi','vtxprob','sigmarv','sigmawv']

#load models
altDiphoModel = xg.Booster()
#modelName = '/vols/cms/es811/Stage1categorisation/2016/models/altDiphoModel.model'
#modelName = '/vols/cms/es811/Stage1categorisation/2017/models/altDiphoModel.model'
#modelName = '/vols/cms/es811/Stage1categorisation/Pass1/2016/models/altDiphoModel.model'
modelName = '/vols/cms/es811/Stage1categorisation/Pass1/2017/models/altDiphoModel.model'
xmlName = modelName.split('/')[-1].replace('.model','.xml')
altDiphoModel.load_model(modelName)
print 'Loaded model called %s'%modelName.split('/')[-1]
weightDir = 'WeightFiles/'
if '2016' in modelName:
  weightDir += '2016'
elif '2017' in modelName:
  weightDir += '2017'
else:
  exit('expected year 2016 or 2017 in path')
if not path.isdir(weightDir): 
  system('mkdir -p %s'%weightDir)

#convert!
from mlglue.tree import tree_to_tmva, BDTxgboost, BDTsklearn
target_names = ['bkg','sig']
bdt = BDTxgboost(altDiphoModel, diphoVars, target_names, kind='binary', max_depth=6, learning_rate=0.3)
bdt.to_tmva('%s/%s'%(weightDir,xmlName))
print 'Created xml called %s'%xmlName
