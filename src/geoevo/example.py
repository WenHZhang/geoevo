from geoevo import GeoEvoOptimizer
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import random
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def objfunc(zone_data, zone_val, param):
    feature_name = ['landusenum', 'Main_soil', 'Slope', 'gS', 'mS', 'fS', 'gU', 'mU', 'fU', 'Sand', 'Silt',
                    'Clay', 'pH_H2O', 'EC_H2O', 'TC', 'TOC', 'TN', 'FSS', 'Water_cont', 'Stone_dens', 'BD_bulk',
                    'Rock_fragm']
    var_name = ['CS_0_30']
    zone_data['Main_soil'] = pd.factorize(zone_data['Main_soil'])[0]
    zone_data['Slope'] = pd.factorize(zone_data['Slope'])[0]
    zone_val['Main_soil'] = pd.factorize(zone_val['Main_soil'])[0]
    zone_val['Slope'] = pd.factorize(zone_val['Slope'])[0]
    model = RandomForestRegressor(n_estimators=int(np.ceil(param[0])), max_features=int(np.ceil(param[1])), oob_score=False)
    model.fit(zone_data[feature_name], zone_data[var_name])
    p = model.predict(zone_val[feature_name])
    r2 = r2_score(zone_val[var_name], p)
    return r2

if __name__ == '__main__': 
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    ## load data
    train_df = pd.read_csv('train_soc.csv')
    train_df, val_df = train_test_split(train_df, test_size=0.3, random_state=seed)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    train_df.rename(columns={'County': 'zone'}, inplace = True)
    val_df.rename(columns={'County': 'zone'}, inplace = True)
    zoneslist = train_df['zone'].unique()
    adjmat = pd.read_csv('adj_soc.csv', index_col=0)
    feature_name = ['landusenum', 'Main_soil', 'Slope', 'gS', 'mS', 'fS', 'gU', 'mU', 'fU', 'Sand', 'Silt',
                    'Clay', 'pH_H2O', 'EC_H2O', 'TC', 'TOC', 'TN', 'FSS', 'Water_cont', 'Stone_dens', 'BD_bulk',
                    'Rock_fragm']
    ## start optimization
    print('begin init')
    ## problem definition
    bounds = [(100,500),
            (1,len(feature_name))]
    dim = len(bounds)

    popsize = 5
    its = 2

    optimizer = GeoEvoOptimizer(popsize, its, dim, objfunc, 1, zoneslist, adjmat, True)
    best_param, best_obj = optimizer.optimize(bounds, train_df, val_df)
    print(f'The best hyperparameter is\n {best_param}\n with objective value\n {best_obj}')

    

