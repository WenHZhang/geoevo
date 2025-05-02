# GeoEvo
A software for Geographic Evolutionary Multi-Task Optimization Framework for Automated Hyperparameter Tuning on Spatially Stratified Machine Learning Models (GeoEvo).

## Introduction
The rapid growth of spatial data across various fields has made data-driven machine learning (ML) methods increasingly essential for spatial analysis. However, the performance of ML methods heavily depends on hyperparameter tuning (HPT), which becomes particularly challenging when dealing with spatial stratified heterogeneity (SSH). SSH causes significant variability in statistical characteristics across sub-regions, requiring separate local models. Traditional HPT methods either apply uniform hyperparameters or treat each model independently, ignoring spatial associations between adjacent models. To address this gap, we propose a geographic evolutionary multi-task optimization framework (GeoEvo) that leverages spatial associations for collaborative HPT of local ML models under SSH. GeoEvo formulates a multi-task constrained single objective optimization problem to jointly optimize hyperparameters across multiple local models. The framework introduces a geographic proximity differential evolution operator (Geo-DE) to enable collaboration among spatially adjacent models and a geographic selection operator (Geo-SL) to promote hyperparameter sharing, improving resource utilization and accelerating convergence.

## Potential Application Scenarios
- **Hyperparameter Tuning for Multiple Spatially Localized Models**  
  Ideal for scenarios with Spatially Stratified Heterogeneity (SSH), including:  
  - Climate zone-specific weather prediction models  
  - Real estate price forecasting across urban sub-regions  
  - Traffic flow analysis with road network segmentation  
  - Agricultural yield prediction in heterogeneous soil regions  

- **Algorithmic Extensions for Spatial Optimization**  
  The framework's evolutionary mechanisms can be adapted to:  
  - Distributed urban infrastructure layout optimization  
  - Ecological monitoring network sensor deployment  
  - Collaborative planning systems for logistics distribution routes  

## Installation:
```bash
$ pip install geoevo
```

## Dependencies
- matplotlib: 3.7.1
- numpy: 2.2.1
- pandas: 2.2.3
- scikit-learn: 1.3.0
- scipy: 1.14.1

## Example
A complete example for using this software can be found at the folder './src/geoevo/example.py' with data.
### Simplified step
**1.Import package:**
```python
from geoevo import GeoEvoOptimizer
import pandas as pd
```
**2.Load data:**
```python
train_df = pd.read_csv('xxx') # training dataset
val_df = pd.read_csv('xxx') # validation dataset
adjmat = pd.read_csv('xxx') # 2D array of spatial adjacent matrix
zoneslist = ['1','2'] # list of region identifier (e.g., ['zone1', 'zone2'])
```
**3.Define problem:**
```python
bounds = [(100,500), (1,len(feature_name))] # the range of hyperparameters
dim = len(bounds)
def objfunc(zone_data, zone_val, param):
    ### User-define objective function ###
```
**4.Initialize algorithm**
```python
popsize = 50 # population size
its = 100 # iteration times
cpu = 1 # Number of CPU cores for parallel processing
optimizer = GeoEvoOptimizer(popsize, its, dim, objfunc, cpu, zoneslist, adjmat, True)
```
**5.Start optimization and output results**
```python
best_param, best_obj = optimizer.optimize(bounds, train_df, val_df)
print(f'The best hyperparameter is\n {best_param}\n with objective value\n {best_obj}')
```

## Citations
If this software is useful for your research or application, please cite our paper:
```bash
default
```
We also recommend our previous works below as basis models for GeoEvo:
```bash
Cheng, S., Zhang, W., Luo, P., Wang, L., & Lu, F. (2024). An explainable spatial interpolation method considering spatial stratified heterogeneity. International Journal of Geographical Information Science, 39(3), 600–626. https://doi.org/10.1080/13658816.2024.2426067
```

## Acknowledgement
The example data is from an open dataset：
```bash
Poeplau, C., Don, A., Flessa, H., Heidkamp, A., Jacobs, A., Prietz, R., 2020. Erste Bodenzustandserhebung Landwirtschaft – Kerndatensatz. https://doi.org/10.3220/DATA20200203151139
```