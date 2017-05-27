# -*- coding: utf-8 -*-
"""
Spyder Editor.
"""
#################################### Raster Modeling #######################################
######################## Analyze and predict air temperature with Earth Observation data #######################################
#This script performs analyses to predict air temperature using several coveriates.
#The goal is to predict air temperature using Remotely Sensing data as well as compare measurements
# from the ground station to the remotely sensed measurements.
#
#AUTHORS: Benoit Parmentier
#DATE CREATED: 05/12/2019
#DATE MODIFIED: 05/27/2019
#Version: 1
#PROJECT: General utlity to apply model to raster
#TO DO:
#
#COMMIT: clean up code for workshop
#
#################################################################################################

###### Library used in this script
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns
import rasterio
import subprocess
import pandas as pd
import os, glob
from rasterio import plot
import geopandas as gpd
import georasters as gr
import gdal
import rasterio
import descartes
import pysal as ps
from cartopy import crs as ccrs
from pyproj import Proj
from osgeo import osr
from shapely.geometry import Point
from collections import OrderedDict
import webcolors
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

################ NOW FUNCTIONS  ###################

##------------------
# Functions used in the script
##------------------

def create_dir_and_check_existence(path):
    #Create a new directory
    try:
        os.makedirs(path)
    except:
        print ("directory already exists")

def modelGenerate(avg_df,selected_features,selected_target,mod=None,prop=0.3,random_seed=100):
    #Function to fit a regressio model given a data frame

    X_train, X_test, y_train, y_test = train_test_split(avg_df[selected_features], 
                                                    avg_df[selected_target], 
                                                    test_size=prop, 
                                                    random_state=random_seed)
    
    if mod==None:
        from sklearn.linear_model import LinearRegression
        mod = LinearRegression().fit(X_train,y_train)

    mod.fit(X_train, y_train)

    y_pred_train = mod.predict(X_train) # Note this is a fit!
    y_pred_test = mod.predict(X_test) # Note this is a fit!

    #r2_val_train = regr.score(X_train, y_train) #coefficient of determination (R2)
    #r2_val_test = regr.score(X_test, y_test)

    from sklearn import metrics
    #https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics

    mae_val_test = metrics.mean_absolute_error(y_test, y_pred_test) #MAE
    rmse_val_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)) #RMSE
    mae_val_train = metrics.mean_absolute_error(y_train, y_pred_train) #MAE
    rmse_val_train = np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)) #RMSE
    
    data = np.array([[mae_val_test,rmse_val_test,r2_val_test],
                     [mae_val_train,rmse_val_train,r2_val_train]])
    data_metrics_df = pd.DataFrame(data,columns=['mae','rmse','r2'])
    data_metrics_df['test']=[1,0]
    
    X_test['test'] = 1
    X_train['test'] = 0
    y_test['test'] = 1
    y_train['test'] = 0
    
    X = pd.concat([X_train,X_test],sort=False)
    y = pd.concat([y_train,y_test],sort=False)
    ### return a tuple, could be a dict or list?
    
    residuals_val_test = y_test[selected_target] - y_pred_test
    residuals_val_train = y_train[selected_target] - y_pred_train
    residuals_val_test['test'] = 1   
    residuals_val_train['test'] = 0   
        
    residuals_df = pd.concat([residuals_val_test,residuals_val_train],sort=False)
    
    return X, y, mod, residuals_df,data_metrics_df


def rasterPredict(mod,rast_in,dtype_val=None,out_filename=None,out_dir=None):
    
    out_filename = os.path.join(out_dir,out_filename)
    
    ## Check for type:
    
    src_RP1 = rasterio.open(rast_in)
    #src_RP2 = rasterio.open(os.path.join(in_dir,infile_lst_month7))
    
    #Option  to set the dtype from the predicted val??
    
    
    #window = src_RP1.block_windows(1)
    #block_array = src_RP1.read(window=window)
    #RP1_block = src_RP1.read(window=window, masked=True)
    #shape_block = RP1_block.shape
    #RP1_block = RP1_block.ravel() #only sample of the form (20,), missing feature
    #RP1_block.reshape(-1,1)
 
    #result_block = mod.predict(RP1_block.reshape(-1,1)) # Note this is a fit!
    #result_block.dtype
    
    ## Check if file exists first, still a problem here
    exists = os.path.isfile(out_filename)
    if exists:
        print("File already exists, removing file")
        os.remove(out_filename)
        
        out_profile = src_RP1.profile.copy()
        
        if dtype_val!= None:
            out_profile['dtype']=dtype_val
            nodata_val = rasterio.dtypes.dtype_ranges[dtype_val][0] #take min val of range
            out_profile['nodata']=nodata_val
            
        dst = rasterio.open(out_filename, 
                        'w', 
                        **out_profile)        
    else:
        print("creating file")
        out_profile = src_RP1.profile.copy()
        if dtype_val!= None:
            out_profile['dtype']=dtype_val
            nodata_val = rasterio.dtypes.dtype_ranges[dtype_val][0] #take min val of range
            out_profile['nodata']=nodata_val
        dst = rasterio.open(out_filename, 
                        'w', 
                        **out_profile)
        
    pdb.set_trace() #this sets up break points
    #improve this for multiprocessing!!
    for block_index, window in src_RP1.block_windows(1):
        RP1_block = src_RP1.read(window=window, masked=True)
        shape_block = RP1_block.shape
        RP1_block = RP1_block.ravel() #only sample of the form (20,), missing feature
        #RP1_block.reshape(-1,1)
        #RP2_block = src_RP2.read(window=window, masked=True)

        result_block = mod.predict(RP1_block.reshape(-1,1)) # Note this is a fit!
        result_block = result_block.reshape(shape_block)
        
        
        dst.write(result_block, window=window)
    
    
    #pdb.set_trace()
    
    src_RP1.close()
    #src_RP2.close()
    dst.close()
    
    
    return(out_filename)

############################################################################
#####  Parameters and argument set up ###########

#ARGS 1
in_dir = "/home/bparmentier/Data/Benoit/python/rasterModeling/data"
#ARGS 2
out_dir = "/home/bparmentier/Data/Benoit/python/rasterModeling/outputs"

#in_dir="/nfs/bparmentier-data/Data/workshop_spatial/climate_regression/data/Oregon_covariates"
#out_dir="/nfs/bparmentier-data/Data/workshop_spatial/climate_regression/outputs"

#ARGS 3:
create_out_dir=True #create a new ouput dir if TRUE
#ARGS 7
out_suffix = "rasterModeling_test_05162019" #output suffix for the files and ouptut folder
#ARGS 8
NA_value = -9999 # NA flag balue
file_format = ".tif"

#NLCD coordinate reference system: we will use this projection rather than TX.
CRS_reg = "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
method_proj_val = "bilinear" # method option for the reprojection and resampling
gdal_installed = True #if TRUE, GDAL is used to generate distance files

#epsg 2991
crs_reg = "+proj=lcc +lat_1=43 +lat_2=45.5 +lat_0=41.75 +lon_0=-120.5 +x_0=400000 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"

#infile = "mean_month1_rescaled.rst" # mean LST for January
infile_lst_month1 = "lst_mean_month1_rescaled.tif" 
infile_lst_month7 = "lst_mean_month7_rescaled.tif" 

infile_forest_perc =""
ghcn_filename = "ghcn_or_tmax_covariates_06262012_OR83M.shp" # climate stations

prop = 0.3
random_seed= 100

################# START SCRIPT ###############################

######### PART 0: Set up the output dir ################

#set up the working directory
#Create output directory

if create_out_dir==True:
    #out_path<-"/data/project/layers/commons/data_workflow/output_data"
    out_dir_new = "output_data_"+out_suffix
    out_dir = os.path.join(out_dir,out_dir_new)
    create_dir_and_check_existence(out_dir)
    os.chdir(out_dir)        #set working directory
else:
    os.chdir(create_out_dir) #use working dir defined earlier

###########################################
### PART I: READ AND VISUALIZE DATA #######

data_gpd = gpd.read_file(os.path.join(in_dir,ghcn_filename))

data_gpd.head()  

## Extracting information from raster using raster io object
lst1 = rasterio.open(os.path.join(in_dir,infile_lst_month1))
lst7 = rasterio.open(os.path.join(in_dir,infile_lst_month7))
type(lst1)
lst1.crs # explore Coordinate Reference System 
lst1.shape
lst1.height
plot.show(lst1)
plot.show(lst7)

## Read raster bands directly to Numpy arrays and visualize data
r_lst1 = lst1.read(1,masked=True) #read first array with masked value, nan are assigned for NA
r_lst7 = lst7.read(1,masked=True) #read first array with masked value, nan are assigned for NA

type(r_lst1)
r_lst1.size

r_diff = r_lst7 - r_lst1

plt.imshow(r_diff) # other way to display data
plt.title("Difference in land surface temperature between January and July ", fontsize= 20)
plt.colorbar()

# Explore values distribution
plt.hist(r_lst1.ravel(),
         bins=256,
         range=(259.0,287.0))
## Add panel figures later
         
##### Combine raster layer and geogpanda layer

station_or = gpd.read_file(os.path.join(in_dir,ghcn_filename))

##### How to combine plots with rasterio package
fig, ax = plt.subplots()
rasterio.plot.show(lst1,ax=ax,
                          clim=(259.0,287.0),)
station_or.plot(ax=ax,marker="*",
              color="red",
               markersize=10)

#ax.set_title("Long term mean for January land surface temperature", fontsize= 20)
#fig.colorbar(ax)
               
###########################################
### PART II : Extract information from raster and prepare covariates #######
#raster = './data/slope.tif'

lst1_gr = gr.from_file(os.path.join(in_dir,infile_lst_month1))
lst7_gr = gr.from_file(os.path.join(in_dir,infile_lst_month7))
type(lst1_gr) # check that we have a georaster object
# Plot data

#### Extract information from raster using coordinates
x_coord = station_or.geometry.x # pands.core.series.Series
y_coord = station_or.geometry.y
# Find value at point (x,y) or at vectors (X,Y)
station_or['LST1'] = lst1_gr.map_pixel(x_coord,y_coord)
station_or['LST7'] = lst7_gr.map_pixel(x_coord,y_coord)

station_or.columns #get names of col

station_or['year'].value_counts()
station_or.groupby(['month'])['value'].mean() # average by stations per month
     
print("number of rows:",station_or.station.count(),", number of stations:",len(station_or.station.unique()))
#station_or['LST1'] = station_or['LST1'] - 273.15 #create new column
#station_or['LST7'] = station_or['LST7'] - 273.15 #create new column

station_or_jan = station_or.loc[(station_or['month']==1) & (station_or['value']!=-9999)]
station_or_jul = station_or.loc[(station_or['month']==7) & (station_or['value']!=-9999)]

station_or_jan.head()
station_or_jan.columns
station_or_jan.shape

#avg_df = station_or.groupby(['station'])['value'].mean())
avg_jan_df = station_or_jan.groupby(['station'])['value','LST1','LST7'].mean()
avg_jul_df = station_or_jul.groupby(['station'])['value','LST1','LST7'].mean()

avg_jan_df.head()
avg_jan_df.shape
avg_jul_df.shape
avg_jan_df.head()
avg_jul_df.head()




### rescaling:
lst1_gdal = gdal.Open(os.path.join(in_dir,infile_lst_month1))
#gdal.Warp("dem_md.tif", dem_cropped_baltimore,dstSRS=crs_reg)
#dem_cropped_baltimore = None


import os

#gdal_path = 'whatever/path/gdal/is/installed/at'
#gdal_calc_path = os.path.join(gdal_path, 'gdal_calc.py')
gdal_calc_path =  os.path.join('/usr/bin','gdal_calc.py')

# Arguements.
input_file = os.path.join(in_dir,infile_lst_month1)
output_file = os.path.join(out_dir,'rescaled_lst1.tif')
calc_expr = '"A - 273.15"'
#nodata = '0'
#typeof = '"Byte"'

# Generate string of process.
#gdal_calc_str = 'python {0} -A {1} --outfile={2} --calc={3} --NoDataValue={4} --type={5}'
#gdal_calc_process = gdal_calc_str.format(gdal_calc_path, input_file_path, 
#    output_file_path, calc_expr, nodata, typeof)

gdal_calc_str = 'python {0} -A {1} --outfile={2} --calc={3} '
gdal_calc_process = gdal_calc_str.format(gdal_calc_path, input_file, 
    output_file, calc_expr)

#gdal_calc.py -A input1.tif -B input2.tif --outfile=result.tif --calc="A+B"

# Call process.
os.system(gdal_calc_process)

test = rasterio.open(output_file)
plot.show(test,clim=(-10,50))

############
###  PART III : Fit model and generate prediction

avg_jan_df['T1'] = avg_jan_df['value']/10
avg_jul_df['T7'] = avg_jul_df['value']/10
         
####################################
### Add split training and testing!!!
### Add additionl covariates!!
#selected_covariates_names_updated = selected_continuous_var_names + names_cat 
selected_features = ['LST1'] #selected features
selected_target = ['T1'] #selected dependent variables
## Split training and testing


X_train, X_test, y_train, y_test = train_test_split(avg_jan_df[selected_features], 
                                                    avg_jan_df[selected_target], 
                                                    test_size=prop, 
                                                    random_state=random_seed)
   
X_train.shape

regr = LinearRegression() #create/instantiate object used for linear regresssion
regr.fit(X_train,y_train) #fit model

y_pred_train = regr.predict(X_train) # Note this is a fit!
y_pred_test_regr = regr.predict(X_test) # Note this is a fit!
y_pred_test_regr.dtype
plt.hist(y_pred_test_regr)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = random_seed)
# Train the model on training data
rf.fit(X_train,y_train);
y_pred_train_rf = rf.predict(X_train);
y_pred_test_rf = rf.predict(X_test);
y_pred_train_rf.dtype #float32, need to convert to match data type of input raster!!!!
plt.hist(y_pred_test_rf)


from sklearn import svm
#X = [[0, 0], [2, 2]]
#y = [0.5, 2.5]
clf = svm.SVR()
clf.fit(X_train,y_train) 
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
    gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
    tol=0.001, verbose=False)
#clf.predict([[1, 1]])
#array([1.5])

from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

#x = np.arange(0.0, 1, 0.01).reshape(-1, 1)
#y = np.sin(2 * np.pi * x).ravel()

#reg = MLPRegressor(hidden_layer_sizes=(10,),  activation='relu', solver='adam',    alpha=0.001,batch_size='auto',
#               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
#               random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
#               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
#               epsilon=1e-08)

reg = MLPRegressor(max_iter=1500)
reg = reg.fit(X_train,y_train)

#### Model evaluation

#https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
r2_val_test = regr.score(X_test, y_test)
mae_val_test = metrics.mean_absolute_error(y_test, y_pred_test) #MAE
rmse_val_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)) #RMSE

r2_val_train = regr.score(X_train, y_train) #coefficient of determination (R2)
rmse_val_train = np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)) #RMSE
mae_val_train = metrics.mean_absolute_error(y_test, y_pred_test)

data = np.array([[mae_val_test,rmse_val_test,r2_val_test],[mae_val_train,rmse_val_train,r2_val_train]])
    
data_metrics_df = pd.DataFrame(data,columns=['mae','rmse','r2'])
data_metrics_df['test']=[1,0]
#metrics.r2_scores(y_test, y_pred_test)

plt.scatter(X_train, y_train,  color='black')
plt.plot(X_train, regr.predict(X_train), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.xlabel("LST temperature")
plt.ylabel("Air temperature")
plt.title("Regression between satellite based temperature and air temperature")
plt.show()

print('reg coef',regr.coef_)
print('reg intercept',regr.intercept_)

selected_features = ['LST1'] #selected features
selected_target = ['T1'] #selected dependent variables

fit_ols_jan = fit_ols_reg(avg_df=avg_jan_df,
            selected_features = selected_features,
            selected_target = selected_target,
            prop=0.3,
            random_seed=10)

selected_features = ['LST7'] #selected features
selected_target = ['T7'] #selected dependent variables

fit_ols_jul = fit_ols_reg(avg_df=avg_jul_df,
            selected_features = selected_features,
            selected_target = selected_target,
            prop=0.3,
            random_seed=10)

data_metrics = pd.concat([fit_ols_jan[4],fit_ols_jul[4]])
data_metrics['month'] = [1,1,7,7] 
data_metrics

residuals_jan_df =fit_ols_jan[3]
residuals_jul_df =fit_ols_jul[3]

residuals_jan_df.columns
residuals_jan_df['test'] = residuals_jan_df['test'].astype('category')
outfile = os.path.join(out_dir,"residuals_jan_df_"+out_suffix+".csv")
residuals_jan_df.to_csv(outfile)

residuals_jul_df['test'] = residuals_jul_df['test'].astype('category')
outfile = os.path.join(out_dir,"residuals_jul_df_"+out_suffix+".csv")
residuals_jul_df.to_csv(outfile)
    
#Note that we had to change data type to categorical for the variable used on the x-axis!

f, ax = plt.subplots(1, 2)
sns.boxplot(ax=ax[0],x='test',y='T1',data=residuals_jan_df)#title='January residuals')
ax[0].set(ylim=(-8, 8)) 
ax[0].set(title="Residuals in January") 

sns.boxplot(ax=ax[1],x='test',y='T7',data=residuals_jul_df) #title='July residuals')
ax[1].set(ylim=(-8, 8)) 
ax[1].set(title="Residuals in July") 

plt.tight_layout()

data_metrics.head()

################ APPLY TO RASTER TO PREDICT

#rast_in = rasterio.open(os.path.join(in_dir,rast_in))
rast_in = (os.path.join(in_dir,infile_lst_month1))
#rast_in = output_file

#debugusing pdb
import pdb

dtype_val = None
out_filename = "results_linear_regression6.tif"
l_regr_filename = pdb.runcall(rasterPredict,regr,rast_in,dtype_val,out_filename,out_dir)
l_regr_filename = rasterPredict(regr,rast_in,dtype_val,out_filename,out_dir)
r_regr = rasterio.open(l_regr_filename)
#plot.show(r_regr,clim=(270,310)) 
plot.show(r_regr)

array = r_regr.read()
array.dtype
array.min()
array.max()
np.histogram(array.ravel())
plt.hist(array.ravel())
plt.hist(array.ravel(),
         bins=256,
         range=(0,20))


#https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
#dtype_val=None
dtype_val = 'float64'
out_filename = "results_random_forest2.tif"
test= pdb.runcall(rasterPredict,rf,rast_in,dtype_val,out_filename,out_dir)
r_rf = rasterio.open(test)

plot.show(r_rf)
plot.show(r_rf,clim=(0,20))

array = r_rf.read()
array.dtype
array.min()
array.max()

np.histogram(array.ravel())
plt.hist(array.ravel())
plt.hist(array.ravel(),
         bins=256,
         range=(0,15))


#https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
#dtype_val=None

array = r_mlp.read()
array.dtypedtype_val = 'float64'
out_filename = "results_mlp.tif"
mlp_filename = pdb.runcall(rasterPredict,reg,rast_in,dtype_val,out_filename,out_dir)
r_mlp = rasterio.open(mlp_filename)

plot.show(r_mlp)
plot.show(r_mlp,clim=(0,20))
plot.show(r_mlp,clim=(3,12))

array.min()
array.max()

np.histogram(array.ravel())
plt.hist(array.ravel())
plt.hist(array.ravel(),
         bins=256,
         range=(0,20))

############################# END OF SCRIPT ###################################


###### Additional information ######

# #LAND COVER INFORMATION (Tunamu et al., Jetz lab)

# LC1: Evergreen/deciduous needleleaf trees
# LC2: Evergreen broadleaf trees
# LC3: Deciduous broadleaf trees
# LC4: Mixed/other trees
# LC5: Shrubs
# LC6: Herbaceous vegetation
# LC7: Cultivated and managed vegetation
# LC8: Regularly flooded shrub/herbaceous vegetation
# LC9: Urban/built-up
# LC10: Snow/ice
# LC11: Barren lands/sparse vegetation
# LC12: Open water

## LST information: mm_01, mm_02 ...to mm_12 are monthly mean LST at station locaitons
## LST information: nobs_01, nobs_02 ... to nobs_12 number of valid obs used in mean LST averages
## TMax : monthly mean tmax at meteorological stations
## nbs_stt: number of stations used in the monthly mean tmax at stations