New repo to generate a set of functions to apply machine learning and Deep learning models to raster images within python.

The end goal is to generate a python package that makes it easy to apply models in similar fashion to scikit learn e.g.:

r_out = rasterPredict(mod,raster_variables)

where mod is a general model object and raster_variables a rasterio or file name referring to a raster image.

