New repo to generate a set of functions to apply functions, machine learning models and Deep learning models to raster images within python. 

The end goal is to generate a python package that makes it easy to apply functions, operations or models to every pixels in similar fashion to scikit learn or general image algebra e.g.:

r_out = rasterPredict(mod,raster_variables)
r_out = rasterApply(fun, args, raster_in)

where mod is a general model object and raster_variables a rasterio or file name referring to a raster image.

In contrast to reading a full image band in memory, the processing is done by chunk using python generator (via rasterio image chunk) and should be able to handle rasters of very large size.

The goal is to fill gaps to existing raster packages (rasterio, rasterstats and georasters). The focus is on modeling, operations and efficient processing for large rasters (multicore processing to be added).
