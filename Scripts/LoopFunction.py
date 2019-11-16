#Importing Packages

import pandas as pd
import numpy as np
import pyarrow as pa
#from tqdm import tqdm, trange, tqdm_notebook
from time import sleep
from math import sqrt
import glob
import os
import re

from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

from sklearn.pipeline import make_pipeline
from sklearn import datasets


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scaler = StandardScaler()

import pandas as pd
from sklearn import preprocessing
import boto3
import io
import pandas as pd
import pyarrow.parquet as pq

from pandas import DataFrame
import pyspark
from pyspark.sql import SparkSession
import plotly
#from pyspark.ml.clustering import KMeans
#from pyspark.ml.evaluation import ClusteringEvaluator
#from pyspark.ml.feature import VectorAssembler
#from pyspark.sql import SQLContext
#from pyspark import SparkContext
#from pyspark import SparkConf
#from pyspark.context import SparkContext

from os import listdir
from os.path import isfile, join
from sklearn.externals import joblib
from PIL import Image, ImageDraw
from CCfunctions import *


# \\Users\\kunal\\Desktop\\WORK\\Datathon\\Phase02-DataDelivery

# Importing 
sugarcanetiles = 'C:\\Users\\kunal\\Desktop\\WORK\\Datathon\\Phase02-DataDelivery\\sugarcanetiles'
output_image_folder = 'C:\\Users\\kunal\\Desktop\\WORK\\Datathon\\Images\\'
output_parquet_folder = 'C:\\Users\\kunal\\Desktop\\WORK\\Datathon\\Parquets\\'

onlyfiles = [f for f in listdir(sugarcanetiles) if isfile(join(sugarcanetiles, f))]

# load Model

kms_model_file = '15ImageModel_model.sav'
#joblib.dump(model, kms_model_file)

Kmean = joblib.load(kms_model_file)

# where Unclean means original image (Tile x Time), where an image is a passing of the satelight over a tile of ground
all_Unclean_Images_DF = pd.DataFrame()
all_Unclean_Images_DF['Unclean'] = onlyfiles
all_Unclean_Images_DF['Unclean'].str.split('-',expand=True)
all_Unclean_Images_DF['X Tile'] =  all_Unclean_Images_DF['Unclean'].str.split('-',expand=True)[0]
all_Unclean_Images_DF['Y Tile'] =  all_Unclean_Images_DF['Unclean'].str.split('-',expand=True)[1]
all_Unclean_Images_DF['TileTuple'] = all_Unclean_Images_DF['X Tile'].astype(str) + " "+ all_Unclean_Images_DF['Y Tile'].astype(str)

# create SelectedDF to contain a set of tiles to be processed  : a tile is a square of ground
SelectedTiles = ['7680 10240']
SelecteDF  = all_Unclean_Images_DF[all_Unclean_Images_DF['TileTuple'].isin(SelectedTiles)]
SelecteDF = SelecteDF[['X Tile','Y Tile','TileTuple']].drop_duplicates()

# create the column headings for the resulting dataframe to be saved as parquet
columns = ["tile_x","tile_y", "x","y", "date", "mask", "red","green","blue"]
columns.extend([f'B{b:02d}' for b in range(1,13)])

min_max_scaler = preprocessing.MinMaxScaler()

# columns to be used for the data model
vegi1InputCols = ['Scaled_NDVI', 'Scaled_LCI', 'Scaled_LAI', 'Scaled_GNDVI', 'Scaled_SCI']

for index,row in SelecteDF.iterrows():
    TILE_X = row['X Tile']
    TILE_Y = row['Y Tile']

    (start_x, start_y) = (0, 0)
    (size_x, size_y) = (512, 512)
    cropbox = (start_x, start_y, start_x + size_x, start_y + size_y)
    
    mask_path = get_mask_path(TILE_X, TILE_Y)
    masp = open_image(mask_path, mode = 'P', cropbox = cropbox)
    mask_pixels = pixels_in_mask(7680, 10240)
    tci_list = get_timeseries_image_paths(TILE_X, TILE_Y, 'TCI')

    b_name_list = [f'B{b:02d}' for b in range(1,13)]
    b_path_lol = [get_timeseries_image_paths(TILE_X, TILE_Y, b) for b in b_name_list]

    ModelDF = pd.DataFrame()
    assert len(tci_list) == len(b_path_lol[0])
    (tile_x, tile_y) = (TILE_X, TILE_Y)
    
    
    for day_no in range(0,4): # len(tci_list)):
        tci_path = tci_list[day_no]
        date = last_date_in_path(tci_path)
        b_path_list = [b_path_list[day_no] for b_path_list in b_path_lol]

        tci_img = open_image(tci_path, cropbox = cropbox, verbose=False)
        b_img_list = [open_image(b_path, cropbox = cropbox, verbose=False) for b_path in b_path_list]

        data = read_img_pixel_values(tile_x, tile_y, date, masp, tci_img, *b_img_list)

        df = pd.DataFrame(columns=columns, data=data)

        # Calculate Indices:
        # https://support.micasense.com/hc/en-us/articles/227837307-An-overview-of-the-available-layers-and-indices-in-Atlas
        # https://support.micasense.com/hc/en-us/articles/226531127-Creating-agricultural-indices-NDVI-NDRE-from-an-Atlas-GeoTIFF-in-QGIS-
        # https://earth.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
        # https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/indexdb/
        # NDVI ref: https://medium.com/analytics-vidhya/satellite-imagery-analysis-with-python-3f8ccf8a7c32

        # NDVI - Normalised Difference Vegetation Index (NDVI)
        #  = (NIR - RED) / (NIR + RED)
        df['NDVI'] = df.apply(lambda df : int(10000 * (float(df.B08) - float(df.B04)) / (df.B08 + df.B04)) , axis=1)

        # GNDVI - Green Normalized Difference NDVI
        #  = (NIR - GREEN)/(NIR + GREEN)
        df['GNDVI'] = df.apply(lambda df : int(10000 * (float(df.B08) - float(df.B03)) / (df.B08 + df.B03)) , axis=1)

        # RDVI - Normalised Difference Vegetation Index (NDVI)
        #  = 2*(NIR - RED) / sqrt(NIR + RED)
        df['RDVI'] = df.apply(lambda df : int(10000 * 2 * (float(df.B08) - float(df.B04)) / sqrt(df.B08 + df.B04)) , axis=1)

        # RBNDVI - Red Blue Normalised Difference Vegetation Index (NDVI)
        #  = (NIR - RED -BLUE) / (NIR + RED + BLUE)
        df['RBNDVI'] = df.apply(lambda df : int(10000 * (2*float(df.B08) - float(df.B04) - float(df.B02)) / (2*df.B08 + df.B04 + df.B02)) , axis=1)

        # LCI - Leaf Chlorophyll Index
        #  = (NIR - REDE)/(NIR + REDE)
        df['LCI'] = df.apply(lambda df : int(100 * (float(df.B08) - float(df.B05)) / (df.B08 + df.B05)) , axis=1)

        # LAI - Leaf Area Index
        #  = (REDE - RED)/(REDE + RED)
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3231680/
        df['LAI'] = df.apply(lambda df : int(10000 * (float(df.B05) - float(df.B04)) / (df.B05 + df.B04)) , axis=1)

        # SCI - Soil Composition Index
        #  = (SWIR - NIR)/(SWIR + NIR)
        df['SCI'] = df.apply(lambda df : int(10000 * (float(df.B11) - float(df.B08)) / (df.B11 + df.B08)) , axis=1)

        # SCI - Soil Composition Index
        #  = (SWIR - NIR)/(SWIR + NIR)
        df['SCI'] = df.apply(lambda df : int(10000 * (float(df.B11) - float(df.B08)) / (df.B11 + df.B08)) , axis=1)

        # NDMI - Normalized Difference Moisture Index
        #  = (NIR - SWIR)/(SWIR + NIR)
        df['NDMI'] = df.apply(lambda df : int(10000 * (float(df.B08) - float(df.B11)) / (df.B11 + df.B08)) , axis=1)

        # GLI - Green Leaf Index
        #  = (GREEN - BLUE)/(GREEN + BLUE)
        df['GLI'] = df.apply(lambda df : int(10000 * (float(df.B05) - float(df.B04)) / (df.B05 + df.B04)) , axis=1)

        # ModelDF = pd.concat([ModelDF,df]).reset_index(drop=True)
        df = df[(df['red']<225)&(df['green']<225)&(df['blue']<225)].reset_index(drop=True)

        df['Scaled_NIR'] = min_max_scaler.fit_transform(df[['B08']])
        df['Scaled_RED'] = min_max_scaler.fit_transform(df[['B04']])
        df['Scaled_GRN'] = min_max_scaler.fit_transform(df[['B03']])
        df['Scaled_NDVI'] = min_max_scaler.fit_transform(df[['NDVI']])
        df['Scaled_LCI'] = min_max_scaler.fit_transform(df[['LCI']])
        df['Scaled_LAI'] = min_max_scaler.fit_transform(df[['LAI']])
        df['Scaled_SCI'] = min_max_scaler.fit_transform(df[['SCI']])
        df['Scaled_GNDVI'] = min_max_scaler.fit_transform(df[['GNDVI']])

        df['PixelTuple'] = df['x'].astype(str) + " " + df['y'].astype(str)
        dfMask = df[df['PixelTuple'].isin(mask_pixels)].reset_index(drop=True)
        dfMask['prediction']= Kmean.predict(dfMask[vegi1InputCols])
        tciFilePath = sugarcanetiles + str(TILE_X)+"-"+ str(TILE_Y) + "-TCI-" + date+'.png'
        #tciFilePath = sugarcanetiles + '\\7680-10240-TCI-2017-06-20.png'
        tci = open_image(tciFilePath, mode='RGB')
        numberOfClusters = 2
        vegiClustered_df = dfMask[['x','y','prediction']]

        colours = [
            np.array([0x10,0xAD,0x00], dtype='uint8'),  # limish green
            np.array([0x00,0x26,0xA4], dtype='uint8'),  # blueish
            np.array([0xFF,0x00,0x00], dtype='uint8'),  # red
            np.array([0xFF,0xC6,0x00], dtype='uint8'),  # yellow
            np.array([0x00, 0x00, 0x00],dtype='uint8')
            ]

        imgnp = overlayPredictionImage(vegiClustered_df, tci, colours)
        # imgnp.shape

        clusterImg = Image.fromarray(np.hstack((imgnp, np.array(tci))))

        path = 'C:\\Users\\kunal\\Desktop\\WORK\\Datathon\\Github\\Dash-Application\\Images\\'
        path += f"comp_image_{TILE_X}_{TILE_Y}_{date}.png"
        print(type(clusterImg))
        clusterImg.save(path)
        path = output_parquet_folder
        path += f"image_values_{TILE_X}_{TILE_Y}_{date}.snappy.parquet"
        vegiClustered_df.to_parquet(path)


    p = Pool(50)
    p.map( processTile, [row for index, row in SelecteDF.iterrows()[0:1]])
    