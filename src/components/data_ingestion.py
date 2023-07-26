import os
import sys
print(os.getcwd())
from exceptions import CustomException
from logger import logging
from dataclasses import dataclass

import rasterio as rio
import numpy as np

@dataclass #This allows to define the data class of any variable
class DataIngestionConfig:

    raw_data_path:str=os.path.join('artifacts', "image.png")

class DataIngestion:
    def __init__(self):
          self.ingestion_config = DataIngestionConfig() #The previous paths are stored in this attribute

    def read_image(self):
            '''
            Reads image from path and returns it as an array along with the number of rows and columns

            Returns
            -------
            Original image as numpy ndarray
            Number of rows and columns.

            '''

            logging.info("Starting data ingestion")

            try:
                # Read the image from the repository
                image = rio.open(self.ingestion_config.raw_data_path).read()

                # Bands that have 0 variance will be dropped because they do not add any info
                droppable_bands = np.where(np.var(image, axis=(1, 2))==0)[0]

                image = np.delete(image, droppable_bands, axis=0)

                rows = image.shape[1]
                columns = image.shape[2]

                logging.info("Ingestion of data completed")

                return (image,
                        rows,
                        columns)

            except Exception as e:
                raise CustomException(e, sys)
        


    