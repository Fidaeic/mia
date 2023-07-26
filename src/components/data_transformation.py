import sys
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from exceptions import CustomException
from logger import logging
import os
from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')
    processed_image_file_path=os.path.join('artifacts', 'processed_image.npy')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.preprocessor_object = StandardScaler()
        

    def batchwise(self, image, neighbours):
        '''
        Generates a new matrix with the newighbouring pixels of every pixel of each band.
        The rows of the outcoming matrix represents pixels of the original image and the
        columns are the values of the pixel and the neighbouring ones, given a number of
        neighbours

        Parameters
        ----------
        neighbours : int
            Defines how many neighbours should be considered to build the matrix.

        Returns
        -------
            numpy ndarray with the bands and the neighbouring pixels extended batchwise
        '''
        logging.info("Batchwise method initialized")

        if neighbours<1:
            raise CustomException("Number of neighbours must be at least 1", sys)
        
        window_width = neighbours+2

        try:

            # Gets the final size of the matrix
            final_size = np.sum(np.square(np.arange(3, window_width+1)))

            # To get the transformed matrix, we create a sliding window for each band. The size of the sliding window will correspond to the number of neighbours

            final_matrix = np.concatenate(
                [np.lib.stride_tricks.sliding_window_view(image[band], 
                                                          window_shape=(window_width, window_width)).reshape(-1, final_size) 
                    for band in range(image.shape[0])], axis=1)
            
            
            logging.info("Image converted into batchwise matrix")

            logging.info("Preprocessing batchwise matrix")

            processed_matrix = self.preprocessor_object.fit_transform(final_matrix)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=self.preprocessor_object
            )

            logging.info("Preprocessor object saved")


            np.save(self.data_transformation_config.processed_image_file_path, processed_matrix)

            logging.info("Processed matrix saved")

        except Exception as e:
            raise CustomException(e, sys)