from exceptions import CustomException
from logger import logging
from utils import save_object
import os
import sys
from sklearn.decomposition import PCA
import numpy as np

from dataclasses import dataclass
import json

@dataclass
class ModelTrainerConfig:
    processed_image_file_path=os.path.join('artifacts/processed', 'processed_image.npy')
    model_results_file_path = os.path.join("artifacts/results", "model_results.json")
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def fit(self, ncomps):
        '''
        Applies PCA to the image after extending it batchwise

        Parameters
        ----------
        ncomps : int
            Number of components to be considered for the PCA.

        Returns
        -------
        Scores: numpy ndarray
            Coordinates of the extended image after projecting it on a reduced
            subspace
        Loadings: numpy ndarray
            Set of vectors that define the directions of the principal components
            in the new subspace
        Rsquare: list
            Accumulated explained variance for each component

        '''

        try:
            logging.info("Starting model training")

            X_scaled = np.load(self.model_trainer_config.processed_image_file_path)

            pca_object = PCA(n_components=ncomps)

            scores = pca_object.fit_transform(X_scaled)

            results_dict = {"rsquare": np.cumsum(pca_object.explained_variance_ratio_).tolist(),
                       "explained_variance": pca_object.explained_variance_ratio_.tolist(),
                       "eigenvals": np.power(pca_object.singular_values_, 2).tolist(),
                       "component_variance": np.var(scores, axis=0).tolist(),
                       "scores": scores.tolist(),
                       "loadings": pca_object.components_.tolist()}

            # self.reconstructed_training_X, self.residuals, _ = self.reconstruct(self.extended_image)

            logging.info("Training of the model completed")

            # Save the PCA object after training
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=pca_object)
            
            logging.info("Saved model object")

            # Save the results of the model
            json_object = json.dumps(results_dict, indent=4)
            with open(self.model_trainer_config.model_results_file_path, "w") as outfile:
                outfile.write(json_object)
            logging.info("Saved results of model")
            
            return results_dict['rsquare']
        
        except Exception as e:
            raise CustomException(e, sys)