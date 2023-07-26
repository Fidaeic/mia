from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_training import ModelTrainer


if __name__ =="__main__":
    obj=DataIngestion()
    transf=DataTransformation()
    model_trainer = ModelTrainer()

    im, row, col = obj.read_image()

    transf.batchwise(im, 1)

    print(model_trainer.fit(2))

