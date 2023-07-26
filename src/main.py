from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation


if __name__ =="__main__":
    obj=DataIngestion()
    transf=DataTransformation()

    im, row, col = obj.read_image()

    

    transf.batchwise(im, 1)