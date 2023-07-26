from components.data_ingestion import DataIngestion


if __name__ =="__main__":
    obj=DataIngestion()

    im, row, col = obj.read_image()

    print(im)