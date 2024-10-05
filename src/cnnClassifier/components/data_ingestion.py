import os
import urllib.request as request
import zipfile
import kaggle
import json
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from pathlib import Path
from cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def setup_kaggle_auth(self):
        kaggle_json_path = os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')
        if not os.path.exists(kaggle_json_path):
            raise FileNotFoundError(f"Kaggle API credentials not found at {kaggle_json_path}. Please set up your Kaggle API credentials.")
        
        # Memeriksa izin file
        if os.name != 'nt':  # Untuk sistem non-Windows
            if os.stat(kaggle_json_path).st_mode & 0o777 != 0o600:
                os.chmod(kaggle_json_path, 0o600)
                logger.warning(f"Changed permissions of {kaggle_json_path} to 0600 for security.")
        
        # Menyiapkan environment variables untuk Kaggle API
        with open(kaggle_json_path, 'r') as f:
            kaggle_cred = json.load(f)
        os.environ['KAGGLE_USERNAME'] = kaggle_cred['username']
        os.environ['KAGGLE_KEY'] = kaggle_cred['key']
        
        logger.info("Kaggle authentication setup completed.")

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            if self.config.kaggle_dataset:
                self.setup_kaggle_auth()  # Menyiapkan autentikasi sebelum mengunduh
                kaggle.api.dataset_download_files(
                    self.config.kaggle_dataset,
                    path=os.path.dirname(self.config.local_data_file),
                    unzip=False
                )
                logger.info(f"Dataset downloaded from Kaggle: {self.config.kaggle_dataset}")
            elif self.config.source_URL:
                filename, headers = request.urlretrieve(
                    url = self.config.source_URL,
                    filename = self.config.local_data_file
                )
                logger.info(f"{filename} download! with following info: \n{headers}")
            else:
                logger.info("No Kaggle dataset or source URL specified. Using local file.")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

