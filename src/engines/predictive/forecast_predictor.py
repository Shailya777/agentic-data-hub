import os
import joblib
import pandas as pd
from datetime import datetime

class ForecastPredictor:

    def __init__(self, models_dir: str= '../../../models', data_dir: str= '../../../data/processed'):
        """
        Initializes the predictor by loading the serialized XGBoost pipelines
        and calculating the most recent historical state for feature generation.
        :param models_dir: Path where the models are stored.
        :param data_dir: Path where the data is stored.
        """

        # Loading Models:
        rev_model_path= os.path.abspath(os.path.join(models_dir, 'revenue_forecast_pipeline.pkl'))
        inv_model_path= os.path.abspath(os.path.join(models_dir, 'inventory_forecast_pipeline.pkl'))

        print(rev_model_path)
        print(inv_model_path)
        if not os.path.exists(rev_model_path) or not os.path.exists(inv_model_path):
            raise FileNotFoundError('Forecast Models not found. Please run the Training Notebooks first.')

        print('Models Loaded Successfully')

        self.rev_model= joblib.load(rev_model_path)
        self.inv_model= joblib.load(inv_model_path)

        # Building Historical Data Required for Auto-Regressive / LAG Features:
        self.data_path= os.path.join(data_dir, 'forecasting_training_data.csv')
        self.latest_state= self._build_latest_state()

    def _build_latest_state(self):
        return 'Successful'

if __name__ == '__main__':
    predictor = ForecastPredictor(models_dir= '../../../models', data_dir= '../../../data/processed')

