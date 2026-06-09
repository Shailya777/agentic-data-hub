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
        print('Latest State Loaded Successfully')
        print(self.latest_state)

    def _build_latest_state(self) -> pd.DataFrame:
        """
        Reads the raw data, aggregates it to weekly buckets, calculates the EWMA,
        and isolates the absolute last week of data for each category to use as
        the baseline for future predictions.
        :return: Dataframe of latest weekly stats for each category.
        """

        df= pd.read_csv(self.data_path)
        df['sale_date']= pd.to_datetime(df['sale_date'])

        # Re-Sampling to Weekly Buckets:
        weekly_df= df.groupby('category').resample('W', on= 'sale_date').agg(
            {
                'daily_revenue': 'sum',
                'units_sold': 'sum'
            }
        ).reset_index()

        # Renaming Columns:
        weekly_df.rename(columns={
            'daily_revenue':'weekly_revenue',
            'units_sold':'weekly_units',
            'sale_date':'week-ending_date'
        }, inplace= True)

        # Calculating Current EWMA State (No shift() needed here because we WANT the current week's state)
        weekly_df['ewma_4_revenue']= weekly_df.groupby('category')['weekly_revenue'].transform(
            lambda x: x.ewm(span= 4, adjust= False).mean()
        )

        weekly_df['ewma_4_units']= weekly_df.groupby('category')['weekly_units'].transform(
            lambda x: x.ewm(span= 4, adjust= False).mean()
        )

        # Isolating Most Recent Week for Each Category:
        latest_state= weekly_df.sort_values(by=['week-ending_date']).groupby('category').tail(1).copy()
        return latest_state.set_index('category')

if __name__ == '__main__':
    x= ForecastPredictor()

