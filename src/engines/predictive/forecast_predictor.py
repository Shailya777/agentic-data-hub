import os
import joblib
import pandas as pd
from datetime import timedelta

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
        #print(self.latest_state)

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
            'sale_date':'week_ending_date'
        }, inplace= True)

        # Calculating Current EWMA State (No shift() needed here because we WANT the current week's state)
        weekly_df['ewma_4_revenue']= weekly_df.groupby('category')['weekly_revenue'].transform(
            lambda x: x.ewm(span= 4, adjust= False).mean()
        )

        weekly_df['ewma_4_units']= weekly_df.groupby('category')['weekly_units'].transform(
            lambda x: x.ewm(span= 4, adjust= False).mean()
        )

        # Isolating Most Recent Week for Each Category:
        latest_state= weekly_df.sort_values(by=['week_ending_date']).groupby('category').tail(1).copy()
        return latest_state.set_index('category')

    def predict(self, category: str, forecast_type: str= 'revenue') -> dict:
        """
        Takes a product category and a forecast type (revenue or inventory),
        engineers the temporal features for next week, and returns the prediction.
        :param category: Category to Predict Forecast for.
        :param forecast_type: Revenue or Inventory.
        :return: Dictionary of predictions.
        """

        category_clean= category.lower().replace(' ', '_')

        # Fallback if Category Does not Exist:
        if category_clean not in self.latest_state.index:
            return {
                'status': 'error',
                'message': f"Category {category} not found in Historical Data."
            }

        # Fetching Latest stats for Category:
        state= self.latest_state.loc[category_clean]
        last_date= state['week_ending_date']
        target_date= last_date + timedelta(days=7)

        year= target_date.isocalendar().year
        week_of_year= target_date.isocalendar().week

        # Routing to Correct Model:
        if forecast_type == 'revenue':
            input_df= pd.DataFrame([{
                'category': category_clean,
                'year': year,
                'week_of_year': week_of_year,
                'lag_1_revenue': state['weekly_revenue'],
                'ewma_4_revenue': state['ewma_4_revenue']
            }])
            prediction= self.rev_model.predict(input_df)[0]
            return {
                'status': 'success',
                'category': category,
                'target_date': target_date.strftime('%Y-%m-%d'),
                'metric': 'Projected Revenue',
                'value': f"${prediction:,.2f}"
            }

        elif forecast_type in ['inventory', 'units', 'stock']:
            input_df= pd.DataFrame([{
                'category': category_clean,
                'year': year,
                'week_of_year': week_of_year,
                'lag_1_units': state['weekly_units'],
                'ewma_4_units': state['ewma_4_units']
            }])
            prediction= self.inv_model.predict(input_df)[0]
            return {
                'status': 'success',
                'category': category,
                'target_date': target_date.strftime('%Y-%m-%d'),
                'metric': 'Projected Unit Volume',
                'value': f"{int(prediction)} units"
            }

        else:
            return {
                'status': 'error',
                'message': "Invalid forecast type. Use 'revenue' or 'inventory'."
            }

if __name__ == '__main__':
    # Test:
    predictor= ForecastPredictor(models_dir= '../../../models', data_dir= '../../../data/processed')
    print(predictor.predict(category='health_beauty', forecast_type='revenue'))
    print(predictor.predict(category='health_beauty', forecast_type='inventory'))
