import os
import pandas as pd

class RFMPredictor:

    def __init__(self, data_dir: str= '../../../data/processed'):
        """
        Initializes the predictor by loading the pre-calculated risk profiles.
        :param data_dir: Path to the data directory.
        """

        # Customer Risk Profiles File path:
        self.data_path= os.path.abspath(os.path.join(data_dir, 'customer_risk_profiles.csv'))

        if not os.path.exists(self.data_path):
            raise FileNotFoundError('Customer Risk Profiles not found. Please run the RFM Training Notebook first.')

        # Loading Dataframe and Converting it into Dictionary:
        df= pd.read_csv(self.data_path)
        self.profiles= df.set_index('customer_unique_id')['risk_profile'].to_dict()
        print('RFM Profiles Loaded Successfully.')

if __name__ == '__main__':
    predictor= RFMPredictor(data_dir='../../../data/processed')