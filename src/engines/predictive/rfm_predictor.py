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

    def predict(self, customer_unique_id: str) -> dict:
        """
        Takes a customer_unique_id and returns their churn risk profile.
        :param customer_unique_id: Unique customer ID.
        :return: Dictionary of churn risk profile.
        """
        clean_id= customer_unique_id.strip()

        if clean_id not in self.profiles:
            return {
                'status': 'error',
                'message': f"Customer ID '{clean_id}' not found in the database."
            }

        risk_profile= self.profiles[clean_id]
        return {
            'status': 'success',
            'customer_unique_id': clean_id,
            'metric': 'Churn Risk Profile',
            'value': risk_profile
        }

if __name__ == '__main__':
    # Test:
    predictor= RFMPredictor(data_dir='../../../data/processed')
    print(predictor.predict(customer_unique_id='248ffe10d632bebe4f7267f1f44844c9'))