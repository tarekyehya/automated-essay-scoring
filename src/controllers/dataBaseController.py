import pandas as pd
from sklearn.model_selection import StratifiedKFold
from helpers import get_settings
config = get_settings()

class DataController:

    # if needed
    @staticmethod
    def make_test_set_skf_out(part_size = config.TEST_SIZE):
        # Initialize StratifiedKFold with 1 split (for taking one part)
        df = pd.read_csv(config.DF_PATH)
        skf = StratifiedKFold(n_splits=int(len(df) / part_size))
    
        # Get the indices for the first fold
        for _, test_index in skf.split(df, df['score']):
            part_indices = test_index
            break
    
        # Select the part DataFrame
        test = df.iloc[part_indices]
    
        # Drop the selected part from the original DataFrame
        train = df.drop(part_indices)

        # Save data 
        train.to_csv(config.TRAIN_PATH, index=False)
        test.to_csv(config.TEST_PATH, index=False)

        return test, train
    
