import pandas as pd
from sklearn.model_selection import StratifiedKFold
from helpers import get_settings
config = get_settings()

class DataController:

    def __init__(self):
        super().__init__()
        self.label_column = 'score'

    # if needed
    def make_test_set_skf_out(self,df, part_size = config.TEST_SIZE):
        # Initialize StratifiedKFold with 1 split (for taking one part)
        skf = StratifiedKFold(n_splits=int(len(df) / part_size))
    
        # Get the indices for the first fold
        for _, test_index in skf.split(df, df[self.label_column]):
            part_indices = test_index
            break
    
        # Select the part DataFrame
        test = df.iloc[part_indices]
    
        # Drop the selected part from the original DataFrame
        train = df.drop(part_indices)
    
        return test, train
    
