import re
import polars as pl
from helpers import get_settings
config = get_settings()


class dataPreprocessing:

    def __init__(self):
        self.raw_path = config.RAW_PATH

    
    def load_data(self):
        # to read with paragraph 
        columns = [pl.col("full_text").str.split(by="\n\n").alias("paragraph")]
        train = pl.read_csv(self.raw_path + "train.csv").with_columns(columns)
        test = pl.read_csv(self.raw_path + "train.csv").with_columns(columns)
        return train, test

    def remove_HTML(self,x):
        html=re.compile(r'<.*?>')
        return html.sub(r'',x) 
    
    def data_clean(self,x):
        # Convert words to lowercase
        x = x.lower()
    
        # Remove HTML
        x = self.remove_HTML(x)

        # Delete strings starting with @
        x = re.sub("@\w+", '',x)

        # Delete Numbers
        x = re.sub("'\d+", '',x) # can delete it
        x = re.sub("\d+", '',x)

        # Delete URL
        x = re.sub("http\w+", '',x)

        # Replace consecutive empty spaces with a single space character
        x = re.sub(r"\s+", " ", x)

        # Replace consecutive commas and periods with one comma and period character
        x = re.sub(r"\.+", ".", x)
        x = re.sub(r"\,+", ",", x)

        # Remove empty characters at the beginning and end
        x = x.strip()
        return x
    
    def full_text_preprocess(self,df):
        return df.with_columns(pl.col('full_text').map_elements(self.data_clean,return_dtype=str))
    
    def paragraph_preprocess(self,df):
        return df.with_columns(pl.col('paragraph').map_elements(self.data_clean,return_dtype=str))
