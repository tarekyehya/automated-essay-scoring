import polars as pl
from .dataPreprocessing import dataPreprocessing


class featureEngineering:

    @staticmethod
    def Paragraph_eng(tmp):
    
        # Expand the paragraph list into several lines of data
        tmp = tmp.explode('paragraph')
    
        # Paragraph preprocessing -> if clean it before 
        tmp = dataPreprocessing.paragraph_preprocess(tmp)

        # Calculate the length of each paragraph
        tmp = tmp.with_columns(pl.col('paragraph').map_elements(lambda x: len(x),return_dtype=int).alias("paragraph_len"))

        # Calculate the number of sentences and words in each paragraph
        tmp = tmp.with_columns(pl.col('paragraph').map_elements(lambda x: len(x.split('.')),return_dtype=int).alias("paragraph_sentence_cnt"),
                    pl.col('paragraph').map_elements(lambda x: len(x.split(' ')),return_dtype=int).alias("paragraph_word_cnt"),)
    
        paragraph_fea = ['paragraph_len','paragraph_sentence_cnt','paragraph_word_cnt']
    
    
        aggs = [
            # Count the number of paragraph lengths greater than and less than the i-value
            *[pl.col('paragraph').filter(pl.col('paragraph_len') >= i).count().alias(f"paragraph_{i}_cnt") for i in [50,75,100,125,150,175,200,250,300,350,400,500,600,700] ], 
            *[pl.col('paragraph').filter(pl.col('paragraph_len') <= i).count().alias(f"paragraph_{i}_cnt") for i in [25,49]], 
        
            # other
            *[pl.col(fea).max().alias(f"{fea}_max") for fea in paragraph_fea],
            *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in paragraph_fea],
            *[pl.col(fea).min().alias(f"{fea}_min") for fea in paragraph_fea],
            *[pl.col(fea).first().alias(f"{fea}_first") for fea in paragraph_fea],
            *[pl.col(fea).last().alias(f"{fea}_last") for fea in paragraph_fea],
            ]
        df = tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
        df = df.to_pandas()
    
        return df
    

    # sentence feature
    @staticmethod
    def Sentence_eng(tmp):
    
        # use periods to segment sentences in the text
        tmp = tmp.with_columns(pl.col('full_text').str.split(by=".").alias("sentence"))
    
        tmp = tmp.explode('sentence')
        # Calculate the length of a sentence
        tmp = tmp.with_columns(pl.col('sentence').map_elements(lambda x: len(x),return_dtype=int).alias("sentence_len"))

        # Filter out the portion of data with a sentence length greater than 15
        tmp = tmp.filter(pl.col('sentence_len')>=15)

        # Count the number of words in each sentence
        tmp = tmp.with_columns(pl.col('sentence').map_elements(lambda x: len(x.split(' ')),return_dtype=int).alias("sentence_word_cnt"))
    
        sentence_fea = ['sentence_len','sentence_word_cnt']
    
        aggs = [
            # Count the number of sentences with a length greater than i
            *[pl.col('sentence').filter(pl.col('sentence_len') >= i).count().alias(f"sentence_{i}_cnt") for i in [15,50,100,150,200,250,300] ], 
            # other
            *[pl.col(fea).max().alias(f"{fea}_max") for fea in sentence_fea],
            *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in sentence_fea],
            *[pl.col(fea).min().alias(f"{fea}_min") for fea in sentence_fea],
            *[pl.col(fea).first().alias(f"{fea}_first") for fea in sentence_fea],
            *[pl.col(fea).last().alias(f"{fea}_last") for fea in sentence_fea],
            ]
        df = tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
        df = df.to_pandas()
    
        return df


    # word feature
    @staticmethod
    def Word_eng(tmp):
    
        # use spaces to separate words from the text
        tmp = tmp.with_columns(pl.col('full_text').str.split(by=" ").alias("word"))
    
        tmp = tmp.explode('word')

        # Calculate the length of each word
        tmp = tmp.with_columns(pl.col('word').map_elements(lambda x: len(x),return_dtype=int).alias("word_len"))

        # Delete data with a word length of 0
        tmp = tmp.filter(pl.col('word_len')!=0)
    
    
        aggs = [
            # Count the number of words with a length greater than i+1
            *[pl.col('word').filter(pl.col('word_len') >= i+1).count().alias(f"word_{i+1}_cnt") for i in range(15) ], 
            # other
            pl.col('word_len').max().alias(f"word_len_max"),
            pl.col('word_len').mean().alias(f"word_len_mean"),
            pl.col('word_len').std().alias(f"word_len_std"),
            pl.col('word_len').quantile(0.25).alias(f"word_len_q1"),
            pl.col('word_len').quantile(0.50).alias(f"word_len_q2"),
            pl.col('word_len').quantile(0.75).alias(f"word_len_q3"),
        ]
        df = tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
        df = df.to_pandas()
    
        return df
