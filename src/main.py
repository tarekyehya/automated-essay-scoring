# dataPreocessing test
from controllers import dataPreprocessing

dp = dataPreprocessing()

train,_ = dp.load_data()
train = train.sample(5) # for test

clean_train = dp.full_text_preprocess(train)

print(clean_train[0]['full_text'])
print(clean_train[0])






if __name__ == "__main__":
    pass
    # call the method