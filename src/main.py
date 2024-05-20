from controllers import dataPreprocessing, featureEngineering, EssayScorerClassic

def prepare_data(df):

    df_clean = dataPreprocessing.full_text_preprocess(df)

    # will work in paragraph only
    df_para = featureEngineering.Paragraph_eng(df)

    # will work in the full text
    df_sen = featureEngineering.Sentence_eng(df_clean)
    df_word = featureEngineering.Word_eng(df_clean)

    df_para_sen = df_para.merge(df_sen, on = 'essay_id')
    df_para_sen_word = df_para_sen.merge(df_word, on = 'essay_id')

    final_data = df_para_sen_word.merge(df_clean[['essay_id','full_text','score']].to_pandas(),on = 'essay_id')

    return final_data


def train_model(train):

    # Initialize and train the model
    essay_scorer = EssayScorerClassic()
    essay_scorer.create_vectorizer()
    essay_scorer.create_model()

    # Train the model
    X_train = train.drop(['score','essay_id'], axis=1)
    y_train = train['score']
    essay_scorer.train_model(X_train, y_train)

    return essay_scorer


if __name__ == "__main__":

    # call the methods

    # for load data
    dp = dataPreprocessing()

    # take part for test
    train,test = dp.load_data()
    train = train.sample(1500) # for test

    # fast test
    prepared_train = prepare_data(train)
    prepare_test = prepare_data(test)
    print(f'{len(prepared_train.columns) == 72} preprocssing and feature engineering tests')
    print(prepared_train.columns)
    train_model(prepared_train)

    # fast prediction test
    model = train_model(train=prepared_train)
    results = model.predict(prepare_test)

    print(results)
