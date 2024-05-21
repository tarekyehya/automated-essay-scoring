import os
import dill
from controllers import dataPreprocessing, featureEngineering, EssayScorerClassic,DataController
from sklearn.metrics import f1_score, cohen_kappa_score
from helpers import get_settings

config = get_settings()

def prepare_data(df):
    df_clean = dataPreprocessing.full_text_preprocess(df)

    # Will work on paragraphs
    df_para = featureEngineering.Paragraph_eng(df)

    # Will work on the full text
    df_sen = featureEngineering.Sentence_eng(df_clean)
    df_word = featureEngineering.Word_eng(df_clean)

    df_para_sen = df_para.merge(df_sen, on='essay_id')
    df_para_sen_word = df_para_sen.merge(df_word, on='essay_id')
    final_data = df_para_sen_word.merge(df_clean[['essay_id', 'full_text', 'score']].to_pandas(), on='essay_id')

    return final_data

def train_model(train):
    # Initialize and train the model
    essay_scorer = EssayScorerClassic()
    essay_scorer.create_vectorizer()
    essay_scorer.create_model()

    # Train the model
    X_train = train.drop(['score', 'essay_id'], axis=1)
    y_train = train['score']
    essay_scorer.train_model(X_train, y_train)

    return essay_scorer

def load_models_classic():
    if not os.path.exists(config.MODELS_CLASSIC_PATH):
        print('No fitted models yet')
        return None

    # Load the dictionary
    with open(config.MODELS_CLASSIC_PATH, 'rb') as file:
        loaded_models = dill.load(file)

    return EssayScorerClassic().pass_models_classic(loaded_models)

if __name__ == "__main__":

    debug = False
    # split the df to train and test and save them
    DataController.make_test_set_skf_out()

    # Load data
    dp = dataPreprocessing()
    train, test = dp.load_data()

    if debug:
        test = test.sample(50)
        train = train.sample(500)  # For testing purposes

    # Prepare data
    prepared_train = prepare_data(train)
    prepared_test = prepare_data(test)

    if debug:
        print(f'{len(prepared_train.columns) == 72} preprocssing and feature engineering tests')
        print(prepared_train.columns)

    # Train or load the model
    if not os.path.exists(config.MODELS_CLASSIC_PATH):
        print('Training mode')
        model = train_model(prepared_train)
        model.save_models()
    else:
        print('Inference mode')
        model = load_models_classic()

    # Make predictions
    results = model.predict(prepared_test.drop('score',axis = 1))
    f1_test = f1_score(prepared_test.score.values, results, average='weighted')
    kappa_test = cohen_kappa_score(prepared_test.score.values, results, weights='quadratic')
    print(f'F1 score = {f1_test} and cohen_kappa_score = {kappa_test}')

