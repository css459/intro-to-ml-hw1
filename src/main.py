from src.model.email import Email
from src.model.perceptron_classifier import *
from src.preprocess import get_vocabulary_vector


def create_pandas_dataframes():
    """
    Automatic function to form train and test
    Pandas DataFrames.
    :return: Train and Test set, respectively
    """
    train, test = Email.load_emails_from_data()

    train_y = [int(t.is_spam) for t in train]
    test_y = [int(t.is_spam) for t in test]

    vocab = get_vocabulary_vector(train)
    print("[ INF ] Vocab Size:", len(vocab))

    train = [t.vectorize_tokens(vocab) for t in train]
    test = [t.vectorize_tokens(vocab) for t in test]

    train = pd.DataFrame.from_records(train, columns=vocab)
    test = pd.DataFrame.from_records(test, columns=vocab)

    train['is_spam'] = train_y
    test['is_spam'] = test_y

    return train, test


def perceptron_train(train_df):
    train_x = train_df.drop('is_spam', 1)
    train_y = train_df['is_spam']

    p = AveragedPerceptronClassifier()
    p.fit(train_x, train_y)

    p.save_features()
    p.save_weights()

    return p.weights


def perceptron_test(w, test_df):
    test_x = test_df.drop('is_spam', 1)
    test_y = test_df['is_spam']

    p = AveragedPerceptronClassifier()
    p.weights = w
    p.validate(test_x, test_y)


def train_test():
    train_df, test_df = create_pandas_dataframes()
    train_x = train_df.drop('is_spam', 1)
    train_y = train_df['is_spam']
    test_x = test_df.drop('is_spam', 1)
    test_y = test_df['is_spam']

    p = AveragedPerceptronClassifier(max_iter=50)
    p.fit(train_x, train_y)
    p.validate(test_x, test_y)


def train_final():
    train_df, test_df = create_pandas_dataframes()
    final = pd.concat([train_df, test_df], ignore_index=True)

    final_x = final.drop('is_spam', 1)
    final_y = final['is_spam']

    p = PerceptronClassifier(max_iter=10)
    p.fit(final_x, final_y)

    p.save_weights()
    p.save_features()

    vocab = final_x.columns
    test = Email.load_test_file()
    test_y = [int(t.is_spam) for t in test]
    test = [t.vectorize_tokens(vocab) for t in test]

    test = pd.DataFrame.from_records(test, columns=vocab)
    test['is_spam'] = test_y

    test_x = test_df.drop('is_spam', 1)
    test_y = test_df['is_spam']

    p.validate(test_x, test_y)


if __name__ == '__main__':
    # This is only used to make to consistent with
    # what the homework asks. The way I use my Perceptron
    # is in `train_test()`:
    # train_df, test_df = create_pandas_dataframes()
    # perceptron_test(perceptron_train(train_df), test_df)

    # train_test()

    # Final, full training on all data
    train_final()
