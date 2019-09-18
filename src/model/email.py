# Constants
DATA_TRAIN_FILE = "../data/spam_train.txt"
DATA_TEST_FILE = "../data/spam_test.txt"


class Email:

    def __init__(self, tokens, is_spam):
        self.tokens = tokens
        self.is_spam = int(is_spam) == 1

    def vectorize_tokens(self, vocabulary):
        """
        Given a vector of text information in the `tokens` field,
        transform this into a binary word-occurrence vector for all
        words in the vocabulary.

        :param vocabulary:  A list of words for which this output
                            vector will represent. The output vector
                            will be the same length as this vector,
                            and the order of the vocabulary vector
                            will determine the order of this output
                            vector.

        :return: An ordered binary vector of length `vocabulary`.
        """
        vec = []
        words = set(self.tokens)
        for v in vocabulary:
            if v in words:
                vec.append(1)
            else:
                vec.append(0)

        return vec

    @staticmethod
    def load_emails_from_data(validation_percent=0.20):
        with open(DATA_TRAIN_FILE, "r") as fp:
            emails = []

            # Parse each line in the file into email tokens
            # and Y variable
            for line in fp:
                is_spam = int(line.split()[0])
                tokens = line.split()[1:]

                new_email = Email(tokens, is_spam)
                emails.append(new_email)

            print("[ INF ] Read", len(emails), "samples.")
            print("[ INF ] Train count:", len(emails) * (1 - validation_percent),
                  "Test Count:", len(emails) * validation_percent)

            split_index = int(len(emails) * (1 - validation_percent))
            return emails[:split_index], emails[split_index:]

    @staticmethod
    def load_test_file():
        with open(DATA_TEST_FILE, "r") as fp:
            emails = []

            # Parse each line in the file into email tokens
            # and Y variable
            for line in fp:
                is_spam = int(line.split()[0])
                tokens = line.split()[1:]

                new_email = Email(tokens, is_spam)
                emails.append(new_email)

            print("[ INF ] Read", len(emails), "samples.")
            print("[ INF ] Count:", len(emails))

            return emails