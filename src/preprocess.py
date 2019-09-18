def get_vocabulary_vector(emails, freq_threshold=30):
    """
    For the given list of emails, create
    an ordered vocabulary vector containing
    all words in the corpus that satisfy a frequency
    threshold. The frequency is based upon the number of
    samples that contain the considered word.

    :param emails:          List of Email objects.
    :param freq_threshold:  The minimum number of samples
                            a word must occur in for it to
                            be included in the vocabulary.

    :return:    A list of tokens representing the vocabulary
                for the input list.
    """
    tokens = {}

    for email in emails:
        for token in set(email.tokens):
            if token not in tokens:
                tokens[token] = 1
            else:
                tokens[token] += 1

    vocab = []

    for token, count in tokens.items():
        if count >= freq_threshold:
            vocab.append(token)

    return vocab
