class CountVectorizer():
    """Convert a collection of text to list of words
    and list of words counters of every text"""
    def fit_transform(self, corpus: list = []) -> list[list[int]] | str:
        """Make a matrix of using words from vocabulary of all texts
        in every sentence"""

        self.corpus = corpus

        if not CountVectorizer.checking_input(corpus):
            return 'Error'

        words_2d_ar = []
        for sentence in corpus:
            words_2d_ar.append(sentence.lower().split(' '))

        vocab_dict = CountVectorizer.dict_from_vocab(self)
        matrix = []
        for sentence in words_2d_ar:
            vocab_dict_copy = vocab_dict.copy()
            for word in sentence:
                if word in vocab_dict_copy.keys():
                    vocab_dict_copy[word] += 1
            matrix.append([cnt for cnt in vocab_dict_copy.values()])

        return matrix

    def get_feature_names(self) -> list[str] | str:
        """Make a vocabulary from all texts"""
        corpus = self.corpus
        if not CountVectorizer.checking_input(corpus):
            return 'Error'
        all_words = ' '.join(corpus).lower().split(' ')
        words_w_out_dubs = CountVectorizer.remove_dublicates(all_words)
        self.vocabulary = words_w_out_dubs
        return self.vocabulary

    def dict_from_vocab(self) -> dict:
        """Transform list of words into dict with counter 0"""
        array = CountVectorizer.get_feature_names(self)
        vocab_dict = {word: 0 for word in array}
        return vocab_dict

    @staticmethod
    def remove_dublicates(array: list) -> list[str]:
        """Remove dublicates from list of words"""
        new_array = []
        for word in array:
            if word not in new_array:
                new_array.append(word)
        return new_array

    @staticmethod
    def checking_input(array: list) -> bool:
        """Checks if input is correct"""
        if not isinstance(array, list):
            return False
        for word in array:
            if not isinstance(word, str):
                return False
        return True


if __name__ == '__main__':
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]

    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)

    print(vectorizer.get_feature_names())
    print(count_matrix)
