import numpy as np
import pickle
from Bio import SeqIO

from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_dataset(fastafile, vocab_path, length):
    records = get_record(fastafile)

    sequences = fragmentate(records, length)

    with open(vocab_path, 'rb') as f:
        tokenizer = pickle.load(f)

    vocab = Vocab(tokenizer)
    sequences = vocab.encode(sequences)
    sequences = add_class_token(sequences)

    return sequences


def get_record(fastafile):
    records = []

    with open(fastafile, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            records.append(record)

    return records

def fragmentate(records, length):
    fragments = []

    for record in records:
        for i in range(len(record.seq) - length + 1):
            fragment = record.seq[i:(i + length)]
            fragments.append(str(fragment))

    return fragments

def add_class_token(sequences):
    sequences += 1

    sequences = np.array(sequences)
    mask = (sequences == 1)
    sequences[mask] = 0

    # class_token = 1
    cls_arr = np.ones((len(sequences), 1))     # shape=(len(sequences), 1)
    sequences = np.hstack([cls_arr, sequences]).astype('int64')

    return sequences


class Vocab:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def fit(self, texts):
        """ 単語のベクトル化の準備

        Arg:
            texts([str], ndarray): 単語のリスト
                e.g. ['VPTAPP',
                      'ATSQVP']

        Return:
            self(Vocab instance): 学習を完了したインスタンス

        """
        if type(texts).__module__ == 'numpy':
            texts = np.squeeze(texts)
            texts = texts.tolist()

        self.tokenizer.fit_on_texts(texts)
        return self

    def encode(self, texts):
        """ 単語のリストを整数のリストに変換する

        Arg:
            texts(list, ndarray): 単語のリスト

        Return:
            ndarray: shape=(n_samples, n_words)
                e.g. [[0, 1, 2, 3, 4, 4],
                      [3, 2, 5, 6, 0, 4]]

        """
        if type(texts).__module__ == 'numpy':
            texts = np.squeeze(texts)
            texts = texts.tolist()

        sequences = self.tokenizer.texts_to_sequences(texts)
        sequences = pad_sequences(sequences, padding='post', value=0)

        return sequences

    def decode(self, sequences, class_token=False):
        """ 整数のリストを単語のリストに変換する

        Arg:
            sequences(ndarray): 整数の配列
                shape=(n_samples, n_words)

        Return:
            [str]: 単語のリスト

        """
        if class_token:  # class_tokenを削除
            sequences = np.delete(sequences, 0, axis=-1)

        # ndarrayからlistに変換
        sequences = sequences.tolist()

        for i, seq in enumerate(sequences):
            try:  # 0が存在しない場合はValueError
                pad_idx = seq.index(0)
            except ValueError:
                continue

            sequences[i] = seq[:pad_idx]

        if class_token:
            for i, seq in enumerate(sequences):
                sequences[i] = list(map(lambda x: x-1, seq))

        texts = self.tokenizer.sequences_to_texts(sequences)
        texts = [text.replace(' ', '') for text in texts]

        return texts

    def _texts(self, sequences):
        return ['\t'.join(words) for words in sequences]


if __name__ == '__main__':
    main()