import numpy as np
import pandas as pd
import tensorflow as tf

from dataset import get_dataset
from models.transformer import BinaryClassificationTransformer


fastafile = 'data/raw/protein.faa'
vocab_path = 'data/vocab.pickle'
checkpoint_path = 'models/'
result_path = 'data/result.csv'
length = 26
threshold = 0.5

# parameters of Transformer
num_words = 26
head_num = 8            # Transformerの並列化に関するパラメータ
dropout_rate = 0.04
hopping_num = 2         # Multi-Head Attentionを施す回数
hidden_dim = 904        # 単語ベクトルの次元数
lr = 2.03e-5            # 学習率


def main():
    x = get_dataset(fastafile, vocab_path, length)

    model = create_model()
    model.load_weights(checkpoint_path)

    y_pred = model.predict(x)
    y_pred = np.squeeze(y_pred)
    y_pred = (y_pred >= threshold).astype(int)

    df = pd.DataFrame([x, y_pred], columns=['seq', 'prediction'])

    df.to_csv(result_path)


def create_model():
    """ モデルを定義する """
    model = BinaryClassificationTransformer(
                vocab_size=num_words,
                hopping_num=hopping_num,
                head_num=head_num,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(
                                learning_rate=lr),
                 loss='binary_crossentropy')

    return model

