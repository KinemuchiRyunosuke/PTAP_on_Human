import numpy as np
import pandas as pd

from dataset import get_dataset


fastafile = 'data/raw/protein.faa'
vocab_path = 'data/vocab.pickle'
checkpoint_path = 'models/'
result_path = 'data/result.csv'
length = 26
threshold = 0.5


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
                vocab_size=args.num_words,
                hopping_num=args.hopping_num,
                head_num=args.head_num,
                hidden_dim=args.hidden_dim,
                dropout_rate=args.dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(
                                learning_rate=args.lr),
                 loss='binary_crossentropy')

    return model

