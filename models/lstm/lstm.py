from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Model

MAX_INDEX           = 42
MAX_STRING_LENGTH   = 63
EMBEDDING_DIMENSION = 128


def build_lstm():
    inputs = Input(shape=(MAX_STRING_LENGTH,), dtype='int32', name='input')
    x = Embedding(MAX_INDEX, EMBEDDING_DIMENSION, name='embedding')(inputs)
    x = LSTM(256, name='lstm')(x)
    x = Dropout(0.5, name='dropout')(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = build_lstm()
    model.summary()
