from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Dropout
from tensorflow.keras.models import Model

MAX_INDEX         = 42
MAX_STRING_LENGTH = 63
EMBEDDING_DIMENSION = 128


def build_ann():
    net = {}

    net['input'] = Input((MAX_STRING_LENGTH,), dtype='int32', name='input')

    net['embedding'] = Embedding(
        output_dim=EMBEDDING_DIMENSION,
        input_dim=MAX_INDEX,
        input_length=MAX_STRING_LENGTH,
        name='embedding'
    )(net['input'])

    net['extradense'] = Dense(100, activation='relu', name='extradense')(net['embedding'])

    net['flatten'] = Flatten()(net['extradense'])

    net['dropout'] = Dropout(0.5, name='dropout')(net['flatten'])

    net['output'] = Dense(1, activation='sigmoid', name='output')(net['dropout'])

    model = Model(net['input'], net['output'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = build_ann()
    model.summary()
