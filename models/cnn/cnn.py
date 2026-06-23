from tensorflow.keras.layers import (Input, Embedding, Conv1D, GlobalMaxPool1D,
                                     Dense, Dropout, concatenate)
from tensorflow.keras.models import Model

MAX_INDEX           = 42
MAX_STRING_LENGTH   = 63
EMBEDDING_DIMENSION = 128
NUM_CONV_FILTERS    = 60


def build_cnn():
    net = {}

    net['input'] = Input((MAX_STRING_LENGTH,), dtype='int32', name='input')

    net['embeddingCNN'] = Embedding(
        output_dim=EMBEDDING_DIMENSION,
        input_dim=MAX_INDEX,
        input_length=MAX_STRING_LENGTH,
        name='embeddingCNN'
    )(net['input'])

    # 5 filter size song song (2 → 6 ký tự)
    for size in [2, 3, 4, 5, 6]:
        net[f'conv{size}'] = Conv1D(NUM_CONV_FILTERS, size, name=f'conv{size}')(net['embeddingCNN'])
        net[f'pool{size}'] = GlobalMaxPool1D(name=f'pool{size}')(net[f'conv{size}'])

    net['concatcnn'] = concatenate(
        [net['pool2'], net['pool3'], net['pool4'], net['pool5'], net['pool6']],
        axis=1, name='concatcnn'
    )

    net['dropoutcnnmid'] = Dropout(0.5, name='dropoutcnnmid')(net['concatcnn'])
    net['densecnn']      = Dense(NUM_CONV_FILTERS, activation='relu', name='densecnn')(net['dropoutcnnmid'])
    net['dropoutcnn']    = Dropout(0.5, name='dropoutcnn')(net['densecnn'])
    net['output']        = Dense(1, activation='sigmoid', name='output')(net['dropoutcnn'])

    model = Model(net['input'], net['output'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = build_cnn()
    model.summary()
