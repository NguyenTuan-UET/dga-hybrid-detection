from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPool1D, LSTM, Dense
from tensorflow.keras.models import Model

MAX_INDEX           = 42
MAX_STRING_LENGTH   = 63
EMBEDDING_DIMENSION = 128


def build_mit():
    main_input = Input(shape=(MAX_STRING_LENGTH,), dtype='int32', name='main_input')

    embedding = Embedding(
        input_dim=MAX_INDEX,
        output_dim=EMBEDDING_DIMENSION,
        input_length=MAX_STRING_LENGTH
    )(main_input)

    # CNN lọc feature trước
    conv     = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', strides=1)(embedding)
    max_pool = MaxPool1D(pool_size=2, padding='same')(conv)

    # LSTM đọc output của CNN (nối tiếp — khác Bilbo)
    encode = LSTM(64, return_sequences=False)(max_pool)

    output = Dense(1, activation='sigmoid')(encode)

    model = Model(inputs=main_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = build_mit()
    model.summary()
