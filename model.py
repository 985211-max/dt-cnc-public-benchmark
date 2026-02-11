"""
LSTM model definition for tool wear prediction.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm_classifier(input_shape, lstm_units1=64, lstm_units2=32, dropout_rate=0.3, lr=0.001):
    """
    Build a 2-layer LSTM model for binary classification.
    """
    model = Sequential([
        LSTM(lstm_units1, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

def build_lstm_regressor(input_shape, lstm_units1=64, lstm_units2=32, dropout_rate=0.3, lr=0.001):
    """
    Build a 2-layer LSTM model for remaining useful life regression.
    """
    model = Sequential([
        LSTM(lstm_units1, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='mse',
                  metrics=['mae'])
    return model
