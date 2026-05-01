# -*- coding: utf-8 -*-
"""
Generate synthetic data and train an LSTM model for Action Recognition.
Creates `action_lstm.keras` to use immediately.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import pickle

ACTIONS = [
    "idle",            # 0
    "walk_forward",    # 1
    "walk_backward",   # 2
    "turn_left",       # 3
    "turn_right",      # 4
    "stop",            # 5
    "jump"             # 6
]
NUM_CLASSES = len(ACTIONS)
SEQ_LEN = 30
FEATURES = 132 # 33 landmarks * 4 (x,y,z,v)

def generate_synthetic_data(num_samples_per_class=200):
    print("Generating synthetic data...")
    X = []
    y = []
    
    for label_idx, action in enumerate(ACTIONS):
        for _ in range(num_samples_per_class):
            seq = np.zeros((SEQ_LEN, FEATURES))
            # Base pose: everything centered at 0.5, 0.5
            seq[:, :] = 0.5
            
            # Visibility all 1.0
            for i in range(33):
                seq[:, i*4 + 3] = 1.0
            
            # Modify sequences based on action to make them distinguishable
            if action == "idle":
                # Static pose
                seq += np.random.normal(0, 0.01, seq.shape)
            elif action == "walk_forward":
                # Oscillating legs/arms
                t = np.linspace(0, 4*np.pi, SEQ_LEN)
                seq[:, 15*4 + 1] += 0.2 * np.sin(t) # Left wrist y
                seq[:, 16*4 + 1] -= 0.2 * np.sin(t) # Right wrist y
            elif action == "walk_backward":
                t = np.linspace(0, 4*np.pi, SEQ_LEN)
                seq[:, 15*4 + 1] -= 0.2 * np.sin(t) 
                seq[:, 16*4 + 1] += 0.2 * np.sin(t)
            elif action == "turn_left":
                # Hands to the left (x < 0.5)
                seq[:, 15*4 + 0] = 0.2
                seq[:, 16*4 + 0] = 0.2
            elif action == "turn_right":
                # Hands to the right (x > 0.5)
                seq[:, 15*4 + 0] = 0.8
                seq[:, 16*4 + 0] = 0.8
            elif action == "stop":
                # Hands raised up (y < 0.5)
                seq[:, 15*4 + 1] = 0.1
                seq[:, 16*4 + 1] = 0.1
            elif action == "jump":
                # Whole body moves up and down
                t = np.linspace(0, np.pi, SEQ_LEN)
                for i in range(33):
                    seq[:, i*4 + 1] -= 0.3 * np.sin(t)
                    
            seq += np.random.normal(0, 0.02, seq.shape) # add some noise
            X.append(seq)
            y.append(label_idx)
            
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]

def build_and_train_model(X, y):
    print("Building LSTM model...")
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQ_LEN, FEATURES)),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Training model...")
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/action_lstm.keras')
    print("Model saved to models/action_lstm.keras")
    
    # Save label encoder
    with open('models/actions.pkl', 'wb') as f:
        pickle.dump(ACTIONS, f)
    print("Labels saved to models/actions.pkl")

if __name__ == "__main__":
    X, y = generate_synthetic_data()
    build_and_train_model(X, y)
