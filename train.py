import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Concatenate, 
    BatchNormalization, LeakyReLU, Layer, LayerNormalization,
    Bidirectional, TimeDistributed
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, 
    ModelCheckpoint, TensorBoard
)
from tensorflow.keras.regularizers import l1_l2

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * tf.transpose(a, [0, 2, 1])
        return tf.reduce_sum(output, axis=1)

class OptimizedTrafficPredictor:
    def __init__(self, time_steps=5, feature_dim=9):
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.model = None
        
    def create_optimized_model(self, learning_rate=0.001):
        """Create an optimized hybrid model with advanced architecture"""
        
        # LSTM Input Branch
        lstm_input = Input(shape=(self.time_steps, self.feature_dim), name='lstm_input')
        
        # Enhanced LSTM processing
        lstm_x = Bidirectional(LSTM(128, return_sequences=True))(lstm_input)
        lstm_x = LayerNormalization()(lstm_x)
        lstm_x = Dropout(0.3)(lstm_x)
        
        lstm_x = Bidirectional(LSTM(64, return_sequences=True))(lstm_x)
        lstm_x = LayerNormalization()(lstm_x)
        lstm_x = Dropout(0.3)(lstm_x)
        
        # Apply attention
        lstm_x = AttentionLayer()(lstm_x)
        
        # Dense Branch for current features
        dense_input = Input(shape=(self.feature_dim,), name='dense_input')
        
        # Enhanced dense processing
        dense_x = Dense(64, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dense_input)
        dense_x = BatchNormalization()(dense_x)
        dense_x = LeakyReLU()(dense_x)
        dense_x = Dropout(0.3)(dense_x)
        
        dense_x = Dense(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dense_x)
        dense_x = BatchNormalization()(dense_x)
        dense_x = LeakyReLU()(dense_x)
        dense_x = Dropout(0.3)(dense_x)
        
        # Combine branches
        combined = Concatenate()([lstm_x, dense_x])
        
        # Final processing
        x = Dense(64, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(combined)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        output = Dense(1)(x)
        
        # Create model
        model = Model(inputs=[lstm_input, dense_input], outputs=output)
        
        # Compile with optimized settings
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model

    def train_model(self, X_train_lstm, X_train_dense, y_train, 
                   X_val_lstm=None, X_val_dense=None, y_val=None,
                   batch_size=32, epochs=100, patience=15):
        """Train the model with advanced callbacks and monitoring"""
        
        # Prepare validation data
        validation_data = None
        if X_val_lstm is not None and X_val_dense is not None and y_val is not None:
            validation_data = ([X_val_lstm, X_val_dense], y_val)
        
        # Callbacks
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=patience,
                restore_best_weights=True,
                mode='min'
            ),
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6,
                mode='min'
            ),
            # Model checkpoint
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss' if validation_data else 'loss',
                save_best_only=True,
                mode='min'
            ),
            # TensorBoard logging
            TensorBoard(log_dir='./logs')
        ]
        
        # Train model
        history = self.model.fit(
            [X_train_lstm, X_train_dense],
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

def prepare_data_for_training(X_scaled, y, time_steps=5, val_split=0.2):
    """Prepare data for the optimized model"""
    
    # Create sequences for LSTM
    X_lstm = []
    X_dense = []
    y_new = []
    
    for i in range(len(X_scaled) - time_steps):
        X_lstm.append(X_scaled[i:i+time_steps])
        X_dense.append(X_scaled[i+time_steps])
        y_new.append(y[i+time_steps])
    
    X_lstm = np.array(X_lstm)
    X_dense = np.array(X_dense)
    y_new = np.array(y_new)
    
    # Split into train and validation
    train_size = int(len(X_lstm) * (1 - val_split))
    
    X_train_lstm = X_lstm[:train_size]
    X_train_dense = X_dense[:train_size]
    y_train = y_new[:train_size]
    
    X_val_lstm = X_lstm[train_size:]
    X_val_dense = X_dense[train_size:]
    y_val = y_new[train_size:]
    
    return (X_train_lstm, X_train_dense, y_train,
            X_val_lstm, X_val_dense, y_val)
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore')

class TrafficSignalPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.time_steps = 3
        
    def load_and_preprocess_data(self):
        """Load and preprocess the traffic data"""
        print("Loading and preprocessing data...")
        
        # Load data
        data = pd.read_csv(self.data_path)
        
        # Convert date and time columns
        data['current_date'] = pd.to_datetime(data['current_date'], format='%d-%m-%Y')
        data['current_time'] = pd.to_datetime(data['current_time'], format='%H:%M:%S')
        
        # Extract features
        data['day_of_week'] = data['current_date'].dt.dayofweek
        data['hour_of_day'] = data['current_time'].dt.hour
        data['minute_of_day'] = data['current_time'].dt.minute
        data['duration_in_traffic_sec'] = data['duration_in_traffic'].apply(
            lambda x: int(x.split()[0]) * 60 if x != 'N/A' else 0
        )
        data['distance_km'] = data['distance'].apply(lambda x: float(x.split()[0]))
        
        # Define features
        self.features = [
            'day_of_week', 'hour_of_day', 'minute_of_day', 
            'duration_in_traffic_sec', 'distance_km',
            'origin_lat', 'origin_lng', 'destination_lat', 'destination_lng'
        ]
        
        # Scale features
        X = data[self.features]
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare targets
        y_green = data['green_light'].values
        y_red = data['red_light'].values
        
        return X_scaled, y_green, y_red

    def prepare_sequence_data(self, X, y):
        """Prepare sequential data for LSTM"""
        X_seq = []
        y_new = []
        X_direct = []
        
        for i in range(len(X) - self.time_steps):
            X_seq.append(X[i:(i + self.time_steps)])
            y_new.append(y[i + self.time_steps])
            X_direct.append(X[i + self.time_steps])
            
        return np.array(X_seq), np.array(X_direct), np.array(y_new)

    def create_hybrid_model(self, lstm_input_shape, ml_input_shape):
        """Create a hybrid LSTM-Dense model"""
        # LSTM Branch
        lstm_input = Input(shape=lstm_input_shape)
        lstm_x = LSTM(64, return_sequences=True)(lstm_input)
        lstm_x = Dropout(0.2)(lstm_x)
        lstm_x = LSTM(32)(lstm_x)
        lstm_x = Dropout(0.2)(lstm_x)
        lstm_x = Dense(16, activation='relu')(lstm_x)
        
        # Dense Branch
        ml_input = Input(shape=ml_input_shape)
        ml_x = Dense(32, activation='relu')(ml_input)
        ml_x = Dropout(0.2)(ml_x)
        ml_x = Dense(16, activation='relu')(ml_x)
        
        # Combine branches
        combined = Concatenate()([lstm_x, ml_x])
        x = Dense(32, activation='relu')(combined)
        x = Dropout(0.2)(x)
        output = Dense(1)(x)
        
        model = Model(inputs=[lstm_input, ml_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        return model

    def train_models(self, X_scaled, y_target, signal_type):
        """Train both XGBoost and hybrid models"""
        print(f"\nTraining {signal_type} signal models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_target, test_size=0.2, random_state=42
        )
        
        # Prepare sequence data
        X_train_seq, X_train_direct, y_train_new = self.prepare_sequence_data(X_train, y_train)
        X_test_seq, X_test_direct, y_test_new = self.prepare_sequence_data(X_test, y_test)
        
        # Print shapes for verification
        print("\nData shapes:")
        print(f"X_train_seq: {X_train_seq.shape}")
        print(f"X_train_direct: {X_train_direct.shape}")
        print(f"y_train_new: {y_train_new.shape}")
        
        # Train XGBoost
        print("\nTraining XGBoost model...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        xgb_model.fit(X_train_direct, y_train_new)
        
        # Train Hybrid Model
        print("\nTraining Hybrid model...")
        lstm_input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        ml_input_shape = (X_train_direct.shape[1],)
        
        hybrid_model = self.create_hybrid_model(lstm_input_shape, ml_input_shape)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = hybrid_model.fit(
            [X_train_seq, X_train_direct],
            y_train_new,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate Models
        xgb_pred = xgb_model.predict(X_test_direct)
        hybrid_pred = hybrid_model.predict([X_test_seq, X_test_direct]).flatten()
        
        # Calculate metrics
        metrics = {
            'XGBoost': {
                'MAE': mean_absolute_error(y_test_new, xgb_pred),
                'MSE': mean_squared_error(y_test_new, xgb_pred),
                'R2': r2_score(y_test_new, xgb_pred)
            },
            'Hybrid': {
                'MAE': mean_absolute_error(y_test_new, hybrid_pred),
                'MSE': mean_squared_error(y_test_new, hybrid_pred),
                'R2': r2_score(y_test_new, hybrid_pred)
            }
        }
        
        # Plot results
        self.plot_results(y_test_new, xgb_pred, hybrid_pred, history, signal_type)
        
        return hybrid_model, xgb_model, metrics

    def plot_results(self, y_true, xgb_pred, hybrid_pred, history, signal_type):
        """Plot model performance and predictions"""
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Plot XGBoost predictions
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, xgb_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.title(f'XGBoost Predictions - {signal_type} Signal')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        
        # Plot Hybrid predictions
        plt.subplot(2, 2, 2)
        plt.scatter(y_true, hybrid_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.title(f'Hybrid Model Predictions - {signal_type} Signal')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        
        # Plot training history
        plt.subplot(2, 2, 3)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss - {signal_type} Signal')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def save_models(self, hybrid_model, xgb_model, signal_type):
        """Save trained models to disk"""
        hybrid_model.save(f'hybrid_model_{signal_type.lower()}.h5')
        with open(f'xgb_model_{signal_type.lower()}.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)
        print(f"\nModels for {signal_type} signal saved successfully!")

def main():
    # Initialize predictor
    predictor = TrafficSignalPredictor('output_traffic_data.csv')
    
    # Load and preprocess data
    X_scaled, y_green, y_red = predictor.load_and_preprocess_data()
    
    # Train and evaluate green signal models
    hybrid_green, xgb_green, metrics_green = predictor.train_models(
        X_scaled, y_green, 'Green'
    )
    
    # Train and evaluate red signal models
    hybrid_red, xgb_red, metrics_red = predictor.train_models(
        X_scaled, y_red, 'Red'
    )
    
    # Print final metrics
    print("\nFinal Results:")
    print("\nGreen Signal Metrics:")
    print("XGBoost:", metrics_green['XGBoost'])
    print("Hybrid:", metrics_green['Hybrid'])
    
    print("\nRed Signal Metrics:")
    print("XGBoost:", metrics_red['XGBoost'])
    print("Hybrid:", metrics_red['Hybrid'])
    
    # Save models
    predictor.save_models(hybrid_green, xgb_green, 'Green')
    predictor.save_models(hybrid_red, xgb_red, 'Red')

if __name__ == "__main__":
    main()