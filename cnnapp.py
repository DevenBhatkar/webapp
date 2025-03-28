import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template

# Import necessary classes from devvtry.py
from devvtry import ModelWrapper, SimpleModel, preprocess_spectrogram, pad_spectrogram

# Load model wrapper
with open('model_wrapper.pkl', 'rb') as f:
    model_wrapper = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        if request.is_json:
            data = request.get_json()
            spectrogram_id = data.get('spectrogram_id')
            csv_features = np.array(data.get('csv_features'))
            
            # Load spectrogram from parquet file
            spectrogram_path = os.path.join('train_spectrograms', f"{spectrogram_id}.parquet")
            spectrogram = pd.read_parquet(spectrogram_path).values
            
            # Make prediction
            prediction = model_wrapper.predict(spectrogram, csv_features)
            
            return jsonify({
                'success': True,
                'predicted_class': prediction['predicted_class'],
                'probabilities': prediction['probabilities']
            })
            
        else:
            # Handle form data (for web interface)
            spectrogram_id = request.form.get('spectrogram_id')
            
            # Parse CSV features
            csv_features = []
            for feature in ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']:
                csv_features.append(float(request.form.get(feature, 0)))
            
            # Load spectrogram from parquet file
            spectrogram_path = os.path.join('train_spectrograms', f"{spectrogram_id}.parquet")
            spectrogram = pd.read_parquet(spectrogram_path).values
            
            # Make prediction
            prediction = model_wrapper.predict(spectrogram, np.array(csv_features))
            
            # Format probabilities for display
            probabilities = {k: f"{v*100:.2f}%" for k, v in prediction['probabilities'].items()}
            
            return render_template(
                'result.html',
                spectrogram_id=spectrogram_id,
                prediction=prediction['predicted_class'],
                probabilities=probabilities
            )
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/sample', methods=['GET'])
def sample():
    """
    Sample endpoint for demonstration that doesn't require a spectrogram file
    """
    try:
        # Get a random sample ID from the validation set
        import random
        
        # Get all spectrogram IDs available in train_spectrograms
        spectrogram_files = []
        if os.path.exists('train_spectrograms'):
            spectrogram_files = [f.replace('.parquet', '') for f in os.listdir('train_spectrograms') if f.endswith('.parquet')]
        
        if spectrogram_files:
            # Pick a random spectrogram
            spectrogram_id = random.choice(spectrogram_files)
            spectrogram_path = os.path.join('train_spectrograms', f"{spectrogram_id}.parquet")
            
            if os.path.exists(spectrogram_path):
                # Load the spectrogram
                spectrogram = pd.read_parquet(spectrogram_path).values
                
                # Generate random CSV features for demonstration
                csv_features = np.random.rand(6)
                
                # Make prediction
                prediction = model_wrapper.predict(spectrogram, csv_features)
                
                # Format probabilities for display
                probabilities = {k: f"{v*100:.2f}%" for k, v in prediction['probabilities'].items()}
                
                return render_template(
                    'result.html',
                    spectrogram_id=spectrogram_id,
                    prediction=prediction['predicted_class'],
                    probabilities=probabilities
                )
            
        # Return a mock result if no real samples available
        mock_probabilities = {
            'GPD': '10.25%',
            'GRDA': '5.30%',
            'LPD': '15.10%',
            'LRDA': '8.75%',
            'Other': '12.60%',
            'Seizure': '48.00%'
        }
        
        return render_template(
            'result.html',
            spectrogram_id='sample_123',
            prediction='Seizure',
            probabilities=mock_probabilities
        )
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True) 