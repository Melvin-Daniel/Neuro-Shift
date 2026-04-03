import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import sys
sys.path.append('../signal-simulator')
from eeg_generator import SyntheticEEGGenerator
from signal_preprocessor import SignalPreprocessor
from feature_extractor import FeatureExtractor

class CommandClassifier:
    """ML-based classifier for brain command recognition"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Initialize preprocessing pipeline
        self.preprocessor = SignalPreprocessor()
        self.feature_extractor = FeatureExtractor()
        
        # Command mapping
        self.commands = ['light_on', 'light_off', 'fan_on', 'fan_off', 'tv_on', 'idle']
    
    def generate_training_data(self, samples_per_command=50):
        """Generate synthetic training data"""
        generator = SyntheticEEGGenerator()
        
        X = []  # Features
        y = []  # Labels
        
        print("Generating training data...")
        for command in self.commands:
            print(f"  Generating {samples_per_command} samples for: {command}")
            for i in range(samples_per_command):
                # Generate signal
                signal_data = generator.generate_command_signal(command, duration=2.0)
                
                # Preprocess
                preprocessed = self.preprocessor.preprocess_pipeline(signal_data['signal'])
                
                # Extract features
                features = self.feature_extractor.extract_all_features(preprocessed['processed'])
                feature_vector = self.feature_extractor.features_to_vector(features)
                
                X.append(feature_vector)
                y.append(command)
        
        return np.array(X), np.array(y)
    
    def train(self, X=None, y=None, test_size=0.2):
        """Train the classifier"""
        # Generate data if not provided
        if X is None or y is None:
            X, y = self.generate_training_data(samples_per_command=50)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        print(f"\nTraining classifier on {len(X_train)} samples...")
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTraining completed!")
        print(f"Accuracy on test set: {accuracy * 100:.2f}%")
        
        # Detailed report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return accuracy
    
    def predict(self, signal_data):
        """Predict command from raw signal"""
        if not self.is_trained:
            raise Exception("Model not trained! Call train() first.")
        
        # Preprocess signal
        preprocessed = self.preprocessor.preprocess_pipeline(signal_data)
        
        # Extract features
        features = self.feature_extractor.extract_all_features(preprocessed['processed'])
        feature_vector = self.feature_extractor.features_to_vector(features)
        
        # Predict
        prediction_encoded = self.model.predict([feature_vector])[0]
        prediction_proba = self.model.predict_proba([feature_vector])[0]
        
        # Decode prediction
        predicted_command = self.label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = np.max(prediction_proba) * 100
        
        return {
            'command': predicted_command,
            'confidence': confidence,
            'probabilities': dict(zip(self.label_encoder.classes_, prediction_proba))
        }
    
    def save_model(self, filepath='command_classifier_model.pkl'):
        """Save trained model to file"""
        if not self.is_trained:
            raise Exception("Model not trained! Nothing to save.")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'commands': self.commands
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath='command_classifier_model.pkl'):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.commands = model_data['commands']
        self.is_trained = True
        
        print(f"Model loaded from: {filepath}")


# Test the classifier
# Test the classifier
if __name__ == "__main__":
    print("=" * 60)
    print("NEURO-SHIFT COMMAND CLASSIFIER - TRAINING")
    print("=" * 60)
    
    # Create and train classifier
    classifier = CommandClassifier()
    accuracy = classifier.train()
    
    print("\n" + "=" * 60)
    print("TESTING REAL-TIME PREDICTION")
    print("=" * 60)
    
    # Test with new signals
    generator = SyntheticEEGGenerator()
    test_commands = ['light_on', 'fan_off', 'tv_on', 'idle']
    
    print("\nTesting predictions on new signals:")
    for cmd in test_commands:
        signal_data = generator.generate_command_signal(cmd, duration=2.0)
        result = classifier.predict(signal_data['signal'])
        
        print(f"\nActual: {cmd}")
        print(f"Predicted: {result['command']} (Confidence: {result['confidence']:.2f}%)")
        
        # Show top 3 predictions
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        print("Top 3 predictions:")
        for i, (command, prob) in enumerate(sorted_probs[:3], 1):
            print(f"  {i}. {command}: {prob*100:.2f}%")
    
    # Save model
    print("\n" + "=" * 60)
    classifier.save_model('command_classifier_model.pkl')
    print("=" * 60)
