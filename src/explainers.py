
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
import joblib
import os
from src.models.predict import MaternalRiskPredictor


class ModelExplainer:
    
    def __init__(self, model_type='best'):
        # Load model and predictor
        self.predictor = MaternalRiskPredictor(model_type=model_type)
        self.model = self.predictor.model
        self.scaler = self.predictor.scaler
        self.feature_names = self.predictor.feature_names
        self.is_keras = self.predictor.is_keras
        
        # Load training data for explainers
        self.X_train = np.load('data/processed/X_train.npy')
        self.X_test = np.load('data/processed/X_test.npy')
        self.y_test = np.load('data/processed/y_test.npy')
        
        # Lazy initialization of explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        print(f"Explainer initialized for {model_type} model")
        
    def initialize_shap(self, background_samples=100):
        print("Initializing SHAP explainer...")
        
        # Sample background data for SHAP
        background = shap.sample(self.X_train, background_samples)
        
        # Choose explainer based on model type
        if self.is_keras:
            self.shap_explainer = shap.DeepExplainer(self.model, background)
        else:
            model_type = type(self.model).__name__
            # Tree-based models can use TreeExplainer (faster)
            if 'Forest' in model_type or 'XGB' in model_type or 'LGBM' in model_type or 'Gradient' in model_type:
                try:
                    self.shap_explainer = shap.TreeExplainer(self.model)
                except:
                    self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background)
            else:
                self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background)
        
        print("SHAP explainer initialized")
        
    def initialize_lime(self):
        print("Initializing LIME explainer with raw units...")
        
        # Use unscaled data for LIME initialization to show raw units
        if self.scaler:
            X_train_raw = self.scaler.inverse_transform(self.X_train)
        else:
            X_train_raw = self.X_train
            
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            X_train_raw,
            feature_names=self.feature_names,
            class_names=['Low Risk', 'Mid Risk', 'High Risk'],
            mode='classification',
            discretize_continuous=True,
            sample_around_instance=True
        )
        
        print("LIME explainer initialized with raw units")
        
    def get_shap_values(self, X, max_samples=None):
        if self.shap_explainer is None:
            self.initialize_shap()
            
        # Limit samples for performance if needed
        if max_samples and len(X) > max_samples:
            X = X[:max_samples]
            
        print(f"Computing SHAP values for {len(X)} samples...")
        shap_values = self.shap_explainer.shap_values(X)
        
        return shap_values
        
    def plot_shap_summary(self, X=None, max_samples=100, save_path='reports/shap_summary.png'):
        if X is None:
            X = self.X_test[:max_samples]
            
        shap_values = self.get_shap_values(X, max_samples)
        
        plt.figure(figsize=(10, 8))
        
        # Handle multiclass vs binary classification
        if isinstance(shap_values, list):
            # For multiclass, show high risk class
            shap.summary_plot(shap_values[2], X, feature_names=self.feature_names, 
                            show=False, plot_type='bar')
            plt.title('SHAP Feature Importance - High Risk Class')
        else:
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, 
                            show=False)
            plt.title('SHAP Feature Importance')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP summary plot saved to {save_path}")
        plt.close()
        
    def plot_shap_beeswarm(self, X=None, max_samples=100, save_path='reports/shap_beeswarm.png'):
        if X is None:
            X = self.X_test[:max_samples]
            
        shap_values = self.get_shap_values(X, max_samples)
        
        plt.figure(figsize=(10, 8))
        
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[2], X, feature_names=self.feature_names, 
                            show=False)
            plt.title('SHAP Feature Impact - High Risk Class')
        else:
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, 
                            show=False)
            plt.title('SHAP Feature Impact')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP beeswarm plot saved to {save_path}")
        plt.close()
        
    def plot_shap_waterfall(self, patient_data, save_path='reports/shap_waterfall.png'):
        if self.shap_explainer is None:
            self.initialize_shap()
            
        # Get SHAP values for this specific prediction
        features = self.predictor.preprocess_input(patient_data)
        shap_values = self.shap_explainer.shap_values(features)
        prediction = self.predictor.predict(patient_data)
        risk_class = prediction['risk_level_numeric']
        
        plt.figure(figsize=(10, 6))
        
        # Extract SHAP values for the predicted class
        if isinstance(shap_values, list):
            shap_values_class = shap_values[risk_class][0]
        else:
            shap_values_class = shap_values[0]
            
        # Get expected value (baseline)
        expected_value = self.shap_explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[risk_class]
            
        explanation = shap.Explanation(
            values=shap_values_class,
            base_values=expected_value,
            data=features[0],
            feature_names=self.feature_names
        )
        
        shap.waterfall_plot(explanation, show=False)
        plt.title(f'SHAP Explanation - {prediction["risk_level"]}')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP waterfall plot saved to {save_path}")
        plt.close()
        
    def explain_with_lime(self, patient_data, num_features=6, target_class=None):
        if self.lime_explainer is None:
            self.initialize_lime()
            
        # patient_data should be a dict of raw values
        if isinstance(patient_data, dict):
            raw_features = np.array([[patient_data[col] for col in self.feature_names]])
        elif isinstance(patient_data, pd.DataFrame):
            raw_features = patient_data[self.feature_names].values
        else:
            raw_features = patient_data
            
        # Create prediction function that scales input before passing to model
        def predict_fn_scaled(x):
            if self.scaler:
                x_scaled = self.scaler.transform(x)
            else:
                x_scaled = x
                
            if self.is_keras:
                return self.model.predict(x_scaled, verbose=0)
            else:
                return self.model.predict_proba(x_scaled)
            
        # If target_class is not specified, explain the predicted class
        if target_class is None:
            prediction = self.predictor.predict(patient_data)
            target_class = prediction['risk_level_numeric']
            
        # Generate LIME explanation using raw features
        explanation = self.lime_explainer.explain_instance(
            raw_features[0],
            predict_fn_scaled,
            num_features=num_features,
            labels=(0, 1, 2)
        )
        
        return explanation
        
    def plot_lime_explanation(self, patient_data, save_path='reports/lime_explanation.png'):
        explanation = self.explain_with_lime(patient_data)
        prediction = self.predictor.predict(patient_data)
        
        plt.figure(figsize=(10, 6))
        
        # Extract feature contributions
        exp_list = explanation.as_list()
        features = [item[0] for item in exp_list]
        values = [item[1] for item in exp_list]
        
        # Color code: green for increases risk, red for decreases
        colors = ['green' if v > 0 else 'red' for v in values]
        plt.barh(features, values, color=colors, alpha=0.7)
        plt.xlabel('Feature Contribution')
        plt.title(f'LIME Explanation - {prediction["risk_level"]}')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"LIME explanation plot saved to {save_path}")
        plt.close()
        
        return explanation
        
    def get_text_explanation(self, patient_data):
        prediction = self.predictor.predict(patient_data)
        interpretation = self.predictor.get_risk_interpretation(prediction)
        
        # Get top 3 contributing factors from LIME
        lime_exp = self.explain_with_lime(patient_data, num_features=3)
        exp_list = lime_exp.as_list()
        
        # Format text explanation
        text = f"{interpretation}\n\n"
        text += "Key Contributing Factors:\n"
        
        for i, (feature, contribution) in enumerate(exp_list[:3], 1):
            direction = "increases" if contribution > 0 else "decreases"
            text += f"{i}. {feature} {direction} risk (contribution: {contribution:.3f})\n"
            
        return text
        
    def generate_global_report(self, save_dir='reports'):
        print("\n" + "="*60)
        print("GENERATING EXPLAINABILITY REPORT")
        print("="*60)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate SHAP plots
        print("\n1. Generating SHAP visualizations...")
        self.plot_shap_summary(save_path=f'{save_dir}/shap_summary.png')
        self.plot_shap_beeswarm(save_path=f'{save_dir}/shap_beeswarm.png')
        
        # Generate LIME example for random patient
        print("\n2. Generating LIME example...")
        example_idx = np.random.choice(len(self.X_test))
        example_features = self.X_test[example_idx:example_idx+1]
        example_df = pd.DataFrame(example_features, columns=self.feature_names)
        self.plot_lime_explanation(example_df, save_path=f'{save_dir}/lime_example.png')
        
        # Compute and save feature importance
        print("\n3. Computing feature importance...")
        shap_values = self.get_shap_values(self.X_test[:200])
        
        # Average importance across classes if multiclass
        if isinstance(shap_values, list):
            feature_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            feature_importance = np.abs(shap_values).mean(axis=0)
            
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        importance_df.to_csv(f'{save_dir}/feature_importance.csv', index=False)
        print(f"\nFeature Importance:\n{importance_df.to_string()}")
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title('Global Feature Importance (SHAP)')
        plt.xlabel('Mean |SHAP value|')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n" + "="*60)
        print("REPORT GENERATION COMPLETE!")
        print(f"All files saved to {save_dir}/")
        print("="*60)


if __name__ == "__main__":
    print("Initializing Model Explainer...")
    
    try:
        explainer = ModelExplainer(model_type='best')
        explainer.generate_global_report()
        
        print("\n" + "="*60)
        print("EXAMPLE INDIVIDUAL EXPLANATION")
        print("="*60)
        
        example_patient = {
            'Age': 30,
            'SystolicBP': 140,
            'DiastolicBP': 90,
            'BS': 12.0,
            'BodyTemp': 99.0,
            'HeartRate': 88
        }
        
        print("\nPatient Data:")
        for key, value in example_patient.items():
            print(f"  {key}: {value}")
        
        explanation = explainer.get_text_explanation(example_patient)
        print(f"\n{explanation}")
        
        explainer.plot_shap_waterfall(example_patient, 'reports/example_shap_waterfall.png')
        explainer.plot_lime_explanation(example_patient, 'reports/example_lime.png')
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure data is processed and models are trained first.")

