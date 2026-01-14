

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predict import MaternalRiskPredictor
import matplotlib.pyplot as plt

# Try to import explainability features (optional)
try:
    from src.explainers import ModelExplainer
    import shap
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False
    ModelExplainer = None
    shap = None

# Configure Streamlit page
st.set_page_config(
    page_title="Maternal Health - Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: #166534;
    }
    .risk-mid {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: #92400e;
    }
    .risk-high {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: #991b1b;
    }
    .clinical-audit {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 2rem;
        border-top: 1px solid #e2e8f0;
        padding-top: 1rem;
    }
    .cds-section {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1.5rem;
        color: #1e293b;
    }
    .cds-section h3 {
        color: #991b1b;
        margin-top: 0;
    }
    </style>
""", unsafe_allow_html=True)


# Cache model loading to avoid reloading on each prediction
@st.cache_resource
def load_predictor():
    try:
        return MaternalRiskPredictor(model_type='best')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Cache explainer loading (SHAP/LIME can be slow to initialize)
@st.cache_resource
def load_explainer():
    if not EXPLAINABILITY_AVAILABLE or ModelExplainer is None:
        return None
    try:
        return ModelExplainer(model_type='best')
    except Exception as e:
        st.warning(f"Could not load explainer: {str(e)}")
        return None


def main():
    st.title("🏥 Clinical Decision Support: Maternal Health")
    st.markdown("### Evidence-based risk assessment for obstetric care")
    
    predictor = load_predictor()
    
    # Sidebar settings
    st.sidebar.header("⚙️ Configuration")
    
    if predictor:
        st.sidebar.info(f"""
        **Audit Information**
        - Model Version: `{predictor.VERSION}`
        - Last Modified: `{predictor.model_timestamp}`
        - Model Type: `{predictor.model_type}`
        """)
    
    show_explanation = st.sidebar.checkbox("XAI: Feature Contributions", value=True)
    show_probabilities = st.sidebar.checkbox("Clinical Probability Distribution", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Clinical Definitions
    - **Green/Low**: Standard prenatal pathway
    - **Amber/Mid**: Increased surveillance
    - **Red/High**: Immediate clinical review
    """)
    
    # Clinical Case Input
    st.header("📋 Clinical Case Data")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            "Maternal Age (years)",
            min_value=10,
            max_value=100,
            value=25,
            step=1,
            help="Patient's age in years"
        )
        
        systolic_bp = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=70,
            max_value=200,
            value=120,
            step=1,
            help="Upper blood pressure reading"
        )
        
        diastolic_bp = st.number_input(
            "Diastolic Blood Pressure (mmHg)",
            min_value=40,
            max_value=120,
            value=80,
            step=1,
            help="Lower blood pressure reading"
        )
    
    with col2:
        blood_sugar = st.number_input(
            "Blood Sugar (mmol/L)",
            min_value=3.0,
            max_value=25.0,
            value=7.5,
            step=0.1,
            help="Blood glucose level"
        )
        
        body_temp = st.number_input(
            "Body Temperature (°F)",
            min_value=95.0,
            max_value=105.0,
            value=98.6,
            step=0.1,
            help="Body temperature in Fahrenheit"
        )
        
        heart_rate = st.number_input(
            "Heart Rate (bpm)",
            min_value=40,
            max_value=150,
            value=72,
            step=1,
            help="Beats per minute"
        )
    
    st.markdown("---")
    
    # Session state for SHAP and results to avoid the "redirect to Square 1" bug
    if 'shap_requested' not in st.session_state:
        st.session_state.shap_requested = False
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None

    # Assessment button
    if st.button("🔍 Run Clinical Assessment", type="primary", use_container_width=True):
        st.session_state.shap_requested = False # Reset SHAP on new prediction
        
        # --- NEW: Clinical Validation ---
        valid_input = True
        
        # 1. Pulse Pressure Check
        if systolic_bp <= diastolic_bp:
            st.error("🚨 **Physiological Impossibility Detected**: Systolic blood pressure must be higher than Diastolic blood pressure. Please re-check vitals.")
            valid_input = False
            
        # 2. Extreme Vitals Warning (Safety Layer)
        if body_temp > 103 or heart_rate < 50 or heart_rate > 130 or age > 60:
            st.warning("⚠️ **Clinical Alert**: Some parameters entered are extreme (e.g., severe fever, bradycardia, or advanced age). The model may proceed, but these require immediate clinical attention regardless of the predicted risk score.")

        if valid_input:
            predictor = load_predictor()
            if predictor is None:
                st.error("❌ **Model not found!** Please train the models first.")
                return
            
            # Prepare clinical data dictionary
            patient_data = {
                'Age': age,
                'SystolicBP': systolic_bp,
                'DiastolicBP': diastolic_bp,
                'BS': blood_sugar,
                'BodyTemp': body_temp,
                'HeartRate': heart_rate
            }
            
            with st.spinner("Analyzing clinical data..."):
                try:
                    result = predictor.predict(patient_data)
                    st.session_state.last_prediction = (result, patient_data)
                except Exception as e:
                    st.error(f"❌ Assessment failed: {str(e)}")

    # Display results if available in session state
    if st.session_state.last_prediction:
        result, patient_data = st.session_state.last_prediction
        risk_level = result['risk_level']
        confidence = result['confidence']
        interpretation = predictor.get_risk_interpretation(result)
        
        # Check for pulse pressure again to ensure valid state
        if patient_data['SystolicBP'] <= patient_data['DiastolicBP']:
             st.warning("⚠️ The current results are based on invalid physiological data (Systolic <= Diastolic). Please update inputs.")
        
        st.markdown("---")
        st.header("⚖️ Clinical Risk Assessment")
        
        # Set styling based on risk level
        if "Low" in risk_level:
            risk_class = "risk-low"
            risk_icon = "✅"
            risk_color = "#22c55e"
        elif "Mid" in risk_level:
            risk_class = "risk-mid"
            risk_icon = "⚠️"
            risk_color = "#f59e0b"
        else:
            risk_class = "risk-high"
            risk_icon = "🚨"
            risk_color = "#ef4444"
        
        st.markdown(f"""
        <div class="{risk_class}">
            <h2>{risk_icon} {risk_level} Assessment</h2>
            <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                Model Confidence: <strong>{confidence*100:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
                
        # Clinical Decision Support Section for High Risk
        high_prob = result['probabilities']['High Risk']
        threshold = 0.5 # Clinical threshold for intervention
        
        if high_prob > threshold:
            st.markdown(f"""
            <div class="cds-section">
                <h3>🔴 Clinical Decision Support (High Risk Trigger)</h3>
                <p><strong>Trigger:</strong> High-risk probability ({high_prob*100:.1f}%) > Clinical Threshold ({threshold*100:.0f}%)</p>
                <p><strong>Immediate Actions:</strong></p>
                <ul>
                    <li><strong>Urgent senior OB review</strong> and continuous maternal/fetal monitoring.</li>
                    <li><strong>Confirmatory Checks:</strong>
                        <ul>
                            <li>Repeat Blood Pressure properly (rested, correct cuff size) and check trend.</li>
                            <li>Screen for "Red Flag" symptoms: Headache, visual changes, epigastric pain, reduced fetal movement.</li>
                            <li>Review glucose status and temperature context.</li>
                        </ul>
                    </li>
                    <li><strong>Escalation:</strong> Follow local maternity emergency/Pre-eclampsia pathways; consider admission per protocol.</li>
                </ul>
                <p style="font-size: 0.9rem; font-style: italic;">Note: Record vitals, model probability, and top explainability features in the medical record.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed explanation section
        if show_explanation:
            st.subheader("🔬 Clinical Feature Contributions (XAI)")
            st.info(interpretation)
            
            explainer = load_explainer()
            if explainer is not None:
                with st.spinner("Analyzing feature contributions..."):
                    try:
                        # Get LIME explanation with raw units
                        lime_explanation = explainer.explain_with_lime(patient_data, num_features=6)
                        exp_list = lime_explanation.as_list()
                        
                        st.markdown("#### 🔍 Key Contributing Factors (Raw Units)")
                        st.markdown("Positive values increase risk, negative values decrease risk.")
                        
                        # Extract and sort
                        feature_names_sorted = [item[0] for item in exp_list]
                        contributions_sorted = [item[1] for item in exp_list]
                        
                        # Create feature contribution chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['#ef4444' if v > 0 else '#22c55e' for v in contributions_sorted]
                        ax.barh(feature_names_sorted, contributions_sorted, color=colors, alpha=0.7)
                        ax.set_xlabel('Contribution to Risk Prediction', fontsize=12)
                        ax.set_title('Feature Contribution Analysis (LIME)', fontsize=14, fontweight='bold')
                        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                        ax.grid(True, alpha=0.3, axis='x')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # SHAP Section with persistence fix
                        if shap is not None:
                            with st.expander("🔬 Advanced SHAP Waterfall Explanation"):
                                if st.button("Generate SHAP Waterfall Plot"):
                                    st.session_state.shap_requested = True
                                
                                if st.session_state.shap_requested:
                                    with st.spinner("Computing SHAP values..."):
                                        try:
                                            if explainer.shap_explainer is None:
                                                explainer.initialize_shap(background_samples=50)
                                            features_array = explainer.predictor.preprocess_input(patient_data)
                                            shap_values = explainer.shap_explainer.shap_values(features_array)
                                            
                                            target_idx = result['risk_level_numeric']
                                            
                                            # Handle different SHAP output formats (list for older versions/specific models, array for others)
                                            if isinstance(shap_values, list):
                                                sv = shap_values[target_idx][0]
                                                ev = explainer.shap_explainer.expected_value[target_idx]
                                            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                                                # Format: (observations, features, classes)
                                                sv = shap_values[0, :, target_idx]
                                                ev = explainer.shap_explainer.expected_value[target_idx]
                                            else:
                                                # fallback for binary or single-output
                                                sv = shap_values[0]
                                                ev = explainer.shap_explainer.expected_value
                                            
                                            explanation = shap.Explanation(
                                                values=sv, base_values=ev, data=features_array[0],
                                                feature_names=explainer.feature_names
                                            )
                                            plt.figure(figsize=(10, 8))
                                            shap.waterfall_plot(explanation, show=False)
                                            plt.title(f'SHAP Waterfall Plot - {risk_level}', fontsize=14, fontweight='bold')
                                            st.pyplot(plt)
                                            plt.close()
                                        except Exception as e:
                                            st.error(f"SHAP Error: {e}")
                    except Exception as e:
                        st.warning(f"⚠️ Explanation features currently limited: {str(e)}")
            else:
                st.info("ℹ️ Advanced explainability features (SHAP/LIME) are not available.")
        
        # Probability distribution visualization
        if show_probabilities:
            st.subheader("📈 Clinical Probability Distribution")
            
            # Create probability bar chart
            prob_df = pd.DataFrame({
                'Risk Level': ['Low', 'Mid', 'High'],
                'Probability': [result['probabilities']['Low Risk'], 
                               result['probabilities']['Mid Risk'], 
                               result['probabilities']['High Risk']]
            })
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(prob_df['Risk Level'], prob_df['Probability'], 
                   color=['#22c55e', '#f59e0b', '#ef4444'], alpha=0.7, edgecolor='black')
            
            # Add clinical threshold line
            ax.axvline(x=threshold, color='#ef4444', linestyle='--', linewidth=1.5, label=f'High-Risk Trigger ({threshold*100}%)')
            ax.legend(loc='lower right')
            
            ax.set_xlabel('Probability', fontsize=12)
            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3, axis='x')
            
            for i, prob in enumerate(prob_df['Probability']):
                ax.text(prob + 0.02, i, f'{prob*100:.1f}%', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        # Clinical Summary for records
        with st.expander("📄 View Assessment Summary (for clinical records)"):
            st.markdown("### Clinical Case Summary")
            summary_df = pd.DataFrame({
                'Clinical Descriptor': list(patient_data.keys()),
                'Measured Value': list(patient_data.values())
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            st.markdown("### Assessment Analytics")
            st.markdown(f"""
            - **Risk Level**: {risk_level}
            - **Model Confidence**: {confidence*100:.2f}%
            - **Low Risk Prob**: {result['probabilities']['Low Risk']*100:.2f}%
            - **Mid Risk Prob**: {result['probabilities']['Mid Risk']*100:.2f}%
            - **High Risk Prob**: {result['probabilities']['High Risk']*100:.2f}%
            """)
    
    # Batch Audit section
    st.markdown("---")
    st.header("📁 Clinical Audit Upload")
    
    with st.expander("Upload CSV for multi-case audit"):
        st.markdown("""
        The CSV must include: Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate.
        """)
        
        uploaded_file = st.file_uploader("Clinical data CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                if st.button("Run Batch Audit"):
                    predictor = load_predictor()
                    if predictor is not None:
                        with st.spinner("Processing audit..."):
                            results = predictor.predict_batch(batch_df)
                            batch_df['Clinical_Risk'] = [r['risk_level'] for r in results]
                            batch_df['Confidence'] = [f"{r['confidence']*100:.1f}%" for r in results]
                            st.success(f"✅ Audit complete for {len(results)} cases")
                            st.dataframe(batch_df, use_container_width=True)
                            
                            csv = batch_df.to_csv(index=False)
                            st.download_button("📥 Download Audit Results", csv, "audit_results.csv", "text/csv")
            except Exception as e:
                st.error(f"Audit Error: {e}")

    # Clinical Disclaimer and Research Prototype Notice
    st.markdown(f"""
    <div class="clinical-audit">
        <p><strong>IMPORTANT NOTICE:</strong> This system is a <strong>research prototype</strong> designed for pilot validation and prospective evaluation. 
        It is NOT a clinical diagnostic tool and is NOT approved for clinical deployment.</p>
        <p><strong>Clinical Use Disclaimer:</strong> This tool is a decision-support system only and does not constitute a clinical diagnosis. 
        It must be used in conjunction with a full clinical assessment, locally approved maternity guidelines, and professional medical judgment. 
        All predictions must be verified by qualified healthcare professionals.</p>
        <p><strong>Limitations:</strong> Model trained on limited dataset (n~1000). Performance metrics are based on retrospective data. 
        Requires prospective validation before any clinical consideration. Not evaluated for bias across different demographic groups.</p>
        <p>Model Version: <code>{predictor.VERSION if predictor else 'N/A'}</code> | 
        Ref: <code>MHR-V1-Research-Prototype</code> | 
        Timestamp: <code>{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</code></p>
    </div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
