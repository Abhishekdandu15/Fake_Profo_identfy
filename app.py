import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
from pathlib import Path

# ---------------------------
# Session State Initialization
# ---------------------------
def init_session_state():
    defaults = {
        "users": {},
        "logged_in": False,
        "current_user": None,
        "otp": None,
        "temp_user": {},
        "show_otp_input": False,
        "prediction_history": [],
        "model_trained": False,
        "train_columns": []  # to store feature names from training
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state immediately
init_session_state()

# ---------------------------
# Cache and Data Functions
# ---------------------------
@st.cache_data
def load_and_preprocess_data(file):
    """Load and preprocess data with caching"""
    df = pd.read_csv(file, encoding="ISO-8859-1")
    return preprocess_data(df)

@st.cache_data
def preprocess_data(df):
    """Preprocess the dataset efficiently with caching"""
    drop_columns = ["id", "name", "screen_name", "created_at", "updated", "profile_image_url", 
                    "profile_banner_url", "profile_background_image_url_https", "profile_image_url_https", 
                    "profile_background_image_url", "description"]
    
    df_cleaned = df.copy()
    # Drop unnecessary columns
    df_cleaned = df_cleaned.drop(columns=[col for col in drop_columns if col in df_cleaned.columns])
    
    # Efficient null handling and conversion
    if "location" in df_cleaned.columns:
        df_cleaned.loc[:, "location"] = df_cleaned["location"].fillna("Unknown")
        # Convert location to categorical codes (numeric)
        df_cleaned.loc[:, "location"] = df_cleaned["location"].astype("category").cat.codes
    if "default_profile" in df_cleaned.columns:
        df_cleaned.loc[:, "default_profile"] = df_cleaned["default_profile"].fillna(0)
    if "profile_background_tile" in df_cleaned.columns:
        df_cleaned.loc[:, "profile_background_tile"] = df_cleaned["profile_background_tile"].fillna(0)
    
    return df_cleaned

@st.cache_data
def train_models_with_cache(X_train, X_test, y_train, y_test):
    """
    Train models with caching.
    Convert features using get_dummies and align train/test sets.
    Return trained models, performance metrics, and the list of training columns.
    """
    # Convert categorical/numeric features into dummy variables
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    # Save the training feature names for later use
    train_columns = list(X_train.columns)
    
    # Initialize SVM Classifier
    svm_clf = SVC(
        C=1.0,
        kernel='rbf',
        probability=True,
        random_state=42
    )
    
    # Initialize Naive Bayes Classifier
    nb_clf = GaussianNB()
    
    # Train both models
    svm_clf.fit(X_train, y_train)
    nb_clf.fit(X_train, y_train)
    
    # Predictions and metrics
    y_pred_svm = svm_clf.predict(X_test)
    y_pred_nb = nb_clf.predict(X_test)
    
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    report_svm = classification_report(y_test, y_pred_svm)
    report_nb = classification_report(y_test, y_pred_nb)
    
    return svm_clf, nb_clf, accuracy_svm, report_svm, accuracy_nb, report_nb, train_columns

@st.cache_data
def create_prediction_visualizations(input_data, probability, prediction_history):
    """Create cached visualizations for predictions"""
    # Pie chart of prediction probabilities
    fig_pie = px.pie(
        values=[probability[1], probability[0]],
        names=['Fake', 'Real'],
        title='Prediction Probability Distribution',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    
    # Feature distribution chart
    feature_names = ['Statuses', 'Followers', 'Friends', 'Location', 'Default Profile', 'Background Tile']
    fig_features = go.Figure()
    fig_features.add_trace(go.Scatter(
        x=feature_names,
        y=input_data[0],
        mode='lines+markers',
        name='Current Profile',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=10)
    ))
    fig_features.update_layout(
        title='Feature Distribution',
        xaxis_title='Features',
        yaxis_title='Value',
        hovermode='x'
    )
    
    # History chart if available
    if prediction_history:
        history_df = pd.DataFrame(prediction_history)
        fig_history = px.line(
            history_df,
            x='timestamp',
            y='confidence',
            color='prediction',
            title='Prediction History',
            labels={'confidence': 'Confidence Score', 'prediction': 'Prediction Type'},
            color_discrete_map={'Fake Profile': '#FF6B6B', 'Real Profile': '#4ECDC4'}
        )
    else:
        fig_history = None
    
    return fig_pie, fig_features, fig_history

# ---------------------------
# Helper Functions
# ---------------------------
def generate_otp():
    """Generate a random 6-digit OTP"""
    return random.randint(100000, 999999)

def save_model(model, filename):
    """Save trained model to disk"""
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, f"models/{filename}")

def load_model(filename):
    """Load trained model from disk"""
    try:
        return joblib.load(f"models/{filename}")
    except Exception as e:
        return None

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Fake Profile Detection",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Navigation
# ---------------------------
page = st.sidebar.selectbox("Navigation", ["Login", "Register", "Fake Profile Detection"])

# ---------------------------
# Registration Page
# ---------------------------
if page == "Register":
    st.title("User Registration")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username", key="reg_username")
            password = st.text_input("Password", type="password", key="reg_password")
        with col2:
            phone_number = st.text_input("Phone Number (with country code)", key="reg_phone")
        
        if st.button("Register"):
            if username in st.session_state["users"]:
                st.error("Username already exists!")
            elif not username or not phone_number or not password:
                st.error("Please fill in all fields.")
            else:
                otp = generate_otp()
                st.session_state["otp"] = otp
                st.session_state["temp_user"] = {
                    "username": username,
                    "phone_number": phone_number,
                    "password": password
                }
                st.session_state["show_otp_input"] = True
                st.info(f"🔐 Your OTP is: {otp}")

        if st.session_state.get("show_otp_input", False):
            otp_input = st.text_input("Enter OTP")
            if st.button("Verify OTP"):
                if otp_input and st.session_state.get("otp"):
                    if int(otp_input) == st.session_state["otp"]:
                        user_details = st.session_state["temp_user"]
                        st.session_state["users"][user_details["username"]] = user_details
                        st.success("✅ Registration successful! Please proceed to login.")
                        st.session_state["otp"] = None
                        st.session_state["show_otp_input"] = False
                        st.session_state["temp_user"] = {}
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ Incorrect OTP. Please try again.")

# ---------------------------
# Login Page
# ---------------------------
if page == "Login":
    st.title("User Login")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username", key="login_username")
        with col2:
            password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            users = st.session_state.get("users", {})
            if username in users and users[username]["password"] == password:
                st.session_state["logged_in"] = True
                st.session_state["current_user"] = username
                st.success(f"Welcome back, {username}! 👋")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

# ---------------------------
# Fake Profile Detection Page
# ---------------------------
if page == "Fake Profile Detection":
    if not st.session_state.get("logged_in", False):
        st.error("⚠️ Please log in to access this page.")
    else:
        st.title("🕵️‍♂️ Fake Profile Detection App")
        
        with st.sidebar:
            st.write(f"👤 Logged in as: {st.session_state.get('current_user', 'Unknown')}")
            if st.button("Logout"):
                st.session_state["logged_in"] = False
                st.session_state["current_user"] = None
                st.experimental_rerun()
            
            if st.button("Clear History"):
                st.session_state["prediction_history"] = []
                st.success("History cleared!")
        
        uploaded_file = st.file_uploader("📤 Upload Training Data (CSV)", type=["csv"])
        
        if uploaded_file is not None:
            with st.spinner("Processing data..."):
                df = load_and_preprocess_data(uploaded_file)
                st.write("### Data Preview")
                st.dataframe(df.head())
                
                features = ["statuses_count", "followers_count", "friends_count", 
                            "location", "default_profile", "profile_background_tile"]
                
                X = df[features]
                y = df["Label"].astype(int)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if st.button("Train Models"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        status_text.text(f"Training in progress: {i+1}%")
                        time.sleep(0.01)
                    
                    svm_clf, nb_clf, accuracy_svm, report_svm, accuracy_nb, report_nb, train_columns = train_models_with_cache(
                        X_train, X_test, y_train, y_test
                    )
                    
                    # Save the expected feature names in session state
                    st.session_state["train_columns"] = train_columns
                    
                    save_model(svm_clf, "svm_model.joblib")
                    save_model(nb_clf, "nb_model.joblib")
                    
                    st.session_state["model_trained"] = True
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("### Support Vector Machine Results:")
                        st.metric("Accuracy", f"{accuracy_svm:.2%}")
                        st.text(report_svm)
                    
                    with col2:
                        st.write("### Naïve Bayes Results:")
                        st.metric("Accuracy", f"{accuracy_nb:.2%}")
                        st.text(report_nb)
        
        st.write("## 🎯 Predict Fake Profile")
        
        # Add model selection option
        model_choice = st.radio("Select Model for Prediction", ["Support Vector Machine", "Naïve Bayes"])
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                statuses_count = st.number_input("Statuses Count", min_value=0)
                followers_count = st.number_input("Followers Count", min_value=0)
            with col2:
                friends_count = st.number_input("Friends Count", min_value=0)
                location = st.number_input("Location Code", min_value=0)
            with col3:
                default_profile = st.selectbox("Default Profile", [0, 1])
                profile_background_tile = st.selectbox("Profile Background Tile", [0, 1])
            
            submitted = st.form_submit_button("Predict")
            if submitted:
                if not st.session_state.get("model_trained", False):
                    st.error("⚠️ Please train the model first!")
                else:
                    # Load the selected model
                    model_filename = "svm_model.joblib" if model_choice == "Support Vector Machine" else "nb_model.joblib"
                    selected_model = load_model(model_filename)
                    
                    if selected_model is None:
                        st.error(f"⚠️ {model_choice} model not found. Please train the model first!")
                    else:
                        # Create input DataFrame from form values
                        input_df = pd.DataFrame([{
                            "statuses_count": statuses_count,
                            "followers_count": followers_count,
                            "friends_count": friends_count,
                            "location": location,
                            "default_profile": default_profile,
                            "profile_background_tile": profile_background_tile
                        }])
                        
                        # Convert input data using get_dummies and reindex to match training columns
                        input_data = pd.get_dummies(input_df)
                        train_columns = st.session_state.get("train_columns", [])
                        input_data = input_data.reindex(columns=train_columns, fill_value=0)
                        
                        prediction = selected_model.predict(input_data)[0]
                        probability = selected_model.predict_proba(input_data)[0]
                        
                        result = "Fake Profile" if prediction == 1 else "Real Profile"
                        confidence = probability[1] if prediction == 1 else probability[0]
                        
                        st.session_state["prediction_history"].append({
                            'timestamp': pd.Timestamp.now(),
                            'prediction': result,
                            'confidence': confidence,
                            'features': input_df.iloc[0].tolist(),
                            'model_used': model_choice
                        })
                        
                        st.write("### 📊 Prediction Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Prediction", result)
                        with col2:
                            st.metric("Confidence", f"{confidence:.2%}")
                        with col3:
                            st.metric("Model Used", model_choice)
                        
                        fig_pie, fig_features, fig_history = create_prediction_visualizations(
                            input_df.values, probability, st.session_state["prediction_history"]
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig_pie, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig_features, use_container_width=True)
                        
                        if fig_history:
                            st.plotly_chart(fig_history, use_container_width=True)
        
        if st.session_state["prediction_history"]:
            st.write("### 📜 Prediction History")
            history_df = pd.DataFrame(st.session_state["prediction_history"])
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            st.dataframe(
                history_df[['timestamp', 'prediction', 'confidence', 'model_used']].style.format({
                    'confidence': '{:.2%}'
                }).background_gradient(subset=['confidence'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            st.write("### 📤 Export Options")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Export Prediction History"):
                    csv = history_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="prediction_history.csv",
                        mime="text/csv"
                    )
            with col2:
                if st.button("Export SVM Model"):
                    if st.session_state.get("model_trained", False):
                        model = load_model("svm_model.joblib")
                        if model is not None:
                            with open("models/svm_model.joblib", "rb") as f:
                                model_bytes = f.read()
                                st.download_button(
                                    label="Download SVM Model",
                                    data=model_bytes,
                                    file_name="fake_profile_detector_svm.joblib",
                                    mime="application/octet-stream"
                                )
                        else:
                            st.error("SVM model file not found. Please train the model first.")
                    else:
                        st.error("Please train the model first!")
            with col3:
                if st.button("Export Naïve Bayes Model"):
                    if st.session_state.get("model_trained", False):
                        model = load_model("nb_model.joblib")
                        if model is not None:
                            with open("models/nb_model.joblib", "rb") as f:
                                model_bytes = f.read()
                                st.download_button(
                                    label="Download Naïve Bayes Model",
                                    data=model_bytes,
                                    file_name="fake_profile_detector_nb.joblib",
                                    mime="application/octet-stream"
                                )
                        else:
                            st.error("Naïve Bayes model file not found. Please train the model first.")
                    else:
                        st.error("Please train the model first!")
        
        st.write("---")
        if st.session_state["prediction_history"]:
            st.write("### 📊 Overall Statistics")
            
            # Create tabs for overall stats and model comparison
            tab1, tab2 = st.tabs(["Overall Statistics", "Model Comparison"])
            
            with tab1:
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                total_predictions = len(st.session_state["prediction_history"])
                fake_profiles = sum(1 for p in st.session_state["prediction_history"] if p["prediction"] == "Fake Profile")
                real_profiles = total_predictions - fake_profiles
                avg_confidence = sum(p["confidence"] for p in st.session_state["prediction_history"]) / total_predictions
                with stats_col1:
                    st.metric("Total Predictions", total_predictions)
                with stats_col2:
                    st.metric("Fake Profiles Detected", fake_profiles)
                with stats_col3:
                    st.metric("Real Profiles Detected", real_profiles)
                with stats_col4:
                    st.metric("Average Confidence", f"{avg_confidence:.2%}")
            
            with tab2:
                if 'model_used' in history_df.columns:
                    # Group by model and calculate statistics
                    model_stats = history_df.groupby('model_used').agg(
                        count=('prediction', 'count'),
                        avg_confidence=('confidence', 'mean'),
                        fake_profiles=('prediction', lambda x: sum(x == 'Fake Profile')),
                        real_profiles=('prediction', lambda x: sum(x == 'Real Profile'))
                    )
                    
                    # Create a bar chart for model comparison
                    fig_model_comp = px.bar(
                        model_stats.reset_index(),
                        x='model_used',
                        y=['fake_profiles', 'real_profiles'],
                        title='Model Comparison: Fake vs Real Profiles',
                        labels={'value': 'Count', 'variable': 'Profile Type'},
                        barmode='group',
                        color_discrete_map={'fake_profiles': '#FF6B6B', 'real_profiles': '#4ECDC4'}
                    )
                    st.plotly_chart(fig_model_comp, use_container_width=True)
                    
                    # Display model statistics
                    st.write("### Model Performance Comparison")
                    st.dataframe(
                        model_stats.style.format({
                            'avg_confidence': '{:.2%}'
                        }),
                        use_container_width=True
                    )
        
        with st.expander("ℹ️ Help & Instructions"):
            st.write("""
            ### How to use this app:
            1. **Upload Training Data**: Start by uploading a CSV file containing profile data.
            2. **Train the Model**: Click the 'Train Models' button to train the detection models.
            3. **Select Model**: Choose either Support Vector Machine or Naïve Bayes for prediction.
            4. **Make Predictions**: Enter profile details and click 'Predict' to analyze.
            5. **View Results**: Check the visualizations and prediction history.
            6. **Export Data**: Download your prediction history or trained models.
            
            ### Feature Descriptions:
            - **Statuses Count**: Number of posts/statuses.
            - **Followers Count**: Number of followers.
            - **Friends Count**: Number of friends/following.
            - **Location Code**: Encoded location value.
            - **Default Profile**: Whether the profile uses default settings (0/1).
            - **Profile Background Tile**: Whether the profile has a tiled background (0/1).
            
            ### Model Information:
            - **Support Vector Machine (SVM)**: A powerful algorithm that finds the optimal hyperplane to separate classes. Works well with high-dimensional data.
            - **Naïve Bayes**: A probabilistic classifier based on Bayes' theorem. Efficient and performs well with smaller datasets.
            
            ### Tips:
            - Ensure your training data is properly formatted.
            - Monitor the confidence scores for reliability.
            - Compare the performance of both models to determine which works best for your data.
            - Regularly export your prediction history.
            - Retrain the models with new data periodically.
            """)
        
        with st.expander("ℹ️ About"):
            st.write("""
            ### Fake Profile Detection App
            This application uses machine learning to detect potentially fake social media profiles. 
            It employs both Support Vector Machine (SVM) and Naïve Bayes algorithms for high accuracy prediction.
            
            **Features:**
            - Real-time profile analysis
            - Model selection (SVM or Naïve Bayes)
            - Interactive visualizations
            - Historical tracking with model comparison
            - Model export capabilities
            - Comprehensive statistics
            
            **Version:** 1.0.0
            """)

st.markdown("""
    <style>
    .stMetric .label {
        font-size: 14px !important;
    }
    .stProgress .st-bo {
        background-color: #4CAF50 !important;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
