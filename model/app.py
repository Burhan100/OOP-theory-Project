import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------------------------
# Requirements:
#   None initially. Uses Streamlit's session_state to manage navigation and logic.
#
# Description:
#   Initializes necessary session state variables for user authentication, 
#   page control, and model storage across app pages.
#
# Purpose:
#   To ensure user state is maintained across pages and logic flows properly.
# ---------------------------
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'page' not in st.session_state:
    st.session_state['page'] = 'Login'
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'columns' not in st.session_state:
    st.session_state['columns'] = None
if 'target' not in st.session_state:
    st.session_state['target'] = None

# ---------------------------
# LOGIN PAGE
# ---------------------------
# Requirements:
#   Username and password input from the user.
#
# Description:
#   Renders login UI. Validates credentials and sets login session if matched.
#
# Purpose:
#   To restrict access to the app and control page flow securely.
# ---------------------------
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == "admin" and password == "123":
            st.session_state['authenticated'] = True
            st.session_state['page'] = 'Upload'
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")

# ---------------------------
# UPLOAD + MODEL TRAINING PAGE
# ---------------------------
# Requirements:
#   A CSV file containing at least 2 numeric columns.
#
# Description:
#   Lets user upload a CSV, choose the target column,
#   and trains a Linear Regression model on the rest.
#
# Purpose:
#   To allow dynamic model training based on uploaded user data.
# ---------------------------
def upload_page():
    st.title("Upload CSV & Train Model")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:", df.head())

        # Extract numeric columns for training
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("CSV must contain at least 2 numeric columns")
            return

        # Let user select target variable
        target_column = st.selectbox("Select target column (to predict)", numeric_cols)
        input_columns = [col for col in numeric_cols if col != target_column]

        # Train/test split and model training
        X = df[input_columns]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Save model and metadata
        st.session_state['model'] = model
        st.session_state['columns'] = input_columns
        st.session_state['target'] = target_column
        st.success(f"Model trained to predict `{target_column}` from {input_columns}")

        if st.button("Go to Prediction Page"):
            st.session_state['page'] = 'Predict'
            st.rerun()

# ---------------------------
# PREDICTION PAGE
# ---------------------------
# Requirements:
#   Trained model and input column list must be available in session_state.
#
# Description:
#   Takes user input values for each feature and uses the trained model
#   to predict the target value (e.g., student marks).
#
# Purpose:
#   To provide a user-friendly way to make predictions from the model.
# ---------------------------
def predict_page():
    st.title("Predict Student Score")

    if st.session_state['model'] is None:
        st.warning("Please upload and train model first.")
        st.session_state['page'] = 'Upload'
        st.rerun()


    # Input fields for each feature
    inputs = {}
    for col in st.session_state['columns']:
        inputs[col] = st.number_input(f"Enter value for `{col}`", step=0.1)

    # Make prediction and show result
    if st.button("Predict"):
        input_df = pd.DataFrame([inputs])
        prediction = st.session_state['model'].predict(input_df)[0]
        st.success(f"Predicted `{st.session_state['target']}`: {prediction:.2f}")

    # ---------------------------
    # LOGOUT SECTION
    # ---------------------------
    # Requirements:
    #   None. Accessed from within a logged-in session.
    #
    # Description:
    #   Provides a logout button to reset the session and return to login.
    #
    # Purpose:
    #   To safely end the session and prevent further access without re-login.
    # ---------------------------
    st.markdown("---")
    st.subheader("Logout")
    col1, col2 = st.columns([4, 1])
    with col1:
        st.info("Click the button below to logout and return to the login screen.")
    with col2:
        if st.button("Logout"):
            st.session_state['authenticated'] = False
            st.session_state['page'] = 'Login'
            st.success("You have been logged out.")
            st.rerun()
 
# ---------------------------
# MAIN CONTROL FLOW
# ---------------------------
# Requirements:
#   None (controls navigation between pages).
#
# Description:
#   Controls which page to show based on the session state.
#
# Purpose:
#   To manage multi-page logic flow in a single-page Streamlit app.
# ---------------------------
def main():
    st.set_page_config(page_title="Student Score Predictor", layout="centered")

    if not st.session_state['authenticated']:
        login_page()
    elif st.session_state['page'] == 'Upload':
        upload_page()
    elif st.session_state['page'] == 'Predict':
        predict_page()

if __name__ == "__main__":
    main()
