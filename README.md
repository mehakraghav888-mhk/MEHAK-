# Credit Card Fraud Detection

A beginner-friendly Machine Learning project that predicts whether a credit-card transaction is potentially fraudulent.

## Technology
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## Project structure
- `src/generate_dataset.py` - creates a demo transaction dataset.
- `src/train_model.py` - trains and evaluates the ML model.
- `app.py` - Streamlit web application for single-transaction prediction.
- `requirements.txt` - required Python packages.
- `data/transactions.csv` - generated demo data after running the generator.
- `models/fraud_model.joblib` - trained model created by the training script.
- `reports/` - project documentation.

## How to run

1. Install Python 3.10+.
2. Open a terminal in this project folder.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Generate the demo dataset:
   `python src/generate_dataset.py`
5. Train the model:
   `python src/train_model.py`
6. Start the web app:
   `streamlit run app.py`

The browser will open the Streamlit application.

## Important academic note
The included dataset is synthetic and is intended to make the project reproducible without requiring a large external dataset. For a real-world deployment, use an approved real transaction dataset, stronger validation, class-imbalance handling, monitoring, and security/privacy controls.

## GitHub
Upload all project files to a GitHub repository. Do not upload passwords, API keys, or private financial information.
