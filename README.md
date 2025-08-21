📈 Stock Price Predictor App
This Streamlit application allows users to predict the next day's closing price for a given stock ticker and visualize its historical price trend. It uses a Random Forest Regressor model for predictions and fetches real-time stock data using the yfinance library.

✨ Features
Stock Price Prediction: Predicts the next day's closing price for any valid stock ticker.

Historical Data Visualization: Displays a line chart of the historical closing prices for the entered stock.

Model Accuracy: Shows the Mean Absolute Error (MAE) of the prediction model.

User-Friendly Interface: Built with Streamlit for an interactive web experience.

🚀 Getting Started
Follow these instructions to get a copy of the project up and running on your local machine.

Prerequisites
Before you begin, ensure you have the following installed:

Python 3.7+

pip (Python package installer)

Installation
Clone the repository (or create the file):
If you have a Git repository, clone it:

git clone <your-repository-url>
cd <your-repository-name>

Otherwise, create a new Python file (e.g., stock_predictor.py) and paste the provided code into it.

Create a virtual environment (recommended):

python -m venv venv

Activate the virtual environment:

On Windows:

.\venv\Scripts\activate

On macOS/Linux:

source venv/bin/activate

Install the required libraries:

pip install streamlit yfinance numpy pandas scikit-learn

💻 Usage
Run the Streamlit application:
Open your terminal or command prompt, navigate to the directory where you saved stock_predictor.py, and run:

streamlit run stock_predictor.py

Access the application:
A new tab will automatically open in your web browser, pointing to the Streamlit app (usually http://localhost:8501).

Enter a Stock Ticker:
In the input field, type a valid stock ticker symbol (e.g., AAPL for Apple, TSLA for Tesla, MSFT for Microsoft). The default is AAPL.

Get Prediction:
Click the "Predict Stock Price" button. The app will fetch the data, train the model, and display the predicted price, model accuracy (MAE), and a line chart of the stock's historical closing prices.

🧠 How it Works
The application performs the following steps:

Data Fetching: It uses the yfinance library to download historical stock data (last 5 years by default) for the specified ticker.

Feature Engineering: It calculates the daily percentage return (Return) and drops any rows with missing values.

Model Training:

Features (X): Open, High, Low, and Volume columns are used as input features.

Target (y): The Close price is the target variable to be predicted.

The data is split into training and testing sets.

A Random Forest Regressor model is trained on the training data.

Model Evaluation: The Mean Absolute Error (MAE) is calculated on the test set to provide an indication of the model's prediction accuracy. A lower MAE indicates a more accurate model.

Next Day Prediction: The model uses the latest available stock data (Open, High, Low, Volume) to predict the very next day's closing price.

Visualization: Streamlit's built-in charting capabilities are used to display the historical closing prices.

📚 Dependencies
The following Python libraries are essential for this application:

streamlit

yfinance

numpy

pandas

scikit-learn

All these dependencies are installed via pip install -r requirements.txt if you create one or individually as instructed in the installation section.

🤝 Contributing
Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please feel free to open an issue or submit a pull request.

📝 License
This project is open-source and available under the MIT License.
