# Bank Marketing Classification App
This is a Streamlit app for analyzing and predicting customer responses on bank marketing campaigns, based on the [Kaggle notebook](https://www.kaggle.com/code/amri11/binary-classification-with-a-bank) by Amri.

## Project Overview

The app lets you:
 - Explore the bank marketing dataset with interactive visualizations
 - View statistical summaries and correlation heatmaps
 - Customize plots to dig deeper into the data
 - Make predictions on customer data using a pre-trained model

## Features
 - Data preview and basic statistics
 - Distribution charts (bar and pie) for categorical features
 - Correlation heatmap for numeric variables
 - Custom bar, scatter, and box plots
 - Time series analysis by month
 - Customer response prediction with probability scores

## How to Run
 1. Clone this repository
    ```bash
    git clone https://github.com/AmriDomas/Binary-Classification-with-a-Bank.git
    cd Binary-Classification-with-a-Bank
    ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app
   ```bash
   streamlit run streamlit_bankhf.py
   ```
## File Structure

 - `streamlit_bankhf.py`: Main Streamlit app
 - `bank_model.pkl`: Pre-trained classification model ([Model in Hugingface](https://huggingface.co/11amri/xgboostbank/main/bank_model.pkl))
 - `train.csv`: Bank marketing dataset ([Dataset in Hugingface](https://huggingface.co/datasets/11amri/banktrain/main/train.csv))
 - `requirements`.txt: Python dependencies

## Data Source

The dataset and baseline model are based on the Kaggle notebook [Binary Classification with a Bank Marketing Dataset](https://www.kaggle.com/code/amri11/binary-classification-with-a-bank).

## Notes
 - For large model and data files, consider hosting them externally (e.g., Hugging Face Hub) and loading them dynamically in the app.
 - This project showcases data exploration, visualization, and model inference integrated into a user-friendly interface.

## Author
Amri – Data Science & ML Engineer

 - [Linkedin](https://www.linkedin.com/in/muh-amri-sidiq/)
 - [Kaggle](https://www.kaggle.com/amri11)
