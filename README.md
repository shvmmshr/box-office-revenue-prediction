# Box Office Revenue Prediction System

## Overview
Predict box office revenue using machine learning. Analyze features like titles, genres, directors, actors, and descriptions to forecast financial success. We created our own database from scratch using GoLang and Selenium web scraping with ChromeDriver. Trailer analysis will be added in a future update.

## Kaggle Notebook
Explore the full implementation here:  
[Box Office Revenue Prediction System on Kaggle](https://www.kaggle.com/code/shvmmshra/box-office-revenue-prediction-system)

## Features
- Custom Database (movie metadata, revenue)  
- Trailer analysis planned for future  
- Feature Extraction (NLP on titles, genres, etc.)  
- Machine Learning Models (Logistic Regression, SVM, Decision Trees, etc.)  
- Ensemble Approach for improved accuracy  

## Dataset
- Title  
- Year  
- Rate  
- Duration  
- Genre1, Genre2, Genre3  
- Director, Writer  
- Actor1, Actor2  
- Language  
- Description  
- Verdict (Hit, Flop, etc.)  

## Installation
```bash
git clone https://github.com/shvmmshr/box-office-revenue-prediction.git
cd Notebooks
pip install -r requirements.txt
jupyter notebook box-office-revenue-prediction-system.ipynb
```

## Usage
- Data Preprocessing (missing values, encoding)  
- Model Training (multiple ML models)  
- Model Evaluation (accuracy, F1, etc.)  
- Prediction for new movies  

## Results
| Algorithm               | Accuracy | Precision | Recall | F1 Score |
|-------------------------|----------|-----------|--------|----------|
| Logistic Regression     | 0.55     | 0.68      | 0.55   | 0.60     |
| Support Vector Machine  | 0.11     | 0.01      | 0.11   | 0.02     |
| Decision Tree           | 0.57     | 0.59      | 0.57   | 0.58     |
| Random Forest           | 0.69     | 0.51      | 0.69   | 0.58     |
| XGBoost                 | 0.68     | 0.58      | 0.68   | 0.61     |
| LightGBM                | 0.68     | 0.57      | 0.68   | 0.61     |

## Future Work
- Add trailer data to the database  
- Expand dataset with social media sentiment, cast popularity  
- Advanced models (deep learning)  
- User-friendly interface  

## License
Licensed under the MIT License. See LICENSE for details.