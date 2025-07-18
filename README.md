# IMDB Sentiment Analysis with LSTM

This project uses Long Short-Term Memory (LSTM) networks to classify IMDB movie reviews as positive or negative. The task involves building deep learning models capable of understanding the sequential nature of text and making binary sentiment predictions.

This project was completed as part of a **peer-graded assignment** for the **Introduction to Deep Learning** course.

## Files

**Included in this repository:**
- `imdb_sentiment_analysis_lstm.ipynb`: Jupyter notebook containing exploratory data analysis, text preprocessing, LSTM model building, training, evaluation, and performance comparison.
- `imdb_sentiment_analysis_lstm.pdf`: PDF version of the full notebook run including EDA, model training, evaluation, and final results — ideal for quick review without setting up the environment.
- `baseline_lstm_val_predictions.csv`: CSV file with validation set predictions (labels and probabilities) from the Baseline LSTM model.
- `bidirectional_lstm_val_predictions.csv`: CSV file with validation set predictions from the Bidirectional LSTM model.
- `stacked_lstm_val_predictions.csv`: CSV file with validation set predictions from the Stacked LSTM model.

## Models

Three LSTM-based models were trained and compared:
1. **Baseline LSTM**: Embedding layer + single LSTM layer, dropout = 0.5, learning rate = 0.001  
2. **Bidirectional LSTM**: Bidirectional LSTM layer with dropout = 0.3  
3. **Stacked LSTM**: Two stacked LSTM layers with batch normalization, dropout = 0.3, learning rate = 0.0005  

## Results

The **Baseline LSTM** model achieved the highest ROC AUC (~0.930), demonstrating excellent overall classification performance despite being the simplest architecture. The **Bidirectional LSTM** showed the highest recall and F1 score, indicating strong ability to correctly identify positive sentiment cases by leveraging context from both directions in the text. The **Stacked LSTM** exhibited the highest precision but had lower recall and F1 scores, suggesting a more conservative prediction behavior.

Validation curves, training logs, and detailed performance comparisons are included in the notebook for further analysis.

## Requirements

- Python 3.10 or higher  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

Install dependencies with:

```bash
pip install pandas scikit-learn matplotlib seaborn numpy 
