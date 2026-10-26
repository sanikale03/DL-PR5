# Deep Learning PR 5 — Summary Report
## Sentiment Analysis with RNN, LSTM and GRU

**Institute:** Red & White Skill Education  
**Subject:** Deep Learning  
**Practical:** PR 5  
**Task:** Binary sentiment classification of movie reviews  
**Dataset:** IMDB Dataset of 50K Movie Reviews  
**Framework:** TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib and Seaborn

---

## 1. Project Overview

The purpose of this practical is to classify movie reviews as **positive** or **negative** using recurrent neural network architectures. The notebook compares an order-blind Artificial Neural Network (ANN) baseline with SimpleRNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU) models.

The supplied practical brief specifies the raw Kaggle IMDB CSV with `review` and `sentiment` columns. The expected target encoding is positive = 1 and negative = 0. The provided notebook is designed to load `IMDB Dataset.csv` from the same directory.

## 2. Objectives

- Inspect the dataset, missing values, class balance and duplicate reviews.
- Clean HTML fragments and text noise while retaining apostrophes and sentiment-bearing words.
- Split the data into training and test sets before fitting the tokenizer.
- Tokenize and pre-pad/truncate reviews to a consistent sequence length.
- Explain RNN sequence processing and manually verify a NumPy RNN forward pass against Keras.
- Train and evaluate ANN, SimpleRNN, LSTM and GRU models.
- Study vanishing/exploding gradients, sequence-length sensitivity and gradient clipping.
- Compare model metrics, inspect errors and implement a reusable sentiment prediction function.

## 3. Dataset Description

The practical brief describes 50,000 reviews, balanced between 25,000 positive and 25,000 negative reviews. The CSV is expected to contain:

| Column | Description |
|---|---|
| `review` | Raw English movie-review text |
| `sentiment` | Positive or negative target label |

Reviews may contain HTML tags, mixed case, punctuation, contractions and varying lengths. Duplicate removal is performed before the train/test split to reduce data leakage risk.

**Dataset source:** [Kaggle — IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
**Original source cited in the brief:** Maas et al. (2011), Stanford Large Movie Review Dataset.

## 4. Methodology

1. **Load and inspect:** print the dataset shape, sample rows, class counts and null counts.
2. **Remove duplicates:** count exact duplicate rows and remove them before splitting.
3. **Clean text:** lowercase, remove HTML tags, retain letters and apostrophes, and collapse repeated whitespace.
4. **Explore lengths:** calculate words per review and plot a histogram and sentiment-wise boxplot.
5. **Split data:** use an 80/20 stratified train/test split with `random_state=42`.
6. **Tokenize:** fit `Tokenizer(num_words=10000, oov_token='<OOV>')` on training text only.
7. **Pad sequences:** use pre-padding and pre-truncation at the selected `MAXLEN`.
8. **Train models:** compare the ANN baseline, SimpleRNN, LSTM, GRU and a stacked LSTM.
9. **Evaluate:** calculate accuracy, precision, recall, F1 and ROC-AUC; display confusion matrices and ROC curves.
10. **Analyse errors:** inspect false positives and false negatives and test custom sentences with the inference function.

## 5. Model Architectures

| Model | Main architecture | Key property |
|---|---|---|
| ANN baseline | Embedding → Flatten → Dense → Sigmoid | Does not explicitly model word-to-word recurrence |
| SimpleRNN | Embedding → SimpleRNN → Sigmoid | Shares recurrent weights across time steps |
| LSTM | Embedding → LSTM → Sigmoid | Uses a cell state and forget/input/output gates |
| GRU | Embedding → GRU → Sigmoid | Uses update/reset gates and no separate cell state |
| Stacked LSTM | Embedding → LSTM(64, sequences) → LSTM(32) → Sigmoid | Adds recurrent depth and computational cost |

For an embedding dimension of 32 and 64 recurrent units, the practical brief gives these recurrent-layer parameter formulas:

- SimpleRNN: `units × (input_dim + units + 1)` = **6,208**
- LSTM: `4 × units × (input_dim + units + 1)` = **24,832**
- GRU with `reset_after=True`: `3 × (units × input_dim + units × units + 2 × units)` = **18,816**

These are recurrent-layer counts, not total model parameter counts; the embedding and output layers add parameters.

## 6. Core Concepts

### RNN and sequence order
An RNN reads tokens sequentially and updates a hidden state at each time step. Reusing the same recurrent weights keeps the recurrent parameter count independent of sequence length.

### Backpropagation Through Time
BPTT unrolls the recurrent network over the sequence and propagates gradients backwards through time. Products of recurrent Jacobians can cause gradients to vanish or explode.

### LSTM
The LSTM cell state provides an additive information path. The forget gate controls retention, the input gate controls new information, and the output gate controls what is exposed as the hidden state.

### GRU
The GRU combines memory control into update and reset gates. It has fewer gate parameter sets than an LSTM and can offer a useful efficiency trade-off.

### Gradient clipping
Gradient clipping limits the norm of large gradients and can reduce exploding-gradient instability. It does not, by itself, solve vanishing gradients.

## 7. Evaluation Plan

The notebook produces the following outputs after execution with the real IMDB CSV:

- Review-length histogram and boxplot by sentiment.
- Manual NumPy RNN hidden-state heatmap and Keras verification.
- Training/validation loss and accuracy curves.
- Confusion matrices and classification reports.
- Gradient-norm plots and sequence-length experiment.
- High-learning-rate comparison with and without gradient clipping.
- Results table and combined ROC curves.
- Accuracy by review length, misclassified review examples and custom-sentence predictions.

**Results note:** No model scores are reported in this document because the full training pipeline has not been run here against the real Kaggle dataset. Do not treat demo-mode output as final experimental evidence. After running all notebook cells, transfer the actual values from `results_table.csv` into the final report if your institute requires numerical findings.

## 8. Expected Interpretation

The experiment is intended to show whether gated recurrent architectures retain useful context better than a basic SimpleRNN, and how the trade-off in accuracy, parameter count and training time changes across models. The final recommendation must be based on the metrics and error examples produced by the actual run, not on a pre-assumed winner.

## 9. Limitations

- The embeddings are learned from the training data and are not pretrained.
- The models read the review in one direction.
- Sarcasm, mixed sentiment, negation and very long reviews can still be difficult to classify.
- Results depend on the dataset, sequence length, training settings and available hardware.

## 10. Conclusion

This practical provides an end-to-end workflow for binary movie-review sentiment analysis, from raw-text cleaning and tokenization through recurrent-model training, evaluation and inference. The final comparison of ANN, SimpleRNN, LSTM and GRU should be completed after executing the notebook on the required IMDB CSV and reviewing the generated plots and metrics.

## 11. References

1. Maas, A. L. et al. (2011). *Learning Word Vectors for Sentiment Analysis*. Association for Computational Linguistics.
2. Kaggle dataset: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
3. TensorFlow/Keras documentation: https://www.tensorflow.org/
