# 🧠 Deep Learning PR-5 — Sentiment Analysis with RNN, LSTM and GRU

## 📌 Project Overview

This practical implements a text-classification workflow using recurrent neural network architectures and the **IMDB Dataset of 50K Movie Reviews**. The project follows the seven tasks in the PR-5 brief, moving from raw-text cleaning and tokenisation to RNN forward propagation, gradient analysis, LSTM and GRU models, evaluation, and sentiment prediction.

The implementation is provided in Jupyter Notebook format using Python and TensorFlow/Keras. It is designed to generate training curves, confusion matrices, ROC curves, gradient plots, comparison tables, and inference examples when the notebook is executed.

**Important:** model accuracy, loss, ROC-AUC and other performance results must come from running the notebook on the real Kaggle dataset. This README does not claim unverified model-performance numbers.

---

## 🎯 Objectives

- Load and inspect the IMDB review dataset.
- Check missing values, class balance and duplicate reviews.
- Remove duplicates before the train/test split to reduce data leakage.
- Clean raw review text while retaining apostrophes and sentiment-bearing words.
- Explore review lengths with a histogram and sentiment-wise boxplot.
- Split the data using stratification and `random_state=42`.
- Fit a tokenizer on training text only and pad/truncate sequences.
- Build an order-blind ANN baseline and compare it with recurrent models.
- Implement and verify a manual NumPy RNN forward pass against Keras.
- Explain RNN sequence types, Backpropagation Through Time (BPTT), vanishing gradients and exploding gradients.
- Train SimpleRNN, LSTM, GRU and a stacked LSTM model.
- Compare models using accuracy, precision, recall, F1 and ROC-AUC.
- Analyse misclassified reviews and test a reusable sentiment prediction function.
- Save plots and results for the final submission.

---


## 📺 Presentation Video

👉 **https://drive.google.com/file/d/1kmwR3SfuSy2SD7GK4B7VeyXmdcdiLr2x/view?usp=sharing**
------------------------------------------------------------------------

## 📂 Project Input — IMDB Dataset

Download `IMDB Dataset.csv` from:

👉 https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Place the CSV in the same working directory as `DL_PR5.ipynb`.

### Dataset details

| Property | Description |
|---|---|
| Dataset | IMDB Dataset of 50K Movie Reviews |
| Expected file | `IMDB Dataset.csv` |
| Expected rows | 50,000 before duplicate removal |
| Columns | `review`, `sentiment` |
| Labels | `positive`, `negative` |
| Task | Binary text classification |
| Input | Raw English movie reviews |

The practical brief cites the original Large Movie Review Dataset by Maas et al. (2011). Reviews may contain HTML fragments, punctuation, contractions and different lengths. Duplicate reviews are removed before splitting to reduce the risk of data leakage.

**Note:** The notebook may use a small demo dataset if the CSV is missing, so it can demonstrate code flow. Demo-mode results are not valid final experimental results. Use the real CSV for your submission.

---

## 🧠 Concepts Covered

### 1. Text cleaning and tokenisation

The `clean_text()` function lowercases text, removes HTML tags, retains letters and apostrophes, and collapses repeated whitespace. Apostrophes and words such as `not` and `no` are retained because they can change sentiment.

The tokenizer is configured with `num_words=10000` and an out-of-vocabulary token. It is fitted only on training text. Sequences are pre-padded and pre-truncated to a fixed length.

### 2. ANN baseline vs RNN

The ANN baseline uses an embedding layer, flattening, and dense layers. It does not explicitly process a review as a sequence. An RNN reads one token at a time and carries a hidden state forward. Its recurrent weights are shared across time steps.

A basic RNN hidden-state equation is:

\[
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)
\]

### 3. RNN sequence mappings

- **One-to-one:** one input vector maps to one output.
- **One-to-many:** one input maps to an output sequence, such as image captioning.
- **Many-to-one:** a sequence maps to one label; this project uses this pattern for sentiment analysis.
- **Many-to-many:** a sequence maps to a sequence, such as token tagging.

### 4. BPTT and gradient problems

Backpropagation Through Time unrolls the RNN over time and propagates gradients backwards through its steps. Repeated gradient factors can shrink, causing vanishing gradients, or grow, causing exploding gradients. Gradient clipping limits large gradient norms but does not directly solve vanishing gradients.

### 5. LSTM

LSTM uses a cell state and forget, input and output gates. The cell state provides a path for information to be retained over many time steps.

### 6. GRU

GRU uses update and reset gates and has no separate cell state. Its gate structure is simpler than LSTM's and can provide a useful parameter-count and training-speed trade-off.

---

## 🚀 Project Workflow

```text
Start
  ↓
Install Dependencies and Import Libraries
  ↓
Task 1 — Load, Inspect and Clean Reviews
  ↓
Remove Duplicates → Split Data → Tokenise → Pad Sequences
  ↓
Task 2 — ANN Baseline and Manual RNN Forward Pass
  ↓
Task 3 — RNN Types and SimpleRNN Sentiment Model
  ↓
Task 4 — BPTT, Gradient Flow, Sequence Length and Clipping
  ↓
Task 5 — LSTM Model and Stacked LSTM
  ↓
Task 6 — GRU Model and Hidden-Unit Sensitivity
  ↓
Task 7 — Results, ROC Curves, Error Analysis and Inference
  ↓
Save Plots, Results and Final Submission Files
```

---

## 🧪 Practical Tasks — 1 to 7

### Task 1 — Loading, cleaning, tokenisation and padding

This section loads the CSV, prints its shape, class counts and null counts, displays raw reviews, counts/removes duplicates, cleans text and explores review lengths.

**Expected outputs:**
- Dataset inspection and duplicate-removal summary.
- Before/after text-cleaning example.
- Review-length histogram with mean, median and 95th-percentile lines.
- Review-length boxplot by sentiment.
- Tokenizer vocabulary examples and final train/test shapes.

### Task 2 — RNN intuition, ANN baseline and forward propagation

This section trains an order-blind ANN baseline, implements an RNN forward pass using NumPy and compares the hidden states with a Keras `SimpleRNN` layer.

**Expected outputs:**
- ANN model summary, parameter count and test metrics.
- Manual hidden-state values and heatmap.
- `np.allclose` verification against Keras.
- ANN vs RNN comparison table.

### Task 3 — RNN types and SimpleRNN sentiment model

This section demonstrates one-to-one, one-to-many, many-to-one and many-to-many output shapes. It then trains and evaluates a SimpleRNN sentiment model with early stopping.

**Expected outputs:**
- Sequence-type output-shape table.
- Training/validation loss and accuracy curves.
- Classification report, confusion matrix and ROC-AUC.
- Embedding weight shape and example word vectors.

### Task 4 — BPTT and gradient problems

This section explains BPTT, vanishing gradients and exploding gradients, and includes experiments involving gradient norms, sequence length and gradient clipping.

**Expected outputs:**
- Gradient norm plot for SimpleRNN.
- Test accuracy versus sequence length.
- Loss curves with and without gradient clipping.
- Summary table of gradient problems and remedies.

### Task 5 — LSTM

This section explains the LSTM cell state and gates, verifies the parameter-count formula, and trains an LSTM and a stacked LSTM.

**Expected outputs:**
- LSTM training curves and evaluation metrics.
- Confusion matrix.
- Stacked LSTM evaluation and training curves.
- Gradient comparison with SimpleRNN, where supported by the executed notebook.

### Task 6 — GRU

This section explains update and reset gates, trains the GRU model, compares gradient flow, and evaluates the effect of changing hidden units.

**Expected outputs:**
- GRU training curves and evaluation metrics.
- Gradient-norm comparison for SimpleRNN, LSTM and GRU.
- Accuracy and parameter-count plots for 32, 64 and 128 hidden units.

### Task 7 — Final comparison, inference and recommendation

This section combines model results, plots ROC curves, evaluates accuracy by review length, displays misclassified examples and tests `predict_sentiment(text)` on custom sentences.

**Expected outputs:**
- Results table for ANN baseline, SimpleRNN, LSTM, GRU and stacked LSTM where available.
- Combined ROC curves.
- Accuracy by short, medium and long review length.
- False-positive and false-negative examples.
- Predictions for at least six custom sentences.
- Final recommendation based on measured results.

---

## 📊 Plots and Outputs

The notebook is designed to generate plots such as:

- `plots/length_distribution.png`
- `plots/length_by_sentiment.png`
- `plots/rnn_forward_heatmap.png`
- `plots/training_curves_simplernn.png`
- `plots/training_curves_lstm.png`
- `plots/training_curves_gru.png`
- `plots/training_curves_stacked_lstm.png`
- `plots/gradient_norms_rnn.png`
- `plots/gradient_norms_all_cells.png`
- `plots/accuracy_vs_sequence_length.png`
- `plots/gradient_clipping.png`
- `plots/hidden_unit_sensitivity.png`
- `plots/roc_curves.png`
- `plots/accuracy_by_length.png`
- `plots/results_table.csv`

Some outputs are created only after their corresponding notebook cells execute successfully. Review the actual files produced by your run before submission.

---

## 📁 Recommended Project Folder Structure

```text
DL_PR5/
├── DL_PR5.ipynb
├── DL_PR5.html
├── README.md
├── Summary_Report.md
├── requirements.txt
├── results_table.csv
└── plots/
    ├── length_distribution.png
    ├── length_by_sentiment.png
    ├── rnn_forward_heatmap.png
    ├── training_curves_simplernn.png
    ├── training_curves_lstm.png
    ├── training_curves_gru.png
    ├── gradient_norms_all_cells.png
    ├── accuracy_vs_sequence_length.png
    ├── gradient_clipping.png
    ├── hidden_unit_sensitivity.png
    ├── roc_curves.png
    └── accuracy_by_length.png
```

The list above is a recommended structure. Include only files that actually exist after running the notebook.

---

## 🛠️ Technologies Used

- **Python** — programming language.
- **TensorFlow / Keras** — neural-network layers, training and evaluation.
- **NumPy** — numerical operations and manual RNN calculations.
- **Pandas** — dataset handling and result tables.
- **Matplotlib** — plotting.
- **Seaborn** — statistical visualisations.
- **Scikit-learn** — train/test split and evaluation metrics.
- **Jupyter Notebook** — interactive practical implementation.

---

## 📦 Installation

Open a terminal in the project directory and run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, if `python` is not recognised, try:

```bash
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open `DL_PR5.ipynb`, ensure `IMDB Dataset.csv` is in the working directory, then choose **Kernel → Restart & Run All**.

### Requirements

The notebook uses these packages:

```text
tensorflow
scikit-learn
pandas
numpy
matplotlib
seaborn
nbformat
```

TensorFlow compatibility depends on your Python version and operating system. If installation fails, use a Python version supported by the TensorFlow release you install.

---

## 📈 Results and Recommendation

Use the actual `results_table.csv` created by the notebook to compare accuracy, precision, recall, F1, ROC-AUC, parameter count and time per epoch. Do not insert estimated or demo-mode metrics as final results.

The best recurrent cell should be selected based on the measured performance, runtime, parameter cost, long-review behaviour and error analysis. LSTM and GRU are designed to manage information over longer sequences than a basic SimpleRNN, but the best choice for this dataset must be established from the executed experiment.

### Limitations

- Word embeddings are learned from this dataset; pretrained embeddings are not used.
- The recurrent models read reviews in one direction.
- Sarcasm, negation, mixed sentiment and very long reviews can still be difficult to classify.
- Results depend on preprocessing, sequence length, model configuration and hardware.

---

## 📚 References

1. Maas, A. L. et al. (2011). *Learning Word Vectors for Sentiment Analysis*. Association for Computational Linguistics.
2. [Kaggle — IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
3. [TensorFlow documentation](https://www.tensorflow.org/)

---

## 🎥 Presentation Video

Add the link to your 5–10 minute face-and-screen explanation video here before submitting.

## 👩‍💻 Author

**Sanika Kale**  
MCA Student | AI/ML & Data Science  
Red & White Skill Education

---

*This README documents the project workflow and expected outputs. Final performance claims should be based on the results generated by running the notebook on the real IMDB dataset.*
