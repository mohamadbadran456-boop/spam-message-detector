# AI Spam Message Detector

A beginner machine learning project that classifies English SMS messages as **spam** (unwanted) or **ham** (normal).

## Tools

- Python and Google Colab
- Pandas for loading and cleaning data
- Scikit-learn: TF-IDF, Multinomial Naive Bayes, and evaluation metrics
- Python standard library for downloading and opening the dataset

## Project file

`spam_message_detector.ipynb` contains the data loading, cleaning, training, evaluation, and interactive prediction cells.

## How to run

1. Open Google Colab: https://colab.research.google.com/
2. Upload `spam_message_detector.ipynb` from this repository.
3. Run the notebook cells in order. Internet access is needed to download the dataset.
4. When the final cell asks for a message, enter an English SMS and press Enter.
5. Run the final cell again to test another message.

If dependencies are missing, run this in a code cell first:

```python
!pip install pandas scikit-learn
```

## Dataset and preparation

The UCI SMS Spam Collection contains labeled SMS messages.
The loaded file had 5,574 rows. Removing 403 duplicate messages left 5,171 messages. No missing values were found in the loaded label and message columns.

| Category | Messages after cleaning |
| --- | ---: |
| Ham | 4,518 |
| Spam | 653 |

The data was split with `test_size=0.2`, `random_state=42`, and `stratify=y`:

- Training: 4,136 messages
- Testing: 1,035 messages

Duplicates were removed before splitting. A pipeline fits TF-IDF and Multinomial Naive Bayes on the training data only. Test messages are used for evaluation, not fitting.

## Model

`TfidfVectorizer()` converts message text into numerical features. `MultinomialNB()` learns to classify those features. Both use their default settings inside a Scikit-learn pipeline.

## Observed test results

These results were obtained from the notebook run on the split described above.

**Overall accuracy: 95.27%**

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Ham | 0.949 | 1.000 | 0.974 | 904 |
| Spam | 1.000 | 0.626 | 0.770 | 131 |

| Actual class | Predicted ham | Predicted spam |
| --- | ---: | ---: |
| Ham | 904 | 0 |
| Spam | 49 | 82 |

The model detected 82 of 131 spam messages and missed 49. No normal messages were classified as spam in this test. High overall accuracy alone does not fully describe performance because most messages are ham.

## Manual examples

| Input | Observed prediction |
| --- | --- |
| hello mohamad | HAM |
| Congratulations! You won a free prize. Call now to claim! | SPAM |

These two examples demonstrate the input feature; they are not a substitute for the test-set evaluation.

## Limitations

- Spam recall was 62.6%, so the model missed a substantial number of spam messages.
- Results describe one test split and do not guarantee future performance.
- This project is intended for English SMS, not all languages or message types.
- A HAM prediction does not guarantee that a message or link is safe.
- The current interface runs inside a notebook; it is not a deployed messaging filter.

## Learning outcomes

I practiced loading and cleaning data, preventing exact duplicate messages from appearing across the split, transforming text with TF-IDF, training a classifier, interpreting precision and recall, and testing user input.

## Possible future improvements

Compare alternative classifiers and parameters using cross-validation on the training data. Keep a separate final test set for evaluating the selected approach. Add a simple graphical interface.

## Dataset credit

Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository.

Source: https://doi.org/10.24432/C5CC84

Dataset license: Creative Commons Attribution 4.0 (CC BY 4.0).
