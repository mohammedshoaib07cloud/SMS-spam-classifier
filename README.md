# SMS Spam Classifier

**[Live demo → https://sms-spam-classifier-yvbziuhxvqb7cqwjoapppck.streamlit.app/]**

![App screenshot](screenshot.png)

A machine learning project that classifies SMS messages as spam or ham (not spam), with a Streamlit web app for trying it out.

## Problem
Spam messages waste time and can be scams. The goal was to build a classifier that catches as much spam as possible without blocking real messages.

## Dataset
SMS Spam Collection (Kaggle), about 5,572 messages, labelled spam or ham. About 86.6% were ham, so the classes are imbalanced. I removed 403 duplicate messages, which shifted the balance to [5,169] ham. Check `df["label"].value_counts(normalize=True)` for your exact number.

## Approach
1. Cleaned the data and removed duplicates
2. Split into 80% train and 20% test (stratified, so both sets keep the same spam ratio)
3. Converted text to numbers with TF-IDF, fitted on training data only to avoid data leakage
4. Trained three models and compared them
5. Analysed the messages the model got wrong

## Results

  Model	                         Accuracy	Precision	Recall	   F1  
 0 TF-IDF + Naive Bayes	           0.966      0.990	     0.740	  0.847
 1 TF-IDF + Logistic Regression	   0.976	  0.921	     0.885	  0.903
 2 + extra features	               0.979	  0.916	     0.916	  0.916


**Why not just accuracy?** About 87% of messages are ham, so a model that always says "ham" would score about 87% accuracy while catching no spam. Precision and recall show what actually matters.

**What I found:**
- Naive Bayes was very cautious. It wrongly blocked only 1 real message, but missed 34 of 131 spam messages in the test set (recall 0.74).
- Logistic Regression with balanced class weights raised recall to 0.885, at the cost of lower precision.
- Adding hand-made features (digit count, share of capital letters, message length) raised recall further, to 0.916. TF-IDF with lowercasing was discarding the capital-letter signal that is obvious to a human reader of spam ("FREE", "WINNER").
- I chose the third model because it made the fewest total mistakes (roughly 22 compared with 35 for Naive Bayes). This assumes flagged messages go to a reviewable spam folder. If messages were deleted outright, the higher-precision Naive Bayes would be the safer choice, since it blocks far fewer real messages.

## Limitations
- The dataset is small and mostly older UK-style texts, so modern scams (fake delivery notices, bank phishing links) may slip through
- The test set has only about 131 spam messages, so small differences between models may not hold up on other splits
- Results come from a single train/test split, not cross-validation

## What I'd improve next
- Use cross-validation for more reliable comparisons
- Try character-level n-gram features
- Tune the decision threshold to trade recall against precision
- Train on a larger, more recent dataset

## Run it locally
```
git clone https://github.com/mohammedshoaib07cloud/SMS-spam-classifier.git
cd SMS-spam-classifier
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack
Python, pandas, scikit-learn, Streamlit