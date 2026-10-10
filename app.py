import joblib
import numpy as np
import streamlit as st
from scipy.sparse import hstack


@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("vectorizer.joblib")
    scaler = joblib.load("scaler.joblib")
    model = joblib.load("model.joblib")
    return vectorizer, scaler, model


def extra_features(texts):
    # Must be IDENTICAL to the function in your notebook
    return np.array([
        [
            sum(c.isdigit() for c in t),
            sum(c.isupper() for c in t) / max(len(t), 1),
            len(t),
        ]
        for t in texts
    ])


vectorizer, scaler, model = load_artifacts()

st.title("SMS Spam Classifier")
st.write("Type or paste a text message and the model will estimate whether it is spam.")

message = st.text_area("Message", height=120)

if st.button("Check message"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        text_vec = vectorizer.transform([message])
        extra = scaler.transform(extra_features([message]))
        features = hstack([text_vec, extra])

        prob_spam = model.predict_proba(features)[0][1]

        if prob_spam >= 0.5:
            st.error(f"Likely SPAM ({prob_spam:.0%} confidence)")
        else:
            st.success(f"Looks like a normal message ({1 - prob_spam:.0%} confidence)")

        st.progress(float(prob_spam))
        st.caption("Bar shows the model's estimated probability of spam.")

st.caption(
    "Trained on the SMS Spam Collection dataset (older UK text messages), "
    "so it may miss modern scam styles."
)