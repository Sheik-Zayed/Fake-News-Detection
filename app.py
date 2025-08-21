import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("bert_fake_news_model")
    tokenizer = DistilBertTokenizerFast.from_pretrained("bert_fake_news_model")
    return model, tokenizer

model, tokenizer = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
        return pred, confidence

# Streamlit UI
st.title("🤖 Fake News Detection with DistilBERT")
st.markdown("Enter a news article below to check whether it's **Real** or **Fake** using a fine-tuned transformer model.")

user_input = st.text_area("📰 Paste News Article", height=200, placeholder="e.g., NASA has confirmed the landing of its latest rover on Mars...")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        with st.spinner("Analyzing with DistilBERT..."):
            label, confidence = predict(user_input)
            if label == 1:
                st.success(f"✅ This looks like **Real News** (Confidence: {confidence:.2%})")
            else:
                st.error(f"🚨 This appears to be **Fake News** (Confidence: {confidence:.2%})")

st.markdown("---")
st.caption("Model: DistilBERT | Trained on TF-IDF preprocessed dataset")
