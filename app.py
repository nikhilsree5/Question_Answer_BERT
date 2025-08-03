import streamlit as st
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Load model and tokenizer
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

st.title("🧠 BERT Question Answering Agent")
st.write("Enter a context and a question. The model will extract the most probable answer.")

context = st.text_area("Context", height=200)
question = st.text_input("Question")

if not context:
    st.write("Enter a valid context")
elif not question:
    st.write("Enter a valid question")
elif context and question:
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt', truncation=True, padding=True, max_length=512)
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=token_type_ids)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits)

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[0][start:end+1])
    )

    st.success(f"Answer: {answer}")