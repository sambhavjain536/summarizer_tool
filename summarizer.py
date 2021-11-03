"""
from transformers import T5ForConditionalGeneration, T5Tokenizer

# initialize the model architecture and weights
model = T5ForConditionalGeneration.from_pretrained("t5-base")
# initialize the model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def Abstractive_Summarizer(article):
    ln = len(article)
    mxl = (ln*50)//100
    req_mxl = (mxl*50)//100
    req_mnl = (mxl*15)//100
    # encode the text into tensor of integers using the appropriate tokenizer
    inputs = tokenizer.encode("summarize: " + article, return_tensors="pt", max_length=mxl, truncation=True)
    # generate the summarization output
    outputs = model.generate(
        inputs, 
        max_length=req_mxl, 
        min_length=req_mnl, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True)
    # just for debugging
    #print(outputs)
    req_summary = tokenizer.decode(outputs[0])
    return str(req_summary)
"""

def Abstractive_Summarizer(article):
    return str(article)