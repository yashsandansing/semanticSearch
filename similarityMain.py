import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

#Tokenizer for punctuation removal
tokenizer = RegexpTokenizer(r'\w+')
#For stop-word removal
stop = set(stopwords.words('english'))
#Bert-encoder
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def cleanData(inputText):
    '''
    Clean the data by the same format that the data in database was cleaned by
    Returns:
        lowered text with punctuations and stopwords removed
    '''
    #Remove punctuations
    inputText = " ".join(tokenizer.tokenize(inputText))
    #Remove stopwords
    inputText = ' '.join([word.lower() for word in inputText.split() if word.lower() not in (stop)])
    return inputText

def findRelevantProducts(cleanedText, top_k):
    '''
    Retrieve top-k results with the help of cosine similarity
    Returns: 
        Top-k similar products
    '''
    #encode the text with the same encoder
    inputEmbedding = model.encode(cleanedText, convert_to_tensor=True)
    #calculate cosine similarity of input text w.r.t the database
    cos_scores = util.cos_sim(inputEmbedding, file["embeddings"])[0]
    #retrieve top-k results
    top_results = torch.topk(cos_scores, k=top_k)
    
    return [list(file[file.index == idx]["productLink"]) for idx in top_results[1].tolist()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input text for the type of dress/shirt/cloth you want to search for")
    parser.add_argument("--N", type=int, default=5, help="Number of products to return")
    args = parser.parse_args()

    #Dataframe in the format [productLinks, description, embeddings using bert encoder]
    file = pd.read_pickle("embeddingsFinal.pkl")
    
    text = cleanData(args.input)
    links = findRelevantProducts(text, args.N)

    print(len(links))