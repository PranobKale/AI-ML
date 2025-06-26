from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embbedings = OpenAIEmbeddings(model='text embedding-3-large', dimensions=300)

document = [
    "Virat Kohli, Batsman, Right-handed, India",
    "Sachin Tendulkar, Batsman, Right-handed, India",
    "Jasprit Bumrah, Bowler, Right-arm fast, India",
    "MS Dhoni, Wicketkeeper-Batsman, Right-handed, India"
]

query = 'tell me anout virat'

doc_embeddings = embbedings.embed_documents(document)
query_embeddings = embbedings.embed_query(document)

# source = cosine_similarity([query_embeddings],doc_embeddings) # need to send the data in the 2d list
# note :- here we get 2d list of the similarity scores eg. [[0.642135810,0.34254,0.321458,0.32147]]
# print(source)   

scores = cosine_similarity([query_embeddings],doc_embeddings)[0] # need to send the data in the 2d list

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(document[index])
print('similarity score :-',score)