from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name='sentance_transformers/all-MiniLM-L6-v2')

# for document through local model
documents = [
    'Delhi is the capital of india',
    'my name is pranob kale',
    'kolcatta is the capital of west bengal',
    'paris is the capital of france'
]

vector1 = embedding_model.embed_documents(documents)

print(vector1)


# for query through local model
vector = embedding_model.embed_query('Delhi is the capital of India')

print(vector)