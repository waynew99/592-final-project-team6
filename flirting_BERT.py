from transformers import pipeline

# Load sentiment analysis pipeline with the pre-trained model
classifier = pipeline('sentiment-analysis', 
                      model="distilbert-base-uncased-finetuned-sst-2-english")

# The piece of text you want to classify
text = "I love using BERT for natural language processing tasks!"

# Perform sentiment analysis
result = classifier(text)

# Print the result
print(result)
