import os
import openai
import json

# Set up your OpenAI API key
openai.api_key = 'OPENAI-API-KEY'

# Define the text data for training
training_data = """
Once upon a time, there was a brave knight.
He embarked on a dangerous quest to rescue the princess from the evil dragon.

In a distant land, a young wizard discovered a hidden spellbook.
With the newfound knowledge, he set out to save his village from an ancient curse.

Deep in the forest, a group of friends found a magical portal.
They entered the portal and were transported to a mystical realm filled with mythical creatures.
"""

# Set the training configuration
training_config = {
    'model': 'gpt-3.5-turbo',  # Model to train with
    'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': training_data}],
    'max_tokens': 1000,  # Maximum number of tokens to train on
    'temperature': 0.9,  # Controls the randomness of the output text
}

# Function to train the model
def train_model(training_data, config):
    # Fine-tune the model with the provided text data
    response = openai.ChatCompletion.create(
        model=config['model'],
        messages=config['messages'],
        max_tokens=config['max_tokens'],
        temperature=config['temperature'],
        n=1,
        stop=None
    )
    return response

# Function to generate predictions from the trained model
def generate_prediction(model_id, user_message):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': user_message}],
        max_tokens=1200,
        temperature=0.9,
        n=1,
        stop=None
    )
    return response.choices[0].message.content

# Train the model and get the response
response = train_model(training_data, training_config)

# Save the trained model to a file
model_id = response['model']
model_path = 'trained_model/model.json'
with open(model_path, 'w') as f:
    json.dump(response, f)

print('Training complete. Model saved.')

# Example usage of the generate_prediction function
user_message = "Pease explain what chatgpt is?"
prediction = generate_prediction(model_id, user_message)
print('User Message:', user_message)
print('Model Prediction:', prediction)
