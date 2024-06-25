# text_prediction.py
import openai

# Set your OpenAI API key
openai.api_key = 'your-api-key-here'

def predict_next_words(input_text, max_length=50):
    """
    Predict the next words in a sequence using GPT-4.
    
    Parameters:
    input_text (str): The input text for which to predict the continuation.
    max_length (int): The maximum number of tokens to generate.
    
    Returns:
    str: The predicted continuation of the input text.
    """
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=input_text,
        max_tokens=max_length,
        n=1,
        stop=None,
        temperature=0.7
    )
    predicted_text = response.choices[0].text.strip()
    return predicted_text

# Example usage
if __name__ == "__main__":
    input_text = "Hello, how can I assist"
    predicted_text = predict_next_words(input_text)
    print("Predicted continuation:", predicted_text)
