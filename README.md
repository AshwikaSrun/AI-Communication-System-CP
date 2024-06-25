# AI-based Communication System for Cerebral Palsy

This project provides an AI-based solution to enhance communication for individuals with cerebral palsy using advanced NLP techniques.

## Overview

The system integrates GPT-4 for text prediction and Tacotron 2 with WaveGlow for text-to-speech synthesis, creating a functional prototype that demonstrates the potential of AI in transforming communication aids.

## Setup

### Install Necessary Libraries

```bash
pip install openai torch transformers scipy
git clone https://github.com/NVIDIA/tacotron2
cd tacotron2
pip install -r requirements.txt

Set Your OpenAI API Key 
Replace 'your-api-key-here' in the scripts with your actual OpenAI API key.

Usage
Text Prediction
Run text_prediction.py to generate text continuations using GPT-4.

Text-to-Speech
Run text_to_speech.py to convert text into speech using Tacotron 2 and WaveGlow.

Interactive Interface
Run interface.py to use the interactive command-line interface for text prediction and text-to-speech synthesis.

# Example usage of text prediction
input_text = "Hello, how can I assist"
predicted_text = predict_next_words(input_text)
print("Predicted continuation:", predicted_text)

# Example usage of text-to-speech
text = "Hello, how can I assist you today?"
text_to_speech(text)
