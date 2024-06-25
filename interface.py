# interface.py
import openai
import numpy as np
import torch
from tacotron2.model import Tacotron2
from tacotron2.hparams import create_hparams
from tacotron2.train import load_model
from waveglow.denoiser import Denoiser
from scipy.io.wavfile import write

# Set your OpenAI API key
openai.api_key = 'your-api-key-here'

# Load pre-trained Tacotron2 model
hparams = create_hparams()
hparams.sampling_rate = 22050
checkpoint_path = "tacotron2_statedict.pt"
tacotron2 = load_model(hparams)
tacotron2.load_state_dict(torch.load(checkpoint_path)['state_dict'])
tacotron2.eval()

# Load pre-trained WaveGlow model
waveglow_path = 'waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval()
denoiser = Denoiser(waveglow)

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

def text_to_speech(text, output_file="output.wav"):
    """
    Convert text to speech using Tacotron 2 and WaveGlow.
    
    Parameters:
    text (str): The text to be converted into speech.
    output_file (str): The file path to save the generated audio.
    """
    sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    mel_outputs, mel_outputs_postnet, _, alignments = tacotron2.inference(sequence)
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    audio = denoiser(audio, strength=0.01)[:, 0]
    audio = audio * 32768.0
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    write(output_file, hparams.sampling_rate, audio)

def main():
    """
    Main function to run the interactive interface for the AI-based Communication System.
    """
    print("Welcome to the AI-based Communication System for Cerebral Palsy")
    while True:
        user_input = input("Enter your text (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        predicted_text = predict_next_words(user_input)
        print("Predicted continuation:", predicted_text)
        text_to_speech(predicted_text)
        print("Speech generated and saved to output.wav")

if __name__ == "__main__":
    main()
