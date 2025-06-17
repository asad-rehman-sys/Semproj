from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize the Flask application
app = Flask(__name__)

# Load a pretrained LLM model and tokenizer (e.g., GPT2 from Hugging Face)
# You can replace 'gpt2' with another model like 'EleutherAI/gpt-neo-125M' if needed
MODEL_NAME = "gpt2"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Set model to evaluation mode and use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

@app.route('/generate', methods=['POST'])
def generate_text():
    """
    Flask route to generate text using the LLM.
    Expects a JSON payload with a 'prompt' field.
    """
    data = request.get_json()

    if 'prompt' not in data:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    prompt = data['prompt']
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate continuation using the model
    outputs = model.generate(
        **inputs,
        max_length=100,  # Maximum number of tokens in the output
        do_sample=True,  # Sampling for diversity
        top_k=50,        # Top-k sampling
        top_p=0.95,      # Nucleus sampling
        temperature=0.9  # Controls randomness
    )

    # Decode and return the result
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": generated_text})

if __name__ == '__main__':
    # Run the Flask server on localhost at port 5000
    app.run(host='0.0.0.0', port=5000)
