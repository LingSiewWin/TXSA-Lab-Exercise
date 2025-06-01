from transformers import pipeline

# Load a pre-trained language model
generator = pipeline('text-generation', model='distilgpt2')

# Get user input
prompt = input("Please enter a phrase: ")
length = int(input("Enter the sentence length (max 50): "))

# Generate text
outputs = generator(
    prompt,
    max_length=length,
    num_return_sequences=1,
    truncation=True
)

# Print result
print("\nGenerated sentence:")
print(outputs[0]['generated_text'])