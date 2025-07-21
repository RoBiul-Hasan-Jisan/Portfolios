from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def generate_answer(model, tokenizer, prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('models/chemistry_gpt2')
    model = GPT2LMHeadModel.from_pretrained('models/chemistry_gpt2')
    model.eval()

    print("Ask questions about Chemistry (type 'exit' to quit):")
    while True:
        question = input("Question: ")
        if question.lower() == 'exit':
            break
        answer = generate_answer(model, tokenizer, question)
        print("Answer:", answer, "\n")

if __name__ == "__main__":
    main()
