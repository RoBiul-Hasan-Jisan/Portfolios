from transformers import pipeline

def main():
    pipe = pipeline("question-answering", model="robiulhasanjisan88/Bangla-QA-BERT")
    
    context = """
    Chemistry is the branch of science that deals with the composition of compounds, structure, properties, uses, etc.
    """
    question = "What is chemistry?"

    result = pipe(question=question, context=context)
    print(f"Answer: {result['answer']} (score: {result['score']:.4f})")

if __name__ == "__main__":
    main()
