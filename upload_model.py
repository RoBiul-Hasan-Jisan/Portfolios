import os
from huggingface_hub import login, upload_folder

def upload_model():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not found in environment variables!")
        return

    repo_id = "robiulhasanjisan88/Bangla-QA-BERT"
    folder_path = "models/chemistry_gpt2"  # Adjust if needed

    login(token=token)
    
    # Assuming repo already created manually on Hugging Face, so skip create_repo

    upload_folder(folder_path=folder_path, repo_id=repo_id, token=token)

    print("Model uploaded successfully!")

if __name__ == "__main__":
    upload_model()
