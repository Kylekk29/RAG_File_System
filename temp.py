import os
import sys
from huggingface_hub import snapshot_download

def download_embedding_models():
    # 1. Primary Model configuration from your .env
    primary_repo = "BAAI/bge-m3"
    primary_local_dir = os.path.join("AImodels", "embedding_model", "BAAI", "bge-m3")
    
    # 2. Fallback Model configuration from your .env
    fallback_repo = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    fallback_local_dir = os.path.join("AImodels", "embedding_model", "sentence-transformers", "paraphrase-multilingual-MiniLM-L12-v2")

    models_to_download = [
        {"repo": primary_repo, "dir": primary_local_dir, "name": "Primary (BGE-M3)"},
        {"repo": fallback_repo, "dir": fallback_local_dir, "name": "Fallback (MiniLM)"}
    ]

    for model in models_to_download:
        print("=" * 60)
        print(f"Starting download for {model['name']}...")
        print(f"Hugging Face Repo: {model['repo']}")
        print(f"Target Directory:  {model['dir']}")
        print("=" * 60)
        
        try:
            # snapshot_download downloads the entire model repository repository
            snapshot_download(
                repo_id=model['repo'],
                local_dir=model['dir'],
                local_dir_use_symlinks=False,  # Copies actual files, ideal for moving across systems
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Skips redundant framework weights
            )
            print(f"✅ Success! {model['name']} downloaded completely.\n")
        except Exception as e:
            print(f"❌ Failed to download {model['name']}: {e}\n")

if __name__ == "__main__":
    download_embedding_models()