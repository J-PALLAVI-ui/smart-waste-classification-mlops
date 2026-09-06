from src.train import train
from src.register_model import register_model

def run_pipeline():
    print("🚀 Starting Pipeline")

    accuracy, model = train()

    print(f"✅ Training done. Accuracy: {accuracy}")

    register_model(model, accuracy)

    print("✅ Pipeline completed")

if __name__ == "__main__":
    run_pipeline()