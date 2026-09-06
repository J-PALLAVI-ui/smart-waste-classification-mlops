import mlflow
import mlflow.tensorflow

MODEL_NAME = "waste_classifier_efficientnet"

def register_model(model, accuracy):
    with mlflow.start_run(run_name="model_registry"):

        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.tensorflow.log_model(
            model,
            artifact_path="best_model",
            registered_model_name=MODEL_NAME
        )

    print("✅ Model registered")