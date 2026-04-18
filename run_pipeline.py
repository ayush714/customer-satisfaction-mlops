from pipelines.training_pipeline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    train_pipeline()

    try:
        tracking_uri = get_tracking_uri()
        print(
            "Now run \n "
            f"    mlflow ui --backend-store-uri '{tracking_uri}'\n"
            "To inspect your experiment runs within the mlflow UI.\n"
            "You can find your runs tracked within the `mlflow_example_pipeline`"
            " experiment. Here you'll also be able to compare the two runs."
        )
    except ValueError:
        print(
            "Pipeline finished successfully, but no MLflow experiment tracker is "
            "configured on the active ZenML stack. Configure an MLflow tracker "
            "to enable the MLflow UI command hint."
        )
