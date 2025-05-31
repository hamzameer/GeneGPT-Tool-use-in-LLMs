import pandas as pd
from . import metrics


def create_log_table(results: dict) -> pd.DataFrame | None:
    """Log the results to a table."""
    # Prepare data for MLflow table
    table_data = []
    for category, questions_answers in results.items():
        for question, details in questions_answers.items():
            answer = details.get("answer")
            prediction = details.get("prediction")
            thoughts = details.get("thoughts")

            exact_match = None
            levenshtein_dist = None
            partial_match_score = None

            try:
                exact_match = metrics.calculate_exact_match(answer, prediction)
            except Exception as e:
                print(f"Error calculating exact_match for question '{question}': {e}")

            try:
                levenshtein_dist = metrics.calculate_levenshtein_distance(
                    answer, prediction
                )
            except Exception as e:
                print(f"Error calculating levenshtein_distance for question '{question}': {e}")

            try:
                partial_match_score = metrics.calculate_partial_match(
                    answer, prediction
                )
            except Exception as e:
                print(f"Error calculating partial_match for question '{question}': {e}")

            table_data.append(
                {
                    "category": category,
                    "question": question,
                    "ground_truth_answer": answer,
                    "thoughts": thoughts,
                    "prediction": prediction,
                    "match": exact_match,
                    "levenshtein_distance": levenshtein_dist,
                    "partial_match": partial_match_score,
                }
            )
    if len(table_data) == 0:
        print("No data to log as a table.")
        return None
    else:
        table_df = pd.DataFrame(table_data)
        return table_df


def log_metrics(results_df: pd.DataFrame) -> dict:
    """Log the metrics to MLflow."""
    overall_accuracy = results_df["match"].mean()
    avg_levenshtein_distance = results_df["levenshtein_distance"].mean()
    overall_partial_match_accuracy = results_df["partial_match"].mean()

    metrics_to_log = {
        "overall_accuracy": overall_accuracy,
        "overall_average_levenshtein_distance": avg_levenshtein_distance,
        "overall_partial_match_accuracy": overall_partial_match_accuracy,
    }

    # Calculate metrics per category
    if "category" in results_df.columns:
        category_metrics = results_df.groupby("category").agg(
            accuracy=("match", "mean"),
            average_levenshtein_distance=("levenshtein_distance", "mean"),
            partial_match_accuracy=("partial_match", "mean"),
        )
        for category, row in category_metrics.iterrows():
            metrics_to_log[f"{category}_accuracy"] = row["accuracy"]
            metrics_to_log[f"{category}_average_levenshtein_distance"] = row[
                "average_levenshtein_distance"
            ]
            metrics_to_log[f"{category}_partial_match_accuracy"] = row["partial_match_accuracy"]

    return metrics_to_log
