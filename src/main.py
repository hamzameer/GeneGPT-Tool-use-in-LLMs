"""
Main script to process datasets with LLMs (Azure OpenAI or Ollama).

Takes a dataset, model provider, and model name as input,
calls the respective LLM API, and appends the results to the dataset.
"""

import argparse
import concurrent.futures
import mlflow
from tqdm import tqdm

from dotenv import load_dotenv

from .file_io import load_json, save_json, load_yaml
from .reporting import create_log_table, log_metrics
from .llm_interface import (
    get_client,
    call_llm,
    call_llm_with_tools,
)

# Default values if not found in config, though config.yaml should provide them
DEFAULT_MAX_TURNS = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5
DEFAULT_MAX_WORKERS = 10

load_dotenv()


def process_single_question(
    client,
    model_name: str,
    question: str,
    tool_use: bool,
    max_turns: int,
    max_retries: int,
    retry_delay: int,
    ground_truth_answer: str,
) -> tuple[str, dict]:
    """Helper function to process a single question. To be run in a thread."""
    try:
        if tool_use:
            llm_response = call_llm_with_tools(
                client,
                model_name,
                question,
                max_turns,
                max_retries,
                retry_delay,
            )
        else:
            llm_response = call_llm(
                client, model_name, question, max_retries, retry_delay
            )
    except Exception as e:
        print(f"Failed to call LLM for question '{question[:50]}...': {e}")
        return question, {
            "answer": ground_truth_answer,
            "thoughts": f"Critical error in processing: {e}",
            "prediction": "ERROR_PROCESSING",
        }

    if llm_response:
        print(f"LLM response: {llm_response.answer}")
        return question, {
            "answer": ground_truth_answer,
            "thoughts": llm_response.thoughts,
            "prediction": llm_response.answer,
        }
    else:
        return question, {
            "answer": ground_truth_answer,
            "thoughts": "LLM call returned no response after retries.",
            "prediction": None,
        }


def process_dataset(
    provider: str, model_name: str, dataset: dict, tool_use: bool, config: dict
) -> dict:
    """
    Processes each question in the dataset using the LLM and appends results.
    Uses ThreadPoolExecutor for concurrent question processing.
    """
    client = get_client(provider)
    results = {}

    max_turns = config.get("MAX_TURNS", DEFAULT_MAX_TURNS)
    max_retries = config.get("MAX_RETRIES", DEFAULT_MAX_RETRIES)
    retry_delay = config.get("RETRY_DELAY", DEFAULT_RETRY_DELAY)
    max_workers = config.get("MAX_WORKERS", DEFAULT_MAX_WORKERS)
    print(f"Using up to {max_workers} concurrent workers for question processing.")

    all_questions_with_category = []
    for category, questions_answers in dataset.items():
        print(f"Processing category: {category}")
        results[category] = {}
        for question, ground_truth_answer in questions_answers.items():
            print(f"Processing question: {question}")
            all_questions_with_category.append(
                (category, question, ground_truth_answer)
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_details = {
            executor.submit(
                process_single_question,
                client,
                model_name,
                question,
                tool_use,
                max_turns,
                max_retries,
                retry_delay,
                ground_truth_answer,
            ): (category, question)
            for category, question, ground_truth_answer in all_questions_with_category
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_details),
            total=len(all_questions_with_category),
            desc="Processing Questions",
        ):
            original_category, original_question_text = future_to_details[future]
            try:
                q_text_processed, q_result = future.result()
                results[original_category][q_text_processed] = q_result
            except Exception as exc:
                print(
                    f"Question '{original_question_text[:50]}...' generated an exception: {exc}"
                )
                results[original_category][original_question_text] = {
                    "answer": dataset[original_category][original_question_text],
                    "thoughts": f"Error during threaded execution: {exc}",
                    "prediction": "ERROR_THREAD_EXECUTION",
                }
    return results


def main():
    """Entry point"""

    parser = argparse.ArgumentParser(
        description="Process a dataset with a specified LLM and append results."
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the JSON dataset file (e.g., data/geneturing_small.json)",
        required=True,
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["azure", "ollama"],
        help="The LLM provider to use ('azure' or 'ollama').",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gpt-4.1", "gpt-4.1-mini", "qwen3:4b"],
        help="The model name to use.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tool-use",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable tool use (e.g., --tool-use). Disable with --no-tool-use. (default: disabled)",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="src/config.yaml",
        help="Path to the YAML configuration file (e.g., src/config.yaml)",
    )

    args = parser.parse_args()
    print("Arguments:")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model}")
    print(f"  Output Path: {args.output_path}")
    print(f"  Tool Use: {args.tool_use}")

    config = load_yaml(args.config_path)
    print(f"  Loaded config from {args.config_path}")

    # Configure MLflow tracking URI and experiment name first
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment("GeneGPT")

    # Start the MLflow run
    with mlflow.start_run():
        # Activate autologging within the run context
        mlflow.openai.autolog()

        # Log all parameters within the run context
        mlflow.log_params(vars(args))  # Logs command-line arguments

        # Log parameters from config
        if "llm_params" in config:
            mlflow.log_params(config["llm_params"])
        mlflow.log_param("max_turns", config.get("MAX_TURNS", DEFAULT_MAX_TURNS))
        mlflow.log_param("max_retries", config.get("MAX_RETRIES", DEFAULT_MAX_RETRIES))
        mlflow.log_param("retry_delay", config.get("RETRY_DELAY", DEFAULT_RETRY_DELAY))
        mlflow.log_param("max_workers", config.get("MAX_WORKERS", DEFAULT_MAX_WORKERS))

        data = load_json(args.dataset_path)
        mlflow.log_artifact(args.dataset_path)
        print(f"Loaded {len(data)} entries from {args.dataset_path}")

        results = process_dataset(
            args.provider, args.model, data, args.tool_use, config
        )
        print(f"Processed {len(results)} entries")

        table_data = create_log_table(results)
        if table_data is not None:
            mlflow.log_table(data=table_data, artifact_file="tabular_results.json")
        else:
            print("Skipping table logging as no data was generated.")

        if table_data is not None and not table_data.empty:
            metrics_dict = log_metrics(table_data)
            mlflow.log_metrics(metrics_dict)
            print("\nOverall Metrics:")
            if "overall_accuracy" in metrics_dict:
                print(f"  Overall Accuracy: {metrics_dict['overall_accuracy']:.4f}")
            if "overall_average_levenshtein_distance" in metrics_dict:
                print(
                    f"  Overall Average Levenshtein Distance: {metrics_dict['overall_average_levenshtein_distance']:.4f}"
                )
            if "overall_partial_match_accuracy" in metrics_dict:
                print(
                    f"  Overall Partial Match Accuracy: {metrics_dict['overall_partial_match_accuracy']:.4f}"
                )
        else:
            print(
                "Skipping metrics logging and printing as no data was generated for the table."
            )

        save_json(results, args.output_path)
        mlflow.log_artifact(args.output_path)
        print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
