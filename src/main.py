"""
Main script to process datasets with LLMs (Azure OpenAI or Ollama).

Takes a dataset, model provider, and model name as input,
calls the respective LLM API, and appends the results to the dataset.
"""

import argparse
import json
import os
import re
import time

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, Field

import metrics
from prompts import FEW_SHOT_PROMPT, SYSTEM_PROMPT, TOOL_USE_SYSTEM_PROMPT
from tools import AVAILABLE_FUNCTIONS, tools_definition

load_dotenv()

MAX_TURNS = 12
MAX_RETRIES = 3  # Maximum number of retries for LLM calls
RETRY_DELAY = 5  # Seconds to wait between retries


class ResponseSchema(BaseModel):
    """Schema for the response from the LLM."""

    thoughts: str = Field(description="The thoughts process to answer the question")
    answer: str = Field(description="The final answer to the question")


def load_json(path: str) -> dict:
    """Load a JSON file from a given path."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str) -> dict:
    """Load a YAML file from a given path."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: str) -> None:
    """Save a dictionary to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def get_client(provider: str):
    """Get a client for a given provider and model."""
    if provider == "azure":
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        return client
    elif provider == "ollama":
        client = OpenAI(
            api_key=os.getenv("OLLAMA_API_KEY"),
            base_url=os.getenv("OLLAMA_API_ENDPOINT"),
        )
        return client
    else:
        raise ValueError(f"Invalid provider: {provider}")


def make_messages(question: str, system_prompt: str, few_shot_prompt: str) -> list:
    """Make the messages for the LLM."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": few_shot_prompt},
        {"role": "user", "content": question},
    ]
    return messages


def make_messages_tool_use(
    question: str, system_prompt: str, few_shot_prompt: str
) -> list:
    """Make the messages for the LLM with tool use."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": few_shot_prompt},
        {"role": "user", "content": question},
    ]

    return messages


def create_log_table(results: dict) -> None:
    """Log the results to a table."""
    # Prepare data for MLflow table
    table_data = []
    for category, questions_answers in results.items():
        for question, details in questions_answers.items():
            answer = details.get("answer")
            prediction = details.get("prediction")
            thoughts = details.get("thoughts")
            table_data.append(
                {
                    "category": category,
                    "question": question,
                    "ground_truth_answer": answer,
                    "thoughts": thoughts,
                    "prediction": prediction,
                    "match": metrics.calculate_exact_match(answer, prediction),
                    "levenshtein_distance": metrics.calculate_levenshtein_distance(
                        answer, prediction
                    ),
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

    metrics_to_log = {
        "overall_accuracy": overall_accuracy,
        "overall_average_levenshtein_distance": avg_levenshtein_distance,
    }

    # Calculate metrics per category
    if "category" in results_df.columns:
        category_metrics = results_df.groupby("category").agg(
            accuracy=("match", "mean"),
            average_levenshtein_distance=("levenshtein_distance", "mean"),
        )
        for category, row in category_metrics.iterrows():
            metrics_to_log[f"{category}_accuracy"] = row["accuracy"]
            metrics_to_log[f"{category}_average_levenshtein_distance"] = row[
                "average_levenshtein_distance"
            ]

    return metrics_to_log


def call_llm(client: AzureOpenAI | OpenAI, model: str, question: str) -> str:
    """Call the LLM with a given question and return the response."""

    messages = make_messages(question, SYSTEM_PROMPT, FEW_SHOT_PROMPT)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=ResponseSchema,
            )
            return response.choices[0].message.parsed
        except Exception as e:
            print(f"Error calling LLM (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print("Max retries reached. Failing.")
                return None  # Or raise the exception e
    return None # Should not be reached if MAX_RETRIES > 0


def validate_response_schema(content: str) -> ResponseSchema:
    """Validate and parse the response content against ResponseSchema."""
    try:
        # Remove XML-style thinking tags if present
        content_cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content_cleaned = content_cleaned.strip()

        # Try to parse as JSON first
        response_data = json.loads(content_cleaned)
        # Validate against ResponseSchema
        return ResponseSchema(**response_data)

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Could not parse response as valid JSON schema: {e}")
        # Return a default response if parsing fails
        return ResponseSchema(
            thoughts="Could not parse response properly",
            answer=None,  # Use the raw content as answer
        )


def execute_tool_call(tool_call) -> dict:
    """Execute a single tool call and return the response message."""
    function_name = tool_call.function.name
    tool_call_id = tool_call.id

    print(f"  Function: {function_name}")

    # Parse arguments
    try:
        function_args = json.loads(tool_call.function.arguments)
        print(f"  Arguments: {function_args}")
    except json.JSONDecodeError:
        print(f"  Error: Could not parse arguments for {function_name}")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": function_name,
            "content": json.dumps(
                {"error": f"Invalid arguments format: {tool_call.function.arguments}"}
            ),
        }

    # Check if function exists and execute
    if function_name not in AVAILABLE_FUNCTIONS:
        print(f"  Error: Unknown function '{function_name}'")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": function_name,
            "content": json.dumps({"error": f"Function '{function_name}' not found"}),
        }

    # Execute the function
    try:
        function_response = AVAILABLE_FUNCTIONS[function_name](**function_args)
        print(f"  Tool executed. Response: {function_response[:100]}...")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": function_name,
            "content": function_response,
        }
    except Exception as e:
        print(f"  Error executing {function_name}: {e}")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": function_name,
            "content": json.dumps({"error": f"Error in {function_name}: {str(e)}"}),
        }


def format_tool_calls_for_messages(tool_calls):
    """Convert tool_calls to the format needed for messages."""
    return (
        [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
        if tool_calls
        else None
    )


def call_llm_with_tools(
    client: AzureOpenAI | OpenAI, model: str, question: str, max_turns: int = MAX_TURNS
) -> ResponseSchema:
    """Call the LLM with tools and return a validated ResponseSchema."""
    messages = make_messages_tool_use(question, TOOL_USE_SYSTEM_PROMPT, FEW_SHOT_PROMPT)

    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")

        response_message = None
        for attempt in range(MAX_RETRIES):
            try:
                # Make the LLM call
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools_definition,
                    tool_choice="auto",
                )
                response_message = response.choices[0].message
                break  # Success, exit retry loop
            except Exception as e:
                print(f"Error calling LLM (attempt {attempt + 1}/{MAX_RETRIES}) in turn {turn + 1}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"Max retries reached for LLM call in turn {turn + 1}. Failing turn.")
                    # Return an error response if all retries fail
                    return ResponseSchema(
                        thoughts=f"Error communicating with AI model after {MAX_RETRIES} retries in turn {turn +1}",
                        answer=f"Error: {str(e)}",
                    )
        
        if response_message is None: # Should only happen if all retries failed and loop finished
             return ResponseSchema(
                thoughts=f"Error communicating with AI model after {MAX_RETRIES} retries in turn {turn +1}. No response message.",
                answer="Error: LLM call failed after multiple retries.",
            )


        # Handle the response
        if response_message.tool_calls:
            print("LLM requested tool calls:")

            # Add assistant message with tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": format_tool_calls_for_messages(
                        response_message.tool_calls
                    ),
                }
            )

            # Execute each tool and add responses
            for tool_call in response_message.tool_calls:
                tool_response = execute_tool_call(tool_call)
                messages.append(tool_response)

        else:
            # No tool calls - this should be the final answer
            print("LLM provided direct answer")
            if response_message.content:
                return validate_response_schema(response_message.content)
            else:
                return ResponseSchema(
                    thoughts="LLM provided no content",
                    answer="No response content received",
                )

    # Max turns reached - try to extract last assistant message
    print("Max turns reached")
    return ResponseSchema(
        thoughts="Max turns reached without final answer",
        answer="Conversation ended without a final answer",
    )


def process_dataset(
    client: AzureOpenAI | OpenAI, model_name: str, dataset: dict, tool_use: bool
) -> dict:
    """
    Processes each question in the dataset using the LLM and appends results.
    """
    results = {}
    for category, questions_answers in dataset.items():
        print(f"\nProcessing category: {category}...")
        results[category] = {}
        for question, ground_truth_answer in questions_answers.items():
            try:
                if tool_use:
                    llm_response = call_llm_with_tools(client, model_name, question)
                else:
                    llm_response = call_llm(client, model_name, question)
            except Exception as e:
                print(f"Failed to call LLM: {e}")
                continue

            results[category][question] = {
                "answer": ground_truth_answer,
                "thoughts": llm_response.thoughts,
                "prediction": llm_response.answer,
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
        "--tool_use",
        type=bool,
        default=False,
        help="Whether to use tools defined in tools.py",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="src/config.yaml",  # Default to your config file
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

    # Enable MLflow OpenAI autologging
    mlflow.openai.autolog()

    # Set MLFlow tracking URI
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment("GeneGPT")

    with mlflow.start_run():
        mlflow.log_params(vars(args))

        if "llm_params" in config:
            mlflow.log_params(config["llm_params"])

        # Load the dataset
        data = load_json(args.dataset_path)
        mlflow.log_artifact(args.dataset_path)
        print(f"Loaded {len(data)} entries from {args.dataset_path}")

        # Load the client
        client = get_client(args.provider)
        print(f"Loaded client: using {args.provider}")

        # Process the dataset
        results = process_dataset(client, args.model, data, args.tool_use)
        print(f"Processed {len(results)} entries")

        # Create a table of the results
        table_data = create_log_table(results)
        mlflow.log_table(data=table_data, artifact_file="tabular_results.json")

        # Log the metrics
        metrics_dict = log_metrics(table_data)
        mlflow.log_metrics(metrics_dict)

        # Save the results
        save_json(results, args.output_path)
        mlflow.log_artifact(args.output_path)
        print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
