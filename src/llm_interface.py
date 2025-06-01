import json
import os
import re
import time

from openai import AzureOpenAI, OpenAI
from .models import ResponseSchema
from .prompts import FEW_SHOT_PROMPT, SYSTEM_PROMPT, TOOL_USE_SYSTEM_PROMPT
from .tools import AVAILABLE_FUNCTIONS, tools_definition


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
            thoughts=content_cleaned,
            answer=content_cleaned,  # Use the raw content as answer
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
        print(
            f"  Tool executed. Response: {str(function_response)[:100]}..."
        )  # Ensure response is string for slicing
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


def call_llm(
    client: AzureOpenAI | OpenAI,
    model: str,
    question: str,
    max_retries: int,
    retry_delay: int,
) -> ResponseSchema | None:
    """Call the LLM with a given question and return the response."""
    messages = make_messages(question, SYSTEM_PROMPT, FEW_SHOT_PROMPT)
    for attempt in range(max_retries):
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=ResponseSchema,
            )
            parsed_response = response.choices[0].message.parsed
            if isinstance(parsed_response, ResponseSchema):
                return parsed_response
            elif isinstance(
                parsed_response, dict
            ):  # If it's a dict, try to create ResponseSchema
                return ResponseSchema(**parsed_response)
            else:
                # Fallback or error if type is unexpected
                print(f"Unexpected parsed response type: {type(parsed_response)}")
                # Force to ResponseSchema
                return ResponseSchema(
                    thoughts="Unexpected response structure",
                    answer=str(parsed_response),
                )

        except Exception as e:
            print(f"Error calling LLM (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Failing.")
                return ResponseSchema(
                    thoughts=f"LLM call failed after {max_retries} retries: {e}",
                    answer=None,
                )
    return ResponseSchema(
        thoughts=f"LLM call failed after {max_retries} retries. No response.",
        answer=None,
    )


def call_llm_with_tools(
    client: AzureOpenAI | OpenAI,
    model: str,
    question: str,
    max_turns: int,
    max_retries: int,
    retry_delay: int,
) -> ResponseSchema:
    """Call the LLM with tools and return a validated ResponseSchema."""
    messages = make_messages_tool_use(question, TOOL_USE_SYSTEM_PROMPT, FEW_SHOT_PROMPT)

    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")

        response_message = None
        for attempt in range(max_retries):
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
                print(
                    f"Error calling LLM (attempt {attempt + 1}/{max_retries}) in turn {turn + 1}: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(
                        f"Max retries reached for LLM call in turn {turn + 1}. Failing turn."
                    )
                    return ResponseSchema(
                        thoughts=f"Error communicating with AI model after {max_retries} retries in turn {turn + 1}: {e}",
                        answer=f"Error: {str(e)}",
                    )

        if response_message is None:
            return ResponseSchema(
                thoughts=f"Error communicating with AI model after {max_retries} retries in turn {turn + 1}. No response message.",
                answer="Error: LLM call failed after multiple retries.",
            )

        if response_message.tool_calls:
            print("LLM requested tool calls:")
            messages.append(
                {
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": format_tool_calls_for_messages(
                        response_message.tool_calls
                    ),
                }
            )
            for tool_call in response_message.tool_calls:
                tool_response = execute_tool_call(tool_call)
                messages.append(tool_response)
        else:
            print("LLM provided direct answer")
            if response_message.content:
                return validate_response_schema(response_message.content)
            else:
                return ResponseSchema(
                    thoughts="LLM provided no content",
                    answer="No response content received",
                )

    # Max turns reached - enforce a final response without tool calling
    print("Max turns reached. Attempting to get a final answer without tool use...")
    try:
        # Direct call with existing messages and tool_choice="none".
        final_response = client.chat.completions.create(
            model=model,
            messages=messages,  # Use the accumulated conversation history
            tools=tools_definition,  # Still provide tool definitions as context, but restrict choice
            tool_choice="none",  # Instruct the LLM not to call any tools
        )
        final_response_message = final_response.choices[0].message
        if final_response_message and final_response_message.content:
            print("LLM provided a final direct answer after max turns.")
            return validate_response_schema(final_response_message.content)
        else:
            print("LLM provided no content in the final attempt after max turns.")
            return ResponseSchema(
                thoughts="Max turns reached, LLM provided no content in final attempt",
                answer="Conversation ended, no final answer generated after max turns.",
            )
    except Exception as e:
        print(f"Error during final LLM call after max turns: {e}")
        return ResponseSchema(
            thoughts=f"Max turns reached, error during final LLM call: {e}",
            answer="Conversation ended, error during final answer generation.",
        )
