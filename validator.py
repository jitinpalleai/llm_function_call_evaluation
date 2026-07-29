import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
TOOLS_FILE = BASE_DIR / "tool_schemas.json"
CONVERSATIONS_FILE = BASE_DIR / "conversation_examples.json"


def load_json(file_path: Path) -> dict[str, Any]:
    try:
        with file_path.open("r", encoding="utf8") as file:
            return json.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"File not found: {file_path.name}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {file_path.name}: {error}") from error


def build_tool_registry(tool_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tools = tool_data.get("tools", [])
    return {tool["name"]: tool for tool in tools if "name" in tool}


def validate_tool_call(
    tool_call: dict[str, Any],
    tool_registry: dict[str, dict[str, Any]]
) -> list[str]:
    errors: list[str] = []

    tool_name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})

    if not tool_name:
        errors.append("Tool name is missing")
        return errors

    if tool_name not in tool_registry:
        errors.append(f"Unknown tool: {tool_name}")
        return errors

    tool_schema = tool_registry[tool_name]
    parameter_schema = tool_schema.get("parameters", {})
    required_fields = parameter_schema.get("required", [])
    properties = parameter_schema.get("properties", {})

    for field_name in required_fields:
        if field_name not in arguments:
            errors.append(
                f"Missing required argument '{field_name}' for tool '{tool_name}'"
            )

    for argument_name in arguments:
        if argument_name not in properties:
            errors.append(
                f"Unexpected argument '{argument_name}' for tool '{tool_name}'"
            )

    return errors


def validate_conversation(
    conversation: dict[str, Any],
    tool_registry: dict[str, dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    conversation_id = conversation.get("id", "unknown")
    messages = conversation.get("messages", [])

    if not messages:
        return [f"{conversation_id}: Conversation has no messages"]

    if messages[0].get("role") != "user":
        errors.append(f"{conversation_id}: First message must come from the user")

    tool_calls = 0
    tool_results = 0

    for index, message in enumerate(messages, start=1):
        role = message.get("role")

        if role not in {"user", "assistant", "tool"}:
            errors.append(
                f"{conversation_id}: Message {index} has an invalid role"
            )

        if role == "assistant" and "tool_call" in message:
            tool_calls += 1
            tool_errors = validate_tool_call(
                message["tool_call"],
                tool_registry
            )

            for error in tool_errors:
                errors.append(
                    f"{conversation_id}: Message {index}: {error}"
                )

        if role == "tool":
            tool_results += 1

            if not message.get("name"):
                errors.append(
                    f"{conversation_id}: Message {index} is missing the tool name"
                )

            if "content" not in message:
                errors.append(
                    f"{conversation_id}: Message {index} is missing tool output"
                )

    if tool_calls != tool_results:
        errors.append(
            f"{conversation_id}: Tool call count does not match tool result count"
        )

    if messages[-1].get("role") != "assistant":
        errors.append(
            f"{conversation_id}: Final message must come from the assistant"
        )

    return errors


def main() -> None:
    tool_data = load_json(TOOLS_FILE)
    conversation_data = load_json(CONVERSATIONS_FILE)

    tool_registry = build_tool_registry(tool_data)
    conversations = conversation_data.get("conversations", [])

    all_errors: list[str] = []

    for conversation in conversations:
        all_errors.extend(
            validate_conversation(conversation, tool_registry)
        )

    if all_errors:
        print("Validation failed")
        for error in all_errors:
            print(f"Error: {error}")
        raise SystemExit(1)

    print(f"Validation passed for {len(conversations)} conversations")


if __name__ == "__main__":
    main()
