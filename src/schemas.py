from pydantic import BaseModel, TypeAdapter, ValidationError
from typing import Dict, Literal


GenerationStatus = Literal[
    "function",
    "parameter"
]

JsonTypes = Literal[
    "string",
    "number",
    "boolean"
]


class ParameterDetail(BaseModel):
    type: JsonTypes


class ReturnDetail(BaseModel):
    type: str


class FunctionSchema(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDetail]
    returns: ReturnDetail


class PromptItem(BaseModel):
    prompt: str


class Phase(BaseModel):
    status: GenerationStatus


UNKNOWN_FUNCTION = FunctionSchema(
    name="fn_unknown",
    description="Selected when no function is valid.",
    parameters={},
    returns=ReturnDetail(type="null"),
)


def load_functions(file_path: str) -> list[FunctionSchema]:
    """Load the function definitions from `file_path` and append the
    fixed fallback UNKNOWN_FUNCTION, so it's always available both as
    a prompt option for the model and as a valid decoder target."""
    try:
        with open(file_path) as f:
            functions = TypeAdapter(list[FunctionSchema]).validate_json(f.read())
    except ValidationError:
        raise ValueError("malformed json values")
    return functions + [UNKNOWN_FUNCTION]


def load_prompts(file_path: str) -> list[str]:
    with open(file_path) as f:
        items = TypeAdapter(list[PromptItem]).validate_json(f.read())
    return [item.prompt for item in items]