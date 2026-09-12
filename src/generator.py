import json
from typing import Optional
try:
    from llm_sdk import Small_LLM_Model
except Exception as e:
    print(f"Error: {e}")

from src.decoder import select_next_token
from src.schemas import FunctionSchema, Phase, load_functions, load_prompts

MAX_GENERATION_STEPS = 200


llm_model = Small_LLM_Model()
results: list[dict] = []


def build_prompt(user_prompt: str, functions: list[FunctionSchema]) -> str:
    lines: list[str]

    lines = [
        "You are a function-calling assistant. Do not answer the user's "
        "request yourself. Instead, choose the single best matching "
        "function from the list below and output a JSON object with "
        "the function name and the parameter values found in the request.",
        "",
        "Available functions:",
    ]
    for fn in functions:
        params = ", ".join(f"{name}: {detail.type}" for name, detail in fn.parameters.items())
        lines.append(f"- {fn.name}({params}): {fn.description}")

    lines += [
        "",
        'Example request: "What is the sum of 10 and 20?"',
        'Function call: {"name": "fn_add_numbers", "parameters": {"a": 10, "b": 20}}',
        "",
        f'User request: "{user_prompt}"',
        'Function call: {"name": "',
    ]
    return "\n".join(lines)


def _next_token(prompt_text: str,
                generated_so_far: str,
                phase: Phase,
                function_names: Optional[list[str]] = None,
                parameter_schema: Optional[dict[str, str]] = None) -> str:
    input_ids = llm_model.encode(prompt_text)[0].tolist()
    logits = llm_model.get_logits_from_input_ids(input_ids)
    token, _ = select_next_token(logits, generated_so_far, llm_model, phase,
                                  function_names, parameter_schema)
    return token


def generate_call(user_prompt: str, functions_path: str) -> dict:
    """Generate one function call for a single prompt. Returns {} on failure."""
    functions: list[FunctionSchema] = load_functions(functions_path)
    function_names = [fn.name for fn in functions]
    function_parameter_types = {
        fn.name: {param_name: detail.type for param_name, detail in fn.parameters.items()}
        for fn in functions
    }

    prompt_text = build_prompt(user_prompt, functions)
    phase = Phase(status="function")
    json_text = '{"name": "'

    # generate function name
    for _ in range(MAX_GENERATION_STEPS):
        token = _next_token(prompt_text, json_text, phase, function_names=function_names)
        if '"' in token:
            token = token[: token.index('"') + 1]
            prompt_text += token
            json_text += token
            break
        prompt_text += token
        json_text += token
    else:
        print(f"Gave up generating a function name for: {user_prompt!r}")
        return {}

    selected_function_name = json_text[len('{"name": "'):-1]
    parameter_schema = function_parameter_types.get(selected_function_name, {})

    prompt_text += ', "parameters": {'
    json_text += ', "parameters": {'
    phase.status = "parameter"

    # generate function parameters
    for _ in range(MAX_GENERATION_STEPS):
        token = _next_token(prompt_text, json_text, phase, parameter_schema=parameter_schema)
        if "}" in token:
            token = token[: token.index("}") + 1]
            prompt_text += token
            json_text += token
            if json_text.count("{") > json_text.count("}"):
                json_text += "}"
                prompt_text += "}"
            break
        prompt_text += token
        json_text += token
    else:
        print(f"Gave up generating parameters for: {user_prompt!r}")
        return {}

    try:
        function_call = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"Could not parse function call for {user_prompt!r}: {e}")
        return {}

    return {"prompt": user_prompt} | function_call

def run_generator(prompts_path: str, functions_path: str) -> list[dict]:
    prompts: list[str] = load_prompts(prompts_path)

    for prompt in prompts:
        print("Processing prompt:", prompt)
        result = generate_call(prompt, functions_path)
        if result:
            results.append(result)
    return results

def save_generator_results(file_path: str) -> None:
    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)