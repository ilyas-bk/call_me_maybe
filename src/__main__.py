from src.generator import run_generator, save_generator_results
import sys

FUNCTIONS_PATH = "data/input/functions_definition.json"
PROMPTS_PATH = "data/input/function_calling_tests.json"
RESULTS_PATH = "data/output/function_calling_results.json"


def main():
    run_generator(PROMPTS_PATH, FUNCTIONS_PATH)
    save_generator_results(RESULTS_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error : {e}")
        sys.exit(1)