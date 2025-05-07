import os
import subprocess
import sys
import time


def print_header(title):
    """Print a formatted header for each step in the pipeline"""
    print("\n" + "=" * 80)
    print(f" {title.upper()} ".center(80, "*"))
    print("=" * 80 + "\n")


def run_step(step_number, step_name, command, args=None):
    """Run a single step in the pipeline"""
    print_header(f"Step {step_number}: {step_name}")

    try:
        # If the command is a Python file, run it with the appropriate interpreter
        if command.endswith(".py"):
            cmd = [sys.executable, command]
            if args:
                cmd.extend(args)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Print output in real-time
            for line in process.stdout:
                print(line, end="")

            process.wait()
            return process.returncode == 0
        else:
            # For other commands
            result = subprocess.run(command, shell=True, check=True)
            return True

    except subprocess.CalledProcessError as e:
        print(f"Error in {step_name}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error in {step_name}: {e}")
        return False


def main():
    # Current directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  # Change to script directory

    # Define the pipeline steps
    pipeline_steps = [
        #        {
        #            "number": 1,
        #            "name": "Scraping YouTube Comments",
        #            "command": "1-scrape.py",
        #            "required_files": [],
        #        },
        {
            "number": 2,
            "name": "Processing Emojis",
            "command": "2-emojihandler.py",
            "required_files": ["scraped.txt"],
        },
        {
            "number": 3,
            "name": "Cleaning Text",
            "command": "3-clean.py",
            "required_files": ["scraped.txt"],
        },
        {
            "number": 4,
            "name": "Tagging and Transliterating",
            "command": "4-tagandtransliterate.py",
            "args": [
                "--input",
                "corpus.txt",
                "--model",
                "model/crf_model.pkl",
                "--output",
                "crf-tagged/crf-tagged.csv",
                "--processed-output",
                "output.txt",
            ],
            "required_files": ["corpus.txt", "model/crf_model.pkl"],
        },
    ]

    # Create required directories if they don't exist
    os.makedirs("crf-tagged", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    # Check for model file
    if not os.path.exists("model/crf_model.pkl"):
        if os.path.exists("../analytics/crf_model.pkl"):
            print("Copying CRF model from analytics directory...")
            import shutil

            shutil.copy("../analytics/crf_model.pkl", "model/crf_model.pkl")
        else:
            print(
                "Warning: CRF model not found. Please ensure model/crf_model.pkl exists before running step 4."
            )

    # Run each step in the pipeline
    for step in pipeline_steps:
        # Check for required files
        missing_files = [f for f in step["required_files"] if not os.path.exists(f)]
        if missing_files:
            print(
                f"Cannot run step {step['number']}: Missing required files: {', '.join(missing_files)}"
            )
            user_input = input(
                f"Do you want to skip step {step['number']} and continue? (y/n): "
            )
            if user_input.lower() not in ["y", "yes"]:
                print("Pipeline execution stopped.")
                return
            continue

        # Run the step
        args = step.get("args", None)
        success = run_step(step["number"], step["name"], step["command"], args)

        if not success:
            print(f"Step {step['number']} failed. Pipeline execution stopped.")
            user_input = input(
                "Do you want to continue with the next step anyway? (y/n): "
            )
            if user_input.lower() not in ["y", "yes"]:
                return

    print_header("Pipeline Complete")
    print("All steps have been executed. Results can be found in:")
    print("- Final transliterated output: output.txt")
    print("- Tagged data: crf-tagged/crf-tagged.csv")
    print("- Cleaned corpus: corpus.txt")
    print("- Raw scraped data: scraped.txt")
    print("- Emojis: emojis.json")


if __name__ == "__main__":
    main()
