import sys
import os

# Set LM Studio env vars BEFORE importing open-interpreter (LiteLLM reads these)
os.environ["OPENAI_API_KEY"]  = "lm-studio"
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"

# Ensure we can import the locally installed interpreter
try:
    import interpreter
except ImportError:
    print("Error: open-interpreter is not installed in this environment.")
    print("Please install it using: pip install open-interpreter")
    sys.exit(1)

def main():
    print("========================================")
    print("🤖 Henry - Open Interpreter Baseline")
    print("========================================")
    
    # Configure LM Studio
    # Using 'openai/' prefix tells litellm to use the OpenAI API format
    interpreter.interpreter.llm.model = "openai/lmstudio" 
    interpreter.interpreter.llm.api_base = "http://localhost:1234/v1"
    interpreter.interpreter.llm.api_key = "lm-studio"
    
    # User requirement: Do NOT enable auto_run initially.
    interpreter.interpreter.auto_run = False
    
    # Recommended settings for local LLMs
    interpreter.interpreter.llm.max_tokens = 2000
    interpreter.interpreter.llm.context_window = 8192
    
    # System message configuration to shape the persona
    interpreter.interpreter.system_message += (
        "\nYour name is Henry. You are a helpful and highly capable AI assistant running locally."
        "\nYou have full control over the user's computer via Python and shell execution."
        "\nWhen asked to open applications, control the browser, or interact with the OS, write the necessary Python code."
        "\n\nCRITICAL YOUTUBE INSTRUCTION: "
        "\nIf the user asks you to play a specific song or video on YouTube, DO NOT GUESS THE URL (you will accidentally Rickroll them)."
        "\nInstead, write a Python script that searches YouTube and opens the first result. Use this exact code:"
        "\n```python"
        "\nimport urllib.request, urllib.parse, re, webbrowser"
        "\nquery = urllib.parse.quote('SONG NAME HERE')"
        "\nhtml = urllib.request.urlopen(f'https://www.youtube.com/results?search_query={query}', timeout=5).read().decode()"
        "\nvideo_ids = re.findall(r'watch\?v=(\S{11})', html)"
        "\nif video_ids: webbrowser.open(f'https://www.youtube.com/watch?v={video_ids[0]}')"
        "\n```"
    )

    print(f"✅ LM Studio Endpoint: {interpreter.interpreter.llm.api_base}")
    print(f"✅ Auto-run Status: {'ON' if interpreter.interpreter.auto_run else 'OFF (Approval Required)'}")
    print("========================================\n")
    print("Ready! You can now start typing your commands (e.g., 'Open Chrome and go to youtube.com').")
    print("Type 'exit' to quit.\n")
    
    # Start the interactive chat
    interpreter.interpreter.chat()

if __name__ == "__main__":
    main()
