import subprocess
import os
import time
import re
from dataclasses import dataclass, field
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Mapping, Any
import requests
from langchain.llms.base import LLM
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ExecutionStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"

@dataclass
class ExecutionResult:
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""
    return_code: int = 0

@dataclass
class CodeGeneration:
    code: str = ""
    installations: List[str] = field(default_factory=list)
    attempt_count: int = 0
    max_attempts: int = 30
    error_message: str = ""

@dataclass
class ConversationHistory:
    """Class to store conversation history."""
    prompts: List[str] = field(default_factory=list)
    codes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    installations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))


class OllamaLLM(LLM):
    """Ollama LLM integration - keeping original implementation"""
    model_name: str = "llama3.1"     # llama3.1   llama3.1:70b
    api_url: str = "http://192.168.0.149:11434/api/generate"  # llama3.2
    #api_url: str = "http://34.87.65.42:5055/api/generate"
    max_tokens: int = 500
    temperature: float = 0.5

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": stop or [],

            "stream": False
        }

        try:
            response = requests.post(
                self.api_url, json=payload, headers=headers, timeout=300
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API request failed: {e}")

    @property
    def _llm_type(self) -> str:
        return "ollama_llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "api_url": self.api_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

@dataclass
class CodeGenerator:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        self.previous_attempts = []

    def generate_code_with_context(self, prompt: str, error_message: str = "", historical_context: str = "") -> str:
        """Generate Python code based on user prompt with improved context handling."""
        code_template = """
You are a Python code generator. Your task is to create a complete, working Python program that solves this specific request:
{user_prompt}

Previous Error (if any):
{error_info}

Historical Context:
{context}

Instructions:
1.(this is just an example do not use any variabel in main code) Use this EXACT structure and implement the solution within it:
```python
def main():
    try:
        print("Starting execution...", flush=True)
        
        # Get user inputs - ALWAYS use this format:
        print("Enter ...: ", end="", flush=True)  # Prompt
        variable = input()  # Separate input line
        
        # For numbers, ALWAYS convert after input:
        print("Enter number: ", end="", flush=True)
        number = float(input())  # or int(input()) for integers
        
        # Process data and show results
        result = # Your calculation here
        print(f"Result: {{result}}", flush=True)
        
    except Exception as exc:
        print(f"Error: {{exc}}", flush=True)
        raise  # Required
    finally:
        print("Successfully", flush=True)  # Required

if __name__ == "__main__":
    main()
```

2. Critical Rules:
   - ALL print statements must include flush=True
   - NEVER combine print and input - ALWAYS separate them
   - ALL numeric inputs must use float() or int() after input()
   - ALWAYS handle errors in try-except block
   - ALWAYS include the "Successfully" message in finally
   - ALWAYS show clear, labeled outputs

3. Format Requirements:
   - Correct indentation (4 spaces)
   - Proper error handling
   - Clear variable names
   - Descriptive prompts

Generate the complete solution for: {user_prompt}
Return ONLY the working Python code with no additional text or explanations.
"""

        try:
            # Generate code with context
            response = self.llm.invoke(code_template.format(
                user_prompt=prompt,
                error_info=error_message if error_message else "None",
                context=historical_context if historical_context else "None"
            ))
            
            # Process and clean the code
            raw_code = str(response)
            cleaned_code = self._clean_code(raw_code)
            validated_code = self._validate_code_structure(cleaned_code)
            
            # Store the attempt
            self.previous_attempts.append({
                'prompt': prompt,
                'code': validated_code,
                'error': error_message,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return validated_code
            
        except Exception as e:
            self.logger.error(f"Code generation error: {str(e)}")
            raise

    def _clean_code(self, code: str) -> str:
        """Clean and format the generated code."""
        # Remove markdown formatting
        code = code.strip()
        code = code.replace('\t', '    ')  # Replace tabs with 4 spaces
        
        if code.startswith("```") and code.endswith("```"):
            lines = code.split("\n")
            if lines[0].startswith("```python"):
                code = "\n".join(lines[1:-1])
            else:
                code = "\n".join(lines[1:-1])

        # Remove any remaining markdown markers
        code = code.replace("```python", "").replace("```", "")

        # Fix common string formatting issues
        code = code.replace("**name**", "__name__")
        code = code.replace("'__main__'", '"__main__"')
        
        # Process and indent code
        lines = []
        indent = "    "
        current_indent = 0
        
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped:
                lines.append("")
                continue
                
            # Handle basic indentation
            if stripped.startswith(("def ", "class ", "if __name__")):
                current_indent = 0
            elif stripped == "try:":
                current_indent = 1
            elif stripped in ["except Exception as exc:", "finally:", "else:", "elif"]:
                current_indent = 1
            elif stripped == "main()":
                current_indent = 1
                
            # Add the line with proper indentation
            lines.append(indent * current_indent + stripped)
            
            # Adjust indentation for next line
            if stripped.endswith(":"):
                current_indent += 1

        # Ensure final newline
        code = "\n".join(lines)
        if not code.endswith("\n"):
            code += "\n"
            
        return code

    def _validate_code_structure(self, code: str) -> str:
        """Validate and fix the code structure."""
        lines = code.splitlines()
        validated_lines = []
        indent = "    "
        in_main = False
        has_main = False
        has_try = False
        has_except = False
        has_finally = False
        
        for line in lines:
            stripped = line.strip()
            
            # Track code structure
            if stripped.startswith("def main():"):
                in_main = True
                has_main = True
            elif stripped == "try:":
                has_try = True
            elif stripped == "except Exception as exc:":
                has_except = True
            elif stripped == "finally:":
                has_finally = True
                
            # Fix input statements
            if 'input(' in line and 'input()' not in line:
                current_indent = ""
                while line.startswith(indent):
                    current_indent += indent
                    line = line[len(indent):]
                    
                if '=' in line:
                    var_name, input_part = line.split('=', 1)
                    prompt_text = input_part.strip().split('input(')[1].split(')')[0].strip(' "\'")')
                    
                    validated_lines.append(f'{current_indent}print({prompt_text}, end="", flush=True)')
                    if 'float(' in line:
                        validated_lines.append(f'{current_indent}{var_name}= float(input())')
                    elif 'int(' in line:
                        validated_lines.append(f'{current_indent}{var_name}= int(input())')
                    else:
                        validated_lines.append(f'{current_indent}{var_name}= input()')
                else:
                    prompt_text = line.split('input(')[1].split(')')[0].strip(' "\'")')
                    validated_lines.append(f'{current_indent}print({prompt_text}, end="", flush=True)')
                    validated_lines.append(f'{current_indent}input()')
            else:
                validated_lines.append(line)
                
        # Check and fix structure if needed
        if not all([has_main, has_try, has_except, has_finally]):
            return self._generate_basic_structure()
            
        return "\n".join(validated_lines)

    def _generate_basic_structure(self) -> str:
        """Generate a basic code structure."""
        return '''def main():
    try:
        print("Starting execution...", flush=True)
        
        # Get input
        print("Enter first number: ", end="", flush=True)
        num1 = float(input())
        print("Enter second number: ", end="", flush=True)
        num2 = float(input())
        
        # Process and show result
        result = num1 + num2
        print(f"Result: {result}", flush=True)
        
    except Exception as exc:
        print(f"Error: {exc}", flush=True)
        raise
    finally:
        print("Successfully", flush=True)

if __name__ == "__main__":
    main()
'''

    def generate_installation_commands(self, error_message: str, previous_installations: List[str]) -> List[str]:
        """Generate pip install commands based on error message."""
        if not error_message:
            return []

        install_template = """
Analyze this Python error and determine required pip packages:

Error: {error_msg}

Previous installations:
{prev_installs}

Rules:
1. Only suggest pip installable Python packages
2. Only if clearly missing from error
3. One package per line
4. Use 'pip install package' format
5. No system packages (apt, brew, etc.)
6. No packages already installed

Return only pip install commands or empty string."""

        try:
            response = self.llm.invoke(install_template.format(
                error_msg=error_message,
                prev_installs="\n".join(previous_installations)
            ))
            commands = str(response)
        except AttributeError:
            commands = self.llm(install_template.format(
                error_msg=error_message,
                prev_installs="\n".join(previous_installations)
            ))

        return [cmd.strip() for cmd in commands.splitlines() 
                if cmd.strip().startswith("pip install")
                and cmd.strip() not in previous_installations]

    def reset_attempts(self):
        """Reset the previous attempts history."""
        self.previous_attempts = []

class TerminalExecutor:
    def __init__(self):
        """Initialize the Terminal Executor with working directory and logger."""
        self.logger = logging.getLogger(__name__)
        self.current_dir = os.getcwd()  # Store current working directory
        self.output_dir = os.path.join(self.current_dir, "temp_outputs")
        
        # Create output directory if it doesn't exist
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        except Exception as e:
            self.logger.warning(f"Could not create output directory: {e}")
            self.output_dir = self.current_dir

    def run_installation(self, command: str) -> ExecutionResult:
        """Run installation command in new terminal."""
        timestamp = int(time.time())
        install_script = os.path.join(self.output_dir, f"install_{timestamp}.sh")
        
        # Enhanced installation script with better error handling and output capture
        script_content = f'''#!/bin/bash
    set -e  # Exit on error

    cd "{self.current_dir}"

    # Function to handle errors
    handle_error() {{
        echo "Installation failed at line $1"
        exit 1
    }}

    trap 'handle_error $LINENO' ERR

    echo "Starting installation: {command}"
    echo "----------------------------------------"

    # Run installation with error capture
    if ! {command}; then
        echo "Installation failed with exit code $?"
        exit 1
    fi

    echo "----------------------------------------"
    echo "Installation completed successfully"
    echo
    echo "Press Enter to close..."
    read
    '''
        
        try:
            # Create and set up installation script
            with open(install_script, 'w') as f:
                f.write(script_content)
            os.chmod(install_script, 0o755)
            
            # Execute installation script in new terminal
            result = self.execute_in_new_terminal(install_script)
            
            # Clean up installation script
            try:
                os.remove(install_script)
            except Exception as e:
                self.logger.warning(f"Failed to remove installation script: {e}")
            
            # Enhanced result handling
            if result.stdout and "Installation completed successfully" in result.stdout:
                result.status = ExecutionStatus.SUCCESS
                result.error_message = ""
            elif result.error_message:
                self.logger.error(f"Installation error: {result.error_message}")
            elif result.stderr:
                result.status = ExecutionStatus.ERROR
                result.error_message = f"Installation failed: {result.stderr}"
            else:
                result.status = ExecutionStatus.ERROR
                result.error_message = "Installation failed without specific error message"
            
            return result
                
        except Exception as e:
            error_msg = f"Installation setup failed: {str(e)}"
            self.logger.error(error_msg)
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=error_msg
            )

    def execute_in_new_terminal(self, script_path: str, timeout: int = 30) -> ExecutionResult:
        """Execute script in a new terminal window and capture output."""
        # Generate unique names for output files
        base_name = os.path.splitext(os.path.basename(script_path))[0]
        timestamp = int(time.time())
        output_file = os.path.join(self.output_dir, f"{base_name}_{timestamp}.output")
        error_file = os.path.join(self.output_dir, f"{base_name}_{timestamp}.error")
        completion_file = os.path.join(self.output_dir, f"{base_name}_{timestamp}.complete")
        execution_status_file = os.path.join(self.output_dir, f"{base_name}_{timestamp}.status")

        # Clean up any existing files
        for file in [output_file, error_file, completion_file, execution_status_file]:
            if os.path.exists(file):

                try:
                    os.remove(file)
                except Exception as e:
                    self.logger.warning(f"Could not remove file {file}: {e}")

        # Create wrapper script for terminal execution
        wrapper_script = self.create_wrapper_script(  # Changed from _create_wrapper_script
            script_path, output_file, error_file, completion_file, execution_status_file
        )
        wrapper_path = os.path.join(self.output_dir, f"{base_name}_{timestamp}_wrapper.sh")
        
        try:
            with open(wrapper_path, 'w') as f:
                f.write(wrapper_script)
            os.chmod(wrapper_path, 0o755)
        except Exception as e:
            self.logger.error(f"Failed to create wrapper script: {e}")
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=f"Failed to create wrapper script: {str(e)}"
            )

        # Launch in new terminal
        try:
            subprocess.Popen(['gnome-terminal', '--', '/bin/bash', wrapper_path])
        except Exception as e:
            self.logger.error(f"Failed to open new terminal: {e}")
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=f"Failed to open terminal: {str(e)}"
            )

        # Wait for completion and collect results
        result = self.wait_for_completion(  # Changed from _wait_for_completion
            completion_file, output_file, error_file, execution_status_file, timeout
        )

        # Cleanup temporary files
        for file in [output_file, error_file, completion_file, execution_status_file, wrapper_path]:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except Exception as e:
                self.logger.warning(f"Failed to remove temporary file {file}: {e}")

        return result

    def create_wrapper_script(self, script_path: str, output_file: str,  # Changed from _create_wrapper_script
                            error_file: str, completion_file: str, 
                            execution_status_file: str) -> str:
        """Create the wrapper script for terminal execution with improved error detection."""
        return f'''#!/bin/bash
cd "{self.current_dir}"

# Function to capture and analyze output
capture_and_analyze() {{
    # Temporary files for capturing streams separately
    stdout_tmp="{output_file}.stdout"
    stderr_tmp="{output_file}.stderr"
    
    # Execute with separate stream capture
    if [[ "{script_path}" == *.sh ]]; then
        bash "{script_path}" 2> >(tee "$stderr_tmp" >&2) | tee "$stdout_tmp"
        exit_code=${{PIPESTATUS[0]}}
    else
        python3 -u "{script_path}" 2> >(tee "$stderr_tmp" >&2) | tee "$stdout_tmp"
        exit_code=${{PIPESTATUS[0]}}
    fi
    
    # Analyze output for different types of errors
    {{
        # Check for runtime errors
        grep -i "error:" "$stdout_tmp" || true
        grep -i "exception" "$stdout_tmp" || true
        grep -i "traceback" "$stdout_tmp" || true
        # Check for syntax errors
        grep -i "syntaxerror" "$stdout_tmp" || true
        # Check for import errors
        grep -i "importerror" "$stdout_tmp" || true
        grep -i "modulenotfounderror" "$stdout_tmp" || true
        
        # Also check stderr
        if [ -f "$stderr_tmp" ]; then
            cat "$stderr_tmp"
        fi
    }} > "{error_file}"
    
    # Combine outputs
    cat "$stdout_tmp" > "{output_file}"
    
    # Cleanup temporary files
    rm -f "$stdout_tmp" "$stderr_tmp"
    
    # Analyze execution status
    if [ $exit_code -ne 0 ] || [ -s "{error_file}" ] || grep -qiE "error:|exception|traceback" "{output_file}"; then
        {{
            echo "Error occurred during execution"
            echo "----------------------------------------"
            if [ -s "{error_file}" ]; then
                echo "Error details:"
                cat "{error_file}"
            fi
            echo "----------------------------------------"
            echo "Exit Code: $exit_code"
        }} > "{execution_status_file}"
    else
        if grep -q "Successfully" "{output_file}"; then
            echo "Success" > "{execution_status_file}"
        else
            echo "Program completed but 'Successfully' message not found" > "{execution_status_file}"
        fi
    fi
    
    echo $exit_code > "{completion_file}"
    return $exit_code
}}

# Handle unexpected termination
cleanup() {{
    if [ ! -f "{completion_file}" ]; then
        echo "Terminal closed unexpectedly" > "{error_file}"
        echo "Execution terminated by user" > "{execution_status_file}"
        echo "1" > "{completion_file}"
    fi
}}

trap cleanup EXIT
trap 'exit 1' INT TERM

# Execute with improved error capture
capture_and_analyze

# Display final status with improved formatting
if [ -f "{execution_status_file}" ]; then
    echo -e "\\n==========================================="
    echo "Execution Status:"
    echo "==========================================="
    cat "{execution_status_file}"
    echo -e "===========================================\\n"
fi

echo -e "Press Enter to close..."
read
'''

    def wait_for_completion(self, completion_file: str, output_file: str,  # Changed from _wait_for_completion
                          error_file: str, execution_status_file: str,
                          timeout: int) -> ExecutionResult:
        """Wait for script completion with improved error detection."""
        start_time = time.time()
        last_output_size = 0
        last_error_size = 0

        while not os.path.exists(completion_file):
            if time.time() - start_time > timeout:
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    error_message="Execution timed out"
                )

            # Show real-time output with better error highlighting
            for file_path, last_size, is_error in [
                (output_file, last_output_size, False),
                (error_file, last_error_size, True)
            ]:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            if len(content) > last_size:
                                new_content = content[last_size:]
                                if is_error:
                                    new_content = re.sub(
                                        r'(Error|Exception|Traceback|Failed).*',
                                        lambda m: f"\033[91m{m.group()}\033[0m",
                                        new_content,
                                        flags=re.IGNORECASE
                                    )
                                    last_error_size = len(content)
                                else:
                                    last_output_size = len(content)
                                print(new_content, end='', flush=True)
                    except Exception as e:
                        self.logger.warning(f"Error reading output: {e}")

            time.sleep(0.1)

        # Get execution results with comprehensive error checking
        try:
            with open(completion_file, 'r') as f:
                return_code = int(f.read().strip())
        except Exception as e:
            self.logger.error(f"Error reading completion status: {e}")
            return_code = 1

        stdout = stderr = execution_status = ""
        try:
            with open(output_file, 'r') as f:
                stdout = f.read()
            with open(error_file, 'r') as f:
                stderr = f.read()
            with open(execution_status_file, 'r') as f:
                execution_status = f.read().strip()
        except Exception as e:
            self.logger.error(f"Error reading output files: {e}")

        # Enhanced error detection
        error_patterns = [
            r'Error:.*',
            r'Exception:.*',
            r'Traceback.*',
            r'SyntaxError:.*',
            r'ImportError:.*',
            r'ModuleNotFoundError:.*',
            r'TypeError:.*',
            r'ValueError:.*',
            r'NameError:.*',
            r'AttributeError:.*',
            r'IndexError:.*',
            r'KeyError:.*',
            r'failed with exit code.*'
        ]

        has_error = any([
            return_code != 0,
            bool(stderr.strip()),
            'Error' in execution_status,
            'error' in execution_status.lower(),
            any(re.search(pattern, stdout, re.IGNORECASE | re.MULTILINE) 
                for pattern in error_patterns),
            'Successfully' not in stdout
        ])

        if has_error:
            status = ExecutionStatus.ERROR
            error_messages = []
            
            # Collect all error information
            if execution_status and execution_status != "Success":
                error_messages.append(execution_status)
            
            if stderr.strip():
                error_messages.append(stderr.strip())
            
            # Find all error patterns in stdout
            for pattern in error_patterns:
                matches = re.finditer(pattern, stdout, re.IGNORECASE | re.MULTILINE)
                error_messages.extend(match.group(0) for match in matches)
            
            error_message = '\n'.join(sorted(set(error_messages))) if error_messages else "Unknown error occurred"
            self.logger.error(f"Execution failed:\n{error_message}")
        else:
            status = ExecutionStatus.SUCCESS
            error_message = ""

        return ExecutionResult(
            status=status,
            stdout=stdout,
            stderr=stderr,
            error_message=error_message,
            return_code=return_code
        )
    
@dataclass
@dataclass
class CodeGenerationSystem:
    def __init__(self, llm: LLM):
        self.generator = CodeGenerator(llm)
        self.executor = TerminalExecutor()
        self.logger = logging.getLogger(__name__)
        self.history_dir = os.path.join(os.getcwd(), "conversation_history")
        self.current_history = ConversationHistory()
        self.current_context = ""  # Add this to store current session context
        
        # Create history directory if it doesn't exist
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)

    def run(self, user_prompt: str) -> None:
        """Main execution loop with conversation memory."""
        generation = CodeGeneration()
        script_path = os.path.join(self.executor.output_dir, "generated_script.py")
        
        # Add prompt to history
        self.current_history.prompts.append(user_prompt)
        
        # Load recent history for context if this is a new session
        if not self.current_context:
            recent_histories = self._load_recent_history()
            self.current_context = self._create_context_from_history(recent_histories, user_prompt)
        
        while True:
            if generation.attempt_count >= generation.max_attempts:
                print("\nMaximum attempts reached. Exiting.")
                self._save_history()
                return

            # Generate code with historical and current session context
            print(f"\n[Attempt {generation.attempt_count + 1}] Generating code...")
            
            # Create combined context including current session
            current_session_context = "\n\nCurrent session modifications:\n"
            for i, prompt in enumerate(self.current_history.prompts[-3:]):  # Last 3 prompts
                if i > 0:  # Skip the first prompt as it's already in historical context
                    current_session_context += f"- {prompt}\n"
            
            combined_context = self.current_context + current_session_context
            
            generation.code = self.generator.generate_code_with_context(
                user_prompt,
                generation.error_message,
                combined_context
            )
            
            # Save code to history
            self.current_history.codes.append(generation.code)
            
            # Save and display code
            self.save_code(generation.code, script_path)
            print("\nGenerated Code:")
            print("-" * 40)
            print(generation.code)
            print("-" * 40)

            # Execute code
            print("\nExecuting code in new terminal...")
            result = self.executor.execute_in_new_terminal(script_path)
            
            if result.error_message:
                self.current_history.errors.append(result.error_message)
            
            # Check success conditions
            success_conditions = (
                result.status == ExecutionStatus.SUCCESS and
                "Successfully" in result.stdout and
                not result.stderr.strip() and
                result.return_code == 0
            )

            if success_conditions:
                print("\nExecution successful!")
                self._save_history()
                if self._should_continue():
                    # Preserve context by keeping the same generation object
                    # but updating the user prompt
                    user_prompt = input("Enter your modifications: ")
                    self.current_history.prompts.append(user_prompt)
                    # Reset error message but keep other context
                    generation.error_message = ""
                    continue
                return

            # Handle errors
            generation.attempt_count += 1
            
            # Handle installations
            new_installations = self.generator.generate_installation_commands(
                result.error_message,
                generation.installations
            )
            
            if new_installations:
                print("\nInstalling required packages...")
                for cmd in new_installations:
                    install_result = self.executor.run_installation(cmd)
                    if install_result.status == ExecutionStatus.SUCCESS:
                        generation.installations.append(cmd)
                        self.current_history.installations.append(cmd)
                        print(f"Successfully installed: {cmd}")
                    else:
                        print(f"Installation failed: {install_result.error_message}")
            
            generation.error_message = result.error_message or result.stderr or "Unknown error occurred"
    def save_code(self, code: str, filename: str) -> None:
        """Save generated code to file."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                f.write(code)
        except Exception as e:
            self.logger.error(f"Error saving code: {e}")
            raise

    def _save_history(self) -> None:
        """Save conversation history to a file."""
        history_file = os.path.join(self.history_dir, f"history_{self.current_history.timestamp}.json")
        history_data = {
            "timestamp": self.current_history.timestamp,
            "prompts": self.current_history.prompts,
            "codes": self.current_history.codes,
            "errors": self.current_history.errors,
            "installations": self.current_history.installations
        }
        
        try:
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save history: {e}")

    def _load_recent_history(self, limit: int = 5) -> List[ConversationHistory]:
        """Load recent conversation histories."""
        histories = []
        try:
            if os.path.exists(self.history_dir):
                history_files = sorted(
                    [f for f in os.listdir(self.history_dir) if f.startswith("history_")],
                    reverse=True
                )[:limit]
                
                for file in history_files:
                    with open(os.path.join(self.history_dir, file), 'r') as f:
                        data = json.load(f)
                        history = ConversationHistory(
                            prompts=data["prompts"],
                            codes=data["codes"],
                            errors=data["errors"],
                            installations=data["installations"],
                            timestamp=data["timestamp"]
                        )
                        histories.append(history)
        except Exception as e:
            self.logger.error(f"Failed to load history: {e}")
        
        return histories

    def _create_context_from_history(self, histories: List[ConversationHistory], current_prompt: str) -> str:
        """Create context string from historical conversations."""
        context_parts = []
        
        for history in histories:
            successful_examples = self._find_successful_examples(history)
            if successful_examples:
                context_parts.extend(successful_examples)
        
        # Find similar prompts from history
        similar_examples = self._find_similar_examples(histories, current_prompt)
        if similar_examples:
            context_parts.extend(similar_examples)
        
        if context_parts:
            return "\n\nPrevious successful examples:\n" + "\n\n".join(context_parts)
        return ""

    def _find_successful_examples(self, history: ConversationHistory) -> List[str]:
        """Find successful code examples from history."""
        examples = []
        for prompt, code in zip(history.prompts, history.codes):
            # Basic check for successful code
            if "Successfully" in code and prompt:
                examples.append(f"Prompt: {prompt}\nCode:\n{code}")
        return examples

    def _find_similar_examples(self, histories: List[ConversationHistory], current_prompt: str) -> List[str]:
        """Find similar examples based on prompt similarity."""
        similar_examples = []
        current_words = set(current_prompt.lower().split())
        
        for history in histories:
            for prompt, code in zip(history.prompts, history.codes):
                prompt_words = set(prompt.lower().split())
                # Simple word overlap similarity
                similarity = len(current_words & prompt_words) / len(current_words | prompt_words)
                if similarity > 0.3:  # Threshold for similarity
                    similar_examples.append(f"Similar example:\nPrompt: {prompt}\nCode:\n{code}")
                    
        return similar_examples[:2]  # Limit to 2 similar examples

    def _should_continue(self) -> bool:
        """Check if user wants to continue with modifications."""
        response = input("\nWould you like to modify the program? (yes/no): ").lower()
        return response in ('y', 'yes', 'ye')
    
def cleanup_previous_runs():
    """Clean up previous conversation history and temp outputs."""
    def remove_directory(path: str) -> None:
        try:
            if os.path.exists(path):
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            os.rmdir(file_path)
                    except Exception as e:
                        logging.warning(f"Error while deleting {file_path}: {e}")
                os.rmdir(path)
                logging.info(f"Cleaned up directory: {path}")
        except Exception as e:
            logging.warning(f"Error while cleaning up {path}: {e}")

    # Clean up previous runs
    current_dir = os.getcwd()
    history_dir = os.path.join(current_dir, "conversation_history")
    temp_dir = os.path.join(current_dir, "temp_outputs")
    
    remove_directory(history_dir)
    remove_directory(temp_dir)

def main():
    """Main entry point."""
    try:
        # Clean up previous runs first
        cleanup_previous_runs()
        
        llm = OllamaLLM()
        system = CodeGenerationSystem(llm)
        
        user_prompt = input("Enter your prompt to write a Python program: ")
        system.run(user_prompt)
        
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    main()