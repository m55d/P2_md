# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "pillow",
#   "python-dotenv",
#   "requests",
# ]
# ///

import os
import csv
import json
import logging
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
import base64
import io
from datetime import datetime
import traceback
import numpy as np
import sys

try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import ChatCompletionMessage
except ImportError:
    print("openai not found, install with 'uv pip install openai'")
    exit(1)


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
WAIT_SECONDS = 10
IMAGE_SIZE = (512, 512) # width, height
API_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1"


def get_api_key():
    """Retrieves the API key from environment variables."""
    api_key = os.environ.get("AIPROXY_TOKEN")
    if not api_key:
        logging.error("AIPROXY_TOKEN environment variable not set.")
        raise EnvironmentError("AIPROXY_TOKEN not found")
    return api_key


def create_openai_client():
    """Creates an OpenAI client with error handling."""
    try:
          return OpenAI(api_key=get_api_key(), base_url=API_BASE_URL)
    except Exception as e:
        logging.error(f"Error creating OpenAI client: {e}")
        raise


client = create_openai_client()


def safe_execute_code(code_string, local_vars=None):
    """Safely executes Python code string, capturing output and errors."""
    try:
        if local_vars is None:
            local_vars = {}
            
        # Extract code from markdown-style code blocks if present
        if "```" in code_string:
            code_parts = code_string.split("```")
            code_string = ""
            for part in code_parts:
                if part.strip().startswith("python"):
                    code_string = part.replace("python", "").strip()
                    break
        
        # Clean up the code string
        code_string = code_string.strip()
        if not code_string:
            return None, "Empty code string"
            
        # Validate code string
        try:
            compile(code_string, '<string>', 'exec')
        except SyntaxError as e:
            return None, f"Syntax error in code: {str(e)}"

        # Define execution context
        exec_locals = {
            'pd': pd, 
            'plt': plt, 
            'sns': sns, 
            'np': np,
            **local_vars
        }

        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # Execute the code
            exec(code_string, {}, exec_locals)
            
            # Handle matplotlib plots
            if 'plt' in exec_locals:
                plt_instance = exec_locals['plt']
                if plt_instance.get_fignums():
                    plt.close('all')
                    
            return None, None

        finally:
            # Restore stdout
            sys.stdout = old_stdout
            plt.close('all')  # Ensure all plots are closed

    except Exception as e:
        plt.close('all')  # Clean up any open plots
        # Only log actual errors, not code execution details
        if not str(e).startswith("plt."):  # Don't log matplotlib commands
            logging.error(f"Error executing code: {str(e)}")
        return None, str(e)


def analyze_data(file_path):
    """Analyzes data using pandas, returning summaries and the DataFrame."""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logging.info(f"Successfully loaded data with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logging.error(f"Error loading data with {encoding} encoding: {str(e)}")
                continue
        
        if df is None:
            raise ValueError("Could not read file with any of the attempted encodings")

        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Handle any special characters in data
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.encode('ascii', 'ignore').str.decode('ascii')
        
        logging.info(f"Loaded data from {file_path} successfully.")
        
        # Data Summary
        data_summary = {
            "filename": file_path,
            "columns": list(df.columns),
            "data_types": df.dtypes.apply(str).to_dict(),
            "sample_values": df.head(5).to_dict(orient='list'),
            "missing_values": df.isnull().sum().to_dict(),
            "describe": df.describe().to_dict()
        }

        return df, data_summary, list(df.columns)
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return None, None, None


def generate_llm_prompt(data_summary, step_description, user_message):
    """Generates prompts for LLM with a system instruction."""
    system_instruction = (
        "You are an expert data analyst and storyteller. Your goal is to analyze the provided data and create "
        "a comprehensive narrative. When generating code:"
        "\n1. Do not include explanatory text or comments"
        "\n2. Only return the pure Python code block"
        "\n3. Use pandas, matplotlib, or seaborn only"
        "\n4. Always close plots after saving"
        "\n5. Use clear variable names"
        "\nBe concise and direct in your responses."
    )
    
    prompt = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"{step_description}\nData Summary: {json.dumps(data_summary, indent=2)}\nUser Message: {user_message}"},
    ]
    return prompt


def get_llm_response(prompt, use_function_call=False, functions=None):
    """Sends prompt to LLM, handles retries, and returns the response."""

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_fixed(WAIT_SECONDS), retry_error_callback=lambda _: True)
    def _get_llm_response():
        try:
            if use_function_call:
                response = client.chat.completions.create(
                    model=MODEL, 
                    messages=prompt, 
                    functions=functions, 
                    function_call={"name": "generate_chart"}  # Force function call
                )
            else:
                response = client.chat.completions.create(
                    model=MODEL, 
                    messages=prompt
                )
            return response
        except Exception as e:
             logging.error(f"LLM call failed with exception : {e}")
             raise e

    try:
        response = _get_llm_response()

        if isinstance(response, ChatCompletion):
            if use_function_call:
                message = response.choices[0].message
                if hasattr(message, 'function_call') and message.function_call:
                    return message.function_call, None
                else:
                    logging.error("Expected function call but got regular message")
                    return None, "No function call in response"
            else:
                return response.choices[0].message.content, None
        else:
            logging.error(f"LLM response is of type : {type(response)}")
            return None, "invalid response type"

    except Exception as e:
        logging.error(f"LLM call failed after retries: {e}")
        return None, str(e)


def create_chart(df, chart_type, x_col, y_col, title, file_name, additional_params={}):
    """Creates and saves a chart using matplotlib/seaborn. If the chart is more complex or requires code, return None"""
    try:
        plt.figure(figsize=(10, 6))
        if chart_type == "histplot":
            sns.histplot(df[x_col], **additional_params)
        elif chart_type == "scatterplot":
            sns.scatterplot(x=x_col, y=y_col, data=df, **additional_params)
        elif chart_type == "boxplot":
            sns.boxplot(y=df[y_col], **additional_params)
        elif chart_type == "heatmap":
            corr_matrix = df.corr(numeric_only=True)
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        elif chart_type == "pie_chart":
            df[x_col].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, **additional_params)
            plt.ylabel("")  # Remove default y-label for pie chart
        else:
            plt.close()
            return None
        
        plt.title(title)
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot after saving to free up resources
        return file_name
    except Exception as e:
        logging.error(f"Error creating chart: {e}")
        plt.close()
        return None


def analyze_and_visualize(file_path):
    """Orchestrates the analysis, visualization, and storytelling process."""
    df, data_summary, columns = analyze_data(file_path)
    if df is None:
        return None, None, None

    markdown_content = ""
    image_paths = []

    # Suppress matplotlib debug messages
    plt.set_loglevel('warning')
    
    # --- Initial Analysis ---
    user_message = "Summarize the provided dataset, identify potential areas of interest, and suggest initial analysis to explore the data."
    prompt = generate_llm_prompt(data_summary, "Initial Analysis of the data.", user_message)
    llm_response, err = get_llm_response(prompt)
    if err:
        logging.error(f"Error during initial analysis prompt: {err}")
        return None, None, None

    markdown_content += f"## Initial Data Analysis\n\n{llm_response}\n\n"

    # --- Generic Visualizations ---
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) > 1:
        for i in range(min(len(numeric_cols), 2)):
            x_col = numeric_cols[i]
            file_name = f"histogram_{x_col}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            chart_path = create_chart(df, "histplot", x_col=x_col, y_col=None, title=f"Distribution of {x_col}", file_name=file_name)
            if chart_path:
                image_paths.append(chart_path)
                markdown_content += f"![{x_col} Distribution]({chart_path})\n\n"
        if len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            file_name = f"scatterplot_{x_col}_{y_col}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            chart_path = create_chart(df, "scatterplot", x_col=x_col, y_col=y_col, title=f"Scatter plot of {x_col} vs {y_col}", file_name=file_name)
            if chart_path:
                image_paths.append(chart_path)
                markdown_content += f"![{x_col} vs {y_col} Scatter]({chart_path})\n\n"
        if len(numeric_cols) > 2:
            file_name = f"heatmap_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            chart_path = create_chart(df, "heatmap", x_col=None, y_col=None, title="Correlation Heatmap", file_name=file_name)
            if chart_path:
                image_paths.append(chart_path)
                markdown_content += f"![Correlation Heatmap]({chart_path})\n\n"

    # --- LLM-Driven Analysis and Visualization ---
    function_schema = {
        "name": "generate_chart",
        "description": "Generates a chart based on the data",
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {"type": "string", "enum": ["histplot", "scatterplot", "boxplot", "pie_chart"], "description": "The type of chart to create"},
                "x_col": {"type": "string", "description": "The column for the x-axis"},
                "y_col": {"type": "string", "description": "The column for the y-axis"},
                "title": {"type": "string", "description": "Title of the chart"},
                "file_name": {"type": "string", "description": "File name of chart"},
                "additional_params": {"type": "object", "description": "Additional parameters of the chart"}
            },
            "required": ["chart_type", "title", "file_name"]
        }
    }

    user_message = "Suggest a specific data analysis that includes visualization and explain how it would help us understand the dataset, based on the provided summary. You should use the generate_chart function call"
    prompt = generate_llm_prompt(data_summary, "Ask for specific analysis", user_message)
    llm_function_call, err = get_llm_response(prompt, use_function_call=True, functions=[function_schema])
    if err:
        logging.error(f"Error during function call prompt: {err}")
        return markdown_content, image_paths, None

    if llm_function_call:
        try:
            arguments = json.loads(llm_function_call.arguments)
            chart_type = arguments.get("chart_type")
            x_col = arguments.get("x_col", None)
            y_col = arguments.get("y_col", None)
            title = arguments.get("title")
            file_name = arguments.get("file_name", f"custom_chart_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
            additional_params = arguments.get("additional_params", {})
            chart_path = create_chart(df, chart_type, x_col, y_col, title, file_name, additional_params)
            if chart_path:
                image_paths.append(chart_path)
                markdown_content += f"![Custom Chart]({chart_path})\n\n"
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing function call arguments: {e}")

    # --- LLM-Driven Code Generation Analysis ---
    user_message = "Generate python code to create a more specific analysis, with clear instructions. Always save your charts in png format using plt.savefig(). Only use pandas, matplotlib or seaborn."
    prompt = generate_llm_prompt(data_summary, "Ask for specific code", user_message)
    llm_code, err = get_llm_response(prompt)
    if err:
        logging.error(f"Error getting code from LLM: {err}")
        return markdown_content, image_paths, None
    if llm_code:
        _, code_error = safe_execute_code(llm_code, local_vars={'df': df})
        if code_error:
            logging.error(f"Error executing LLM generated code: {code_error}")
        else:
            for file in os.listdir():
                if file.lower().endswith(('.png')):
                    image_paths.append(file)
                    markdown_content += f"![Custom Code Generated Chart]({file})\n\n"

    # --- Story Generation ---
    user_message = "Create a concise story based on the analysis performed, including insights and implications of findings. Integrate the analysis and visualizations we have so far and make sure the images are referenced with markdown syntax"
    prompt = generate_llm_prompt(data_summary, "Create the Story", user_message)
    llm_narrative, err = get_llm_response(prompt)
    if err:
        logging.error(f"Error getting narrative from LLM: {err}")
        return markdown_content, image_paths, None

    markdown_content += f"\n\n## Narrative\n\n{llm_narrative}"

    return markdown_content, image_paths, True


def main(file_path):
    """Main function to process the CSV, analyze, and generate outputs."""
    logging.info(f"Starting analysis for {file_path}")
    markdown_content, image_paths, success = analyze_and_visualize(file_path)

    if not success:
        logging.error(f"Analysis failed for {file_path}.")
        return

    # Save the markdown file
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)
    logging.info("README.md created.")

    logging.info("Script execution completed.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    main(csv_file)





