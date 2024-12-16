# # /// script
# # requires-python = ">=3.11"
# # dependencies = [
# #   "pandas",
# #   "numpy",
# #   "matplotlib",
# #   "seaborn",
# #   "pillow",
# #   "python-dotenv",
# #   "requests",
# # ]
# # ///

# import os
# import csv
# import json
# import logging
# import random
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
# import base64
# import io
# from datetime import datetime
# import traceback
# import numpy as np
# import sys

# try:
#     from openai import OpenAI
#     from openai.types.chat import ChatCompletion
#     from openai.types.chat.chat_completion import ChatCompletionMessage
# except ImportError:
#     print("openai not found, install with 'uv pip install openai'")
#     exit(1)


# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Constants
# MODEL = "gpt-4o-mini"
# MAX_RETRIES = 3
# WAIT_SECONDS = 10
# IMAGE_SIZE = (512, 512) # width, height
# API_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1"


# def get_api_key():
#     """Retrieves the API key from environment variables."""
#     api_key = os.environ.get("AIPROXY_TOKEN")
#     if not api_key:
#         logging.error("AIPROXY_TOKEN environment variable not set.")
#         raise EnvironmentError("AIPROXY_TOKEN not found")
#     return api_key


# def create_openai_client():
#     """Creates an OpenAI client with error handling."""
#     try:
#           return OpenAI(api_key=get_api_key(), base_url=API_BASE_URL)
#     except Exception as e:
#         logging.error(f"Error creating OpenAI client: {e}")
#         raise


# client = create_openai_client()


# def safe_execute_code(code_string, local_vars=None):
#     """Safely executes Python code string, capturing output and errors."""
#     try:
#         if local_vars is None:
#             local_vars = {}
            
#         # Extract code from markdown-style code blocks if present
#         if "```" in code_string:
#             code_parts = code_string.split("```")
#             code_string = ""
#             for part in code_parts:
#                 if part.strip().startswith("python"):
#                     code_string = part.replace("python", "").strip()
#                     break
        
#         # Clean up the code string
#         code_string = code_string.strip()
#         if not code_string:
#             return None, "Empty code string"
            
#         # Validate code string
#         try:
#             compile(code_string, '<string>', 'exec')
#         except SyntaxError as e:
#             return None, f"Syntax error in code: {str(e)}"

#         # Define execution context
#         exec_locals = {
#             'pd': pd, 
#             'plt': plt, 
#             'sns': sns, 
#             'np': np,
#             **local_vars
#         }

#         # Redirect stdout to capture print statements
#         old_stdout = sys.stdout
#         sys.stdout = io.StringIO()

#         try:
#             # Execute the code
#             exec(code_string, {}, exec_locals)
            
#             # Handle matplotlib plots
#             if 'plt' in exec_locals:
#                 plt_instance = exec_locals['plt']
#                 if plt_instance.get_fignums():
#                     plt.close('all')
                    
#             return None, None

#         finally:
#             # Restore stdout
#             sys.stdout = old_stdout
#             plt.close('all')  # Ensure all plots are closed

#     except Exception as e:
#         plt.close('all')  # Clean up any open plots
#         # Only log actual errors, not code execution details
#         if not str(e).startswith("plt."):  # Don't log matplotlib commands
#             logging.error(f"Error executing code: {str(e)}")
#         return None, str(e)


# def analyze_data(file_path):
#     """Analyzes data using pandas, returning summaries and the DataFrame."""
#     try:
#         # Try different encodings
#         encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
#         df = None
        
#         for encoding in encodings:
#             try:
#                 df = pd.read_csv(file_path, encoding=encoding)
#                 logging.info(f"Successfully loaded data with {encoding} encoding")
#                 break
#             except UnicodeDecodeError:
#                 continue
#             except Exception as e:
#                 logging.error(f"Error loading data with {encoding} encoding: {str(e)}")
#                 continue
        
#         if df is None:
#             raise ValueError("Could not read file with any of the attempted encodings")

#         # Clean column names
#         df.columns = df.columns.str.strip()
        
#         # Handle any special characters in data
#         for col in df.select_dtypes(include=['object']).columns:
#             df[col] = df[col].astype(str).str.encode('ascii', 'ignore').str.decode('ascii')
        
#         logging.info(f"Loaded data from {file_path} successfully.")
        
#         # Data Summary
#         data_summary = {
#             "filename": file_path,
#             "columns": list(df.columns),
#             "data_types": df.dtypes.apply(str).to_dict(),
#             "sample_values": df.head(5).to_dict(orient='list'),
#             "missing_values": df.isnull().sum().to_dict(),
#             "describe": df.describe().to_dict()
#         }

#         return df, data_summary, list(df.columns)
        
#     except Exception as e:
#         logging.error(f"Error loading data: {str(e)}")
#         return None, None, None


# def generate_llm_prompt(data_summary, step_description, user_message):
#     """Generates prompts for LLM with a system instruction."""
#     system_instruction = (
#         "You are an expert data analyst and storyteller. Your goal is to analyze the provided data and create "
#         "a comprehensive narrative. When generating code:"
#         "\n1. Do not include explanatory text or comments"
#         "\n2. Only return the pure Python code block"
#         "\n3. Use pandas, matplotlib, or seaborn only"
#         "\n4. Always close plots after saving"
#         "\n5. Use clear variable names"
#         "\nBe concise and direct in your responses."
#     )
    
#     prompt = [
#         {"role": "system", "content": system_instruction},
#         {"role": "user", "content": f"{step_description}\nData Summary: {json.dumps(data_summary, indent=2)}\nUser Message: {user_message}"},
#     ]
#     return prompt


# def get_llm_response(prompt, use_function_call=False, functions=None):
#     """Sends prompt to LLM, handles retries, and returns the response."""

#     @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_fixed(WAIT_SECONDS), retry_error_callback=lambda _: True)
#     def _get_llm_response():
#         try:
#             if use_function_call:
#                 response = client.chat.completions.create(
#                     model=MODEL, 
#                     messages=prompt, 
#                     functions=functions, 
#                     function_call={"name": "generate_chart"}  # Force function call
#                 )
#             else:
#                 response = client.chat.completions.create(
#                     model=MODEL, 
#                     messages=prompt
#                 )
#             return response
#         except Exception as e:
#              logging.error(f"LLM call failed with exception : {e}")
#              raise e

#     try:
#         response = _get_llm_response()

#         if isinstance(response, ChatCompletion):
#             if use_function_call:
#                 message = response.choices[0].message
#                 if hasattr(message, 'function_call') and message.function_call:
#                     return message.function_call, None
#                 else:
#                     logging.error("Expected function call but got regular message")
#                     return None, "No function call in response"
#             else:
#                 return response.choices[0].message.content, None
#         else:
#             logging.error(f"LLM response is of type : {type(response)}")
#             return None, "invalid response type"

#     except Exception as e:
#         logging.error(f"LLM call failed after retries: {e}")
#         return None, str(e)


# def create_chart(df, chart_type, x_col, y_col, title, file_name, additional_params={}):
#     """Creates and saves a chart using matplotlib/seaborn. If the chart is more complex or requires code, return None"""
#     try:
#         plt.figure(figsize=(10, 6))
#         if chart_type == "histplot":
#             sns.histplot(df[x_col], **additional_params)
#         elif chart_type == "scatterplot":
#             sns.scatterplot(x=x_col, y=y_col, data=df, **additional_params)
#         elif chart_type == "boxplot":
#             sns.boxplot(y=df[y_col], **additional_params)
#         elif chart_type == "heatmap":
#             corr_matrix = df.corr(numeric_only=True)
#             sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
#         elif chart_type == "pie_chart":
#             df[x_col].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, **additional_params)
#             plt.ylabel("")  # Remove default y-label for pie chart
#         else:
#             plt.close()
#             return None
        
#         plt.title(title)
#         plt.savefig(file_name, dpi=300, bbox_inches='tight')
#         plt.close()  # Close the plot after saving to free up resources
#         return file_name
#     except Exception as e:
#         logging.error(f"Error creating chart: {e}")
#         plt.close()
#         return None


# def analyze_and_visualize(file_path):
#     """Orchestrates the analysis, visualization, and storytelling process."""
#     df, data_summary, columns = analyze_data(file_path)
#     if df is None:
#         return None, None, None

#     markdown_content = ""
#     image_paths = []

#     # Suppress matplotlib debug messages
#     plt.set_loglevel('warning')
    
#     # --- Initial Analysis ---
#     user_message = "Summarize the provided dataset, identify potential areas of interest, and suggest initial analysis to explore the data."
#     prompt = generate_llm_prompt(data_summary, "Initial Analysis of the data.", user_message)
#     llm_response, err = get_llm_response(prompt)
#     if err:
#         logging.error(f"Error during initial analysis prompt: {err}")
#         return None, None, None

#     markdown_content += f"## Initial Data Analysis\n\n{llm_response}\n\n"

#     # --- Generic Visualizations ---
#     numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     if len(numeric_cols) > 1:
#         for i in range(min(len(numeric_cols), 2)):
#             x_col = numeric_cols[i]
#             file_name = f"histogram_{x_col}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
#             chart_path = create_chart(df, "histplot", x_col=x_col, y_col=None, title=f"Distribution of {x_col}", file_name=file_name)
#             if chart_path:
#                 image_paths.append(chart_path)
#                 markdown_content += f"![{x_col} Distribution]({chart_path})\n\n"
#         if len(numeric_cols) >= 2:
#             x_col = numeric_cols[0]
#             y_col = numeric_cols[1]
#             file_name = f"scatterplot_{x_col}_{y_col}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
#             chart_path = create_chart(df, "scatterplot", x_col=x_col, y_col=y_col, title=f"Scatter plot of {x_col} vs {y_col}", file_name=file_name)
#             if chart_path:
#                 image_paths.append(chart_path)
#                 markdown_content += f"![{x_col} vs {y_col} Scatter]({chart_path})\n\n"
#         if len(numeric_cols) > 2:
#             file_name = f"heatmap_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
#             chart_path = create_chart(df, "heatmap", x_col=None, y_col=None, title="Correlation Heatmap", file_name=file_name)
#             if chart_path:
#                 image_paths.append(chart_path)
#                 markdown_content += f"![Correlation Heatmap]({chart_path})\n\n"

#     # --- LLM-Driven Analysis and Visualization ---
#     function_schema = {
#         "name": "generate_chart",
#         "description": "Generates a chart based on the data",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "chart_type": {"type": "string", "enum": ["histplot", "scatterplot", "boxplot", "pie_chart"], "description": "The type of chart to create"},
#                 "x_col": {"type": "string", "description": "The column for the x-axis"},
#                 "y_col": {"type": "string", "description": "The column for the y-axis"},
#                 "title": {"type": "string", "description": "Title of the chart"},
#                 "file_name": {"type": "string", "description": "File name of chart"},
#                 "additional_params": {"type": "object", "description": "Additional parameters of the chart"}
#             },
#             "required": ["chart_type", "title", "file_name"]
#         }
#     }

#     user_message = "Suggest a specific data analysis that includes visualization and explain how it would help us understand the dataset, based on the provided summary. You should use the generate_chart function call"
#     prompt = generate_llm_prompt(data_summary, "Ask for specific analysis", user_message)
#     llm_function_call, err = get_llm_response(prompt, use_function_call=True, functions=[function_schema])
#     if err:
#         logging.error(f"Error during function call prompt: {err}")
#         return markdown_content, image_paths, None

#     if llm_function_call:
#         try:
#             arguments = json.loads(llm_function_call.arguments)
#             chart_type = arguments.get("chart_type")
#             x_col = arguments.get("x_col", None)
#             y_col = arguments.get("y_col", None)
#             title = arguments.get("title")
#             file_name = arguments.get("file_name", f"custom_chart_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
#             additional_params = arguments.get("additional_params", {})
#             chart_path = create_chart(df, chart_type, x_col, y_col, title, file_name, additional_params)
#             if chart_path:
#                 image_paths.append(chart_path)
#                 markdown_content += f"![Custom Chart]({chart_path})\n\n"
#         except json.JSONDecodeError as e:
#             logging.error(f"Error parsing function call arguments: {e}")

#     # --- LLM-Driven Code Generation Analysis ---
#     user_message = "Generate python code to create a more specific analysis, with clear instructions. Always save your charts in png format using plt.savefig(). Only use pandas, matplotlib or seaborn."
#     prompt = generate_llm_prompt(data_summary, "Ask for specific code", user_message)
#     llm_code, err = get_llm_response(prompt)
#     if err:
#         logging.error(f"Error getting code from LLM: {err}")
#         return markdown_content, image_paths, None
#     if llm_code:
#         _, code_error = safe_execute_code(llm_code, local_vars={'df': df})
#         if code_error:
#             logging.error(f"Error executing LLM generated code: {code_error}")
#         else:
#             for file in os.listdir():
#                 if file.lower().endswith(('.png')):
#                     image_paths.append(file)
#                     markdown_content += f"![Custom Code Generated Chart]({file})\n\n"

#     # --- Story Generation ---
#     user_message = "Create a concise story based on the analysis performed, including insights and implications of findings. Integrate the analysis and visualizations we have so far and make sure the images are referenced with markdown syntax"
#     prompt = generate_llm_prompt(data_summary, "Create the Story", user_message)
#     llm_narrative, err = get_llm_response(prompt)
#     if err:
#         logging.error(f"Error getting narrative from LLM: {err}")
#         return markdown_content, image_paths, None

#     markdown_content += f"\n\n## Narrative\n\n{llm_narrative}"

#     return markdown_content, image_paths, True


# def main(file_path):
#     """Main function to process the CSV, analyze, and generate outputs."""
#     logging.info(f"Starting analysis for {file_path}")
#     markdown_content, image_paths, success = analyze_and_visualize(file_path)

#     if not success:
#         logging.error(f"Analysis failed for {file_path}.")
#         return

#     # Save the markdown file
#     with open("README.md", "w", encoding="utf-8") as f:
#         f.write(markdown_content)
#     logging.info("README.md created.")

#     logging.info("Script execution completed.")


# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) != 2:
#         print("Usage: python autolysis.py <csv_file>")
#         sys.exit(1)

#     csv_file = sys.argv[1]
#     main(csv_file)




# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "chardet",
#   "scikit-learn",
#   "tabulate",
#   "scikit-learn"
# ]
# ///
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from dateutil import parser
import chardet
import json
import re
import hashlib

# Environment variable for AI Proxy token
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise EnvironmentError("AIPROXY_TOKEN is not set. Please set it before running the script.")

def detect_encoding(file_path):
    """
    Detects the file encoding using chardet.
    """
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        return result['encoding']

def parse_date_with_regex(date_str):
    """
    Attempts to parse date string using regular expressions.
    """
    if not isinstance(date_str, str):  
        return date_str  

    if not re.search(r'\d', date_str):  # Check for digits in the string
        return np.nan

    patterns = [
        (r"\d{2}-[A-Za-z]{3}-\d{4}", "%d-%b-%Y"), 
        (r"\d{2}-[A-Za-z]{3}-\d{2}", "%d-%b-%y"),  
        (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),  
        (r"\d{2}/\d{2}/\d{4}", "%m/%d/%Y"),  
        (r"\d{2}/\d{2}/\d{4}", "%d/%m/%Y")
    ]

    for pattern, date_format in patterns:
        if re.match(pattern, date_str):
            try:
                return pd.to_datetime(date_str, format=date_format, errors='coerce')
            except Exception as e:
                print(f"Error parsing date: {date_str} with format {date_format}. Error: {e}")
                return np.nan

    try:
        return parser.parse(date_str, fuzzy=True, dayfirst=False)
    except Exception as e:
        print(f"Error parsing date with dateutil: {date_str}. Error: {e}")
        return np.nan

def detect_date_column(column):
    """
    Identifies columns that are likely to contain date values.
    """
    if isinstance(column, str):
        if any(keyword in column.lower() for keyword in ['date', 'time', 'timestamp']):
            return True

    sample_values = column.dropna().head(10)
    date_patterns = [r"\d{2}-[A-Za-z]{3}-\d{2}", r"\d{2}-[A-Za-z]{3}-\d{4}", r"\d{4}-\d{2}-\d{2}", r"\d{2}/\d{2}/\d{4}"]

    for value in sample_values:
        if isinstance(value, str):
            for pattern in date_patterns:
                if re.match(pattern, value):
                    return True
    return False

def read_csv(file_path):
    """
    Reads a CSV file, detects its encoding, and attempts to parse date columns.
    """
    try:
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding, encoding_errors='replace')
        for column in df.columns:
            if df[column].dtype == object and detect_date_column(df[column]):
                df[column] = df[column].apply(parse_date_with_regex)
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)


def statistical_tests(df):
    """
    Performs t-tests and ANOVA on numerical columns to find statistically significant differences.
    """
    numeric_data = df.select_dtypes(include=[np.number])
    results = {}

    # Pairwise t-tests
    for col1 in numeric_data.columns:
        for col2 in numeric_data.columns:
            if col1 != col2:
                stat, p_value = ttest_ind(numeric_data[col1].dropna(), numeric_data[col2].dropna())
                results[f"T-test: {col1} vs {col2}"] = {"Statistic": stat, "P-value": p_value}
    
    # ANOVA
    if numeric_data.shape[1] > 2:
        anova_stat, anova_p_value = f_oneway(*[numeric_data[col].dropna() for col in numeric_data.columns])
        results["ANOVA"] = {"Statistic": anova_stat, "P-value": anova_p_value}

    return results

def perform_advanced_analysis(df):
    """
    Performs a comprehensive analysis of the dataframe, including summary statistics and outlier detection.
    """
    analysis = {
        "shape": df.shape,
        "columns": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_statistics": df.describe(include="all").to_dict(),
    }
    for column in df.select_dtypes(include=[np.datetime64]).columns:
        df[column] = df[column].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    outliers = detect_outliers(df)
    if outliers is not None:
        analysis["outliers"] = outliers.value_counts().to_dict()
    
    return analysis

def detect_outliers(df):
    """
    Detects outliers in numeric columns using Isolation Forest.
    """
    numeric_data = df.select_dtypes(include=[np.number]).dropna()
    if numeric_data.empty:
        return None
    iso = IsolationForest(contamination=0.05, random_state=42)
    numeric_data["outliers"] = iso.fit_predict(numeric_data)
    return numeric_data["outliers"]

def regression_analysis(df):
    """
    Performs regression analysis on numeric columns.
    """
    numeric_data = df.select_dtypes(include=[np.number])
    if numeric_data.shape[1] < 2:
        return None
    x = numeric_data.iloc[:, :-1]
    y = numeric_data.iloc[:, -1]
    model = LinearRegression()
    model.fit(x, y)
    predictions = model.predict(x)
    feature_importance = dict(zip(x.columns, np.abs(model.coef_)))
    return {
        "MSE": mean_squared_error(y, predictions),
        "R2": r2_score(y, predictions),
        "Feature Importance": feature_importance,
    }

def clustering_analysis(df):
    """
    Performs clustering analysis on numeric columns using KMeans.
    """
    numeric_data = df.select_dtypes(include=[np.number]).dropna()
    if numeric_data.empty:
        return None, None
    kmeans = KMeans(n_clusters=3, random_state=42)
    numeric_data['Cluster'] = kmeans.fit_predict(numeric_data)
    return numeric_data['Cluster'], numeric_data.index

def visualize_advanced(df, output_folder):
    """
    Creates and saves various visualizations such as correlation heatmap, pairplot, and clustering scatter plot.
    """
    visualizations = []

    numeric_data = df.select_dtypes(include=[np.number]).dropna()
    
    sns.set_palette("colorblind")
    
    if not numeric_data.empty:
        # Correlation Heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = numeric_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={'size': 12},
                    linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title("Correlation Heatmap", fontsize=18, pad=20)
        plt.xticks(rotation=45, fontsize=12, ha='right')
        plt.yticks(fontsize=12)
        file_path = os.path.join(output_folder, "correlation_heatmap.png")
        plt.savefig(file_path, bbox_inches='tight')
        visualizations.append("Correlation Heatmap: Shows correlations between numerical features.")
        plt.close()

    
    # Clustering Scatter Plot with improved legend and axis labels
    clusters, valid_indices = clustering_analysis(df)
    if clusters is not None:
        df_with_clusters = numeric_data.loc[valid_indices].copy()
        df_with_clusters["Cluster"] = clusters.values
        plt.figure(figsize=(10, 8))
        palette = sns.color_palette("Set1", n_colors=len(np.unique(clusters)))
        for cluster in np.unique(clusters):
            subset = df_with_clusters[df_with_clusters["Cluster"] == cluster]
            plt.scatter(subset.iloc[:, 0], subset.iloc[:, 1], 
                        label=f"Cluster {cluster}", s=150, alpha=0.8, c=[palette[cluster]], edgecolor="k")
        plt.title("Clustering Scatter Plot", fontsize=16)
        plt.xlabel(df_with_clusters.columns[0], fontsize=12)
        plt.ylabel(df_with_clusters.columns[1], fontsize=12)
        plt.legend(title="Clusters", fontsize=12, loc="best")
        plt.grid(True, linestyle="--", alpha=0.7)
        file_path = os.path.join(output_folder, "clustering_scatter.png")
        plt.savefig(file_path, bbox_inches='tight')
        visualizations.append("Clustering Scatter Plot: Shows the clustering of data points in the 2D space.")
        
        # Pairplot with improved aesthetics
    sns.set(style="whitegrid")
    pairplot = sns.pairplot(df_with_clusters, hue="Cluster", palette="Set2", plot_kws={'alpha': 0.8})
    pairplot.fig.suptitle("Pairplot of Numerical Features", fontsize=16)
    pairplot.fig.tight_layout()
    pairplot.fig.subplots_adjust(top=0.95)  # Title adjustment
    file_path = os.path.join(output_folder, "pairplot.png")
    pairplot.savefig(file_path, bbox_inches='tight')
    visualizations.append("Pairplot: Shows pairwise relationships between numerical features.")

    return visualizations

def generate_summary(clusters, visualizations, analysis_results):
    """
    Generates a detailed summary of the analysis, visualizations, and clustering results.
    """
    summary = f"### Data Overview\n\nShape of Data: {analysis_results['shape']}\n"
    summary += f"Columns and Data Types:\n{analysis_results['columns']}\n\n"
    summary += f"Missing Values:\n{analysis_results['missing_values']}\n\n"
    summary += f"Summary Statistics:\n{analysis_results['summary_statistics']}\n\n"
    
    if "outliers" in analysis_results:
        summary += f"\nOutlier Analysis:\n{analysis_results['outliers']}\n"
    
    for vis in visualizations:
        summary += f"\n- {vis}"
    
    return summary

def query_llm(function_call):
    """
    Queries the LLM with the function call for dynamic analysis-based prompts.
    """
    prompt = f"""
    Use the following information to generate a detailed analysis report:
    - {function_call}
    """
    try:
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",  # Supported chat model
            "messages": [
                {"role": "system", "content": "You are a helpful data analysis assistant. Provide insights, suggestions, and implications based on the given analysis and visualizations."},
                {"role": "user", "content": prompt},
            ],
        }
        payload_json = json.dumps(payload)
        curl_command = [
            "curl",
            "-X", "POST", url,
            "-H", f"Authorization: Bearer {AIPROXY_TOKEN}",
            "-H", "Content-Type: application/json",
            "-d", payload_json
        ]
        result = subprocess.run(curl_command, capture_output=True, text=True)
        if result.returncode == 0:
            response_data = json.loads(result.stdout)
            return response_data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error in curl request: {result.stderr}")
    except Exception as e:
        print(f"Error querying AI Proxy: {e}")
        return "Error: Unable to generate narrative."

def cache_llm_query(function_call, cache_dir="llm_cache"):
    """
    Caches LLM query results to avoid redundant calls.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Hash the function_call to create a unique key
    query_hash = hashlib.md5(function_call.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{query_hash}.json")

    # Check if cache exists
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    # If not cached, query the LLM
    response = query_llm(function_call)
    with open(cache_file, "w") as f:
        json.dump(response, f)

    return response
        
        
def create_readme(analysis, visualizations, story, output_folder):
    """
    Create a structured and detailed README file to summarize the analysis, visualizations, and narrative.
    """
    readme_path = os.path.join(output_folder, "README.md")
    with open(readme_path, "w") as f:
        # Introduction Section
        f.write("# Comprehensive Data Analysis Report\n\n")
        f.write("This report provides an in-depth analysis of the dataset, including data structure, insights, visualizations, and actionable recommendations.\n\n")

        # Data Overview Section
        f.write("## Data Overview\n\n")
        f.write(f"### Dataset Shape\n\n**Rows**: {analysis['shape'][0]}, **Columns**: {analysis['shape'][1]}\n\n")
        f.write("### Columns and Data Types\n\n")
        f.write(pd.DataFrame(list(analysis['columns'].items()), columns=["Column", "Data Type"]).to_markdown(index=False))
        f.write("\n\n")
        f.write("### Missing Values\n\n")
        f.write(pd.DataFrame(list(analysis['missing_values'].items()), columns=["Column", "Missing Values"]).to_markdown(index=False))
        f.write("\n\n")

        # Summary Statistics Section
        f.write("## Summary Statistics\n\n")
        summary_stats = pd.DataFrame(analysis["summary_statistics"])
        f.write(summary_stats.to_markdown())
        f.write("\n\n")

        # Visualizations Section
        f.write("## Visualizations\n\n")
        f.write("Below are the key visualizations generated during the analysis:\n\n")
        for vis in visualizations:
            vis_filename = os.path.basename(vis)
            if "correlation_heatmap" in vis_filename:
                explanation = "The heatmap highlights the correlations between numerical features. Strong correlations may indicate predictive relationships."
            elif "clustering_scatter" in vis_filename:
                explanation = "The scatter plot shows clustering patterns, which can help identify natural groupings in the data."
            elif "pairplot" in vis_filename:
                explanation = "The pairplot provides pairwise visualizations of feature relationships, which are useful for identifying trends and dependencies."
            else:
                explanation = "This visualization provides additional insights into the dataset."
    
            f.write(f"- **{explanation}**\n  ![Visualization]({vis_filename})\n\n")


        # Narrative Section
        f.write("## Key Insights and Narrative\n\n")
        f.write("### Highlights\n\n")
        f.write(story)
        f.write("\n\n")

        # Recommendations Section
        f.write("## Conclusions and Recommendations\n\n")
        f.write("### Summary\n\n")
        f.write("The analysis revealed significant trends and patterns that are critical for understanding the dataset. Recommendations for further exploration and potential action items are outlined below:\n\n")
        f.write("- Address missing data through imputation or collection improvements.\n")
        f.write("- Focus on high-correlation features for predictive modeling.\n")
        f.write("- Investigate outliers to understand their context and impact.\n")
        f.write("- Use clustering insights for targeted interventions or segmentation.\n")
        f.write("\n")
        

def main(file_path):
    """
    Main function to process the data and generate results.
    """
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_folder = os.path.join(os.getcwd(), dataset_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = read_csv(file_path)        
    
    # Perform analysis
    analysis_results = perform_advanced_analysis(df)
    clusters, cluster_indices = clustering_analysis(df)
    analysis_results["statistical_tests"] = statistical_tests(df)

    # Generate visualizations
    visualizations = visualize_advanced(df, output_folder)
    
    # Generate summary
    summary = generate_summary(clusters, visualizations, analysis_results)
    
    # Generate Story
    story = cache_llm_query(summary)
    
    # Create the README.md
    create_readme(analysis_results, visualizations, story, output_folder)

    print(f"Analysis Complete. Visualizations and README saved in '{output_folder}' folder.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    main(input_file)




