import os
import subprocess

from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.tools import agent_tool
from google.api_core import exceptions
import vertexai
from vertexai import model_garden

NotFound = exceptions.NotFound
InvalidArgument = exceptions.InvalidArgument
GoogleAPIError = exceptions.GoogleAPIError
ServiceUnavailable = exceptions.ServiceUnavailable

vertexai.init(
    project=os.environ.get("GOOGLE_CLOUD_PROJECT", None),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", None),
)

search_agent = Agent(
    model="gemini-2.5-flash",
    name="search_agent",
    description="""
    An agent tool in a multi-agent system that specializes in running Google searches to retrieve information about specific Vertex AI Model Garden models that a user might be interested in.
    """,
    instruction="""
    You're a search agent tool that specializes in running Google searches to retrieve information about specific Vertex AI Model Garden models that a user is interested in learning about.
    Your purpose is to help users discover and compare specific AI models from Vertex AI Model Garden.
    ALWAYS cite sources when providing information, like the model name and the source of the information directly.
    Dont return any information that is not directly available in the sources.

   Preferred sources:
      - Vertex AI Model Garden documentation
      - Google Cloud blog/model comparison posts (only if relevant to Vertex AI)
      - GitHub repos linked from Vertex AI Model Garden
      - Hugging Face model pages
      
    - Stick to concise summaries and avoid general platform details or features unrelated to the models themselves.
    - Avoid making up any model names or capabilities not found in documentation
    """,
    tools=[google_search],
)

def list_deployable_models(model_filter: str) -> dict:
    """Lists all deployable models on vertex model garden filtered by the given filter string.

    Args:
      model_filter (str): A string for filtering the resulting list of deployable
        models. The string can only contain letters, numbers, hyphens (-),
        underscores (_), and periods (.) The string will be matched against
        specific model names and must therefore not include anything that would
        not be found in a model name.

    Returns:
      dict: status and content or error message.
    """
    result = {}
    try:
        all_model_garden_models = model_garden.list_deployable_models(
            model_filter="", list_hf_models=False
        )
        model_garden_results = [
            model for model in all_model_garden_models if model_filter.lower() in model
        ]
        huggingface_results = model_garden.list_deployable_models(
            model_filter=model_filter.lower(), list_hf_models=True
        )
        model_search_results = model_garden_results + huggingface_results
        if not model_search_results:
            result["status"] = "error"
            result["error_message"] = (
                "No deployable models with the given filter were found. Please try"
                " searching again with a different filter."
            )
        else:
            result["status"] = "success"
            result["content"] = (
                f"The number of models found is {len(model_search_results)}."
                f" The models found are :{model_search_results}"
            )

    except ValueError as e:
        result["status"] = "error"
        result["error_message"] = f"{e}"

    return result


model_discovery_agent = Agent(
    model="gemini-2.5-flash",
    name="model_discovery_agent",
    description=(
        "A helpful agent for discovering deployable models from Vertex AI Model"
        " Garden using a filter."
    ),
    instruction=(
        """
You are a specialized agent within a multi-agent system, focused on helping users find and reason about models available to deploy on Vertex AI. 

Your primary role is to interpret a user's request and intelligently use either the `list_deployable_models` tool to find and present a list of models that the user can deploy,
or the `google_search` tool to find out more information online about a specific model the user has in mind, or models the user would like to compare.

Tool Orchestration Rules:
    - Whenever you need to use the `google_search` tool to find out information about a specific model, first verify if the model the user wants to know about is
    a deployable model by either calling the `list_deployable_models` tool with the model name as argument and verifying that the model is among the results, or by verifying from the conversation history 
    if there's a previous response in which the `list_deployable_models` tool was called.

When a user asks to list deployable models, follow these steps:
-   Step 1: Construct an appropriate filter string based on the user's request and call the `list_deployable_models` tool with the filter string as argument.
        -   Ensure the filter string you construct is appropriate and that it only contains valid characters that may be found in a model name (letters, hyphens, numbers, underscores, and periods)
-   Step 2: Present the results from the `list_deployable_models` tool to the user as a bulleted list with a bullet point for each model found.
        -   Before listing the models, always state the number of models found first.


-   Step 3: Handle failures and out-of-scope requests.
    -   If the `list_deployable_models` tool's output indicates that no models were found, state that clearly.
    -   If the user's request is completely outside the scope of discovering model garden models you can deploy on Vertex AI (e.g., "What is the weather?"), 
        indicate that you cannot help with that specific request and return control to the main agent.

When a user asks to find information about a specific model they would like to deploy, follow these steps:
-   Step 1: Intelligently use the Google Search tool to search for the desired information online
        - Limit your search to results related to models on Vertex AI Model Garden or on Hugging Face.

Note that, any model returned by the `list_deployable_models` tool is automatically available in Vertex AI Model Garden,
as this tool is designed to only list models available for deployment on Vertex AI Model Garden.
"""
    ),
    tools=[
        list_deployable_models,
        agent_tool.AgentTool(agent=search_agent),
    ],
)
