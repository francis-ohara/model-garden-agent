import os
import subprocess

from google.adk.agents import Agent
from google.api_core import exceptions
import requests
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


def make_model_garden_search_request(query: str) -> bool:
    """Makes an authenticated GET request to the Model Garden search API, and handles all outputting of results and errors.

    Args:
        query: The search query string.

    Returns:
        True if the request and output were successful, False otherwise.
    """
    # Get the Google Cloud access token
    try:
        process = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            check=True,
        )
        access_token = process.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting access token: {e}")
        return None

    # Define the request parameters
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", None)
    publisher = "google"
    endpoint = "us-central1-aiplatform.googleapis.com"

    # Construct the full URL with query parameters
    url = f"https://{endpoint}/ui/publishers/{publisher}:search"
    params = {"query": query}

    print(f"Making request to: {url} with query: '{query}'")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Goog-User-Project": project_id,
    }

    # Make the GET request
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        search_results = response.json()
        raw_data = search_results.get("publisherModels", [])

        # formatted list for direct output
        formatted_output = ""
        if raw_data:
            formatted_lines = [
                f"I found {len(raw_data)} model(s) matching your request:"
            ]
            for model in raw_data:
                display_name = model.get("displayName", "N/A")
                overview = model.get("overview", "No description available.")
                formatted_lines.append(f"\n- **Model Name:** {display_name}")
                formatted_lines.append(f"  **Description:** {overview}")
            formatted_output = "\n".join(formatted_lines)
        else:
            formatted_output = (
                f"I could not find any models matching the query: '{query}'."
            )

        return {
            "raw_data": raw_data,
            "formatted_list": formatted_output,
        }

    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return {
            "raw_data": [],
            "formatted_list": f"An API request error occurred: {e}",
        }
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {
            "raw_data": [],
            "formatted_list": f"An unexpected error occurred: {e}",
        }


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
You are a specialized agent within a multi-agent system, focused on helping users find and reason about models in the Vertex AI Model Garden catalog. 
You should not perform any web searches or answer general knowledge questions. Your knowledge is strictly limited to the Model Garden catalog.

Your primary role is to interpret a user's request and intelligently use the `make_model_garden_search_request` tool to find and present model information.

When a user asks to find a model, follow these steps:

-   Step 1: Use the `make_model_garden_search_request` tool to retrieve data.
    -   Call the `make_model_garden_search_request` tool with the user's query as the `query` argument.

-   Step 2: Determine how to present the output based on user intent.
    -   If the user's query was a Direct Search** (e.g., "list models with keyword `gemma`" or a specific model name), 
        take the `formatted_list` string from the tool's output and present it directly to the user. Do not add any extra analysis.
    -   If the user's query was a Reasoning Search** (e.g., "best lightweight model for text generation"), ignore the `formatted_list`. 
        Instead, analyze the `raw_data` list to make a recommendation.
        -   Filter and rank the models based on the user's criteria (e.g., "lightweight" implies low resource requirements, 
            "best" implies a high trending score or popular downloads).
        -   Construct a final conversational response that recommends the model(s) and explains the reasoning behind the choice.

-   Step 3: Handle failures and out-of-scope requests.
    -   If the `make_model_garden_search_request` tool's output indicates that no models were found, state that clearly.
    -   If the user's request is completely outside the scope of Model Garden (e.g., "What is the weather?"), 
        indicate that you cannot help with that specific request and return control to the main agent.
"""
    ),
    tools=[
        list_deployable_models,
        make_model_garden_search_request,
    ],
)
