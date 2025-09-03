import os
from typing import Any
from google.adk.agents import Agent
from google.api_core import exceptions
from google.cloud import aiplatform
import vertexai
from vertexai import model_garden

NotFound = exceptions.NotFound
InvalidArgument = exceptions.InvalidArgument
GoogleAPIError = exceptions.GoogleAPIError
ServiceUnavailable = exceptions.ServiceUnavailable

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", None)
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", None)


vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
)


def get_recommended_deployment_config(model_id: str) -> dict[str, Any]:
    """Fetches and formats the recommended deployment configurations for a Model Garden model.

    Args:
        model_id: The ID of the model in Model Garden (e.g.,
          "google/gemma@gemma-2b").

    Returns:
        dict: status and content or error message with deployment options listed
        and indexed.
    """
    project_id = PROJECT_ID
    location = LOCATION
    model_id = model_id.lower()

    aiplatform.init(project=project_id, location=location)

    try:
        model = model_garden.OpenModel(model_id)
        deploy_options = model.list_deploy_options()

        if not deploy_options:
            return {
                "status": "warning",
                "content": (
                    f"No specific deployment options found for model '{model_id}'."
                    " This might mean the model has default configurations or is not"
                    " directly deployable via this method."
                ),
            }

        formatted_options = []
        for i, option in enumerate(deploy_options):
            option_str = [f"**Option {i}:**"]
            if option.dedicated_resources:
                spec = option.dedicated_resources.machine_spec
                option_str.append(f"  - Machine Type: {spec.machine_type}")
                if spec.accelerator_type and spec.accelerator_count:
                    option_str.append(
                        f"  - Accelerator Type: {spec.accelerator_type.name}"
                    )
                    option_str.append(
                        f"  - Accelerator Count: {spec.accelerator_count}"
                    )
            if option.container_spec:
                option_str.append(
                    f"  - Container Image: {option.container_spec.image_uri}"
                )
            formatted_options.append("\n".join(option_str))

        return {
            "status": "success",
            "content": (
                f"Recommended deployment options for '{model_id}':\n\n"
                + "\n\n".join(formatted_options)
            ),
        }

    except NotFound as e:
        return {
            "status": "error",
            "error_message": (
                f"Model '{model_id}' not found in Model Garden. Cannot fetch"
                f" deployment recommendations. Details: {e}"
            ),
        }
    except InvalidArgument as e:
        return {
            "status": "error",
            "error_message": (
                f"Invalid model ID format: {e}. Please provide a valid model ID to"
                " get deployment recommendations."
            ),
        }
    except GoogleAPIError as e:
        return {
            "status": "error",
            "error_message": (
                "Google Cloud API error when fetching deployment recommendations:"
                f" {e}. Please check your project's permissions."
            ),
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": (
                "An unexpected error occurred while fetching deployment"
                f" recommendations: {e}"
            ),
        }


setup_rec_agent = Agent(
    model="gemini-2.5-flash",
    name="setup_rec_agent",
    description=(
        "A helpful agent for providing setup recommendations for deploying AI"
        " models."
    ),
    instruction=(
        """
You are a sub-agent in a multi-agent system that helps users deploy and manage AI models using Vertex AI Model Garden.
User requests are routed to this agent when they mention deploying or deleting endpoints. 
Do not refer to yourself as a sub-agent or mention transfers.
Only respond to requests that fall within the scope of this agent. 
If the user asks for something outside of this agent's scope, return control to the main agent.
Your purpose is to provide setup recommendations for deploying AI models. 

You are capable of the following:
- Listing all recommended deployment configurations for a given model ID.

When listing deployment options:
- Clearly show each one with a numbered index (e.g., "Option 0", "Option 1").
- Include relevant details like machine type and accelerator (if available).
-Once the user selects an option and wants to deploy or do something else, transfer control to the root agent.
"""
    ),
    tools=[
        get_recommended_deployment_config,
    ],
)
