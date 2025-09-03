from datetime import datetime
import os
from typing import Any, Optional
from google.adk.agents import Agent
from google.api_core import exceptions
from google.cloud import aiplatform
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


def deploy_model_to_endpoint(
    model_id: str,
    endpoint_display_name: Optional[str] = None,
    model_display_name: Optional[str] = None,
    option_index: Optional[int] = None,
) -> dict[str, Any]:
    """Deploys a Vertex AI Model Garden model to an endpoint.

    Args:
        model_id: The ID of the model in Model Garden (e.g.,
          "google/gemma@gemma-2b").
        endpoint_display_name: The display name for the new endpoint.
        model_display_name: The display name for the deployed model.
        option_index: The index of the deployment option to use. If not provided,
          the default deployment option will be used.

    Returns:
        dict: status and content or error message.
    """
    print(f"[DEBUG] Option index: {option_index}")

    project_id = os.environ["GOOGLE_CLOUD_PROJECT"].lower()
    location = os.environ["GOOGLE_CLOUD_LOCATION"].lower()
    model_id = model_id.lower()
    if endpoint_display_name:
        endpoint_display_name = endpoint_display_name.lower()
    if model_display_name:
        model_display_name = model_display_name.lower()

    aiplatform.init(project=project_id, location=location)

    try:
        model = model_garden.OpenModel(model_id)
        if option_index is not None:
            deploy_options = model.list_deploy_options()
            if option_index >= len(deploy_options):
                return {
                    "status": "error",
                    "error_message": (
                        f"Invalid option index {option_index} for model '{model_id}'."
                    ),
                }
            selected_option = deploy_options[option_index]

            print(f"[DEBUG] Selected option: {selected_option}")
            machine_type = selected_option.dedicated_resources.machine_spec.machine_type
            accelerator_type = (
                selected_option.dedicated_resources.machine_spec.accelerator_type
            )
            accelerator_count = (
                selected_option.dedicated_resources.machine_spec.accelerator_count
            )

            print(f"[DEBUG] Machine type: {machine_type}")
            print(f"[DEBUG] Accelerator type: {accelerator_type}")
            print(f"[DEBUG] Accelerator count: {accelerator_count}")
            endpoint = model.deploy(
                endpoint_display_name=endpoint_display_name,
                model_display_name=model_display_name,
                machine_type=machine_type,
                accelerator_type=accelerator_type,
                accelerator_count=accelerator_count,
            )
        else:
            endpoint = model.deploy(
                endpoint_display_name=endpoint_display_name,
                model_display_name=model_display_name,
            )
        return {
            "status": "success",
            "Deployed model to endpoint: ": endpoint.resource_name,
        }
    except InvalidArgument as e:
        # If the model_id format is incorrect or the model doesn't exist.
        return {
            "status": "error",
            "error_message": (
                f"Invalid model ID or deployment parameters: {e}. Please check the"
                " model ID and try again."
            ),
        }
    except NotFound as e:
        # If the model_id cannot be found.
        return {
            "status": "error",
            "error_message": (
                f"Model '{model_id}' not found in Model Garden. Please verify the"
                f" model ID and try again. Details: {e}"
            ),
        }
    except ServiceUnavailable as e:  # Specific catch for 503 errors
        return {
            "status": "error",
            "error_message": (
                "Deployment failed due to service unavailability (503 error) for"
                f" model '{model_id}'. This often means the requested resources"
                " (based on the model's default/recommended configuration) are"
                f" temporarily overloaded or unavailable in the '{location}'"
                " region. Please try deploying again, or consider exploring"
                " different deployment configurations or regions using the"
                " 'get_recommended_deployment_config' tool if the issue persists."
                f" Details: {e}"
            ),
        }
    except GoogleAPIError as e:
        # Catch broader API errors, e.g., permission issues, quota limits, etc.
        return {
            "status": "error",
            "error_message": (
                f"Google Cloud API error during deployment: {e}. Please check your"
                " project's permissions and quota."
            ),
        }
    except Exception as e:
        # Catch any other unexpected errors during deployment
        return {
            "status": "error",
            "error_message": (
                f"An unexpected error occurred during model deployment: {e}"
            ),
        }


def list_endpoints() -> dict:
    """Lists all Vertex AI Model Garden Endpoints in the current project and location.

    Returns:
        dict: A dictionary containing status and a list of endpoint details,
              or an error message.
    """
    project_id = os.environ["GOOGLE_CLOUD_PROJECT"].lower()
    location = os.environ["GOOGLE_CLOUD_LOCATION"].lower()

    aiplatform.init(project=project_id, location=location)

    try:
        filter_str = "labels.mg-deploy:* OR labels.mg-one-click-deploy:*"
        endpoints = aiplatform.Endpoint.list(filter=filter_str, location=location)

        if not endpoints:
            return {
                "status": "success",
                "content": (
                    "No Model Garden endpoints found in this project and location."
                ),
            }

        endpoint_list = []
        print(f"[DEBUG] endpoints: {endpoints}")
        for ep in endpoints:
            raw_time = ep.create_time.isoformat()
            dt = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
            formatted_time = dt.strftime("%B %d, %Y at %I:%M %p %Z")

            # Determine deployment status
            if ep.traffic_split:
                status = "Active"
            else:
                status = "Inactive"

            endpoint_list.append(
                f"- ID: {ep.name.split('/')[-1]}\n"
                f"  Display Name: {ep.display_name}\n"
                f"  Status: {status}\n"
                f"  Created: {formatted_time}"
            )

        formatted_output = "Here are your Model Garden endpoints:\n\n" + "\n\n".join(
            endpoint_list
        )

        return {
            "status": "success",
            "content": formatted_output,
        }
    except GoogleAPIError as e:
        return {
            "status": "error",
            "error_message": (
                f"Google Cloud API error while listing endpoints: {e}. Please check"
                " your project's permissions and network connectivity."
            ),
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": (
                f"An unexpected error occurred while listing endpoints: {e}"
            ),
        }


def delete_endpoint(endpoint_id: str) -> str:
    """Deletes a Vertex AI endpoint by ID.

    Args:
        endpoint_id: The ID of the endpoint to delete.

    Returns:
        A confirmation string if successful.
    """
    project_id = os.environ["GOOGLE_CLOUD_PROJECT"].lower()
    location = os.environ["GOOGLE_CLOUD_LOCATION"].lower()
    endpoint_id = endpoint_id.lower()

    aiplatform.init(project=project_id, location=location)
    try:
        endpoint = aiplatform.Endpoint(
            endpoint_name=(
                f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
            )
        )
        endpoint.delete(force=True)
        return {"status": "success", "content": f"Deleted endpoint: {endpoint_id}"}
    except NotFound as e:
        # This exception is raised if the endpoint with the given ID doesn't exist.
        return {
            "status": "error",
            "error_message": (
                f"Endpoint with ID '{endpoint_id}' not found. Please verify the"
                f" endpoint ID and try again. Details: {e}"
            ),
        }
    except InvalidArgument as e:
        # This could happen if the endpoint_id format is malformed
        return {
            "status": "error",
            "error_message": (
                f"Invalid endpoint ID format: {e}. Please provide a valid endpoint"
                " ID."
            ),
        }
    except GoogleAPIError as e:
        # Catch broader API errors for deletion
        return {
            "status": "error",
            "error_message": (
                f"Google Cloud API error during endpoint deletion: {e}. Please"
                " check your project's permissions."
            ),
        }
    except Exception as e:
        # Catch any other unexpected errors during deletion
        return {
            "status": "error",
            "error_message": (
                f"An unexpected error occurred during endpoint deletion: {e}"
            ),
        }


deploy_model_agent = Agent(
    model="gemini-2.5-flash",
    name="deploy_model_agent",
    description=(
        "A helpful agent for deploying AI models with Vertex Model Garden and"
        " deletes them when no longer needed."
    ),
    instruction=(
        """
You are a sub-agent in a multi-agent system that helps users deploy and manage AI models using Vertex AI Model Garden.
User requests are routed to this agent when they mention deploying or deleting endpoints. 
Do not refer to yourself as a sub-agent or mention transfers.
Only respond to requests that fall within the scope of this agent. 
If the user asks for something outside of this agent's scope, return control to the main agent.
Your purpose is to deploy AI models on Vertex Model Garden to Vertex AI endpoints, 
using either a default or a recommended configuration.

You are capable of the following functions:
- Deploying selected models using a default or recommended configuration.
- Listing all endpoints in the current project and location.
- Deleting deployed endpoints when the user is done with them.

When deploying:
- If the user selects a specific option (e.g., "option 1"), use that exact configuration 
  from the recommendations.
- DO NOT fall back to the default deployment if a config is specified but fails, 
  unless the user explicitly asks for it.
- Assume the default endpoint and model display name is sufficient.

After deploying:
- Inform the user that you can help them run inference on the model they just deployed.

When listing Model Garden endpoints:
-If there are no endpoints, return a friendly message to the user informing them 
  that they have no model garden endpoints in this project and location.
- If there are model garden endpoints, return a list of endpoints with their ID, display name, 
  and create time.

Before deleting an endpoint:
- Always ask the user to confirm the endpoint ID and their intent to delete.
- Do not call the deletion tool without explicit confirmation.
"""
    ),
    tools=[
        deploy_model_to_endpoint,
        delete_endpoint,
        list_endpoints,
    ],
)
