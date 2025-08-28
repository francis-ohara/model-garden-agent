import os
from typing import Any
from google import genai
from google.adk.agents import Agent
from google.api_core.exceptions import GoogleAPIError
from google.api_core.exceptions import NotFound
from google.api_core.exceptions import ServiceUnavailable
import vertexai
from vertexai import model_garden

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", None)
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", None)

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
)


def run_inference(endpoint_id: str, prompt: str) -> dict[str, Any]:
    """Runs inference on a deployed model given the model name and a text prompt

    and returns a dict containing the model's response as a string if successful.

    Args:
      endpoint_id (str): A string that is the endpoint ID of the Vertex AI
        endpoint to which the model was deployed. It typically follows the format:
        mg-[0-9]{10} (e.g. mg-endpoint-1234567890). It will be concatenated to
        construct the full endpoint resource name which follows the format:
        projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{endpoint_id}.
      prompt (str): A string prompt to be used to run inference on the model.

    Returns:
      dict: status and content containing the model's response or an error
      message if unsuccessful.
    """
    try:
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
        )
        response = client.models.generate_content(
            model=f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{endpoint_id}",
            contents=prompt,
        ).text
        return {"status": "success", "content": response}

    except NotFound as e:
        return {
            "status": "error",
            "error_message": (
                "This error is likely due to an invalid endpoint ID being used to"
                " run inference. Please ensure the endpoint ID provided is valid."
                f" Details: {e}"
            ),
        }

    except ServiceUnavailable as e:
        return {
            "status": "error",
            "error_message": (
                "The Vertex AI Service or the specific endpoint on which inference"
                " is being run is temporarily unavailable. Please try again."
                f" Details: {e}"
            ),
        }

    except GoogleAPIError as e:
        return {
            "status": "error",
            "error_message": (
                f"A Google API Error occurred while running inference. Details: {e}"
            ),
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": (
                "An unexpected error occurred during model deployment." f" Details: {e}"
            ),
        }


def inference_request_guide(model_name: str, endpoint_id: str):
    """Returns detailed information on how to run inference for a specific deployed model

    given the model name and endpoint ID of the model.
    It specifically shows code snippets on how to run inference on a deployed
    model through:
      1. The Vertex AI SDK
      2. The ChatCompletion API of the OpenAI SDK
      3. The GenAI SDK

    Args:
      model_name (str): Model Garden model resource name in the format of
        publishers/{publisher}/models/{model}@{version}, or a simplified resource
        name in the format of {publisher}/{model}@{version}, or a Hugging Face
        model ID in the format of {organization}/{model}.
      endpoint_id (str): A string denoting the endpoint ID of the Vertex AI
        endpoint to which the model was deployed. It typically follows the format:
        mg-[0-9]{10,} (e.g. mg-endpoint-1234567890).

    Returns:
      dict: status and content or error message.
            If successful, the content will be a string with detailed instructions
            on how the user can run inference on the deployed model
    """
    response = f"""This is how you can run inference on the model {model_name} deployed
        to the endpoint {endpoint_id}:

        """

    try:
        sample_request = (
            model_garden.OpenModel(model_name)
            .list_deploy_options()[0]
            .deploy_metadata.sample_request
        )

        response += f"""The sample request for the model is as follows:

```{sample_request}```


Based on this sample request, you can run inference on the model using:
(1) The Vertex AI Python SDK:
  The following code snippet demonstrates how to do this:
  
```
      from google.cloud import aiplatform

      endpoint_name = "projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{endpoint_id}"
      endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)
      prediction = endpoint.predict(\n{sample_request[1:-2]}\n)
      print(prediction.predictions[0])
```


(2) The ChatCompletion API of the OpenAI SDK:
  The following code snippet demonstrates this:
```
      import openai
      import google.auth

      creds, project = google.auth.default()
      auth_req = google.auth.transport.requests.Request()
      creds.refresh(auth_req)

      endpoint_url = f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{endpoint_id}"

      client = openai.OpenAI(base_url=endpoint_url, api_key=creds.token)


      # TODO: replace with prompt you would like to use to run inference.
      prompt = "Tell me a joke"  

      prediction = client.chat.completions.create(
          model="",
          messages=[{{"role": "user", "content": prompt}}],  
      )
      print(prediction.choices[0].message.content)
``` 


(3) The GenAI Python SDK
  The code snippet below also demonstrates how to run inference using the GenAI Python SDK:
```
      from google import genai

      client = genai.Client(
          vertexai=True,
          project={PROJECT_ID},
          location={LOCATION},
      )

      # TODO: replace with prompt you would like to use to run inference.
      prompt = "Tell me a joke"

      response = client.models.generate_content(
          model=f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{endpoint_id}",
          contents=prompt,
      ).text
      print(response)
```
    """
        return {"status": "success", "content": response}

    except ValueError as e:
        return {
            "status": "error",
            "error_message": (
                "This error is likely due to an invalid model_name. "
                "Please ensure the model name provided is valid."
                f" Details: {e}"
            ),
        }

    except GoogleAPIError as e:
        return {
            "status": "error",
            "error_message": (
                f"A Google API Error occurred while running inference. Details: {e}"
            ),
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": (
                "An unexpected error occurred during model deployment." f" Details: {e}"
            ),
        }


model_inference_agent = Agent(
    model="gemini-2.5-flash",
    name="model_inference_agent",
    description=(
        """A helpful agent for assisting the user to run inference on a deployed model."""
    ),
    instruction=(
        """
                You are a sub-agent in a multi-agent system that helps users deploy and manage AI models using Vertex AI Model Garden.
User requests are routed to this agent when they mention running inference on a deployed model. 
Do not refer to yourself as a sub-agent or mention transfers and only respond to requests that fall within the scope of this agent. 
If the user asks for something outside of this agent's scope, return control to the main agent.
Your purpose is to run inference on a deployed model and to guide the user on how they can run inference on a model they have deployed.

You are currently capable of:
  - Running inference directly on a deployed model given the model's endpoint ID and the string prompt to be used to run inference.
  - Giving detailed instructions to the user on how they can run inference requests through one of the following methods:
    VertexAI Python SDK, OpenAI SDK, and GenAI Python SDK.

RULES
  1. In your interactions with the user, start by clarifying if the user would like you to run inference directly on the deployed model or if they would
     instead like you to guide them on how they can run inference using the Vertex AI SDK, OpenAI SDK, or GenAI SDK.
  2. If the user provides an endoint ID that is a full endpoint resource name following the format: 
     projects/[PROJECT_ID]/locations/[LOCATION]/endpoints/[endpoint_id], extract `endpoint_id` specifically 
     from the resource name and use that as the ID when calling the `run_inference` tool or the `inference_request_guide` tool.
  3. If after asking the user to provide a prompt for running inference, you are unsure if their response is a direct
     question to you or a prompt for running inference, ask the user for clarification before running inference or answering their question.
     For example, you can ask "Is the above the prompt you would like to use to run inference?"
  4. When guiding the user on how to run inference request, be sure to extract the appropriate model name and endpoint ID from your previous conversations with the user before calling the `inference_request_guide` tool.
  5. When guiding the user on how to run inference request, format all content nested within backticks ``` as a code block. 
     Do not include any backticks ``` literally in your output.
  6. When guiding the user on how to run inference request, do not format pound signs # as headings. Use them as literal pound signs in your output.  
"""
    ),
    tools=[run_inference, inference_request_guide],
)
