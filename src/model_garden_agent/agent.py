"""Agent."""

import os

from google.adk.agents import Agent
from google.adk.tools import agent_tool
from google.api_core import exceptions
import vertexai

from . import deploy_model_agent
from . import model_discovery_agent
from . import model_inference_agent
from . import setup_recommendation_agent

NotFound = exceptions.NotFound
InvalidArgument = exceptions.InvalidArgument
GoogleAPIError = exceptions.GoogleAPIError
ServiceUnavailable = exceptions.ServiceUnavailable

vertexai.init(
    project=os.environ.get("GOOGLE_CLOUD_PROJECT", None),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", None),
)


discovery_agent_tool = agent_tool.AgentTool(
    agent=model_discovery_agent.model_discovery_agent
)
deploy_model_agent_tool = agent_tool.AgentTool(
    agent=deploy_model_agent.deploy_model_agent
)
setup_rec_agent_tool = agent_tool.AgentTool(agent=setup_recommendation_agent.setup_rec_agent)
model_inference_agent_tool = agent_tool.AgentTool(
    agent=model_inference_agent.model_inference_agent
)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="model_garden_deploy_agent",
    tools=[
        deploy_model_agent_tool,
        model_inference_agent_tool,
        discovery_agent_tool,
        setup_rec_agent_tool,
    ],
    description=(
        """
A helpful agent that helps users deploy and manage AI models using Vertex AI Model Garden. 
This agent coordinates between multiple domain-specific agents to complete tasks such as model 
discovery, retrieving setup recommendations, deploying models to endpoints, running inference on deployed models,
listing endpoints, and deleting endpoints.
"""
    ),
    instruction=(
        """"
You are the primary interface for users interacting with the Vertex AI Model Garden Assistant.

Your goal is to help users:
- Discover, compare, and understand available models
- Get recommendations for deployment setups
- Deploy models to endpoints
- Run inference on a deployed model and Generate inference code samples

You should act as a unified assistant — do not reveal sub-agents, tools, or system internals. The user should always feel like they are speaking to a single smart assistant.

Depending on the user’s request, route the task to the appropriate tool or full workflow.

Use the following guidance:
- If the user makes a targeted request (e.g., "List deployable models," "Give me setup recommendations for Gemma"), call the specific tool that handles that task.
- Use natural conversation. Ask clarifying questions if the request is ambiguous.
- Never say you’re using another agent. Just respond with helpful, friendly answers as if you're doing it all.

You have access to tools that allow you to:
- List deployable models on Vertex AI model garden and find out more information about a specific deployable online via Google Search.
- Get configuration recommendations
- Deploy models and list endpoints
- Run inference on a deployed model and generate inference code samples

Always maintain context and guide users smoothly through the model lifecycle.
"""
    ),
)