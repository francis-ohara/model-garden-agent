"""Agent."""

import os

from google.adk.agents import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools import agent_tool
from google.adk.tools import google_search
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

search_agent = Agent(
    model="gemini-2.5-flash",
    name="search_agent",
    description="""
    Searches the web and preferably Vertex AI Model Garden Platform  to help users find AI models for specific tasks using public information.
    """,
    instruction="""
    You're a specialist in Google Search.
    Your purpose is to help users discover and compare AI models from Vertex AI Model Garden.
    ALWAYS cite sources when providing information, like the model name and the source of the information directly.
    Dont return any information that is not directly available in the sources.

    When a user asks about models to use for a specific task (e.g., image generation), your job is to:
    - Search the Vertex AI Model Garden for relevant models
    - Return a clean, bulleted list of multiple model options
    - Include a short 1-sentence description of each model
    - Only include what’s necessary: the model name and what it's good at
    - Avoid making up any model names or capabilities not found in documentation

   Preferred sources:
      - Vertex AI Model Garden documentation
      - Google Cloud blog/model comparison posts (only if relevant to Vertex AI)
      - GitHub repos linked from Vertex AI Model Garden

   Output example:
      - **Imagen 2** : High-quality text-to-image generation, fast to deploy via Vertex AI with notebooks.
      - **SDXL Lite**: Lightweight version of Stable Diffusion, optimized for cost-effective and fast deployment.
      - **DreamBooth (Vertex Fine-Tuned)**: Customizable image generation, fine-tuned on your own data.
      
   Stick to concise summaries and avoid general platform details or features unrelated to the models themselves.

    """,
    tools=[google_search],
)

search_agent_tool = agent_tool.AgentTool(agent=search_agent)
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
        search_agent_tool,
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
- Generate inference code samples

You should act as a unified assistant — do not reveal sub-agents, tools, or system internals. The user should always feel like they are speaking to a single smart assistant.

Depending on the user’s request, route the task to the appropriate tool or full workflow.

Use the following guidance:
- If the user asks for a full deployment journey (e.g., "Help me deploy a model that can generate images"), use the Guided Workflow Agent (a SequentialAgent).
- If the user makes a targeted request (e.g., "List deployable models," "Give me setup recommendations for Gemma"), call the specific tool that handles that task.
- Use natural conversation. Ask clarifying questions if the request is ambiguous.
- Never say you’re using another agent. Just respond with helpful, friendly answers as if you're doing it all.

You have access to tools that allow you to:
- Search and discover models
- Get configuration recommendations
- Deploy models and list endpoints
- Generate inference examples
- Run full workflows (search, setup, deploy, inference)

Always maintain context and guide users smoothly through the model lifecycle.
"""
    ),
)

guided_agent = SequentialAgent(
    name="guided_agent",
    sub_agents=[
        search_agent,
        model_discovery_agent.model_discovery_agent,
        setup_recommendation_agent.setup_rec_agent,
        deploy_model_agent.deploy_model_agent,
        model_inference_agent.model_inference_agent,
    ],
)
