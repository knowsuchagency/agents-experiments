import asyncio
import os

from agents import Agent, Runner, set_default_openai_key
from agents.extensions.models.litellm_model import LitellmModel
from openai.types.responses import ResponseTextDeltaEvent


async def main(
    model="openrouter/mistralai/devstral-small",
    instructions="You only respond in haikus.",
):
    model = LitellmModel(model=model, api_key=os.environ["OPENROUTER_API_KEY"])

    agent = Agent(
        name="Assistant",
        instructions=instructions,
        model=model,
    )

    result = Runner.run_streamed(agent, "What sound does a cat make?")

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            print(event.data.delta, end="", flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(main())
