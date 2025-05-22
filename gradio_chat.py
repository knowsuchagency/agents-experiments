import asyncio
import os
import gradio as gr
from typing import List, Dict, Any, AsyncGenerator, Tuple

from agents import Agent, Runner, set_default_openai_key
from agents.extensions.models.litellm_model import LitellmModel
from openai.types.responses import ResponseTextDeltaEvent


class ChatInterface:
    def __init__(self):
        self.conversation_history = []
        self.current_agent = None

    def create_agent(self, model_name: str, instructions: str) -> Agent:
        """Create a new agent with specified model and instructions"""
        model = LitellmModel(
            model=model_name, api_key=os.getenv("API_KEY", "")
        )

        agent = Agent(
            name="Assistant",
            instructions=instructions,
            model=model,
        )
        return agent

    async def stream_response(
        self, message: str, model_name: str, instructions: str
    ) -> AsyncGenerator[str, None]:
        """Stream response from agent with conversation memory"""
        # Create agent if needed or if settings changed
        self.current_agent = self.create_agent(model_name, instructions)

        # Prepare input with conversation history
        if self.conversation_history:
            # Use conversation memory pattern from docs
            input_data = self.conversation_history + [
                {"role": "user", "content": message}
            ]
        else:
            input_data = message

        # Stream the response
        result = Runner.run_streamed(self.current_agent, input_data)

        response_text = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                chunk = event.data.delta
                response_text += chunk
                yield response_text

        # Update conversation history after streaming is complete
        self.conversation_history = result.to_input_list()

    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []


# Global chat interface instance
chat_interface = ChatInterface()


async def respond(
    message: str,
    history: List[Dict[str, str]],
    model_name: str,
    instructions: str,
):
    """Handle chat response with streaming"""

    if not model_name.startswith("openrouter/"):
        model_name = f"openrouter/{model_name}"

    if not message.strip():
        yield history, ""
        return

    # Add user message to history
    history.append({"role": "user", "content": message})

    # Stream response
    history.append({"role": "assistant", "content": ""})
    async for partial_response in chat_interface.stream_response(
        message, model_name, instructions
    ):
        # Update the last assistant message in history
        history[-1]["content"] = partial_response
        yield history, ""


def clear_chat():
    """Clear chat history"""
    chat_interface.clear_conversation()
    return [], ""


def create_interface():
    """Create the Gradio interface"""
    with gr.Blocks(title="AI Chat Assistant", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 AI Chat Assistant")
        gr.Markdown(
            "Customize your AI assistant with different models and instructions, with full conversation memory."
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_input = gr.Textbox(
                    label="Model",
                    value="openrouter/mistralai/devstral-small",
                    placeholder="e.g., openrouter/mistralai/devstral-small",
                    info="LiteLLM-compatible model identifier",
                )

                instructions_input = gr.Textbox(
                    label="Instructions",
                    value="You are a helpful assistant. Reply concisely and clearly.",
                    placeholder="Customize the assistant's behavior...",
                    lines=3,
                    info="System instructions for the assistant",
                )

                clear_btn = gr.Button("🗑️ Clear Conversation", variant="secondary")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_copy_button=True,
                    type="messages",
                )

                msg = gr.Textbox(
                    label="Message",
                    placeholder="Type your message here...",
                    container=False,
                    scale=7,
                )

                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary", scale=1)

        # Event handlers
        msg.submit(
            respond,
            inputs=[msg, chatbot, model_input, instructions_input],
            outputs=[chatbot, msg],
        )

        send_btn.click(
            respond,
            inputs=[msg, chatbot, model_input, instructions_input],
            outputs=[chatbot, msg],
        )

        clear_btn.click(clear_chat, outputs=[chatbot, msg])

        # Example prompts
        gr.Examples(
            examples=[
                ["What sound does a cat make?"],
                ["Explain quantum computing in simple terms"],
                ["Write a haiku about programming"],
                ["What's the weather like on Mars?"],
            ],
            inputs=msg,
        )

    return interface


def main():
    """Launch the Gradio interface"""
    interface = create_interface()
    interface.launch(
        server_name="127.0.0.1", server_port=7860, share=False, show_error=True
    )


if __name__ == "__main__":
    main()
