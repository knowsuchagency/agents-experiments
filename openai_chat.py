import asyncio
import gradio as gr
from typing import List, Dict, AsyncGenerator
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent


class ChatInterface:
    def __init__(self):
        self.current_agent = None
        self.last_response_id = None

    def create_agent(self, model_name: str, instructions: str) -> Agent:
        """Create a new agent with specified model and instructions"""
        agent = Agent(
            name="Assistant",
            instructions=instructions,
            model=model_name,
        )
        return agent

    async def stream_response(
        self, message: str, model_name: str, instructions: str
    ) -> AsyncGenerator[str, None]:
        """Stream response from agent with conversation memory"""
        # Create agent if needed or if settings changed
        self.current_agent = self.create_agent(model_name, instructions)

        # Stream the response using the original previous_response_id approach
        result = Runner.run_streamed(
            self.current_agent, message, previous_response_id=self.last_response_id
        )

        response_text = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                chunk = event.data.delta
                response_text += chunk
                yield response_text

        # Store the response ID for conversation continuity
        self.last_response_id = result.last_response_id

    def clear_conversation(self):
        """Reset the conversation by clearing the response ID"""
        self.last_response_id = None


# Global chat interface instance
chat_interface = ChatInterface()


async def respond(
    message: str,
    history: List[Dict[str, str]],
    model_name: str,
    instructions: str,
):
    """Handle chat response with streaming"""
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
    """Create and configure the Gradio interface"""
    with gr.Blocks(title="OpenAI Chat Assistant", theme=gr.themes.Origin()) as demo:
        gr.Markdown("# 🤖 OpenAI Chat Assistant")
        gr.Markdown(
            "Chat with an AI assistant powered by OpenAI's API with streaming and conversation continuity."
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_input = gr.Textbox(
                    label="Model",
                    value="gpt-4o-mini",
                    placeholder="e.g., gpt-4o, gpt-4o-mini, gpt-3.5-turbo",
                    info="OpenAI model to use for the conversation",
                )

                instructions_input = gr.Textbox(
                    label="System Instructions",
                    value="You are a helpful assistant. Be concise but friendly.",
                    placeholder="Customize the assistant's behavior...",
                    lines=3,
                    info="System instructions for the assistant",
                )

                clear_btn = gr.Button("🗑️ Clear Conversation", variant="secondary")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    placeholder="Start a conversation...",
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

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
