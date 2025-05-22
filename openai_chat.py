import asyncio
import gradio as gr
from agents import Agent, Runner


class ChatInterface:
    def __init__(self):
        self.agent = Agent(
            name="Assistant",
            instructions="You are a helpful assistant. Be concise but friendly.",
        )
        self.last_response_id = None

    async def chat_async(self, message, history):
        """Async chat function that handles the conversation"""
        try:
            # Run the agent with the user's message
            result = await Runner.run(
                self.agent, message, previous_response_id=self.last_response_id
            )

            # Store the response ID for conversation continuity
            self.last_response_id = result.last_response_id

            # Return the response
            return result.final_output

        except Exception as e:
            return f"Error: {str(e)}"

    def chat(self, message, history):
        """Sync wrapper for the async chat function"""
        return asyncio.run(self.chat_async(message, history))

    def clear_conversation(self):
        """Reset the conversation by clearing the response ID"""
        self.last_response_id = None
        return []


def create_interface():
    """Create and configure the Gradio interface"""
    chat_interface = ChatInterface()

    with gr.Blocks(title="OpenAI Chat Assistant") as demo:
        gr.Markdown("# OpenAI Chat Assistant")
        gr.Markdown(
            "Chat with an AI assistant powered by OpenAI's API with conversation continuity."
        )

        chatbot = gr.Chatbot(
            height=500, placeholder="Start a conversation...", show_copy_button=True
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your message here...", container=False, scale=4
            )
            submit_btn = gr.Button("Send", scale=1, variant="primary")

        with gr.Row():
            clear_btn = gr.Button("Clear Conversation", variant="secondary")

        # Handle message submission
        def respond(message, history):
            if not message.strip():
                return history, ""

            # Get bot response
            bot_message = chat_interface.chat(message, history)

            # Add both user message and bot response to history
            history.append((message, bot_message))
            return history, ""

        # Handle clear conversation
        def clear_chat():
            chat_interface.clear_conversation()
            return []

        # Event handlers
        submit_btn.click(respond, inputs=[msg, chatbot], outputs=[chatbot, msg])

        msg.submit(respond, inputs=[msg, chatbot], outputs=[chatbot, msg])

        clear_btn.click(clear_chat, outputs=[chatbot])

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
