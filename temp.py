import os
import base64
from datetime import datetime
from PIL import ImageGrab
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from whisper_mic import WhisperMic


class ScreenAssistant:
    def __init__(self):
        # Initialize Whisper Mic
        self.mic = WhisperMic()

        # Initialize Ollama LLM
        self.llm = ChatOllama(model="llama3.2-vision", temperature=0.7)

        # Flag to control the main loop
        self.is_running = True

    def capture_screenshot(self):
        """Capture screenshot and convert to Base64"""
        # Capture screenshot
        screenshot = ImageGrab.grab()

        # Convert to base64 in memory
        from io import BytesIO
        buffered = BytesIO()
        screenshot.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_str

    def process_question(self, question):
        """Process question with Llama Vision"""
        try:
            # Capture screenshot and convert to base64
            image_base64 = self.capture_screenshot()

            # Prepare content for the model
            content_parts = [
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_base64}"
                },
                {
                    "type": "text",
                    "text": f"Please answer this question about the screenshot: {question}"
                }
            ]

            # Create message and get response
            message = HumanMessage(content=content_parts)
            response = self.llm.invoke([message])

            # Print the response (no audio generation)
            print(f"\nAssistant: {response.content}")

            return response.content

        except Exception as e:
            print(f"Error processing question: {e}")
            return "Sorry, I encountered an error processing your question."

    def run(self):
        """Main loop"""
        try:
            print("Listening for questions... (Press Ctrl+C to stop)")

            while self.is_running:
                # Listen for speech using WhisperMic
                print("\nListening...")
                transcribed_text = self.mic.listen()

                if transcribed_text:
                    print(f"You said: {transcribed_text}")

                    # Check if text contains a question
                    if any(q in transcribed_text.lower() for q in
                           ["what", "how", "why", "when", "where", "which", "who"]):
                        print("Detected question! Processing...")

                        # Process with LLM
                        response = self.process_question(transcribed_text)
                        print(f"\nAssistant: {response}")
                    else:
                        print("No question detected. Continue speaking...")

        except KeyboardInterrupt:
            print("\nStopping assistant...")
        finally:
            self.is_running = False


if __name__ == "__main__":
    assistant = ScreenAssistant()
    assistant.run()