"""Interactive command-line interface for the Weather Agent."""

from rich.console import Console
from rich.markdown import Markdown

from agent import weather_assistant as agent


def interactive_agent():
    """Run an interactive command-line interface for the Weather Agent."""
    print("\n📁 Weather Agent\n")
    print("Ask a question about the weather forecast or alerts.\n\n")

    print("You can try following queries:")
    print("- What's the weather in Orlando, Florida this week?")
    print("- Any weather alerts for Las Vegas today?")
    print("Type 'exit' to quit.")

    # Interactive loop
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() == "/quit":
                print("\nGoodbye! 👋")
                break

            response = agent(user_input)

            print("\n\n=== RENDERED MARKDOWN ===\n")
            console = Console()
            console.print(Markdown(response))
            print("\n=== END OF MARKDOWN ===\n")

        except KeyboardInterrupt:
            print("\n\nExecution interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try asking a different question.")


if __name__ == "__main__":
    interactive_agent()
