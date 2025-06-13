# Importing required libraries
from textblob import TextBlob  # For sentiment analysis
import colorama  # For colored terminal output
from colorama import Fore, Style
import sys
import time

# Initialize colorama
colorama.init(autoreset=True)

# Global variables
user_name = ""
conversation_history = []
positive_count = 0
negative_count = 0
neutral_count = 0

# Animation function
def show_processing_animation():
    print(f"{Fore.CYAN} Detecting sentiment clues", end="")
    for _ in range(3):
        time.sleep(0.5)
        print(",", end="")
        sys.stdout.flush()

# Sentiment analysis function
def analyze_sentiment(text):
    global positive_count, negative_count, neutral_count

    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        conversation_history.append(text)

        if sentiment > 0.75:
            positive_count += 1
            return f"\n{Fore.GREEN} Very Positive sentiment detected, Agent {user_name}! (Score: {sentiment:.2f})"
        elif 0.25 < sentiment:
            positive_count += 1
            return f"\n{Fore.GREEN} Positive sentiment detected, Agent {user_name}! (Score: {sentiment:.2f})"
        elif -0.25 <= sentiment <= 0.25:
            neutral_count += 1
            return f"\n{Fore.YELLOW} Neutral sentiment detected, Agent {user_name}! (Score: {sentiment:.2f})"
        elif -0.75 <= sentiment < -0.25:
            negative_count += 1
            return f"\n{Fore.RED} Negative sentiment detected, Agent {user_name}! (Score: {sentiment:.2f})"
        else:
            negative_count += 1
            return f"\n{Fore.RED} Very Negative sentiment detected, Agent {user_name}! (Score: {sentiment:.2f})"

    except Exception as e:
        return f"{Fore.RED}An error occurred during sentiment analysis: {str(e)}"

# Command handler function
def execute_command(command):
    global conversation_history, positive_count, negative_count, neutral_count

    if command == "summary":
        return (f"{Fore.CYAN} Mission Report:\n"
                f"{Fore.GREEN} Positive messages: {positive_count}\n"
                f"{Fore.RED} Negative messages: {negative_count}\n"
                f"{Fore.YELLOW} Neutral messages: {neutral_count}\n")

    elif command == "reset":
        conversation_history.clear()
        positive_count = negative_count = neutral_count = 0
        return f"{Fore.CYAN}Mission Reset! All previous data has been cleared."

    elif command == "history":
        return "\n".join([f"{Fore.CYAN} Message {i+1}: {msg}" for i, msg in enumerate(conversation_history)]) \
            if conversation_history else f"{Fore.YELLOW} No conversation history available."

    elif command == "help":
        return (f"{Fore.CYAN} Available Commands:\n"
                f"- Type any sentence to analyze sentiment\n"
                f"- Type 'summary' to see sentiment stats\n"
                f"- Type 'reset' to clear all data\n"
                f"- Type 'history' to view message log\n"
                f"- Type 'exit' to end the mission\n")

    else:
        return f"{Fore.RED}Unknown command. Type 'help' for available options."

# Function to get a valid name
def get_valid_name():
    while True:
        name = input("What's your name? ").strip()
        if name and name.isalpha():
            return name
        else:
            print(f"{Fore.RED}Please enter a valid name with only alphabetic characters.")

# Main chat loop
def start_sentiment_chat():
    global user_name

    print(f"{Fore.CYAN}{Style.BRIGHT} Welcome to Sentiment Spy! Your personal emotion detective is here.")
    user_name = get_valid_name()

    print(f"\n{Fore.CYAN}Nice to meet you, Agent {user_name}! Type your sentences to analyze emotions. Type 'help' for options.")

    while True:
        user_input = input(f"\n{Fore.MAGENTA}{Style.BRIGHT} Agent {user_name}: {Style.RESET_ALL}").strip()

        if not user_input:
            print(f"{Fore.RED} Please enter a non-empty message or type 'help' for available commands.")
            continue

        if user_input.lower() == "exit":
            print(f"\n{Fore.BLUE} 🔚Mission complete! Exiting Sentiment Spy. Farewell, Agent {user_name}!")
            print(execute_command("summary"))
            break
        elif user_input.lower() in ["summary", "reset", "history", "help"]:
            print(execute_command(user_input.lower()))
        else:
            show_processing_animation()
            print(analyze_sentiment(user_input))

# Start program
if __name__ == "__main__":
    start_sentiment_chat()
