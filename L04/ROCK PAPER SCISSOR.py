import random
from colorama import init, Fore

# Initialize Colorama for colored console output
init(autoreset=True)

def display_choices(player_name, player_choice, ai_choice):
    """
    Displays the choices made by the player and the AI.

    Args:
        player_name (str): The name of the player.
        player_choice (str): The player's choice (Rock, Paper, or Scissors).
        ai_choice (str): The AI's choice (Rock, Paper, or Scissors).
    """
    print(Fore.CYAN + f"{player_name} chose: {player_choice}")
    print(Fore.RED + f"AI chose: {ai_choice}")

def get_player_choice(player_name):
    
    valid_choices = ['ROCK', 'PAPER', 'SCISSORS']
    choice = ''
    while choice not in valid_choices:
        choice = input(Fore.GREEN + f"{player_name}, your move (R/P/S): ").upper()
        if choice == 'R':
            choice = 'ROCK'
        elif choice == 'P':
            choice = 'PAPER'
        elif choice == 'S':
            choice = 'SCISSORS'
        
        if choice not in valid_choices:
            print(Fore.RED + "Invalid. R, P, or S.")
    return choice

def get_ai_choice():
    
    return random.choice(['ROCK', 'PAPER', 'SCISSORS'])

def determine_winner(player_choice, ai_choice):
    
    if player_choice == ai_choice:
        return "tie"
    elif (player_choice == 'ROCK' and ai_choice == 'SCISSORS') or \
         (player_choice == 'PAPER' and ai_choice == 'ROCK') or \
         (player_choice == 'SCISSORS' and ai_choice == 'PAPER'):
        return "player"
    else:
        return "AI"

def rock_paper_scissors():
    
    print(Fore.YELLOW + "Welcome to Rock Paper Scissors Game!")
    player_name = input(Fore.GREEN + "Your name: ")

    while True:
        player_score = 0
        ai_score = 0
        
        print(Fore.CYAN + f"\n--- New game, {player_name}! ---")

        while True:
            # Get choices
            player_move = get_player_choice(player_name)
            ai_move = get_ai_choice()

            # Display choices
            display_choices(player_name, player_move, ai_move)

            # Determine and announce winner
            winner = determine_winner(player_move, ai_move)

            if winner == "player":
                print(Fore.GREEN + f"🎉 {player_name} wins!")
                player_score += 1
            elif winner == "AI":
                print(Fore.RED + "🤖 AI wins!")
                ai_score += 1
            else:
                print(Fore.YELLOW + "🤝 It's a Tie!")
            
            # Ask to play another round or end the game
            play_another_round = input(Fore.GREEN + f"Do you want to play again {player_name} ? (yes/no): ").lower()
            if play_another_round not in ['y', 'yes']:
                break # Exit the inner round loop

        # After the rounds, announce overall winner
        print(Fore.CYAN + "\n--- Game Over! ---")
        print(Fore.CYAN + f"Final Score: {player_name}: {player_score} | AI: {ai_score}")

        if player_score > ai_score:
            print(Fore.GREEN + f"🏆 {player_name} , you won the game!")
        elif ai_score > player_score:
            print(Fore.RED + "AI won the game!")
        else:
            print(Fore.YELLOW + "Overall it's a tie!")

        # Ask to play a new game (reset scores) or quit
        play_again_game = input(Fore.GREEN + "New game? (y/n): ").lower()
        if play_again_game not in ['y', 'yes']:
            print(Fore.CYAN + "Thanks for playing!")
            break # Exit the outer game loop

if __name__ == "__main__":
    rock_paper_scissors()
