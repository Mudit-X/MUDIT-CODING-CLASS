import random
from colorama import init, Fore

# Initialize Colorama
init(autoreset=True)

def display_board(board):
    print()
    print(' ' + format_symbol(board[0]) + ' | ' + format_symbol(board[1]) + ' | ' + format_symbol(board[2]))
    print(Fore.CYAN + '-----------')
    print(' ' + format_symbol(board[3]) + ' | ' + format_symbol(board[4]) + ' | ' + format_symbol(board[5]))
    print(Fore.CYAN + '-----------')
    print(' ' + format_symbol(board[6]) + ' | ' + format_symbol(board[7]) + ' | ' + format_symbol(board[8]))
    print()

def format_symbol(symbol):
    if symbol == 'X':
        return Fore.RED + symbol + Fore.RESET
    elif symbol == 'O':
        return Fore.BLUE + symbol + Fore.RESET
    else:
        return Fore.YELLOW + symbol + Fore.RESET

def player_choice():
    

def player_move(board, symbol, player_name):
    

def ai_move(board, ai_symbol, player_symbol):
    # Check if AI can win in the next move
    

    # Check if player could win on their next move, and block them
    
    
    # Choose a random move
    

def check_win(board, symbol):
    

def check_full(board):
    

def tic_tac_toe():
    

