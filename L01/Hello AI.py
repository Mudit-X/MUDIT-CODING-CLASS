while True:
    print("Hello! I'm your friendly chatbot")
    name = input("What is your name? ")

    print(f"Nice to meet you, {name}!")

    print()

    feeling = input("How are you feeling today? ").lower()

    if "good" in feeling or "great" in feeling or "fine" in feeling:  # Positive tone
        print("I'm glad to hear that!")
    elif "bad" in feeling or "sad" in feeling:  # Negative tone
        print("I'm sorry to hear that. Hope things get better.")
    else:
        print("I see. Sometimes it's hard to put feelings into words.")

    print()

    # Ask about hobby
    hobby = input("What's your favorite hobby? ").lower()
    print(f"That sounds fun! I hope you get time to enjoy {hobby} often.")

    print()

    # Ask about food
    food = input("What's your favorite food? ").lower()
    print(f"Yum! I love {food} too!")

    print()

    # Ask about movie or show
    show = input("What's a movie or show you really like? ").lower()
    print(f"{show} sounds interesting! I might watch it sometime.")

    print()

    # Ask about sports
    sports = input("What's your favorite sport that you enjoy playing? ").lower()

    if "football" in sports or "soccer" in sports:
        print(f"That sounds fun! I hope you get time to play {sports} often.")
    elif "volleyball" in sports or "basketball" in sports:
        print("That's what I love playing as well!")
    else:
        print("That sounds fun, I will try it out...")

    print()

    # Ask if user wants to chat again
    again = input("Do you want to chat again? (yes/no) ").lower()
    print()
    if again != "yes":
        print("Thanks for chatting with me! Have a great day!")
        break
