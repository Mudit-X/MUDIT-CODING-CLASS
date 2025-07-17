import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from colorama import init, Fore
import time
import sys

# Initialize colorama
init(autoreset=True)

# Load movie data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\MUDIT CODING CLASS\L06\imdb_top_1000.csv")
        df['combined_features'] = df['Genre'].fillna('') + ' ' + df['Overview'].fillna('')
        return df
    except FileNotFoundError:
        print(Fore.RED + "Error: The file 'imdb_top_1000.csv' was not found.")
        exit()

movies_df = load_data()

# Vectorize and compute similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# List all genres
def list_genres(df):
    return sorted(set(genre.strip() for sublist in df['Genre'].dropna().str.split(', ') for genre in sublist))

genres = list_genres(movies_df)

# Recommend movies
def recommend_movies(genre=None, mood=None, rating=None, top_n=5):
    filtered_df = movies_df

    if genre:
        filtered_df = filtered_df[filtered_df['Genre'].str.contains(genre, case=False, na=False)]
    if rating:
        filtered_df = filtered_df[filtered_df['IMDB_Rating'] >= rating]

    filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)  # Shuffle

    recommendations = []
    user_mood_polarity = TextBlob(mood).sentiment.polarity if mood else 0

    for _, row in filtered_df.iterrows():
        overview = row['Overview']
        if pd.isna(overview):
            continue
        polarity = TextBlob(overview).sentiment.polarity

        # New logic: Match sentiment direction or if no mood specified
        if not mood or (user_mood_polarity * polarity >= 0):
            recommendations.append((row['Series_Title'], polarity))

        if len(recommendations) == top_n:
            break

    return recommendations if recommendations else "No suitable movie recommendations found."

# Display recommendations
def display_recommendations(recs, name):
    print(Fore.YELLOW + f"\n🍿 AI-Analyzed Movie Recommendations for {name}:")
    for idx, (title, polarity) in enumerate(recs, 1):
        sentiment = "Positive 😊" if polarity > 0 else "Negative 😞" if polarity < 0 else "Neutral 😐"
        print(f"{Fore.CYAN}{idx}. {title} (Polarity: {polarity:.2f}, {sentiment})")

# Animation
def processing_animation():
    for _ in range(3):
        print(Fore.YELLOW + ".", end="", flush=True)
        time.sleep(0.5)

# Main interaction
def handle_ai(name):
    print(Fore.BLUE + "\nLet's find the perfect movie for you!\n")

    print(Fore.GREEN + "Available Genres: ")
    for idx, genre in enumerate(genres, 1):
        print(f"{Fore.CYAN}{idx}. {genre}")
    print()

    while True:
        genre_input = input(Fore.YELLOW + "Enter genre number or name: ").strip()
        if genre_input.isdigit() and 1 <= int(genre_input) <= len(genres):
            genre = genres[int(genre_input) - 1]
            break
        elif genre_input.title() in genres:
            genre = genre_input.title()
            break
        print(Fore.RED + "Invalid input. Please try again.\n")

    mood = input(Fore.YELLOW + "How do you feel today? (Describe your mood): ").strip()

    print(Fore.BLUE + "\nAnalyzing mood", end="", flush=True)
    processing_animation()
    polarity = TextBlob(mood).sentiment.polarity
    mood_desc = "positive 😊" if polarity > 0 else "negative 😞" if polarity < 0 else "neutral 😐"
    print(f"\n{Fore.GREEN}Your mood is {mood_desc} ({polarity:.2f}).\n")

    while True:
        rating_input = input(Fore.YELLOW + "Enter minimum IMDB rating (7.6 - 9.3) or 'skip': ").strip()
        if rating_input.lower() == 'skip':
            rating = None
            break
        try:
            rating = float(rating_input)
            if 7.6 <= rating <= 9.3:
                break
            else:
                print(Fore.RED + "Rating out of range. Try again.\n")
        except ValueError:
            print(Fore.RED + "Invalid input. Please enter a number or 'skip'.\n")

    print(f"{Fore.BLUE}\nFinding movies for {name}", end="", flush=True)
    processing_animation()

    recs = recommend_movies(genre, mood, rating)
    if isinstance(recs, str):  # If it’s the error message
        print(Fore.RED + recs + "\n")
    else:
        display_recommendations(recs, name)

    while True:
        action = input(Fore.YELLOW + "\nWould you like more recommendations? (yes/no): ").strip().lower()
        if action == 'no':
            print(Fore.GREEN + f"\nEnjoy your movie picks, {name}! Goodbye!")
            break
        elif action == 'yes':
            more_recs = recommend_movies(genre=genre, mood=mood, rating=rating, top_n=5)
            if isinstance(more_recs, str):
                print(Fore.RED + more_recs + "\n")
            else:
                display_recommendations(more_recs, name)
        else:
            print(Fore.RED + "Invalid input. Please answer with 'yes' or 'no'.\n")

# Entry point
def main():
    print(Fore.BLUE + "🎥 Welcome to your Personal Movie Recommendation Assistant! 🎥\n")
    name = input(Fore.YELLOW + "What's your name? ").strip()
    print(f"\n{Fore.GREEN}Great to meet you, {name}!\n")
    handle_ai(name)

if __name__ == "__main__":
    main()
