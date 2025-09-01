from treys import Evaluator, Card, Deck
import random
from card_detector import predict_card_attributes

def convert_to_treys_notation(rank, suit):
    rank_map = {
        "ace": "A", "king": "K", "queen": "Q", "jack": "J",
        "ten": "T", "nine": "9", "eight": "8", "seven": "7",
        "six": "6", "five": "5", "four": "4", "three": "3", "two": "2"
    }
    suit_map = {
        "hearts": "h", "diamonds": "d",
        "spades": "s", "clubs": "c"
    }
    return rank_map[rank.lower()] + suit_map[suit.lower()]

def get_treys_card_objects(image_paths):
    cards = []
    for img_path in image_paths:
        prediction = predict_card_attributes(img_path)
        rank = prediction['rank']
        suit = prediction['suit']
        treys_str = convert_to_treys_notation(rank, suit)
        card_obj = Card.new(treys_str)
        cards.append(card_obj)
    return cards

def evaluate_win_rate(player_card_paths, num_simulations=1000, num_opponents=1):
    evaluator = Evaluator()
    player_hand = get_treys_card_objects(player_card_paths)
    wins = 0

    for _ in range(num_simulations):
        deck = Deck()
        
        # Remove player cards from deck
        for card in player_hand:
            if card in deck.cards:
                deck.cards.remove(card)

        # Draw 5 community cards
        board = deck.draw(5)

        # Draw 2 cards for each opponent
        opponent_hands = []
        for _ in range(num_opponents):
            hand = deck.draw(2)
            opponent_hands.append(hand)

        player_score = evaluator.evaluate(board, player_hand[:2])

        opponent_wins = False
        for opp_hand in opponent_hands:
            opp_score = evaluator.evaluate(board, opp_hand)
            if opp_score < player_score:
                opponent_wins = True
                break

        if not opponent_wins:
            wins += 1

    return wins / num_simulations

# Example
if __name__ == "__main__":
    card_images = [
        r"C:\Users\zizhu\.ssh\ps_Project\.kaggle\archive\train\train\s95.jpg",  
        r"C:\Users\zizhu\.ssh\ps_Project\.kaggle\archive\train\train\s107.jpg"  
    ]
    win_rate = evaluate_win_rate(card_images, num_simulations=2000, num_opponents=5)
    print(f"Estimated Win Rate: {win_rate * 100:.2f}%")
