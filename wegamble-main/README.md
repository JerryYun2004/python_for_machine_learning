## Name  
**Card Master**

## Description  
Card Master is a computer vision pipeline that helps identify the type and strength of playing cards using deep learning. By breaking down card features into separate classification tasks (color, suit, and rank), this project enables reliable card recognition even with limited training data. The final goal is to estimate the player’s winning probability using Monte Carlo simulation. It is built with Raspberry Pi deployment in mind for future wearable integration.

---

## Badges  
Badges (e.g., CI/CD status, code quality) to be added here.

---

## Visuals  
Some resources on Blackjack rules and strategies:  
- https://www.pokerstars.com/casino/games/blackjack/rules  
- https://www.winstar.com/blog/how-to-play-blackjack-a-beginners-guide-to-rules-and-strategy  
- https://www.royalpanda.com/en/blog/guides/how-to-play-blackjack  

---

## Installation
Datasets:
- Keggle: https://www.kaggle.com/datasets/gunhcolab/object-detection-dataset-standard-52card-deck
- Stacked: https://www.kaggle.com/datasets/hugopaigneau/playing-cards-dataset
- Tree: https://www.kaggle.com/datasets/jamesmcguigan/playingcards

### Requirements
- Python 3.10+
- TensorFlow
- OpenCV
- Treys
- NumPy
- scikit-learn
- pickle

### Instructions
Clone the repository and install dependencies:
```bash
pip install -r libraries.txt
```

Download the pre-trained models from the `card_Master/thoughtful_approach` folder for accurate predictions.

---

## Usage

### Predicting Card Attributes
```python
predict_card_attributes(img_path)
```
- Input: Path to a playing card image  
- Output: Dictionary with card `color`, `suit`, and `rank`  
- Internally calls separate CNN models for color, suit (depending on red/black), and rank detection

### Estimating Win Rate
```python
evaluate_win_rate(player_card_paths, num_simulations=1000, num_opponents=1)
```
- Input: List of card image paths, number of simulations, number of opponents  
- Output: Estimated win rate as a float  
- Internally calls `predict_card_attributes` and simulates poker rounds using `treys.Evaluator`

---

## Support  
For help, please contact the author via GitLab Issues or direct message.

---

## Roadmap  
- Integrate with Raspberry Pi camera module  
- Convert model inference pipeline to TensorFlow Lite  
- Build a wearable heads-up display for live card analysis  
- Expand model set to handle multi-card stacked images  
- Improve dataset with synthetic augmentations

---

## Contributing  
Contributions are welcome! Please fork the repository and open a merge request.  
Include a brief description of your change and test it before submitting.

---

## Authors and acknowledgment  
**Main Author:** Zizhuo Yun  
Built with support from ETH Zürich’s Python for Science & ML course team.

---

## License  
This project is for academic and educational purposes only. License terms to be added.

---

## Project status  
**Ongoing development**  
The project is actively being improved and expanded. Contributions and feedback are welcome.
