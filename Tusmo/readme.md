# Tusmo Game Solver Bot

## Overview
This project features an automated Python bot designed to play the word puzzle game Tusmo without human intervention. Tusmo challenges players to guess a target word within six attempts, providing only the word's length and its initial letter as clues. The game offers feedback for each guess in the form of color-coded hints: RED for correctly placed letters, YELLOW for correct letters in the wrong position, and BLUE for incorrect letters.

## Key Features
- **Autonomous Gameplay:** The bot operates independently, navigating through the game's interface, making guesses, and interpreting feedback.
- **Advanced Recognition Algorithms:** Utilizes text and color recognition technologies to analyze the game's feedback, enabling it to understand which letters are correctly placed, misplaced, or not present in the target word.
- **Intelligent Guessing Mechanism:** Employs a strategic algorithm that refines its guesses based on the accumulated feedback, ensuring that each subsequent guess is more informed than the last.

## Technologies Used
- Python for the core bot logic.
- Computer Vision and Color Recognition techniques for interpreting the game's feedback.

## How It Works
1. The bot starts with the initial clue provided by the game (the first letter and the length of the word).
2. It makes a guess and uses color recognition to interpret the feedback from the game.
3. Based on this feedback, the bot narrows down the possible words, focusing only on those that match the known constraints.
4. This process repeats until the bot successfully guesses the word or exhausts its attempts.

## Project Goals
This project was developed as a fun and challenging way to apply and enhance my skills in Python programming, artificial intelligence, and computer vision. It demonstrates the potential of automation and AI in gaming and problem-solving contexts.
