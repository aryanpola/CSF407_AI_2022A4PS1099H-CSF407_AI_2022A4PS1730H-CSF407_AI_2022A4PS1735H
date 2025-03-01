# ======== Hands on 1 (Alternative Implementation) ========

import json
import random
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from typing import Tuple, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Remove OpenAI import and keep only Gemini
from google import genai as googleGenAI

@dataclass
class BoardSnapshot:
    # This holds the current board grid, dimension, last move, and active player's marker
    boardGrid: np.ndarray
    dimension: int
    previousMove: Optional[Tuple[int, int]]
    activeMarker: str

class AbstractParticipant(ABC):
    """
    Base class for both AI Participants (LLM) and a Human Participant.
    """
    def __init__(self, marker: str):
        self.marker = marker  # e.g., 'X' or 'O'

    @abstractmethod
    def perform_turn(self, snapshot: BoardSnapshot) -> Tuple[int, int]:
        pass

class HumanParticipant(AbstractParticipant):
    """
    Human participant: simply asks the user to provide row,col input.
    """
    def perform_turn(self, snapshot: BoardSnapshot) -> Tuple[int, int]:
        while True:
            move_str = input(f"Enter your move as row,col for marker '{self.marker}': ")
            try:
                rIndex, cIndex = map(int, move_str.split(','))
                if (0 <= rIndex < snapshot.dimension) and (0 <= cIndex < snapshot.dimension):
                    if snapshot.boardGrid[rIndex][cIndex] == '':
                        return (rIndex, cIndex)
                    else:
                        print("This cell is not empty. Try again.")
                else:
                    print("Coordinates out of range.")
            except ValueError:
                print("Invalid format. Please type row,col (e.g. 1,1).")

class AiContestant(AbstractParticipant):
    """
    AI-based participant that uses Gemini LLM to make decisions.
    """
    def __init__(self, marker: str, modelHandle: str, apiSecret: str):
        super().__init__(marker)
        self.modelHandle = modelHandle
        self.apiSecret = apiSecret
        self.clientSession = googleGenAI.Client(api_key=apiSecret)

    def prepare_board_string(self, grid: np.ndarray) -> str:
        """
        Turn the board into a string for prompt construction.
        """
        output = "\n"
        for rowBlock in grid:
            output += "|"
            for cellVal in rowBlock:
                output += f" {cellVal if cellVal != '' else ' '} |"
            output += "\n"
        return output

    def compose_prompt(self, snapshot: BoardSnapshot) -> str:
        boardRepr = self.prepare_board_string(snapshot.boardGrid)
        promptDraft = f"""
You are playing a {snapshot.dimension}x{snapshot.dimension} tic-tac-toe as '{self.marker}'.
Board state:
{boardRepr}

Opponent's last move: {snapshot.previousMove if snapshot.previousMove else 'None'}

Decide an optimal move (row,col). You may block the opponent or aim to win.
Reply with only 'row,col' e.g. "2,2" (without quotes).
Valid cells are those that are empty and within [0,{snapshot.dimension - 1}].
"""
        return promptDraft.strip()

    def generate_reply(self, content: str) -> Optional[str]:
        max_retries = 5
        backoff_time = 2
        
        for attemptCount in range(max_retries):
            try:
                result = self.clientSession.models.generate_content(
                    model=self.modelHandle,
                    contents=content
                )
                return result.text
            except Exception as e:
                error_msg = str(e)
                print(f"Gemini error (attempt {attemptCount+1}/{max_retries}): {e}")
                
                # Check if it's a rate limit error
                if 'RESOURCE_EXHAUSTED' in error_msg or '429' in error_msg:
                    wait_time = backoff_time * (2 ** attemptCount)
                    if attemptCount < max_retries - 1:
                        print(f"Rate limit hit. Waiting for {wait_time} seconds before retrying...")
                        sleep(wait_time)
                        continue
                elif attemptCount < max_retries - 1:
                    # For other errors, also retry but with smaller backoff
                    sleep(backoff_time)
                    continue
                    
        print("All Gemini API attempts failed.")
        return None

    def extract_move(self, textResp: str, snapshot: BoardSnapshot) -> Tuple[int, int]:
        """
        Convert a text reply from the LLM into (row, col).
        Fallback: if LLM fails, pick the first empty cell we find.
        """
        try:
            cleaned = textResp.strip().split('\n')[0].strip()
            rIndex, cIndex = map(int, cleaned.split(','))
            # Validate that it's a free cell
            if (0 <= rIndex < snapshot.dimension and
                0 <= cIndex < snapshot.dimension and
                snapshot.boardGrid[rIndex][cIndex] == ''):
                return (rIndex, cIndex)
        except:
            pass

        # If we get here, the provided move was invalid or we couldn't parse it
        for rr in range(snapshot.dimension):
            for cc in range(snapshot.dimension):
                if snapshot.boardGrid[rr][cc] == '':
                    return (rr, cc)
        raise ValueError("No valid moves remain on the board.")

    def perform_turn(self, snapshot: BoardSnapshot) -> Tuple[int, int]:
        statement = self.compose_prompt(snapshot)
        maxAttempts = 3
        backoff = 1

        for attemptCount in range(maxAttempts):
            try:
                reply = self.generate_reply(statement)
                if reply:
                    return self.extract_move(reply, snapshot)
            except Exception as ex:
                print(f"Attempt {attemptCount+1} LLM retrieval error: {ex}")
                if attemptCount < maxAttempts - 1:
                    sleep(backoff)
                    backoff *= 2
                continue

        # If everything fails, pick the first open cell
        for rr in range(snapshot.dimension):
            for cc in range(snapshot.dimension):
                if snapshot.boardGrid[rr][cc] == '':
                    return (rr, cc)

        # If truly no empty cell remains (which shouldn't happen normally):
        raise ValueError("All cells occupied. No moves left.")

class TicTacToeGame:
    """
    Manages a tic-tac-toe game with an NxN board, two participants, turn-taking, etc.
    """
    def __init__(self, dimension: int, playerA: AbstractParticipant, playerB: AbstractParticipant):
        self.dimension = dimension
        self.boardGrid = np.array([['' for _ in range(dimension)] for _ in range(dimension)])
        self.participantA = playerA
        self.participantB = playerB
        self.activeParticipant = random.choice([playerA, playerB])
        self.lastPlaced = None

        print(f"\nStarting new TicTacToeGame on {dimension}x{dimension} board.")
        print(f"First turn: {type(self.activeParticipant).__name__} with marker '{self.activeParticipant.marker}'")

    def assign_move(self, rowIdx: int, colIdx: int) -> bool:
        """
        Place the current participant's marker on the board, if valid.
        """
        if self.boardGrid[rowIdx][colIdx] == '':
            self.boardGrid[rowIdx][colIdx] = self.activeParticipant.marker
            self.lastPlaced = (rowIdx, colIdx)
            return True
        return False

    def swap_participant(self):
        if self.activeParticipant == self.participantA:
            self.activeParticipant = self.participantB
        else:
            self.activeParticipant = self.participantA

    def show_grid(self):
        for rowBlock in self.boardGrid:
            print("|", end="")
            for cellVal in rowBlock:
                print(f" {cellVal if cellVal else ' '} |", end="")
            print("\n" + "-" * (4 * self.dimension + 1))

    def get_snapshot(self) -> BoardSnapshot:
        """
        Provide an up-to-date snapshot of the board, dimension, last move, and who's active.
        """
        return BoardSnapshot(
            boardGrid=self.boardGrid.copy(),
            dimension=self.dimension,
            previousMove=self.lastPlaced,
            activeMarker=self.activeParticipant.marker
        )

def check_win(boardGrid: np.ndarray, dim: int, marker: str) -> bool:
    """
    Check if 'marker' has a winning line in the NxN board.
    """
    # Check rows and columns
    for idx in range(dim):
        if np.all(boardGrid[idx, :] == marker):
            return True
        if np.all(boardGrid[:, idx] == marker):
            return True

    # Check main diagonal
    if np.all(np.diag(boardGrid) == marker):
        return True

    # Check anti-diagonal
    if np.all(np.diag(np.fliplr(boardGrid)) == marker):
        return True

    return False

def check_filled(boardGrid: np.ndarray) -> bool:
    """
    Check if all cells are occupied (board is full).
    """
    for row in boardGrid:
        for c in row:
            if c == '':
                return False
    return True

def conduct_single_game(session: TicTacToeGame) -> str:
    """
    Play one game loop. Return 'X' if X wins, 'O' if O wins, or 'Draw' if no free cells remain.
    """
    while True:
        snapshot = session.get_snapshot()
        rChoice, cChoice = session.activeParticipant.perform_turn(snapshot)
        success = session.assign_move(rChoice, cChoice)

        if not success:
            print("Invalid move attempt. Skipping turn.")
        else:
            print(f"\n{type(session.activeParticipant).__name__} [{session.activeParticipant.marker}] -> ({rChoice},{cChoice})")
            session.show_grid()

            # Check for winner
            if check_win(session.boardGrid, session.dimension, session.activeParticipant.marker):
                return session.activeParticipant.marker

            # Check for draw
            if check_filled(session.boardGrid):
                return "Draw"

        session.swap_participant()

def record_outcomes(results: List[str], fname: str = "Task1.json"):
    """
    Save the outcomes to JSON. Summarize how many times X, O, or Draw occurred.
    """
    data_summary = {
        'results': results,
        'summary': {
            'FirstBrain_wins': results.count('X'),
            'SecondBrain_wins': results.count('O'),
            'tieGames': results.count('Draw')
        }
    }
    with open(fname, 'w') as f:
        json.dump(data_summary, f, indent=2)

def visualize_distribution(results: List[str], outName: str = "Task1.png"):
    """
    Create a bar chart from the outcomes.
    """
    xCount = results.count('X')
    oCount = results.count('O')
    dCount = results.count('Draw')

    categories = ['X Wins', 'O Wins', 'Draws']
    freq = [xCount, oCount, dCount]

    plt.figure(figsize=(8, 6))
    plt.bar(categories, freq)
    plt.title("Game Outcomes Distribution")
    plt.ylabel("Count of Results")
    plt.savefig(outName)
    plt.close()

def repeat_simulations(
    dimension: int,
    players_x: List[AiContestant],  # Now a list of players with different API keys
    players_o: List[AiContestant],
    howMany: int
) -> List[str]:
    """
    Conduct multiple games in a row for Bernoulli-like trials and track outcomes.
    Rotates between available API keys to avoid rate limits.
    """
    outcomes = []
    
    for idx in range(howMany):
        try:
            # Rotate players to distribute API calls
            player_x = players_x[idx % len(players_x)]
            player_o = players_o[idx % len(players_o)]
            
            print(f"\nSimulation round {idx+1} of {howMany}")
            print(f"Using player X with key ending in ...{player_x.apiSecret[-4:]}")
            print(f"Using player O with key ending in ...{player_o.apiSecret[-4:]}")
            
            session = TicTacToeGame(dimension, player_x, player_o)
            gameOutcome = conduct_single_game(session)
            outcomes.append(gameOutcome)
            
            # Add delay between games - increase for more reliability
            sleep(3)  # Increased delay between games
            
        except Exception as e:
            print(f"Error in simulation {idx+1}: {e}")
            print("Skipping to next simulation...")
            outcomes.append("Error")
            
            # Longer pause after errors
            sleep(10)  # Increased recovery time
            
    return outcomes

def launcher():
    """
    Main entry point with user selection for:
    - Board dimension
    - Mode (Human vs AI, AI vs AI)
    - Possibly multiple trials for AI vs AI
    """
    # Use multiple API keys to avoid rate limits
    gemini_keys = [
        'AIzaSyBv95Svw9vSRCEjgG5HxYggo2O1GRCkhVg',  # Replace with real key
        'AIzaSyBHOpiJpg-JPEib8H5oVDA6Jh_X-iisQtQ'  # Replace with real key
    ]

    # Check if we have valid API keys
    if not gemini_keys or not any(gemini_keys):
        print("Missing Gemini API keys. Please set them before running.")
        return

    # Let user choose dimension
    while True:
        try:
            dimension = int(input("Enter Tic-Tac-Toe board size (e.g. 3, 4, 5...): "))
            if dimension <= 0:
                print("Dimension must be positive.")
            else:
                break
        except ValueError:
            print("Invalid dimension, please enter a number.")

    # Let user choose mode
    while True:
        selection = input("Choose mode: [1] Human vs AI, [2] AI vs AI: ")
        if selection in ('1', '2'):
            break
        print("Please type 1 or 2.")

    # Add API check before running experiments
    def test_api_connection(key):
        try:
            client = googleGenAI.Client(api_key=key)
            client.models.generate_content(
                model="gemini-1.5-flash",
                contents="Test"
            )
            return True
        except Exception as e:
            print(f"API connection error for key: {e}")
            return False
    
    # Check API connections
    working_keys = []
    for key in gemini_keys:
        if test_api_connection(key):
            working_keys.append(key)
            print(f"API key validated successfully")
    
    if not working_keys and selection == '2':
        print("Warning: All Gemini API connections failed. Would you like to:")
        print("1. Continue anyway (may fail)")
        print("2. Use random AI players instead")
        print("3. Exit")
        
        choice = input("Enter your choice (1/2/3): ")
        if choice == '3':
            return
        elif choice == '2':
            # Alternative: create random-move AIs instead of API-dependent ones
            class RandomAiPlayer(AbstractParticipant):
                def perform_turn(self, snapshot):
                    empty_cells = []
                    for r in range(snapshot.dimension):
                        for c in range(snapshot.dimension):
                            if snapshot.boardGrid[r][c] == '':
                                empty_cells.append((r, c))
                    if empty_cells:
                        return random.choice(empty_cells)
                    raise ValueError("No valid moves remain on the board.")
                    
            brainX = RandomAiPlayer('X')
            brainO = RandomAiPlayer('O')
            
            # Run simulations with random players
            if selection == '2':
                trialCount = 50
                print(f"Running {trialCount} Random vs Random simulations...")
                outcomes = repeat_simulations(dimension, [brainX], [brainO], trialCount)
                record_outcomes(outcomes, "Task1_Random.json")
                visualize_distribution(outcomes, "Task1_Random.png")
                print(f"\nSimulations done. Results saved in 'Task1_Random.json' and distribution chart in 'Task1_Random.png'.")
            else:
                userPlayer = HumanParticipant('X')
                gameInst = TicTacToeGame(dimension, userPlayer, brainO)
                result = conduct_single_game(gameInst)
                print(f"\nGame Over. Result: {'Draw' if result == 'Draw' else f'Winner is {result}'}")
            return
    elif not working_keys:
        # If no keys work but user wants to continue anyway
        working_keys = gemini_keys

    if selection == '1':
        # Human vs AI
        userPlayer = HumanParticipant('X')
        # Create an AI with Gemini, using the first working key
        geminiAI = AiContestant('O', "gemini-1.5-flash", working_keys[0])
        gameInst = TicTacToeGame(dimension, userPlayer, geminiAI)
        result = conduct_single_game(gameInst)
        print(f"\nGame Over. Result: {'Draw' if result == 'Draw' else f'Winner is {result}'}")

    else:
        # AI vs AI for multiple trials
        trialCount = 20  # Further reduced to prevent rate limit issues
        print(f"Running {trialCount} AI vs AI simulations...")

        # Create multiple AI players with different keys
        players_x = []
        players_o = []
        
        # Create players with each working key
        for key in working_keys:
            players_x.append(AiContestant('X', "gemini-1.5-flash", key))
            players_o.append(AiContestant('O', "gemini-1.5-flash", key))
        
        print(f"Created {len(players_x)} players for X and {len(players_o)} players for O")
        
        outcomes = repeat_simulations(dimension, players_x, players_o, trialCount)
        record_outcomes(outcomes, "Task1.json")
        visualize_distribution(outcomes, "Task1.png")
        print(f"\nSimulations done. Results saved in 'Task1.json' and distribution chart in 'Task1.png'.")

if __name__ == "__main__":
    launcher()
