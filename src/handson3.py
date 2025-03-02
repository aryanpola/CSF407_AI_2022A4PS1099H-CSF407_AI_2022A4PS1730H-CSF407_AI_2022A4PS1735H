import random
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from typing import Tuple, Optional
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from google import genai as googleGenAI

logging.getLogger("google_genai.models").setLevel(logging.WARNING)

@dataclass
class BoardSnapshot:
    boardGrid: np.ndarray
    dimension: int
    previousMove: Optional[Tuple[int, int]]
    activeMarker: str

class AbstractParticipant(ABC):
    def __init__(self, marker: str):
        self.marker = marker

    @abstractmethod
    def perform_turn(self, snapshot: BoardSnapshot) -> Tuple[int, int]:
        pass

class AiContestant(AbstractParticipant):
    def __init__(self, marker: str, modelHandle: str, apiSecret: str):
        super().__init__(marker)
        self.modelHandle = modelHandle
        self.apiSecret = apiSecret
        self.clientSession = googleGenAI.Client(api_key=apiSecret)

    def prepare_board_string(self, grid: np.ndarray) -> str:
        output = "\n"
        for rowBlock in grid:
            output += "|"
            for cellVal in rowBlock:
                output += f" {cellVal if cellVal != '' else ' '} |"
            output += "\n"
        return output

    def compose_prompt(self, snapshot: BoardSnapshot) -> str:
        boardRepr = self.prepare_board_string(snapshot.boardGrid)
        return f"""
You are playing {snapshot.dimension}x{snapshot.dimension} tic-tac-toe as '{self.marker}'.
Board state:
{boardRepr}
Opponent's last move: {snapshot.previousMove if snapshot.previousMove else 'None'}
Decide an optimal move (row,col). Reply with only 'row,col' e.g. "2,2".
Valid cells: [0,{snapshot.dimension - 1}] for both coordinates.
"""

    def perform_turn(self, snapshot: BoardSnapshot) -> Tuple[int, int]:
        prompt = self.compose_prompt(snapshot)
        for _ in range(3):
            try:
                result = self.clientSession.models.generate_content(
                    model=self.modelHandle,
                    contents=prompt
                )
                move = tuple(map(int, result.text.strip().split(',')))
                if 0 <= move[0] < snapshot.dimension and 0 <= move[1] < snapshot.dimension:
                    return move
            except Exception as e:
                sleep(1)
        return self.find_first_empty(snapshot.boardGrid)

    def find_first_empty(self, grid: np.ndarray) -> Tuple[int, int]:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i][j] == '':
                    return (i, j)
        raise ValueError("No empty cells")

class HumanContestant(AbstractParticipant):
    def __init__(self, marker: str):
        super().__init__(marker)

    def perform_turn(self, snapshot: BoardSnapshot) -> Tuple[int, int]:
        print("Current board state:")
        print(self.display_board(snapshot.boardGrid))
        print(f"Your marker is '{self.marker}'.")
        if snapshot.previousMove:
            print(f"Opponent's last move: {snapshot.previousMove}")
        else:
            print("No opponent move yet.")
        while True:
            user_input = input("Enter your move as row,col (e.g., 0,1): ")
            try:
                move = tuple(map(int, user_input.strip().split(',')))
                if 0 <= move[0] < snapshot.dimension and 0 <= move[1] < snapshot.dimension:
                    return move
                else:
                    print(f"Invalid move. Enter numbers between 0 and {snapshot.dimension - 1}.")
            except Exception as e:
                print("Invalid input format. Please try again.")

    def display_board(self, grid: np.ndarray) -> str:
        output = "\n"
        for row in grid:
            output += "|" + "|".join(f" {cell if cell != '' else ' '} " for cell in row) + "|\n"
        return output

class TicTacToeGame:
    def __init__(self, dimension: int, players: Tuple[AbstractParticipant, AbstractParticipant]):
        self.dimension = dimension
        self.board = np.full((dimension, dimension), '', dtype=object)
        self.players = players
        self.current_player = random.choice(players)
        self.last_move = None

    def make_move(self, row: int, col: int) -> bool:
        if self.board[row][col] == '':
            self.board[row][col] = self.current_player.marker
            self.last_move = (row, col)
            return True
        return False

    def get_snapshot(self) -> BoardSnapshot:
        return BoardSnapshot(
            boardGrid=self.board.copy(),
            dimension=self.dimension,
            previousMove=self.last_move,
            activeMarker=self.current_player.marker
        )

    def check_winner(self) -> Optional[str]:
        lines = []
        for i in range(self.dimension):
            lines.append(self.board[i, :])  
            lines.append(self.board[:, i]) 
        lines.append(np.diag(self.board))    
        lines.append(np.diag(np.fliplr(self.board))) 

        for line in lines:
            if len(set(line)) == 1 and line[0] != '':
                return line[0]
        return None

    def play_game(self) -> str:
        for _ in range(self.dimension**2):
            snapshot = self.get_snapshot()
            move = self.current_player.perform_turn(snapshot)
            if self.make_move(*move):
                if winner := self.check_winner():
                    print(f"Tic Tac Toe game finished. Winner: {winner}")
                    return winner
                self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        print("Tic Tac Toe game finished in a draw.")
        return 'Draw'

class BayesWumpus:
    def __init__(self, dimension):
        self.dim = dimension
        self.net = BayesianNetwork([('AdjPit', 'PerceivedBreeze'), 
                                    ('AdjWumpus', 'PerceivedStench')])
        self._build_cpds()

    def _build_cpds(self):
        pit_cpd = TabularCPD('AdjPit', 2, [[0.9], [0.1]])
        wumpus_cpd = TabularCPD('AdjWumpus', 2, [[1 - (1/(self.dim*2))], [1/(self.dim*2)]])
        breeze_cpd = TabularCPD('PerceivedBreeze', 2, 
                                [[0.9, 0.1], [0.1, 0.9]], 
                                evidence=['AdjPit'], evidence_card=[2])
        stench_cpd = TabularCPD('PerceivedStench', 2,
                                [[0.9, 0.1], [0.1, 0.9]],
                                evidence=['AdjWumpus'], evidence_card=[2])
        self.net.add_cpds(pit_cpd, wumpus_cpd, breeze_cpd, stench_cpd)

    def infer_risk(self, percepts):
        infer = VariableElimination(self.net)
        evidence = {'PerceivedBreeze': percepts['Breeze'],
                    'PerceivedStench': percepts['Stench']}
        pit_prob = infer.query(['AdjPit'], evidence=evidence).values[1]
        wumpus_prob = infer.query(['AdjWumpus'], evidence=evidence).values[1]
        return {'Pit': pit_prob, 'Wumpus': wumpus_prob}

class WumpusWorld:
    def __init__(self, size):
        self.size = size
        self.grid = np.full((size, size), "Empty", dtype=object)
        self.agent_pos = (0, 0)
        self.visited = {self.agent_pos}
        self.gold_pos = None
        self.wumpus_pos = None
        self.pits = []
        self.hazard_map = np.zeros((size, size))
        self._initialize_world()

    def _initialize_world(self):
        self.gold_pos = (random.randint(1, self.size-1), random.randint(1, self.size-1))
        self.grid[self.gold_pos] = "Gold"
        
        while True:
            pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if pos != (0,0) and pos != self.gold_pos:
                self.wumpus_pos = pos
                self.grid[pos] = "Wumpus"
                break
        
        # Place pits
        num_pits = max(1, self.size // 3)
        for _ in range(num_pits):
            while True:
                pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
                if pos not in [self.gold_pos, self.wumpus_pos, (0,0)]:
                    self.grid[pos] = "Pit"
                    self.pits.append(pos)
                    break
        
        self._add_percepts()

    def _add_percepts(self):
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        for i in range(self.size):
            for j in range(self.size):
                cell = self.grid[i, j]
                if "Pit" in cell:
                    for dx, dy in directions:
                        x, y = i + dx, j + dy
                        if 0 <= x < self.size and 0 <= y < self.size:
                            self.grid[x, y] += "/Breeze" if self.grid[x, y] != "Empty" else "Breeze"
                if "Wumpus" in cell:
                    for dx, dy in directions:
                        x, y = i + dx, j + dy
                        if 0 <= x < self.size and 0 <= y < self.size:
                            self.grid[x, y] += "/Stench" if self.grid[x, y] != "Empty" else "Stench"

    def move_agent(self, move_type: str, bayes_net: BayesWumpus):
        if move_type == 'best':
            self._best_move(bayes_net)
        else:
            self._random_move()
        self.visited.add(self.agent_pos)
        self._update_hazard_map()

    def _random_move(self):
        neighbors = self._get_neighbors()
        if neighbors:
            self.agent_pos = random.choice(neighbors)
            print(f"Random move to {self.agent_pos}")

    def _best_move(self, bayes_net: BayesWumpus):
        neighbors = self._get_neighbors()
        if not neighbors:
            return

        min_risk = float('inf')
        best_move = None
        current_sense = {
            'Breeze': 1 if 'Breeze' in self.grid[self.agent_pos] else 0,
            'Stench': 1 if 'Stench' in self.grid[self.agent_pos] else 0
        }

        for pos in neighbors:
            risk = bayes_net.infer_risk(current_sense)
            total_risk = risk['Pit'] + risk['Wumpus']
            if total_risk < min_risk:
                min_risk = total_risk
                best_move = pos

        if best_move:
            self.agent_pos = best_move
            print(f"Best move to {best_move} with risk {min_risk:.2f}")

    def _get_neighbors(self):
        i, j = self.agent_pos
        return [(i+di, j+dj) for di, dj in [(-1,0), (1,0), (0,-1), (0,1)] 
                if 0 <= i+di < self.size and 0 <= j+dj < self.size]

    def _update_hazard_map(self):
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) in self.visited:
                    self.hazard_map[i, j] = 0
                else:
                    self.hazard_map[i, j] = 0.2  

    def visualize(self, step: int):
        plt.figure(figsize=(8, 6))
        plt.imshow(self.hazard_map, cmap='Reds', vmin=0, vmax=1)
        plt.colorbar(label='Danger Probability')
        plt.title(f'Wumpus World - Step {step}')
        plt.scatter(self.gold_pos[1], self.gold_pos[0], c='gold', s=200, marker='o')
        plt.scatter(self.wumpus_pos[1], self.wumpus_pos[0], c='black', s=200, marker='x')
        plt.savefig(f'wumpus_step_{step}.png')
        plt.close()

class IntegratedSystem:
    def __init__(self, ttt_size: int, wumpus_size: int, player1: AbstractParticipant, player2: AbstractParticipant):
        self.ttt_size = ttt_size
        self.players = (player1, player2)
        self.ttt_game = TicTacToeGame(ttt_size, self.players)
        self.wumpus_world = WumpusWorld(wumpus_size)
        self.bayes_net = BayesWumpus(wumpus_size)
        self.step = 0

    def run_simulation(self):
        while True:
            self.ttt_game = TicTacToeGame(self.ttt_size, self.players)
            result = self.ttt_game.play_game()
            if result == 'Draw':
                continue  
            move_type = 'best' if result == 'X' else 'random'
            self.wumpus_world.move_agent(move_type, self.bayes_net)
            self.step += 1
            self.wumpus_world.visualize(self.step)
            if self.wumpus_world.agent_pos == self.wumpus_world.gold_pos:
                print(f"Gold found at step {self.step}!")
                break

def main():
    try:
        ttt_size = int(input("Enter Tic Tac Toe board size (e.g., 3 for 3x3): "))
        wumpus_size = int(input("Enter Wumpus World grid size (>=4): "))
        if wumpus_size < 4:
            print("Wumpus World size must be at least 4. Setting size to 4.")
            wumpus_size = 4
    except ValueError:
        print("Invalid input. Using default sizes: Tic Tac Toe=3, Wumpus World=6.")
        ttt_size = 3
        wumpus_size = 6

    print("Select game mode:")
    print("1: LLM vs LLM")
    print("2: Human vs LLM")
    mode = input("Enter 1 or 2: ").strip()

    GEMINI_API_KEY = "AIzaSyBv95Svw9vSRCEjgG5HxYggo2O1GRCkhVg"  
    if mode == "1":
        player1 = AiContestant('X', 'gemini-1.5-flash', GEMINI_API_KEY)
        player2 = AiContestant('O', 'gemini-1.5-pro', GEMINI_API_KEY)
    elif mode == "2":
        print("You selected Human vs LLM mode.")
        human_marker = input("Choose your marker (X/O): ").strip().upper()
        if human_marker == 'X':
            player1 = HumanContestant('X')
            player2 = AiContestant('O', 'gemini-1.5-pro', GEMINI_API_KEY)
        else:
            player1 = AiContestant('X', 'gemini-1.5-flash', GEMINI_API_KEY)
            player2 = HumanContestant('O')
    else:
        print("Invalid mode selected. Defaulting to LLM vs LLM.")
        player1 = AiContestant('X', 'gemini-1.5-flash', GEMINI_API_KEY)
        player2 = AiContestant('O', 'gemini-1.5-pro', GEMINI_API_KEY)

    system = IntegratedSystem(ttt_size, wumpus_size, player1, player2)
    system.run_simulation()

if __name__ == "__main__":
    main()
