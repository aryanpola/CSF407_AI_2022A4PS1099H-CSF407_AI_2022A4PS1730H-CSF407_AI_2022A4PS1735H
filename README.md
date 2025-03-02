You should install the following libraries on your PC:
pip install numpy matplotlib pgmpy google-generativeai









Question 1) output of AI vs AI for some iterations:
Enter Tic-Tac-Toe board size (e.g. 3, 4, 5...): 3
Choose mode: [1] Human vs AI, [2] AI vs AI: 2
API key validated successfully
API key validated successfully
Running 20 AI vs AI simulations...
Created 2 players for X and 2 players for O

Simulation round 1 of 20
Using player X with key ending in ...khVg
Using player O with key ending in ...khVg

Starting new TicTacToeGame on 3x3 board.
First turn: AiContestant with marker 'O'

AiContestant [O] -> (1,1)
|   |   |   |
-------------
|   | O |   |
-------------
|   |   |   |
-------------

AiContestant [X] -> (0,0)
| X |   |   |
-------------
|   | O |   |
-------------
|   |   |   |
-------------

AiContestant [O] -> (0,1)
| X | O |   |
-------------
|   | O |   |
-------------
|   |   |   |
-------------

AiContestant [X] -> (0,2)
| X | O | X |
-------------
|   | O |   |
-------------
|   |   |   |
-------------

AiContestant [O] -> (1,0)
| X | O | X |
-------------
| O | O |   |
-------------
|   |   |   |
-------------

AiContestant [X] -> (2,0)
| X | O | X |
-------------
| O | O |   |
-------------
| X |   |   |
-------------

AiContestant [O] -> (1,2)
| X | O | X |
-------------
| O | O | O |
-------------
| X |   |   |
-------------

Simulation round 2 of 20
Using player X with key ending in ...sQtQ
Using player O with key ending in ...sQtQ

Starting new TicTacToeGame on 3x3 board.
First turn: AiContestant with marker 'X'

AiContestant [X] -> (1,1)
|   |   |   |
-------------
|   | X |   |
-------------
|   |   |   |
-------------

AiContestant [O] -> (0,0)
| O |   |   |
-------------
|   | X |   |
-------------
|   |   |   |
-------------





Question 2) Output of question 2 for N=4
Enter board size (N>=4): 4

World Layout:
Empty          Stench         Breeze         Pit
Stench         Wumpus         Stench         Breeze
Empty          Stench         Empty          Empty
Empty          Gold           Empty          Empty

Best Agent at: (0, 0)
Random Agent at: (0, 0)
Heatmap for step 0 saved.

=== Step 1 ===
Heatmap for step 1 saved.
Best mover advanced to (1, 0)
Step 1 concluded.

=== Step 2 ===
Heatmap for step 2 saved.
Best mover advanced to (2, 0)
Random mover encountered danger at (1, 1) and reverted to (1, 0)
Random mover nearly met the Wumpus!
Step 2 concluded.

=== Step 3 ===
Heatmap for step 3 saved.
Best mover advanced to (3, 0)
Step 3 concluded.

=== Step 4 ===
Heatmap for step 4 saved.
Best mover advanced to (3, 1)
Step 4 concluded.

=== Step 5 ===
Heatmap for step 5 saved.

Best mover reached Gold at (3, 1)
Step 5 concluded.

Simulation complete! Best mover reached the Gold!

Final World Layout:

World Layout:
Empty          Stench         Breeze         Pit
Stench         Wumpus         Stench         Breeze
Empty          Stench         Empty          Empty
Empty          Gold           Empty          Empty

Best Agent at: (3, 1)
Random Agent at: (0, 0)


Question 3)Partial output for a couple of minutes of running
Enter Tic Tac Toe board size (e.g., 3 for 3x3): 3
Enter Wumpus World grid size (>=4): 4
Select game mode:
1: LLM vs LLM
2: Human vs LLM
Enter 1 or 2: 1
Tic Tac Toe game finished in a draw.
Tic Tac Toe game finished. Winner: O
Random move to (1, 0)
Tic Tac Toe game finished. Winner: X
Best move to (0, 0) with risk 0.03
