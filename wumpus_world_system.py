import numpy as np
import random
import matplotlib.pyplot as plt
import os
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

if not os.path.isdir("heatmaps"):
    os.makedirs("heatmaps")

class BayesWumpus:
    def __init__(self, dimension):
        self.dim = dimension
        self.net = self._create_network()
    
    def _create_network(self):
        net = BayesianNetwork([('AdjPit', 'PerceivedBreeze'),
                               ('AdjWumpus', 'PerceivedStench')])
        pit_cpd = TabularCPD(variable='AdjPit', variable_card=2,
                             values=[[0.9],
                                     [0.1]])
        wumpus_cpd = TabularCPD(variable='AdjWumpus', variable_card=2,
                                values=[[1 - (1/(self.dim**2))],
                                        [1/(self.dim**2)]])
        breeze_cpd = TabularCPD(variable='PerceivedBreeze', variable_card=2,
                                values=[[0.9, 0.1],
                                        [0.1, 0.9]],
                                evidence=['AdjPit'], evidence_card=[2])
        stench_cpd = TabularCPD(variable='PerceivedStench', variable_card=2,
                                values=[[0.9, 0.1],
                                        [0.1, 0.9]],
                                evidence=['AdjWumpus'], evidence_card=[2])
        net.add_cpds(pit_cpd, wumpus_cpd, breeze_cpd, stench_cpd)
        return net

    def infer_risk(self, percepts):
        infer = VariableElimination(self.net)
        evidence = {'PerceivedBreeze': percepts['Breeze'],
                    'PerceivedStench': percepts['Stench']}
        pit_query = infer.query(variables=['AdjPit'], evidence=evidence)
        wumpus_query = infer.query(variables=['AdjWumpus'], evidence=evidence)
        return {'Pit': pit_query.values[1], 'Wumpus': wumpus_query.values[1]}

class WumpusWorld:
    def __init__(self, n):
        self.n = n
        self.grid = np.full((n, n), "Empty", dtype=object)
        self.agent_best = (0, 0)
        self.agent_rand = (0, 0)
        self.visited_best = {self.agent_best}
        self.visited_rand = {self.agent_rand}
        self.hazard_map = np.ones((n, n)) * 0.2
        self.hazard_map[0, 0] = 0
        self.wumpus_loc = None
        self.gold_loc = None
        self.blocked = set()
        self._setup_world()

    def _setup_world(self):
        pit_count = max(1, self.n // 3)
        self._drop_item("Gold", 1)
        self._drop_item("Wumpus", 1)
        self._drop_item("Pit", pit_count)
        self._insert_percepts()
        for i in range(self.n):
            for j in range(self.n):
                cell = self.grid[i, j]
                if "Wumpus" in cell:
                    self.wumpus_loc = (i, j)
                if "Gold" in cell:
                    self.gold_loc = (i, j)

    def _drop_item(self, item, count):
        placed = 0
        while placed < count:
            i = random.randint(0, self.n - 1)
            j = random.randint(0, self.n - 1)
            if (i, j) != (0, 0) and self.grid[i, j] == "Empty":
                self.grid[i, j] = item
                placed += 1

    def _insert_percepts(self):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for i in range(self.n):
            for j in range(self.n):
                if "Pit" in self.grid[i, j]:
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < self.n and 0 <= nj < self.n:
                            if "Breeze" not in self.grid[ni, nj]:
                                self.grid[ni, nj] = (self.grid[ni, nj] + "/Breeze") if self.grid[ni, nj] != "Empty" else "Breeze"
                if "Wumpus" in self.grid[i, j]:
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < self.n and 0 <= nj < self.n:
                            if "Stench" not in self.grid[ni, nj]:
                                self.grid[ni, nj] = (self.grid[ni, nj] + "/Stench") if self.grid[ni, nj] != "Empty" else "Stench"

    def show_grid(self):
        print("\nWorld Layout:")
        for i in range(self.n):
            row = ""
            for j in range(self.n):
                row += f"{self.grid[i, j]:15}"
            print(row)
        print(f"\nBest Agent at: {self.agent_best}")
        print(f"Random Agent at: {self.agent_rand}")

    def get_neighbors(self, pos):
        i, j = pos
        nbrs = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + dx, j + dy
            if 0 <= ni < self.n and 0 <= nj < self.n:
                nbrs.append((ni, nj))
        return nbrs

    def sense_cell(self, pos):
        i, j = pos
        return {'Breeze': 1 if 'Breeze' in self.grid[i, j] else 0,
                'Stench': 1 if 'Stench' in self.grid[i, j] else 0}

    def is_dangerous(self, pos):
        i, j = pos
        return ("Pit" in self.grid[i, j]) or ("Wumpus" in self.grid[i, j])

    def update_hazard_map(self, bayes):
        for (i, j) in self.visited_best:
            self.hazard_map[i, j] = 0
        for i in range(self.n):
            for j in range(self.n):
                if (i, j) not in self.visited_best:
                    adjacent = False
                    for (vi, vj) in self.visited_best:
                        if (i, j) in self.get_neighbors((vi, vj)):
                            adjacent = True
                            senses = self.sense_cell((vi, vj))
                            risk = bayes.infer_risk(senses)
                            self.hazard_map[i, j] = risk.get('Pit', 0.2)
                            break
                    if not adjacent:
                        self.hazard_map[i, j] = 0.2

    def draw_heatmap(self, step):
        plt.figure(figsize=(10, 8))
        disp_map = self.hazard_map.copy()
        cmap_mod = plt.cm.Blues_r.copy()
        cmap_mod.set_under('lightgreen')
        masked = np.ma.array(disp_map, mask=np.zeros_like(disp_map, dtype=bool))
        im = plt.imshow(masked, cmap=cmap_mod, vmin=0.01, vmax=1.0)
        wx, wy = self.wumpus_loc
        gx, gy = self.gold_loc
        w_mask = np.zeros_like(disp_map, dtype=bool)
        w_mask[wx, wy] = True
        plt.imshow(np.ma.array(np.ones_like(disp_map), mask=~w_mask),
                   cmap=plt.cm.colors.ListedColormap(['darkred']), alpha=0.7)
        g_mask = np.zeros_like(disp_map, dtype=bool)
        g_mask[gx, gy] = True
        plt.imshow(np.ma.array(np.ones_like(disp_map), mask=~g_mask),
                   cmap=plt.cm.colors.ListedColormap(['gold']), alpha=0.7)
        plt.colorbar(im, label='Pit Probability')
        for i in range(self.n):
            for j in range(self.n):
                if self.hazard_map[i, j] == 0:
                    txt = "Safe"
                    col = "darkgreen"
                else:
                    txt = f"{self.hazard_map[i, j]:.2f}"
                    if (i, j) == (wx, wy):
                        col = "white"
                    elif (i, j) == (gx, gy):
                        col = "black"
                    else:
                        col = "black" if self.hazard_map[i, j] < 0.7 else "white"
                if (i, j) == (wx, wy):
                    plt.text(j, i, "W", ha="center", va="center", color=col, fontweight="bold")
                elif (i, j) == (gx, gy):
                    plt.text(j, i, "G", ha="center", va="center", color=col, fontweight="bold")
                else:
                    plt.text(j, i, txt, ha="center", va="center", color=col)
        ai, aj = self.agent_best
        plt.scatter(aj, ai, marker='*', s=200, color='red', label='Best Agent')
        for k in range(self.n + 1):
            plt.axhline(k - 0.5, color='black', linewidth=1)
            plt.axvline(k - 0.5, color='black', linewidth=1)
        plt.title(f'Pit Probability Heatmap - Step {step}')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.xticks(range(self.n))
        plt.yticks(range(self.n))
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"heatmaps/step_{step}_pit_probability.png")
        plt.close()
        print(f"Heatmap for step {step} saved.")

    def move_random(self):
        i, j = self.agent_rand
        nbrs = self.get_neighbors((i, j))
        if not nbrs:
            print("\nRandom mover has no available moves!")
            return
        prev = self.agent_rand
        choice = random.choice(nbrs)
        self.agent_rand = choice
        self.visited_rand.add(choice)
        if self.is_dangerous(choice):
            self.agent_rand = prev
            print(f"Random mover encountered danger at {choice} and reverted to {prev}")
            content = self.grid[choice[0], choice[1]]
            if "Pit" in content:
                print("Random mover nearly fell into a Pit!")
            if "Wumpus" in content:
                print("Random mover nearly met the Wumpus!")

    def move_best(self, bayes):
        i, j = self.agent_best
        options = self.get_neighbors((i, j))
        candidate = None
        lowest = float("inf")
        current_sense = self.sense_cell((i, j))
        for pos in options:
            if pos in self.visited_best or pos in self.blocked:
                continue
            risk = bayes.infer_risk(current_sense)
            tot = risk.get('Pit', 0) + risk.get('Wumpus', 0)
            if tot < lowest:
                lowest = tot
                candidate = pos
        if candidate is not None:
            prev_pos = self.agent_best
            self.agent_best = candidate
            self.visited_best.add(candidate)
            if self.is_dangerous(candidate):
                self.blocked.add(candidate)
                self.agent_best = prev_pos
                print(f"Best mover hit a hazard at {candidate} and reverted to {prev_pos}")
                cell_info = self.grid[candidate[0], candidate[1]]
                if "Pit" in cell_info:
                    print("Best mover fell into a Pit!")
                if "Wumpus" in cell_info:
                    print("Best mover bumped into the Wumpus!")
            else:
                print(f"Best mover advanced to {candidate}")
        else:
            safe_opts = [p for p in options if p not in self.blocked and 
                         (bayes.infer_risk(current_sense).get('Pit', 0) + bayes.infer_risk(current_sense).get('Wumpus', 0)) < 0.5]
            if safe_opts:
                old = self.agent_best
                self.agent_best = random.choice(safe_opts)
                self.visited_best.add(self.agent_best)
                print(f"Best mover revisited {self.agent_best} from {old}")
            else:
                print("Best mover remains stationary; no safe moves available.")

if __name__ == "__main__":
    user_input = input("Enter board size (N>=4): ")
    try:
        dimension = int(user_input)
        if dimension < 4:
            print("Board size must be at least 4. Using board size 4.")
            dimension = 4
    except ValueError:
        print("Invalid input. Using board size 4.")
        dimension = 4
    world = WumpusWorld(dimension)
    world.show_grid()
    bayes_module = BayesWumpus(dimension)
    for file in os.listdir("heatmaps"):
        path = os.path.join("heatmaps", file)
        if os.path.isfile(path):
            os.remove(path)
    world.update_hazard_map(bayes_module)
    world.draw_heatmap(0)
    step_count = 0
    best_found_gold = False
    rand_found_gold = False
    while not (best_found_gold or rand_found_gold):
        step_count += 1
        print(f"\n=== Step {step_count} ===")
        world.update_hazard_map(bayes_module)
        world.draw_heatmap(step_count)
        bi, bj = world.agent_best
        if "Gold" in world.grid[bi, bj]:
            print(f"\nBest mover reached Gold at {(bi, bj)}")
            best_found_gold = True
        if not best_found_gold:
            world.move_best(bayes_module)
        ri, rj = world.agent_rand
        if "Gold" in world.grid[ri, rj]:
            print(f"\nRandom mover reached Gold at {(ri, rj)}")
            rand_found_gold = True
        if not rand_found_gold:
            world.move_random()
        print(f"Step {step_count} concluded.")
    if best_found_gold:
        print("\nSimulation complete! Best mover reached the Gold!")
    elif rand_found_gold:
        print("\nSimulation complete! Random mover reached the Gold!")
    print("\nFinal World Layout:")
    world.show_grid()
