import numpy as np
import time

class Omok:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)

    def make_move(self, x, y, player):
        if 0 <= x < self.size and 0 <= y < self.size and self.board[x, y] == 0:
            self.board[x, y] = player
            return True
        return False

    def get_legal_moves(self):
        moves = []
        if np.all(self.board == 0):
            return [(self.size // 2, self.size // 2)]
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] != 0:
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx][ny] == 0:
                                moves.append((nx, ny))
        moves.sort(key=lambda move: self.evaluate_move(move), reverse=True)
        return moves

    def evaluate_move(self, move):
        x, y = move
        count = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx][ny] != 0:
                    count += 1
        return count

    def has_winner(self):
        directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] != 0:
                    player = self.board[x][y]
                    for dx, dy in directions:
                        if self.win_check(x, y, dx, dy):
                            return player
        return 0

    def win_check(self, x, y, dx, dy):
        count = 1
        for d in [1, -1]:
            step = 1
            while True:
                nx, ny = x + step * dx * d, y + step * dy * d
                if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx][ny] == self.board[x][y]:
                    count += 1
                    step += 1
                else:
                    break
        return count >= 5

    def print_board(self):
        symbols = {0: '.', 1: 'W', 2: 'B'}
        print("\n".join(" ".join(symbols[self.board[x][y]] for y in range(self.size)) for x in range(self.size))
)
        print()

def iterative_deepening(game, player, time_limit=10):
    best_move = None
    best_score = float('-inf')
    depth = 1
    start_time = time.time()
    while time.time() - start_time < time_limit:
        score, temp_move = minimax(game, depth, float('-inf'), float('inf'), True, player)
        if score > best_score:
            best_score = score
            best_move = temp_move
        depth += 1
    return best_move

def minimax(node, depth, alpha, beta, maximizing_player, player):
    if depth == 0 or node.has_winner():
        return evaluate(node, player), None

    legal_moves = node.get_legal_moves()
    if maximizing_player:
        max_eval = float('-inf')
        best_move = None
        for x, y in legal_moves:
            node.make_move(x, y, player)
            eval, _ = minimax(node, depth - 1, alpha, beta, False, 3 - player)
            node.board[x, y] = 0
            if eval > max_eval:
                max_eval = eval
                best_move = (x, y)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = None
        for x, y in legal_moves:
            node.make_move(x, y, 3 - player)
            eval, _ = minimax(node, depth - 1, alpha, beta, True, player)
            node.board[x, y] = 0
            if eval < min_eval:
                min_eval = eval
                best_move = (x, y)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def evaluate(game, player):
    opponent = 3 - player
    my_score = 0
    opp_score = 0

    def score_line(line):
        score = 0
        count = 1
        open_ends = 0
        last = line[0]
        for cell in line[1:]:
            if cell == last and cell != 0:
                count += 1
            else:
                if last == player and (count >= 2):
                    if open_ends > 0:  # At least one end is open
                        score += count ** 5 + 10 * open_ends
                    else:
                        score += count ** 4
                elif last == opponent and (count >= 2):
                    if open_ends > 0:
                        score -= (count ** 5 + 10 * open_ends) * 1.1  # Slightly prioritize blocking
                    else:
                        score -= count ** 4 * 1.1
                count = 1
                open_ends = 1 if cell == 0 else 0
            last = cell
        # Check the last sequence
        if last == player and (count >= 2):
            score += count ** 5 + 10 * open_ends
        elif last == opponent and (count >= 2):
            score -= (count ** 5 + 10 * open_ends) * 1.1
        return score

    # Check all rows
    for x in range(game.size):
        my_score += score_line(game.board[x, :])

    # Check all columns
    for y in range(game.size):
        my_score += score_line(game.board[:, y])

    # Check all diagonals
    for d in range(-game.size + 1, game.size):
        my_score += score_line(np.diag(game.board, k=d))
        my_score += score_line(np.diag(np.fliplr(game.board), k=d))

    return my_score - opp_score

def play_game():
    game = Omok()
    player_choice = input("Choose your color (Black/White): ").strip().lower()
    human_player = 1 if player_choice == 'white' else 2
    ai_player = 3 - human_player
    current_player = 2  # Black starts first

    while True:
        game.print_board()
        if current_player == human_player:
            valid_move = False
            while not valid_move:
                x, y = map(int, input("Enter the row and column numbers (1-19) separated by space: ").split())
                x, y = x - 1, y - 1
                valid_move = game.make_move(x, y, current_player)
        else:
            move = iterative_deepening(game, ai_player)
            if move:
                game.make_move(move[0], move[1], ai_player)
                print(f"AI ('{'Black' if ai_player == 2 else 'White'}') places at ({move[0] + 1}, {move[1] + 1})")

        winner = game.has_winner()
        if winner:
            print(f"{'White' if winner == 1 else 'Black'} wins!")
            game.print_board()
            break

        current_player = 3 - current_player  # Switch players

play_game()