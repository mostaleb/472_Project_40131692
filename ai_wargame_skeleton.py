from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, Iterable, ClassVar, List
import random
import requests
from collections import deque

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000

global file_name


def set_file_name(alpha_beta: str, max_time: str, max_turns: str) -> str:
    return str("gametrace-" + alpha_beta + "-" + max_time + "-" + max_turns)


class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4


class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker


class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3


##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health: int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table: ClassVar[list[list[int]]] = [
        [3, 3, 3, 3, 1],  # AI
        [1, 1, 6, 1, 1],  # Tech
        [9, 6, 1, 6, 1],  # Virus
        [3, 3, 3, 3, 1],  # Program
        [1, 1, 1, 1, 1],  # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table: ClassVar[list[list[int]]] = [
        [0, 1, 1, 0, 0],  # AI
        [3, 0, 0, 3, 3],  # Tech
        [0, 0, 0, 0, 0],  # Virus
        [0, 0, 0, 0, 0],  # Program
        [0, 0, 0, 0, 0],  # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta: int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()

    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount


##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row: int = 0
    col: int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
            coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
            coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string() + self.col_string()

    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()

    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row - dist, self.row + 1 + dist):
            for col in range(self.col - dist, self.col + 1 + dist):
                yield Coord(row, col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row - 1, self.col)
        yield Coord(self.row, self.col - 1)
        yield Coord(self.row + 1, self.col)
        yield Coord(self.row, self.col + 1)

    @classmethod
    def from_string(cls, s: str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if len(s) == 2:
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None


##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src: Coord = field(default_factory=Coord)
    dst: Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string() + " " + self.dst.to_string()

    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row, self.dst.row + 1):
            for col in range(self.src.col, self.dst.col + 1):
                yield Coord(row, col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0, col0), Coord(row1, col1))

    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0, 0), Coord(dim - 1, dim - 1))

    @classmethod
    def from_string(cls, s: str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if len(s) == 4:
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None

    """@classmethod
    def valid_move(cls, player: Player, coord_pair: CoordPair, unit: Unit, game: Game) -> bool:
      if player is Player.Attacker:
          if unit is UnitType.AI or unit is UnitType.Firewall or unit is UnitType.Program:
            return Game.is_valid_move(game, coord_pair)
        else:"""


##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth: int | None = 4
    min_depth: int | None = 2
    max_time: float | None = 5.0
    game_type: GameType = GameType.AttackerVsDefender
    alpha_beta: bool = True
    max_turns: int | None = 100
    randomize_moves: bool = True
    broker: str | None = None


##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth: dict[int, int] = field(default_factory=dict)
    total_seconds: float = 0.0


##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim - 1
        self.set(Coord(0, 0), Unit(player=Player.Defender, type=UnitType.AI))
        self.set(Coord(1, 0), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(0, 1), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(2, 0), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(0, 2), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(1, 1), Unit(player=Player.Defender, type=UnitType.Program))
        self.set(Coord(md, md), Unit(player=Player.Attacker, type=UnitType.AI))
        self.set(Coord(md - 1, md), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md, md - 1), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md - 2, md), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md, md - 2), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md - 1, md - 1), Unit(player=Player.Attacker, type=UnitType.Firewall))

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord: Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord: Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord: Coord, unit: Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord, None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord: Coord, health_delta: int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords: CoordPair) -> Tuple[bool, bool, bool] | None:
        """Validate a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        """
        --> AI (attacker), Firewall (attacker) and Program (attacker) can only move up and left
        --> AI (defender), Firewall (attacker) andn Program (attacker) can only move down and right
        --> Techs and Viruses can move in all direction
        """
        # Checks for a suicide while confirming the unit is correctly chosen
        if coords.src == coords.dst and self.get(coords.src) is not None and self.next_player == self.get(
                coords.src).player:
            return True, True, True

        # Looks for the surrounding tiles and places them in a list
        adjacent_tiles = list(coords.src.iter_adjacent())

        # if the destination coordinates are not in the adjacent list, return false
        if coords.dst not in adjacent_tiles:
            return False, False, False

        # Checks if the unit at src is the current player's unit
        if self.get(coords.src) is not None and self.get(coords.src).player != self.next_player:
            return False, False, False

        # Whether it's an attack or a healing
        unit = self.get(coords.dst)
        if unit is not None:
            if self.next_player != unit.player:
                # Attacking
                return True, True, False
            elif self.next_player == unit.player:
                # Healing
                return True, False, True

        # Checks if unit is in attack mode
        if self.is_in_attack(coords, adjacent_tiles):
            return False, False, False

        # Checks if the player is an attacker or a defender. Then, looks whether the unit is an
        # AI, program or a firewall and that the unit is trying to move up or left (for attacker)
        # OR down or right (for defender). Returns false if not the case
        if self.next_player == Player.Attacker:
            if ((self.get(coords.src).type == UnitType.AI or self.get(coords.src).type == UnitType.Program or self.get(
                    coords.src).type == UnitType.Firewall)
                    and (coords.dst != adjacent_tiles[0] and coords.dst != adjacent_tiles[1])):  # up or left
                return False, False, False
        else:
            if ((self.get(coords.src).type == UnitType.AI or self.get(coords.src).type == UnitType.Program or self.get(
                    coords.src).type == UnitType.Firewall)
                    and (coords.dst != adjacent_tiles[2] and coords.dst != adjacent_tiles[3])):
                return False, False, False

        # Checks whether the destination or the source is part of the board
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False, False, False

        # Checks whether the unit at the source is the player's unit
        unit = self.get(coords.src)
        if unit is None or unit.player != self.next_player:
            return False, False, False

        # Checks whether the destination is empty
        unit = self.get(coords.dst)
        if unit is None:
            # Movement
            return True, False, False

    def is_in_attack(self, coords: CoordPair, adjacent_tiles: list) -> bool:

        for coord in adjacent_tiles:
            if self.get(coord) is not None and self.get(coord).player is not self.next_player:
                return True
        return False

    def perform_move(self, coords: CoordPair) -> Tuple[bool, str]:
        """Validate and perform a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        """
        --> The code to do is to make sure that the units move from one node to another or not
        """
        # Checks whether to perform a move or not and whether the move was meant to be an attack or a heal to another
        # unit
        if coords.src is None:
            print('weird')
        allowed_move, attack, heal = self.is_valid_move(coords)
        if coords.src is None:
            print('weird')
        if allowed_move:
            if attack and heal:
                self.self_destruct(coords.src)
            elif attack:
                self.attacking(coords)
            elif heal:
                return self.healing(coords)
            else:
                self.set(coords.dst, self.get(coords.src))
                self.set(coords.src, None)

            # Game trace written to a file
            self.gametrace(str(self.turns_played), str(self.next_player), str(coords.src), str(coords.dst),
                           self.to_string())
            return True, ""
        return False, "invalid move"

    def attacking(self, coords: CoordPair):

        # AI = 0, Virus = 1, Tech = 2, Program = 3, Firewall = 4
        unit_src = self.get(coords.src)
        unit_dst = self.get(coords.dst)

        # Reduces the health of the units depending on the damage table
        unit_src.health -= Unit.damage_table[unit_src.type.value][unit_dst.type.value]
        unit_dst.health -= Unit.damage_table[unit_dst.type.value][unit_src.type.value]

        # Checks if units are dead after the attack
        if not unit_src.is_alive():
            self.remove_dead(coords.src)
        if not unit_dst.is_alive():
            self.remove_dead(coords.dst)

    def healing(self, coords: CoordPair) -> Tuple[bool, str]:

        # AI = 0, Virus = 1, Tech = 2, Program = 3, Firewall = 4
        unit_src = self.get(coords.src)
        unit_dst = self.get(coords.dst)

        # Looks if the dst unit is already at 9 of health point
        if unit_dst.health == 9:
            return False, "invalid action"

        # Increases the health of a unit according to the repair table
        unit_dst.health += Unit.repair_table[unit_src.type.value][unit_dst.type.value]

        # Checks that the units did not receive more than 9 health points
        if unit_dst.health > 9:
            unit_dst.health = 9
            return False, "invalid action"
        return True, ""

    def self_destruct(self, coord: Coord) -> None:

        # Takes the area of damage for a suicide
        area_damage = list(coord.iter_range(1))

        # Loops through the area and removes 2 health points to all units in the area and removes the dead units
        for element_dst in area_damage:
            if self.is_valid_coord(element_dst) and self.get(element_dst) is not None:
                unit_dst = self.get(element_dst)
                unit_dst.health -= 2
                if not unit_dst.is_alive():
                    self.remove_dead(element_dst)

        # Removes the self-destructed unit by removing 10 health points
        self.mod_health(coord, -10)

        return None

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')

    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success, result) = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ", end='')
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success, result) = self.perform_move(mv)
                if success:
                    print(f"Player {self.next_player.name}: ", end='')
                    print(result)
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success, result) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ", end='')
                print(result)
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield coord, unit

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker
        return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src, _) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return 0, move_candidates[0], 1
        else:
            return 0, None, 0

    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta. TODO: REPLACE RANDOM_MOVE WITH PROPER GAME LOGIC!!!"""
        start_time = datetime.now()

        # Constructing the tree according to game parameters
        game_clone = self.clone()

        opt = Options()

        head = Node(None, None, None, None)
        self.construct_tree(opt.max_depth, None, head, game_clone)

        score, move_to_make = self.minimax(head, opt.max_depth, -10015,10015,True if self.next_player == Player.Attacker else False)

        # We are not going to be using random_move(), instead, we are going to implement a minimax() function
        # (score, move, avg_depth) = self.random_move()

        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        print(f"Heuristic score: {score}")
        print(f"Evals per depth: ", end='')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ", end='')
        print()
        total_evals = sum(self.stats.evaluations_per_depth.values())
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals / self.stats.total_seconds / 1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        return move_to_make

    def minimax(self, node: Node, depth: int, alpha: int, beta: int, max_: bool) -> Tuple[int, CoordPair]:
        """
        Max = Attacker
        Min = Defender
        """
        if depth == 0:
            return node.value

        if max_:
            max_evaluation = -10015
            for child in node.children:
                evaluation = self.minimax(child, depth - 1, alpha, beta, False)
                max_evaluation = max(max_evaluation, evaluation)
                if Options.alpha_beta:
                    alpha = max(alpha, evaluation)
                    if beta <= alpha:
                        break
                node.value = evaluation
                return max_evaluation, node.move
        else:
            min_evaluation = 10015
            for child in node.children:
                evaluation = self.minimax(child, depth - 1, alpha, beta, True)
                min_evaluation = min(min_evaluation, evaluation)
                if Options.alpha_beta:
                    beta = min(beta, evaluation)
                    if beta <= alpha:
                        break
                node.value = evaluation
                return min_evaluation, node.move

    def evaluate(self, a_units: Iterable[tuple[Coord, Unit]], d_units: Iterable[tuple[Coord, Unit]]) -> int:

        # Takes notes of all the units of each player in their own lists
        attacker_units = [list(tup) for tup in a_units]
        defender_units = [list(tup) for tup in d_units]

        # Keeps track of the number of units (and later their points) in the following order
        # Virus, Tech, Firewall, Program and AI for both players
        attacker_points = [0, 0, 0, 0, 0]
        defender_points = [0, 0, 0, 0, 0]

        """
        AI = 0
        Tech = 1
        Virus = 2
        Program = 3
        Firewall = 4
        
        Sums up the number of each unit of each player in a list
        """
        for element in attacker_units:
            if element[1] == 0:
                attacker_points[4] += 1
            elif element[1] == 1:
                attacker_points[1] += 1
            elif element[1] == 2:
                attacker_points[0] += 1
            elif element[1] == 3:
                attacker_points[3] += 1
            else:
                attacker_points[2] += 1

        for element in defender_units:
            if element[1] == 0:
                defender_points[4] += 1
            elif element[1] == 1:
                defender_points[1] += 1
            elif element[1] == 2:
                defender_points[0] += 1
            elif element[1] == 3:
                defender_points[3] += 1
            else:
                defender_points[2] += 1

        # Returns the value of the heuristic (will be positive if attacker is winning
        # and negative if attacker in losing
        return ((attacker_points[0] * 3 +
                 attacker_points[1] * 3 +
                 attacker_points[2] * 3 +
                 attacker_points[3] * 3 +
                 attacker_points[4] * 9999) -
                (defender_points[0] * 3 +
                 defender_points[1] * 3 +
                 defender_points[2] * 3 +
                 defender_points[3] * 3 +
                 defender_points[4] * 9999))

    from collections import deque

    def construct_tree(self, depth: int, parent: Node, node: Node, game_clone: Game) -> Node:
        # Initialize a stack to keep track of the tree nodes and their state
        stack = deque([(depth, parent, node, game_clone)])

        # Loop as long as there are items in the stack
        while stack:
            # Pop the next item from the stack
            depth, parent, node, game_clone = stack.pop()

            # Check if we've reached the depth limit of the tree
            if depth == 0:
                # If so, evaluate the node and set its parent
                node.value = self.evaluate(game_clone.player_units(Player.Attacker),
                                           game_clone.player_units(Player.Defender))
                node.parent = parent
                print("Leaf Node: Depth=" + str(depth) + ", Value=" + str(node.value))  # Print leaf node details
                continue  # Skip the rest and handle the next item on the stack

            # Set the parent for the current node
            node.parent = parent
            print("Internal Node: Depth=" + str(depth) + ", Parent=" + str(id(parent)) + ", Node=" + str(
                id(node)))  # Print internal node details

            # Get the list of possible moves at this game state
            moves_candidates = list(self.move_candidates())

            # Initialize an empty list to store the child nodes
            children_node = []

            # Loop through each possible move to create child nodes
            for move in moves_candidates:
                # Clone the game state and perform the move
                game_clone_to_perform_move = game_clone.clone()
                game_clone_to_perform_move.perform_move(move)

                # Assign the move to the current node
                node.move = move

                # Create a new child node
                new_child_node = Node(None, [], None, None)

                # Add the new child node to the list of children
                children_node.append(new_child_node)

                # Push the child node onto the stack to process it in a future iteration
                stack.append((depth - 1, node, new_child_node, game_clone_to_perform_move))

            # Assign the list of child nodes to the current node's `children` attribute
            node.children = children_node

        # Return the root of the constructed tree
        return node

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played + 1:
                        move = CoordPair(
                            Coord(data['from']['row'], data['from']['col']),
                            Coord(data['to']['row'], data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None

    def gametrace(self, turn: str, player_name: str, action_src: str, action_dst: str, board_config: str):
        global file_name
        with open(file_name, "a") as f:
            f.write("=============================================================================\n"
                    "Turn #" + turn + "\n"
                                      "Player: " + player_name + "\n"
                                                                 "Move from " + action_src + " to " + action_dst + "\n"
                                                                                                                   "Configuration of the board: \n" +
                    board_config + "\n"
                                   "=============================================================================\n")


##############################################################################################################
@dataclass(slots=True)
class Node:
    parent: Node | None
    move: CoordPair | None
    value: int | None  # heuristic
    children: List[Node] = field(default_factory=list)


##############################################################################################################
def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--game_type', type=str, default="manual", help='game type: auto|attacker|defender|manual')
    parser.add_argument('--max_turns', type=int, default=100, help='maximum turns per game')
    parser.add_argument('--algo', type=bool, default=False,
                        help='wether minimax is used with alpha-beta pruning (True) or not (False)')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker
    if options.max_turns is not None:
        options.max_turns = args.max_turns
    if options.alpha_beta is not None:
        options.alpha_beta = args.algo

    # create a new game
    game = Game(options=options)

    global file_name
    file_name = set_file_name(str(options.alpha_beta), str(options.max_time), str(options.max_turns))

    with open(file_name, "w") as f:
        f.write("=============================================================================\n"
                "Timeout:" + str(options.max_time) + "\n"
                                                     "Max number of turns: " + str(options.max_turns) + "\n"
                                                                                                        "Player 1: " + "Attacker\n" if options.game_type == GameType.AttackerVsDefender or options.game_type == GameType.AttackerVsComp else "Comp\n")

    with open(file_name, "a") as f:
        f.write(
            "Player 2: " + "Defender\n" if options.game_type == GameType.AttackerVsDefender or options.game_type == GameType.CompVsDefender else "Comp\n")

    with open(file_name, "a") as f:
        f.write("Initial configuration: " + game.to_string() + "\n"
                                                               "=============================================================================\n")

    # the main game loop
    while True:
        print()
        print(game)
        winner = game.has_winner()
        if winner is not None:
            print(f"{winner.name} wins!")
            with open(file_name, "a") as f:
                f.write(winner.name + " wins with " + str(game.turns_played) + " move")
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn()
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)


##############################################################################################################

if __name__ == '__main__':
    main()
