from dataclasses import dataclass

import torch
import re

from typing import List, Tuple, Optional

from tqdm import tqdm

PREFIXES_TO_REMOVE = ['|raw', '|inactive', '|c|', '|t:|', '|j|', '|l|', '|request']

NO_TERA = 'null'
NO_MOVE = 'failed'
TEMP_PLAYER = 'pTEMPPLAYER'
NO_LABEL_ID = -100

@dataclass
class PlayerAction:
    player: int
    type: str
    target: Optional[str] = None
    tera: Optional[str] = None
    position: int = 1 # this tells us which line the action should be printed *before*. E.g. 1 means before everything else in the turn (including the |turn text), 6 means before line 6, etc. 1 is a good default (after |turn

    def __str__(self):
        result = f'|action|p{self.player + 1}'

        if self.type == 'move':
            result += f'|move|{self.target if self.target else NO_MOVE}|tera|{self.tera if self.tera else NO_TERA}'
        elif self.type == 'switch':
            result += f'|switch|{self.target}'
        else:
            raise ValueError(f'Got type {self.type} for PlayerAction, must be either `move` or `switch`')

        return result

    def swap_player(self):
        if self.player == 0:
            self.player = 1
        else:
            self.player = 0

class Turn:
    def __init__(self, turn_lines):
        self.turn_lines = turn_lines

        turn_number = '0' if '|start' in turn_lines[0] else turn_lines[0].split('|')[-1] # unused, but could be useful later
        self.turn_number = turn_number

        self.p1_actions: List[PlayerAction]
        self.p2_actions: List[PlayerAction]

        self._parse_turn(turn_lines)

    def __str__(self):
        lines = []

        for i, line in enumerate(self.turn_lines):
            for action in self.p1_actions:
                if action.position == i:
                    lines.append(action)
            for action in self.p2_actions:
                if action.position == i:
                    lines.append(action)

            lines.append(line)

        return '\n'.join([str(action) for action in lines])

    def swap_players(self):
        self.p1_actions, self.p2_actions = self.p2_actions, self.p1_actions

        for action in self.p1_actions:
            action.swap_player()
        for action in self.p2_actions:
            action.swap_player()

        turn_lines = '\n'.join(self.turn_lines)

        turn_lines = turn_lines.replace('p1', TEMP_PLAYER)
        turn_lines = turn_lines.replace('p2', 'p1')
        turn_lines = turn_lines.replace(TEMP_PLAYER, 'p2')

        self.turn_lines = turn_lines.split('\n')

        if self.turn_number == '0': # for the very first turn, the pokemon are switched out in order of player, so we have to swap that as well
            # the last two lines are the sent out pokemon
            self.turn_lines[-1], self.turn_lines[-2] = self.turn_lines[-2], self.turn_lines[-1]

    def _parse_turn(self, turn_lines):
        actions = [[], []]

        move_made = False

        for i, line in enumerate(turn_lines):
            if line.startswith('|move|'):
                # example: |move|p1a: Kilowattrel|Hurricane|p2a: Bellibolt
                move_line = line.split('|')
                player = int(move_line[2][1]) - 1

                if len(actions[player]) == 0: # the player hasn't terra'd
                    action = PlayerAction(type='move', target=move_line[3], player=player)
                    actions[player].append(action)
                else:
                    action = actions[player][0] # there *should* only be one action possible at this point, and that is the one created during terastalization
                    action.target = move_line[3]

                move_made = True

            elif line.startswith('|-terastallize|'):
                # example: |-terastallize|p2a: Swalot|Dark
                tera_line = line.split('|')
                player = int(tera_line[2][1]) - 1
                tera_type = tera_line[3]

                # note: there may not be a move target; a pokemon can tera, get hit with a move, and flinch, faint, etc, and not have anything to populate the move field.
                actions[player].append(PlayerAction(type='move', tera=tera_type, player=player))

                move_made = True
            # this is complicated, because you can switch at the beginning of your turn,
            # in the middle of a turn due to u-turn or similar,
            # or at the end of a turn to replace a fainted pokemon
            elif line.startswith('|switch|'):
                # example: |switch|p2a: Miraidon|Miraidon, L66|112/242
                # if there hasn't been a move made yet, put the action at the beginning like normal,
                if move_made:
                    position = i
                else:
                    position = PlayerAction.position

                switch_line = line.split('|')
                player = int(switch_line[2][1]) - 1
                target = switch_line[3].split(',')[0]

                actions[player].append(PlayerAction(type='switch', target=target, position=position, player=player))

        # player 1 took an action, but it failed due to flinching, fainting, etc, so it doesn't show up on the move list
        p1_took_action = False
        p2_took_action = False

        for action in actions[0]:
            # if we see a 'move' or a switch at the right position (but not a switch at a later position, indicating it was due to fainting) then p1 took an action
            if action.type == 'move' or action.position == PlayerAction.position:
                p1_took_action = True
        for action in actions[1]:
            if action.type == 'move' or action.position == PlayerAction.position:
                p2_took_action = True

        if not p1_took_action:
            actions[0].append(PlayerAction(player=0, type='move'))
        if not p2_took_action:
            actions[1].append(PlayerAction(player=1, type='move'))

        self.p1_actions = actions[0]
        self.p2_actions = actions[1]


class ShowdownBattle:
    def __init__(self, winner : int, ratings : List[str], game_lines: List[str]):
        self.winner = winner
        self.ratings = ratings

        self.turns = []

        # now we can start going through the file and parsing out turns

        line_number = 0
        turn_lines = []

        while line_number < len(game_lines):
            line = game_lines[line_number]

            if line.startswith('|turn'):
                self.turns.append(Turn(turn_lines))
                turn_lines = [line] # reset the turn_lines to the new turn
            else:
                turn_lines.append(line)

            line_number += 1

        self.turns.append(Turn(turn_lines)) # once we get to the end of the file (which will include the win message) we add that turn to the game

    def __len__(self):
        return len(self.turns)

    def __str__(self):
        return self.to_string()

    def __getitem__(self, idx):
        return self.turns[idx]

    def to_string(self, start_turn=0, end_turn=None):
        result = f'|p1|rating|{self.ratings[0]}\n|p2|rating|{self.ratings[1]}\n|\n'

        for turn in self.turns[start_turn:end_turn]:
            result += str(turn) + '\n' # turns don't have an inherent newline at the end

        return result

    def swap_players(self):
        if self.winner == 0:
            self.winner = 1
        else:
            self.winner = 0

        self.ratings[0], self.ratings[1] = self.ratings[1], self.ratings[0]

        for turn in self.turns:
            turn.swap_players()


class ReplayDataset(torch.utils.data.Dataset):
    def __init__(self, replay_files, tokenizer, chunk_size):
        self.tokenizer = tokenizer
        self.replays = []

        # Load the replays
        for replay_file in tqdm(replay_files, total=len(replay_files), desc='Parsing replays'):
            with open(replay_file, "r", encoding='utf-8') as f:
                replay = f.read()

            replay = str(parse_replay(replay))

            # Tokenize the replay
            tokenized_replay = tokenizer(replay)

            # Chunk the tokenized replay
            chunks = []
            for i in range(0, len(tokenized_replay["input_ids"]), chunk_size):
                chunk = tokenized_replay["input_ids"][i:i + chunk_size]

                if len(chunk) > 10:  # no point in running through a batch with so few characters
                    chunks.append(chunk)

            # Add the chunks to the dataset
            for chunk in chunks:
                self.replays.append(chunk)

    def __len__(self):
        return len(self.replays)

    def __getitem__(self, idx):
        return {"input_ids": self.replays[idx]}

class AgentDataset(torch.utils.data.Dataset):
    def __init__(self, replay_files, tokenizer, num_turns):
        self.tokenizer = tokenizer
        self.num_turns = num_turns

        self.games = []

        for replay_file in tqdm(replay_files, total=len(replay_files), desc='Parsing replays'):
            with open(replay_file, "r", encoding='utf-8') as f:
                replay = f.read()

            game = parse_replay(replay)

            if game.winner == 1: # we only want games from the perspective of player 1 when they are the winner
                game.swap_players()

            self.games.append(game)

        self.inputs = []
        self.labels = []
        self.lengths = []

        for game in tqdm(self.games, total=len(self.games), desc='Tokenizing actions'):
            inputs, labels, lengths = self._tokenize_actions_in_game(game)

            self.inputs.extend(inputs)
            self.labels.extend(labels)
            self.lengths.extend(lengths)

    def _tokenize_actions_in_game(self, game) -> Tuple[List[int]]:
        inputs = []
        labels = []
        lengths = []

        if len(game) <= 2:  # not worth parsing a game with less than 2 turns (due to forfeit or whatever)
            return inputs, labels, lengths

        for turn_num in range(1, len(game)): # TODO: right now, we don't start with the opening turn where players send out pokemon, will have to change for different format
            cur_turn = game[turn_num]

            turn_lines = str(cur_turn).split('\n')

            for line_num, line in enumerate(turn_lines):
                if line.startswith('|action|p1|') and NO_MOVE not in line:  # check for lines where the player takes an action (as long as we actually know what it was, not if it failed)
                    action = line.split('|', 3)[-1].strip()  #example |action|p1|move|Stealth Rock|tera|null, only want move and everything after it except newline

                    num_turns_in_seq = self.num_turns
                    input_fits = False

                    while not input_fits:
                        input_str = game.to_string(max(0, turn_num - num_turns_in_seq), turn_num) # TODO: this assumes that, at the very least, 1 turn can fit in the input, which should be true for any reasonable model
                        input_str += '\n'.join(turn_lines[:line_num]) + '\n|action|p1|' # makes the input str everything up until the action we want to predict

                        action_tokens = self.tokenizer.encode(action)
                        input_tokens = self.tokenizer.encode(input_str)

                        input_len = len(input_tokens) + len(action_tokens)
                        # check if the input is too long
                        if input_len > self.tokenizer.model_max_length:
                            num_turns_in_seq -= 1
                        else:
                            input_fits = True

                    inputs.append(input_tokens + action_tokens)
                    label_tokens = len(input_tokens) * [NO_LABEL_ID] + action_tokens
                    labels.append(label_tokens)
                    lengths.append(input_len)

        return inputs, labels, lengths

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "labels": self.labels[idx], "length": self.lengths[idx]}


def parse_replay(replay: str) -> ShowdownBattle:
    # replace all utf-8 characters with ascii (ignore the ones that can't be converted)
    replay = replay.encode('ascii', errors='ignore').decode('ascii')
    replay_lines = replay.split('\n')

    usernames, ratings = get_usernames_and_ratings(replay_lines)

    winner = get_winner(replay_lines, usernames)

    replay_lines = cull_lines(replay_lines)

    replay_lines = replace_usernames(replay_lines, winner, usernames)

    replay_lines = normalize_hp(replay_lines)

    battle = ShowdownBattle(winner, ratings, replay_lines)

    return battle

def get_usernames_and_ratings(replay_lines: List[str]) -> Tuple[List[str]]:
    """Usernames and ratings look like
    |player|p1|p1username|sprite|rating
    |player|p2|p2username|spire|rating
    """

    p1_username = None
    p1_rating = None

    p2_username = None
    p2_rating = None

    for line in replay_lines:
        if line.startswith(f'|player|p1|'):
            split_line = line.split('|')
            p1_username = split_line[3]
            p1_rating = split_line[-1]
        elif line.startswith(f'|player|p2|'):
            split_line = line.split('|')
            p2_username = split_line[3]
            p2_rating = split_line[-1]

        if p1_rating is not None and p2_rating is not None:
            break

    return [p1_username, p2_username], [p1_rating, p2_rating]

def get_winner(replay_lines: List[str], usernames: List[str]) -> int:
    """Winner looks like
    |win|username
    """
    for line in reversed(replay_lines):
        if line.startswith('|win|'):
            winner_username = line.split('|')[2]
            return usernames.index(winner_username)

def cull_lines(replay_lines: List[str]) -> List[str]:
    """Skips unneeded lines, like everything up until |start, any timer messages that begin with |inactive|,
    and any chat messages that begin with |c|"""

    kept_lines = []

    battle_start = False

    for line in replay_lines:
        if line.startswith('|start'):
            battle_start = True

        if battle_start:
            keep_line = True

            if not line.startswith('|'): # we only want lines that start with a |
                keep_line = False

            for prefix in PREFIXES_TO_REMOVE:
                if line.startswith(prefix):
                    keep_line = False
                    break

            if keep_line:
                kept_lines.append(line)

    return kept_lines

def replace_usernames(replay_lines: List[str], winner:int,  usernames: List[str]) -> List[str]:
    replay = '\n'.join(replay_lines)

    # replace any instance of p1: p1username with p1a and p2: p2username with p2a (there are a few cases in the dataset where these exist, mostly after hazards have been set)
    replay = replay.replace(f'p1: {usernames[0]}', 'p1a')
    replay = replay.replace(f'p2: {usernames[1]}', 'p2a')

    # if there's a message with a player's name in it, replace that too ( it could contain some special regex characters, so we have to escape it)
    replay = re.sub(f'\|-message\|(.*){re.escape(usernames[0])}(.*)\n', r'|-message|\1p1\2\n', replay)
    replay = re.sub(f'\|-message\|(.*){re.escape(usernames[1])}(.*)\n', r'|-message|\1p2\2\n', replay)

    # also replace all of the p1as or p2as with p1s and p2s b/c we are only dealing with random battles right now TODO: will change if we get to doubles
    replay = replay.replace('p1a', 'p1')
    replay = replay.replace('p2a', 'p2')

    replay_lines = replay.split('\n')

    # if the last line contains the winner, replace it with the winner
    if replay_lines[-1].startswith('|win|'):
        replay_lines[-1] = f'|win|p{winner + 1}'

    return replay_lines

def normalize_hp(replay_lines):
    # uses regex to replace any instance of x/y to the scaled version of (x*100/y)/100
    # e.g. replaces |switch|p2a: Iron Leaves|Iron Leaves, L81|178/278
    # with |switch|p2a: Iron Leaves|Iron Leaves, L81|64/100

    replay = '\n'.join(replay_lines)

    hp_regex = re.compile(r'(\d+)/(\d+)')

    def normalize_hp_helper(match):
        return f'{int(int(match.group(1)) * 100 / int(match.group(2)))}/100'

    replay = re.sub(hp_regex, normalize_hp_helper, replay)

    return replay.split('\n')

if __name__ == '__main__':
    replay_file = 'dataset/gen9randombattle_rating/replays/gen9randombattle-1859887395.log'
    #replay_file = 'dataset/gen9randombattle_rating/replays/gen9randombattle-1884961761.log'

    with open(replay_file, 'r', encoding='utf-8') as f:
        replay = f.read()

    r = parse_replay(replay)
    print(r)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    dataset = AgentDataset([replay_file], tokenizer, 5)

    print(dataset[0])