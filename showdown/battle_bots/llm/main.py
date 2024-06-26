import json
import logging
import time

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import constants
from data import all_move_json, pokedex
from config import ShowdownConfig
from showdown.battle import Battle
from showdown.engine.helpers import normalize_name
from ..helpers import format_decision

# TODO: figure out a better wy to access agent_dataset to get parse_replay
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agent_dataset import parse_replay, NO_TERA

logger = logging.getLogger(__name__)

MODEL = None
TOKENIZER = None

class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)

        model_path = ShowdownConfig.model_path

        global MODEL, TOKENIZER # the battle bot gets reset after every battle, so we need to make sure we don't reload the model every time
        if MODEL is None:
            if model_path is None:
                raise ValueError('MODEL_PATH must be specified in the config file to use the LLM battle bot.')

            MODEL = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
            MODEL = MODEL.to_bettertransformer()
            MODEL.eval()
            MODEL.cuda()
            TOKENIZER = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"LLM Bot: Loaded model from {model_path}")

        self.model = MODEL
        self.tokenizer = TOKENIZER


        # set the tokenizer's pad token if it is not already set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def find_best_move(self):
        state = self.create_state()
        my_options = self.get_all_options()[0]

        moves = []
        switches = []
        for option in my_options:
            if option.startswith(constants.SWITCH_STRING + " "):
                switches.append(option)
            else:
                moves.append(option)

        # we have a little bit of trouble b/c the names are not normalized in the replays, but in moves and switches they are.
        # The all_move_json has normalized keys and unnormalized values. So we can just use the unnormalized values to get the normalized key.
        # But, the pokedex has normalized keys and *mostly* unnormalized values, but it is all lowercase, so we have to uppercase the first letter, and every letter after a space or -

        moves_unnormalized = [all_move_json[m][constants.NAME] for m in moves]
        switchable_pokemon_normalized = [s.split(" ")[1] for s in switches]
        switchable_pokemon_unnormalized = []

        for pkmn_name in switchable_pokemon_normalized:
            pkmn_name = pokedex[pkmn_name][constants.NAME]
            # capitalize the first letter
            pkmn_name = pkmn_name[0].upper() + pkmn_name[1:]
            # capitalize the letter after a space or -
            pkmn_name = pkmn_name.title()
            switchable_pokemon_unnormalized.append(pkmn_name)

        replay = parse_replay('\n'.join(self.msg_lines))

        if self.user.name == 'p2': # we want to view the game from player 1's perspective
            replay.swap_players()

        # override the ratings if they are specified in the config
        if ShowdownConfig.user_elo:
            replay.ratings[0] = ShowdownConfig.user_elo
        if ShowdownConfig.opponent_elo:
            replay.ratings[1] = ShowdownConfig.opponent_elo

        tera = ''

        if not self.force_switch:
            tera = self.request_json['active'][0].get(constants.CAN_TERASTALLIZE, '') # may not have CAN_TERA even in a move

        choice = choose_best_action(self.model, self.tokenizer, replay, moves_unnormalized, tera, switchable_pokemon_unnormalized, algorithm='sample_forward')

        return format_decision(self, choice)



############# Move Selection Algorithms #############
# These algorithms are used to select a move based from the list of available moves.
# I'm not sure what is the best one yet, so I'll add more over time.
#####################################################

def choose_best_action(model, tokenizer, replay, moves, tera, switches, algorithm='best_forward'):
    """
    Choose the best action from a list of available moves and switches.

    :param model: the model to use to predict the next action
    :param tokenizer: the tokenizer to use to tokenize the replay
    :param replay: the replay of the battle so far (as a agent_dataset.ShowdownBattle) object
    :param moves: a list of moves that the pokemon can make (or an empty list if it can't make any moves)
    :param tera: the tera type of the pokemon if it *can* terastallize, otherwise the empty string
    :param switches: a list of pokemon that the user can switch to (or an empty list if it can't switch)
    :return:
    """

    if algorithm == 'best_forward':
        return best_forward(model, tokenizer, replay, moves, tera, switches)
    elif algorithm == 'sample_forward':
        return sample_forward(model, tokenizer, replay, moves, tera, switches)
    else:
        raise ValueError("Invalid algorithm: {}".format(algorithm))

def best_forward(model, tokenizer, replay, moves, tera, switches):
    sequence_probs, sequence_logits = forward_probs(model, tokenizer, replay, moves, tera, switches)

    best_action_idx = torch.argmax(torch.stack(sequence_logits)).item()

    actions = format_move_strings(moves, tera) + format_switch_strings(switches)

    best_action = actions[best_action_idx]

    return action_to_showdown_decision(best_action)

def sample_forward(model, tokenizer, replay, moves, tera, switches):
    sequence_probs, sequence_logits = forward_probs(model, tokenizer, replay, moves, tera, switches)

    # turn the logits into probabilities
    sequence_probs = torch.softmax(torch.stack(sequence_logits), dim=0)

    best_action_idx = torch.multinomial(sequence_probs, num_samples=1).item()

    actions = format_move_strings(moves, tera) + format_switch_strings(switches)

    best_action = actions[best_action_idx]

    return action_to_showdown_decision(best_action)

def forward_probs(model, tokenizer, replay, moves, tera, switches):
    max_turns = ShowdownConfig.model_turn_limit
    if max_turns is None:
        max_turns = len(replay)

    cur_turn = len(replay)

    actions = format_move_strings(moves, tera) + format_switch_strings(switches)
    num_actions = len(actions)

    input_fits = False

    while not input_fits:
        start_turn = max(0, cur_turn - max_turns)
        replay_str = replay.to_string(start_turn=start_turn, end_turn=cur_turn)

        tokenized_replay = tokenizer(replay_str, return_tensors='pt')
        tokenized_actions = tokenizer(actions, return_tensors='pt', padding=True)

        if tokenized_replay['input_ids'].shape[1] + tokenized_actions['input_ids'].shape[1] < tokenizer.model_max_length:
            input_fits = True
        else:
            max_turns -= 1

    # we only have 1 replay, but we have several actions, so we need to repeat the replay
    tokenized_replay['input_ids'] = torch.cat([tokenized_replay['input_ids'].repeat(num_actions, 1), tokenized_actions['input_ids']], dim=1)
    tokenized_replay['attention_mask'] = torch.cat([tokenized_replay['attention_mask'].repeat(num_actions, 1), tokenized_actions['attention_mask']], dim=1)

    tokenized_replay = tokenized_replay.to('cuda')

    start_time = time.time()

    with torch.no_grad():
        outputs = model(**tokenized_replay)

    forward_pass_time = time.time() - start_time
    logger.debug(f"Time taken for forward pass of the model: {forward_pass_time} seconds")


    sequence_probs, sequence_logits = get_probs_of_seqs(tokenized_replay['input_ids'],
                                                        tokenized_actions['input_ids'],
                                                        outputs.logits,
                                                        tokenizer.pad_token_id)

    return sequence_probs, sequence_logits

def format_move_strings(moves, tera):
    action_strs = []

    for move in moves:
        action_no_tera = f'|action|p1|move|{move}|tera|{NO_TERA}'
        action_strs.append(action_no_tera)

        if tera:
            action_tera = f'|action|p1|move|{move}|tera|{tera}'
            action_strs.append(action_tera)

    return action_strs

def format_switch_strings(switches):
    action_strs = []

    for switch in switches:
        action_str = f'|action|p1|switch|{switch}'
        action_strs.append(action_str)

    return action_strs

def action_to_showdown_decision(best_action):

    split_action = best_action.split('|')
    if split_action[3] == 'move':
        decision = normalize_name(split_action[4])

        if split_action[6] != NO_TERA:
            decision += ' ' + constants.TERASTALLIZE
    else:
        decision = constants.SWITCH_STRING + ' ' + normalize_name(split_action[4])

    # we need to convert the names of the pokemon and moves to their normalized name otherwise the engine starts to have some issues
    return decision

def get_probs_of_seqs(replay_input_ids, action_input_ids, next_token_logits, pad_token_id):
    num_actions, longest_action_seq = action_input_ids.shape
    batch_size, total_seq_len, vocab_size = next_token_logits.shape # many of these values aren't used in the function, but I'm including them to help remember what the shapes are

    # action_input_ids is a 2D tensor of shape (num_actions, seq_len), but some actions are padded, and we don't want to include those in the probabilities
    # so we need to find how long the actual sequences are
    action_seq_lens = torch.sum(action_input_ids != pad_token_id, dim=1)

    sequence_probs = []
    avg_log_probs = []

    start_time = time.time()

    for i, seq_len in enumerate(action_seq_lens):
        token_probs, token_logits = calculate_subsequence_probability(replay_input_ids[i], start=total_seq_len - longest_action_seq,
                                                                      end=total_seq_len - longest_action_seq + seq_len - 1, # -1 b/c end is inclusive
                                                                      next_token_logits=next_token_logits[i])
        sequence_probs.append(torch.prod(token_probs))
        avg_log_probs.append(torch.mean(token_logits))

    get_probs_time = time.time() - start_time
    logger.debug(f"Time taken to get the probability of sequences: {get_probs_time} seconds")

    return sequence_probs, avg_log_probs

def calculate_subsequence_probability(input_ids, start, end, next_token_logits):
    """
    Calculate the probability of a subsequence of an input sequence given a start position, end position, and next_token_logits.

    Parameters:
    input_ids (numpy.ndarray): A 1D numpy array of token IDs.
    start (int): The start position of the subsequence (inclusive).
    end (int): The end position of the subsequence (inclusive).
    next_token_logits (numpy.ndarray): A 2D numpy array where the element at index [i, j] represents the logit for token j at position i+1 in the sequence. These logits are typically obtained as output from a language model, such as GPT-2, during the forward pass of the model.

    Returns:
    token_probabilities (numpy.ndarray): The probabilities of each token in the subsequence.
    token_logits (numpy.ndarray): The logits of each token in the subsequence.

    Raises:
    AssertionError: If the start or end position is not within the valid range.
    """

    assert start > 0 and end < len(input_ids), "Invalid start or end position"

    # Scale the next_token_logits to prevent overflow
    next_token_logits = next_token_logits - torch.max(next_token_logits, dim=-1, keepdim=True).values

    # Apply softmax to the logits to get probabilities
    probabilities = torch.exp(next_token_logits) / torch.sum(torch.nan_to_num(torch.exp(next_token_logits), nan=0.0), dim=-1, keepdim=True)

    token_probabilities = probabilities[range(start-1, end), input_ids[start:end+1]]

    # Convert probabilities back into logits to avoid underflow when multiplying the probabilities together.
    token_logits = torch.log(token_probabilities)

    return token_probabilities, token_logits

    # # Ensure the positions are within the valid range
    # assert start > 0 and end < len(input_ids), "Invalid start or end position"
    #
    # # Scale the next_token_logits to prevent overflow
    # next_token_logits = next_token_logits - np.max(next_token_logits, axis=-1, keepdims=True)
    #
    # # Apply softmax to the logits to get probabilities
    # probabilities = np.exp(next_token_logits) / np.sum(np.nan_to_num(np.exp(next_token_logits), nan=0), axis=-1, keepdims=True) # nan=0 b/c we treat -inf (or close enough to cause an underflow) as 0 probability
    #
    # token_probabilities = probabilities[range(start-1, end), input_ids[start:end+1]]
    #
    # # Wait whaaaaat? Why did you go through all that trouble to convert logits into probabilities just to convert them back into logits?
    # # Well, we couldn't just return the raw logits because each logit is from a different distribution, so they are not comparable.
    # # By converting to probabilities, we can compare probabilities of different subsequences.
    # # The conversion *back* into logits is so that we can avoid underflow when multiplying the probabilities together.
    # token_logits = np.log(token_probabilities)
    #
    # return token_probabilities, token_logits
