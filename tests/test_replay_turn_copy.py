import unittest

import copy

from metamon.backend.replay_parser.replay_state import Move, Pokemon, Turn
from metamon.interface import UniversalPokemon


class TurnCopyTests(unittest.TestCase):
    def test_universal_type_padding_does_not_mutate_parser_state(self):
        live_types = ["Normal"]
        self.assertEqual(
            UniversalPokemon.universal_types(live_types), "normal notype"
        )
        self.assertEqual(live_types, ["Normal"])

    def test_pokemon_deepcopy_preserves_aliases_but_isolates_battle_state(self):
        pokemon = Pokemon("Tauros", lvl=100, gen=1)
        move = Move("Body Slam", gen=1)
        pokemon.moves[move.name] = move
        pokemon.had_moves[move.name] = move
        pokemon.computed_stats["hp"] = 353

        copied = copy.deepcopy(pokemon)

        self.assertIsNot(copied, pokemon)
        self.assertIs(copied.base_stats, pokemon.base_stats)
        self.assertIs(copied.moves[move.name], copied.had_moves[move.name])
        self.assertIsNot(copied.moves[move.name], move)
        copied.moves[move.name].set_pp(3)
        copied.computed_stats["hp"] = 1
        self.assertNotEqual(move.pp, 3)
        self.assertEqual(pokemon.computed_stats["hp"], 353)

    def test_move_deepcopy_preserves_metadata_and_has_independent_pp(self):
        move = Move("Body Slam", gen=1)
        move.set_pp(7)
        move.maximum_pp = 24

        copied = copy.deepcopy(move)

        self.assertIsNot(copied, move)
        self.assertEqual(copied.__dict__, move.__dict__)
        self.assertEqual(copied.id, move.id)
        self.assertEqual(copied.current_pp, move.current_pp)
        self.assertEqual(copied.entry, move.entry)
        copied.set_pp(3)
        self.assertEqual(move.pp, 7)
        self.assertEqual(move.current_pp, 7)
        self.assertEqual(move.maximum_pp, 24)

    def test_next_turn_discards_ephemeral_state_and_preserves_aliases(self):
        turn = Turn(turn_number=7)
        pokemon = {"hp": 100}
        turn.pokemon_1[0] = pokemon
        turn.active_pokemon_1[0] = pokemon
        turn.moves_1[0] = {"move": "Body Slam"}
        turn.choices_2[0] = {"choice": "switch"}
        turn.subturns.append({"temporary": True})
        turn.replacements_1.append(("old", "new", (1, 2)))

        next_turn = turn.create_next_turn()

        self.assertEqual(next_turn.turn_number, 8)
        self.assertEqual(next_turn.moves_1, [None, None])
        self.assertEqual(next_turn.choices_2, [None, None])
        self.assertEqual(next_turn.subturns, [])
        self.assertEqual(next_turn.replacements_1, [])
        self.assertIs(next_turn.pokemon_1[0], next_turn.active_pokemon_1[0])
        self.assertIsNot(next_turn.pokemon_1[0], pokemon)
        next_turn.pokemon_1[0]["hp"] = 50
        self.assertEqual(pokemon["hp"], 100)


if __name__ == "__main__":
    unittest.main()
