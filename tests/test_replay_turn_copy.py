import unittest

import copy

from metamon.backend.replay_parser.replay_state import Move, Pokemon, Turn
from metamon.interface import UniversalMove, UniversalPokemon


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

    def test_replay_move_universal_cache_keeps_pp_live(self):
        move = Move("Thunderbolt", gen=1)
        before = UniversalMove.from_ReplayMove(move, use_snapshot_cache=True)
        same = UniversalMove.from_ReplayMove(move, use_snapshot_cache=True)
        self.assertIs(before, same)
        move.set_pp(before.current_pp - 1)
        after = UniversalMove.from_ReplayMove(move, use_snapshot_cache=True)

        self.assertEqual(before.name, after.name)
        self.assertEqual(before.move_type, after.move_type)
        self.assertEqual(before.category, after.category)
        self.assertEqual(before.current_pp, after.current_pp + 1)
        self.assertIsNot(before, after)

    def test_native_pokemon_snapshot_cache_invalidates_on_live_fields(self):
        pokemon = Pokemon("Tauros", lvl=100, gen=1)
        move = Move("Body Slam", gen=1)
        pokemon.moves[move.name] = move
        pokemon.current_hp = pokemon.max_hp = 100

        first = UniversalPokemon.from_ReplayPokemon(
            pokemon, use_snapshot_cache=True
        )
        same = UniversalPokemon.from_ReplayPokemon(
            pokemon, use_snapshot_cache=True
        )
        self.assertIs(first, same)

        pokemon.current_hp = 50
        move.set_pp(move.pp - 1)
        changed = UniversalPokemon.from_ReplayPokemon(
            pokemon, use_snapshot_cache=True
        )
        self.assertIsNot(first, changed)
        self.assertEqual(changed.hp_pct, 0.5)
        self.assertEqual(changed.moves[0].current_pp, move.pp)

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

    def test_online_inplace_advance_matches_retained_next_turn(self):
        original = Turn(turn_number=7)
        pokemon = Pokemon("Tauros", lvl=100, gen=1)
        move = Move("Body Slam", gen=1)
        pokemon.moves[move.name] = move
        pokemon.had_moves[move.name] = move
        pokemon.last_target = object()
        pokemon.protected = True
        original.pokemon_1[0] = pokemon
        original.active_pokemon_1[0] = pokemon
        original.moves_1[0] = {"move": "Body Slam"}
        original.choices_2[0] = {"choice": "switch"}
        original.subturns.append({"temporary": True})
        original.replacements_1.append(("old", "new", (1, 2)))

        retained_source = copy.deepcopy(original)
        inplace = copy.deepcopy(original)
        expected = retained_source.create_next_turn()
        expected.on_end_of_turn()
        inplace.advance_in_place()

        self.assertEqual(inplace, expected)
        self.assertEqual(inplace.turn_number, 8)
        self.assertEqual(inplace.moves_1, [None, None])
        self.assertEqual(inplace.choices_2, [None, None])
        self.assertEqual(inplace.subturns, [])
        self.assertFalse(inplace.pokemon_1[0].protected)
        self.assertIsNone(inplace.pokemon_1[0].last_target)


if __name__ == "__main__":
    unittest.main()
