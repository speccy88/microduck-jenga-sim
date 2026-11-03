"""Non-billable CPU tests of actual strike/ranking/recording logic, without ML dependencies.

Run: python -m unittest discover -s tests -p test_jenga_preparation.py -v
"""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from mjlab_microduck.jenga_strike import CRITERIA, initial_strike_state, update_strike
from mjlab_microduck.jenga_evaluation import (
    METRICS, checkpoint_iteration, discover_checkpoints, progression, save_json,
    select_best, summarize, video_filename)

spec = importlib.util.spec_from_file_location("record_jenga_under_test", ROOT/"scripts/record_jenga.py")
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


class StrikeTests(unittest.TestCase):
    def setUp(self):
        self.state = initial_strike_state(np.zeros(2), np.zeros((2,3)))
        self.mouth = np.zeros((2,3))
        self.top = np.zeros((2,3))
        self.direction = np.array([[1.,0,0],[1.,0,0]])
        self.sample([False, False])

    def sample(self, contacts, dt=0.02):
        self.state = update_strike(self.state, self.mouth, self.direction,
                                   self.top, np.array(contacts), dt, np)

    def strike(self, speed=0.30):
        self.mouth[:,0] += speed*0.02
        self.sample([True,False])
        self.top[0,0] = 0.04
        self.sample([False,False])

    def test_fast_contact_and_prompt_separation_qualify_only_contacted_env(self):
        self.strike()
        np.testing.assert_array_equal(self.state["fast_strike"], [1,0])
        self.assertAlmostEqual(self.state["strike_head_speed_m_s"][0], 0.30)
        self.assertAlmostEqual(self.state["strike_separation_delay_s"][0], 0.02)

    def test_slow_push_is_not_fast(self):
        self.strike(0.10)
        self.assertEqual(self.state["fast_strike"][0], 0)

    def test_backward_or_sideways_head_speed_does_not_qualify(self):
        for direction in [(-0.3,0,0),(0,0.8,0)]:
            self.setUp()
            self.mouth[0] += np.array(direction)*0.02
            self.sample([True,False])
            self.top[0,0] = 0.04
            self.sample([False,False])
            self.assertEqual(self.state["fast_strike"][0], 0)

    def test_later_fast_motion_cannot_rescue_initial_slow_contact(self):
        self.sample([True,False])
        self.mouth[0,0] += 0.02
        self.sample([True,False])
        self.top[0,0] = 0.04
        self.sample([False,False])
        self.assertEqual(self.state["fast_strike"][0], 0)

    def test_preimpact_sample_is_counted(self):
        self.mouth[0,0] += 0.006
        self.sample([False,False])
        self.sample([True,False])  # head has stopped on impact
        self.top[0,0] = 0.04
        self.sample([False,False])
        self.assertEqual(self.state["fast_strike"][0], 1)

    def test_prolonged_or_repeated_contact_revokes_fast_strike(self):
        self.strike()
        for _ in range(6):
            self.sample([True,False])
        self.assertGreater(self.state["strike_contact_duration_s"][0], 0.12)
        self.assertEqual(self.state["fast_strike"][0], 0)

    def test_delayed_separation_does_not_qualify(self):
        self.mouth[0,0] += 0.006
        self.sample([True,False])
        for _ in range(16):
            self.sample([False,False])
        self.top[0,0] = 0.04
        self.sample([False,False])
        self.assertEqual(self.state["fast_strike"][0], 0)

    def test_preexisting_top_motion_disqualifies_later_touch(self):
        self.top[0,0] = 0.04
        self.sample([False,False])
        self.mouth[0,0] += 0.006
        self.sample([True,False])
        self.assertEqual(self.state["strike_precontact_displaced"][0], 1)
        self.assertEqual(self.state["fast_strike"][0], 0)

    def test_first_sample_contact_without_speed_history_fails_closed(self):
        self.state = initial_strike_state(np.zeros(2), np.zeros((2,3)))
        self.mouth[0,0], self.top[0,0] = 1.0, 0.1
        self.sample([True,False])
        self.assertEqual(self.state["fast_strike"][0], 0)

    def test_nonfinite_head_state_is_latched_invalid(self):
        self.strike()
        self.mouth[0,0] = float("nan")
        self.sample([False,False])
        self.mouth[0,0] = 0.006
        self.sample([False,False])
        self.assertEqual(self.state["strike_invalid"][0], 1)
        self.assertEqual(self.state["fast_strike"][0], 0)

    def test_constant_trunk_frame_head_position_is_not_head_strike(self):
        self.sample([True,False])  # rigid body motion removed by caller's trunk frame
        self.top[0,0] = 0.04
        self.sample([False,False])
        self.assertEqual(self.state["fast_strike"][0], 0)

    def test_reset_and_invalid_dt(self):
        self.strike()
        reset = initial_strike_state(np.zeros(2), np.zeros((2,3)))
        self.assertEqual(reset["fast_strike"].sum(), 0)
        with self.assertRaises(ValueError):
            update_strike(reset,self.mouth,self.direction,self.top,np.array([False,False]),0,np)


def episode(seed, success=0, **changes):
    row = {key:0.0 for key in METRICS}
    row.update(seed=seed, success=success, fast_strike=success,
               strike_separation_delay_s=0.08 if success else -1.0)
    row.update(changes)
    return row


class EvaluationTests(unittest.TestCase):
    def record(self, iteration, successes, **changes):
        rows = [episode(i, int(i < successes), **changes) for i in range(4)]
        return {"checkpoint_iteration":iteration, "summary":summarize(rows, range(4))}

    def test_best_is_success_rate_not_latest_or_total_reward(self):
        records = [self.record(0,0), self.record(100,3), self.record(1999,1)]
        best, role = select_best(records)
        self.assertEqual((best["checkpoint_iteration"],role),(100,"best"))

    def test_tiebreak_uses_failure_union_then_motion_and_earlier_iteration(self):
        a = self.record(200,2)
        rows = [episode(i,int(i<2),robot_fall=int(i==3),base_failure=int(i==3)) for i in range(4)]
        b = {"checkpoint_iteration":100,"summary":summarize(rows,range(4))}
        self.assertEqual(b["summary"]["failure_rate"],0.25)
        self.assertEqual(select_best([b,a])[0]["checkpoint_iteration"],200)
        self.assertEqual(select_best([a,self.record(100,2)])[0]["checkpoint_iteration"],100)
        self.assertEqual(select_best([a,self.record(50,2,base_max_xy_m=0.007)])[0]["checkpoint_iteration"],200)

    def test_no_success_is_labeled_honestly(self):
        self.assertEqual(select_best([self.record(100,0)])[1],"best_available_no_success")

    def test_partial_checkpoint_cannot_win(self):
        partial = {"checkpoint_iteration":10,"summary":summarize([episode(0,1)],range(4))}
        self.assertEqual(select_best([partial,self.record(100,0)])[0]["checkpoint_iteration"],100)
        with self.assertRaises(ValueError):
            select_best([partial])

    def test_duplicate_seeds_nonfinite_and_false_success_are_rejected(self):
        for rows in [[episode(0),episode(0)], [episode(0,max_head_force_n=float("nan"))],
                     [episode(0,1,fast_strike=0)], [episode(0,1,wrong_contact=1)]]:
            with self.assertRaises(ValueError):
                summarize(rows,range(4))

    def test_numeric_discovery_and_interrupted_progression(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"evidence") as temp:
            root = Path(temp)
            for iteration in [1000,0,100,1999]:
                (root/f"model_{iteration}.pt").touch()
            (root/"model_latest.pt").touch()
            (root/"model_2.pt").mkdir()  # a directory is not a checkpoint
            paths = discover_checkpoints(root)
            self.assertEqual([checkpoint_iteration(p) for p in paths],[0,100,1000,1999])
            self.assertEqual([(role,p.name) for role,p in progression(paths)],
                [("early","model_0.pt"),("middle","model_1000.pt"),("late_available","model_1999.pt")])
            self.assertEqual(progression(paths[:1])[0][0],"only_available")
            self.assertEqual(progression(paths[:2])[-1][0],"late_available")

    def test_checkpoint_numbered_names_and_separate_best_file(self):
        names = [video_filename("Mjlab-JengaHead-Vertical3-MicroDuck",1000,role,42)
                 for role in ("middle","best","best_available_no_success")]
        self.assertEqual(len(set(names)),3)
        self.assertTrue(all("checkpoint_01000" in name for name in names))
        with self.assertRaises(ValueError):
            video_filename("other",0,"early",42)

    def test_manifest_atomic_write_rejects_nan_without_losing_existing(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"evidence") as temp:
            path = Path(temp)/"manifest.json"
            save_json(path,{"complete":False})
            with self.assertRaises(ValueError):
                save_json(path,{"metric":float("nan")})
            self.assertFalse(json.loads(path.read_text())["complete"])


class RecordingTests(unittest.TestCase):
    def test_initial_terminal_frames_and_failed_episode_are_retained(self):
        env = NS(step_dt=0.02,max_episode_length=3,reset_terminated=[False],reset_time_outs=[False])
        frames, current = [], [0]
        env.reset = lambda seed:current.__setitem__(0,0)
        env.render = lambda:current[0]
        def step(action):
            current[0] += 1
            row = episode(42,robot_fall=int(current[0]==3))
            env._jenga = {key:np.array([row[key]]) for key in METRICS}
            env.reset_terminated[0] = current[0] == 3
            return {},[0.0],[current[0]==3],{}
        wrapped = NS(get_observations=lambda:{"actor":np.zeros((1,61))},step=step)
        row, trace = recorder.run_episode(env,wrapped,lambda obs:NS(clone=lambda:None),42,
            NS(inference_mode=nullcontext),NS(append_data=frames.append))
        self.assertEqual(frames,[0,1,2,3])
        self.assertEqual(row["robot_fall"],1)
        self.assertEqual(row["success"],0)
        self.assertEqual(len(trace),3)
        self.assertAlmostEqual(row["duration_s"],0.06)

    def test_episode_must_reach_observed_terminal_state(self):
        env = NS(step_dt=0.02,max_episode_length=1,reset=lambda seed:None,
                 _jenga={key:np.zeros(1) for key in METRICS})
        wrapped = NS(get_observations=lambda:{"actor":np.zeros((1,61))},
                     step=lambda action:({},[0.0],[False],{}))
        with self.assertRaises(RuntimeError):
            recorder.run_episode(env,wrapped,lambda obs:NS(clone=lambda:None),42,NS(inference_mode=nullcontext))


if __name__ == "__main__":
    unittest.main()
