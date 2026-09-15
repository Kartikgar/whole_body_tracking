"""CPU tests of action methods with a minimal articulation, without starting Kit."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
import torch

PKG = Path(__file__).resolve().parents[1] / 'source/whole_body_tracking/whole_body_tracking'


class FakeBase:
    def __init__(self, cfg, env):
        self.cfg, self._env, self.device, self.num_envs = cfg, env, 'cpu', env.num_envs
        self._asset = env.asset
        self._num_joints = 29
        self._joint_ids = slice(None)
        self._force_body_ids = [0]
        self._joint_scale = 0.5
        self._offset = torch.ones(env.num_envs, 29)
        self._joint_clip = None
        self._joint_position_targets = torch.zeros(env.num_envs, 29)
        self._motion_command = NS(has_joint_action=True, joint_action=torch.ones(env.num_envs, 29))


def load_classes(path, names, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


CLASSES = load_classes(PKG / 'tasks/tracking/mdp/pelvis_wrench.py',
                       {'DeltaPelvisWrenchAction', 'ExternalDeltaPelvisWrenchAction'},
                       dict(torch=torch, math=math, DeltaComForceAction=FakeBase))


class WrenchTests(unittest.TestCase):
    def make(self, external=False, **overrides):
        cfg = NS(force_scale=300., torque_scale=60., action_clip=1., force_body_name='pelvis',
                 force_clip=None, external_action_buffer_name='delta_external_actions')
        cfg.__dict__.update(overrides)
        env = NS(num_envs=2, asset=NS(set_joint_position_target=Mock(), set_external_force_and_torque=Mock()))
        cls = CLASSES['ExternalDeltaPelvisWrenchAction' if external else 'DeltaPelvisWrenchAction']
        return cls(cfg, env)

    def test_scaling_clipping_and_joint_replay(self):
        term = self.make()
        a = torch.tensor([[2., -.5, 0., -2., .5, 0.]]).repeat(2, 1)
        term.process_actions(a)
        torch.testing.assert_close(term._wrench[0], torch.tensor([300., -150., 0., -60., 30., 0.]))
        torch.testing.assert_close(term._raw_actions, a)
        torch.testing.assert_close(term._joint_position_targets, torch.full((2, 29), 1.5))
        term.apply_actions()
        kw = term._asset.set_external_force_and_torque.call_args.kwargs
        torch.testing.assert_close(kw['forces'], term._wrench[:, None, :3])
        torch.testing.assert_close(kw['torques'], term._wrench[:, None, 3:])
        self.assertEqual(kw['body_ids'], [0])
        stats = term.consume_applied_wrench_log_stats()
        self.assertEqual(stats['saturation_Fx'], 1.)
        self.assertEqual(stats['saturation_Fy'], 0.)
        self.assertEqual(stats['saturation_Tx'], 1.)
        self.assertEqual(term.consume_applied_wrench_log_stats(), {})

    def test_zero_wrench_and_external_joint_targets(self):
        term = self.make(external=True)
        self.assertEqual(term.action_dim, 29)
        term._env.delta_external_actions = torch.zeros(2, 6)
        term.process_actions(torch.full((2, 29), 2.))
        torch.testing.assert_close(term._joint_position_targets, torch.full((2, 29), 2.))
        self.assertEqual(term._wrench.count_nonzero(), 0)
        self.assertEqual(term._scale.shape, (2, 29))

    def test_partial_reset(self):
        term = self.make()
        term.process_actions(torch.ones(2, 6))
        term._env.delta_external_actions = torch.ones(2, 6)
        term._env.delta_base_actions = torch.ones(2, 29)
        term.reset([0])
        self.assertEqual(term._wrench[0].count_nonzero(), 0)
        self.assertEqual(term._wrench[1].count_nonzero(), 6)
        self.assertEqual(term._env.delta_external_actions[0].count_nonzero(), 0)
        self.assertEqual(term._env.delta_external_actions[1].count_nonzero(), 6)
        self.assertEqual(term._asset.set_external_force_and_torque.call_args.kwargs['env_ids'], [0])

    def test_invalid_shapes_and_scales(self):
        for overrides in ({'force_scale': 0}, {'torque_scale': float('nan')}, {'action_clip': -1}):
            with self.assertRaises(ValueError):
                self.make(**overrides)
        with self.assertRaises(ValueError):
            self.make().process_actions(torch.zeros(2, 3))
        with self.assertRaises(RuntimeError):
            self.make(external=True).process_actions(torch.zeros(2, 29))

    def test_reset_buffers_do_not_alias_recorded_policy_outputs(self):
        tree = ast.parse((PKG / 'utils/my_on_policy_runner.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotionOnPolicyRunner')
        for method_name, buffer_name, dim in (
            ('_set_delta_action_buffer', 'delta_external_actions', 6),
            ('_set_delta_base_action_buffer', 'delta_base_actions', 29),
        ):
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
            ns = {'torch': torch}
            exec(compile(ast.Module(body=[method], type_ignores=[]), 'buffers', 'exec'), ns)
            runner = NS(_wrench_metadata=lambda: {}, env=NS(unwrapped=NS(), device='cpu'),
                        delta_policy_action_buffer_name='delta_external_actions',
                        delta_policy_base_action_buffer_name='delta_base_actions')
            action = torch.ones(2, dim)
            ns[method_name](runner, action)
            getattr(runner.env.unwrapped, buffer_name)[0] = 0
            torch.testing.assert_close(action, torch.ones(2, dim))

    def test_checkpoint_contract(self):
        tree = ast.parse((PKG / 'utils/my_on_policy_runner.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotionOnPolicyRunner')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_validate_wrench_checkpoint')
        ns = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'contract', 'exec'), ns)
        expected = dict(role='finetune', contract=self.make().wrench_contract(), observation_layout=['state', 'action'])
        runner = NS(_wrench_metadata=lambda: expected)
        check = lambda loaded, **kw: ns['_validate_wrench_checkpoint'](runner, loaded, **kw)
        check({})  # Initial nominal base checkpoint.
        saved = dict(expected, role='open_loop')
        check({'infos': {'pelvis_wrench': saved}}, frozen=True)
        with self.assertRaises(ValueError):
            check({}, frozen=True)
        bad = dict(saved, contract=dict(saved['contract'], torque_scale=40.))
        with self.assertRaises(ValueError):
            check({'infos': {'pelvis_wrench': bad}}, frozen=True)
        with self.assertRaises(ValueError):
            check({'infos': {'pelvis_wrench': dict(saved, observation_layout=['action', 'state'])}}, frozen=True)


if __name__ == '__main__':
    unittest.main()
