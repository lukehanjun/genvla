from dexmimicgen_pi0_bridge import ACTION_COMPONENTS
from dexmimicgen_pi0_bridge import action_20d_from_action_dict
from dexmimicgen_pi0_bridge import convert_action_20d_to_14d
import numpy as np


class _ArrayGroup(dict):
    pass


def test_action_20d_from_action_dict_uses_canonical_hand_component_order():
    group = _ArrayGroup(
        right_rel_pos=np.asarray([[1, 2, 3]], dtype=np.float32),
        right_rel_rot_6d=np.asarray([[4, 5, 6, 7, 8, 9]], dtype=np.float32),
        right_gripper=np.asarray([[10]], dtype=np.float32),
        left_rel_pos=np.asarray([[11, 12, 13]], dtype=np.float32),
        left_rel_rot_6d=np.asarray([[14, 15, 16, 17, 18, 19]], dtype=np.float32),
        left_gripper=np.asarray([[20]], dtype=np.float32),
    )

    action = action_20d_from_action_dict(group, 0)

    np.testing.assert_array_equal(action, np.arange(1, 21, dtype=np.float32))
    np.testing.assert_array_equal(action[ACTION_COMPONENTS["right"]["position"]], [1, 2, 3])
    np.testing.assert_array_equal(action[ACTION_COMPONENTS["right"]["rotation"]], [4, 5, 6, 7, 8, 9])
    np.testing.assert_array_equal(action[ACTION_COMPONENTS["right"]["gripper"]], [10])
    np.testing.assert_array_equal(action[ACTION_COMPONENTS["left"]["position"]], [11, 12, 13])
    np.testing.assert_array_equal(action[ACTION_COMPONENTS["left"]["rotation"]], [14, 15, 16, 17, 18, 19])
    np.testing.assert_array_equal(action[ACTION_COMPONENTS["left"]["gripper"]], [20])


def test_convert_action_20d_to_14d_preserves_positions_and_grippers():
    action = np.zeros(20, dtype=np.float32)
    action[0:3] = [0.1, 0.2, 0.3]
    action[3:9] = [1, 0, 0, 0, 1, 0]
    action[9] = -1
    action[10:13] = [-0.1, -0.2, -0.3]
    action[13:19] = [1, 0, 0, 0, 1, 0]
    action[19] = 1

    converted = convert_action_20d_to_14d(action)

    np.testing.assert_allclose(converted[0:3], action[0:3])
    np.testing.assert_allclose(converted[6], -1)
    np.testing.assert_allclose(converted[7:10], action[10:13])
    np.testing.assert_allclose(converted[13], 1)
