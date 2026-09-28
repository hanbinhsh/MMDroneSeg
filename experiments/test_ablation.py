import numpy as np
import pytest
import torch

from experiments.ablation_suite import rare_ids_from_train, command_for, JOBS
from experiments.data import DroneData, targeted_crop_origin
from experiments.models import ABLATION_FLAGS, build_model, batch_needs
from experiments.prepare_semantic import DEST
from experiments.train import parser


def test_ablation_common_initialization_and_rng_match_full_reference():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    full = build_model('ours_no_text', False, 22)
    full_state = full.state_dict()
    rng = torch.get_rng_state().clone()
    for name, (detail, boundary) in ABLATION_FLAGS.items():
        torch.manual_seed(42)
        model = build_model(name, False, 22)
        assert torch.equal(torch.get_rng_state(), rng)
        assert (model.use_detail, model.use_boundary, model.use_text) == (detail, boundary, False)
        assert batch_needs(name) == {'detail': detail, 'text': False}
        assert hasattr(model, 'detail_gate') == detail
        assert hasattr(model, 'edge_head') == boundary
        for key, value in model.state_dict().items():
            assert torch.equal(value, full_state[key]), key
        del model


@pytest.mark.parametrize('position', [(0, 0), (32, 60), (18, 27)])
def test_targeted_crop_includes_tiny_objects_at_borders(position):
    label = np.zeros((33, 61), np.uint8)
    label[position] = 11
    label[1, 1] = 255
    for seed in range(20):
        top, left = targeted_crop_origin(label, 16, [11, 13], np.random.default_rng(seed))
        assert 0 <= top <= 17 and 0 <= left <= 45
        assert 11 in label[top:top+16, left:left+16]
    assert targeted_crop_origin(label, 16, [15], np.random.default_rng(0)) is None


def test_crop_policy_default_is_identical_and_targeting_is_reproducible():
    uniform = DroneData(DEST, crop=128, detail=True, limit=1)
    disabled = DroneData(DEST, crop=128, detail=True, limit=1, rare_class_ids=[11], class_aware_crop_prob=0)
    targeted = DroneData(DEST, crop=128, detail=True, limit=1, rare_class_ids=[11], class_aware_crop_prob=1)
    rgb = np.random.default_rng(8).integers(0, 256, (160, 200, 3), dtype=np.uint8)
    label = np.zeros((160, 200), np.uint8)
    label[-1, -1] = 11
    for dataset in (uniform, disabled, targeted):
        dataset.read = lambda _: (rgb.copy(), label.copy())
        dataset.epoch = 3
    a, b = uniform[0], disabled[0]
    for key in ('image', 'mask', 'detail'):
        assert torch.equal(a[key], b[key])
    a, b = targeted[0], targeted[0]
    assert 11 in a['mask']
    for key in ('image', 'mask', 'detail'):
        assert torch.equal(a[key], b[key])
    # Absent rare classes fall back exactly, including the augmentation draws.
    label[-1, -1] = 0
    a, b = uniform[0], targeted[0]
    for key in ('image', 'mask', 'detail'):
        assert torch.equal(a[key], b[key])


def test_sampling_cannot_be_enabled_for_validation_or_ignore_class():
    with pytest.raises(ValueError, match='training crops'):
        DroneData(DEST, 'val', class_aware_crop_prob=0.5, rare_class_ids=[11])
    with pytest.raises(ValueError, match='valid training IDs'):
        DroneData(DEST, rare_class_ids=[255])


def test_rare_classes_selected_only_from_training_and_commands_parse():
    audit = {'prepared_class_pixel_counts': {'train': [999, 1, 0], 'val': [0, 0, 1000]}}
    assert rare_ids_from_train(audit) == [1]
    audit['prepared_class_pixel_counts']['val'] = [999999, 999999, 1]
    assert rare_ids_from_train(audit) == [1]
    for job in JOBS:
        command = command_for(job, {'rare_class_ids': [11, 13]})
        args = parser().parse_args(command[5:])
        assert args.epochs == args.max_epochs == 50
        assert args.seed == 42 and args.horizon == 150 and args.checkpoint_every == 5
        assert args.class_aware_crop_prob == job[-1]
        assert not batch_needs(args.model)['text']
