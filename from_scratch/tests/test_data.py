"""Prompt-sampling tests use synthetic masks and never download the dataset."""

import numpy as np

from tinysam.data import sample_prompt


def rectangle():
    mask = np.zeros((24, 30), dtype=bool)
    mask[6:18, 8:22] = True
    return mask


def test_point_prompts_respect_labels_and_padding():
    mask = rectangle()
    coords, labels = sample_prompt(mask, np.random.default_rng(3), box_probability=0)
    assert labels[0] == 1
    assert mask[int(coords[0, 1]), int(coords[0, 0])]
    for point, label in zip(coords[1:], labels[1:]):
        if label == 1:
            assert mask[int(point[1]), int(point[0])]
        elif label == 0:
            assert not mask[int(point[1]), int(point[0])]
        else:
            assert label == -1


def test_box_prompts_have_ordered_corner_labels_and_padding():
    mask = rectangle()
    coords, labels = sample_prompt(mask, np.random.default_rng(10), box_probability=1)
    assert labels.tolist() == [2, 3, -1]
    assert coords[0, 0] <= coords[1, 0]
    assert coords[0, 1] <= coords[1, 1]
    # Each box edge may move inward/outward by at most ten percent of a side.
    assert coords[0, 0] <= 8 + 1.4 and coords[1, 0] >= 21 - 1.4
    assert coords[0, 1] <= 6 + 1.2 and coords[1, 1] >= 17 - 1.2
    assert np.all(coords[2] == 0)


def test_prompt_sampler_is_seedable():
    mask = rectangle()
    first = sample_prompt(mask, np.random.default_rng(55))
    second = sample_prompt(mask, np.random.default_rng(55))
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
