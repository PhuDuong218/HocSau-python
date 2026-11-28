"""Unit tests for T06 MNIST Summation dataset, CNN+LSTM model, and L1Loss step."""

import unittest

import torch
from torch import nn

from src.lstm_dataset import MnistSummationDataset
from src.lstm_models import MnistCnnLstm, count_parameters


class TestMnistCnnLstm(unittest.TestCase):
    def test_sequence_dataset_shapes_padding_and_reproducibility(self):
        # 20 synthetic grayscale images in [0, 1]
        imgs = torch.rand(20, 1, 28, 28, dtype=torch.float32) + 0.1
        imgs = imgs.clamp(0.0, 1.0)
        labels = torch.arange(20, dtype=torch.int64) % 10

        ds1 = MnistSummationDataset(imgs, labels, num_sequences=12, max_to_add=8, seed=2018)
        ds2 = MnistSummationDataset(imgs, labels, num_sequences=12, max_to_add=8, seed=2018)

        self.assertEqual(len(ds1), 12)
        seq, target = ds1[0]
        seq2, target2 = ds2[0]
        self.assertEqual(tuple(seq.shape), (8, 1, 28, 28))
        self.assertEqual(seq.dtype, torch.float32)
        self.assertEqual(target.dtype, torch.float32)
        self.assertTrue(torch.equal(seq, seq2))
        self.assertTrue(torch.equal(target, target2))

        # Verify padded slots are exact zeros and active slots equal picked labels sum
        n = int(ds1.lengths[0].item())
        self.assertGreaterEqual(n, 1)
        self.assertLessEqual(n, 8)
        if n < 8:
            self.assertEqual(seq[n:].abs().sum().item(), 0.0)
        expected_sum = float(labels[ds1.indices[0, :n]].sum().item())
        self.assertAlmostEqual(target.item(), expected_sum, places=5)

    def test_model_supports_seq_len_7_and_8_and_param_counts(self):
        model = MnistCnnLstm()
        counts = count_parameters(model)
        self.assertEqual(counts["keras_equivalent_params"], 1747073)
        self.assertEqual(counts["pytorch_params"], 1748097)

        for seq_len in (7, 8):
            x = torch.rand(4, seq_len, 1, 28, 28)
            out = model(x)
            self.assertEqual(tuple(out.shape), (4, 1))

    def test_l1_loss_backward_pass(self):
        model = MnistCnnLstm()
        x = torch.rand(4, 8, 1, 28, 28)
        y = torch.tensor([12.0, 25.0, 5.0, 31.0], dtype=torch.float32).view(-1, 1)
        criterion = nn.L1Loss()

        loss = criterion(model(x), y)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main()
