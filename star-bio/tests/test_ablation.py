"""CPU component checks only; no training experiments or optimizer updates."""
import io
from pathlib import Path
import shlex
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import apply_spectral_drift, SpectraDataset
from metadata import model_from_config
from metrics import evaluate

ROOT = Path(__file__).resolve().parents[2]


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, **kwargs):
        return dict(n_classes=7, n_points=129, tsai_root=str(ROOT), patch_len=16,
                    stride=8, layers=1, heads=4, d_model=16, d_ff=32, dropout=0., **kwargs)

    def test_pooling_revin_gradients_and_checkpoint_roundtrip(self):
        for pooling in ('mean', 'segments', 'attention'):
            for revin in (True, False):
                with self.subTest(pooling=pooling, revin=revin):
                    cfg = self.config(pooling=pooling, revin=revin, pool_segments=4)
                    model = model_from_config(cfg).eval()
                    x = torch.randn(2, 3, 129)
                    output = model(x)
                    self.assertEqual(output.shape, (2, 7))
                    output.square().mean().backward()
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            self.assertIsNotNone(param.grad, name)
                            self.assertTrue(torch.isfinite(param.grad).all(), name)
                    stream = io.BytesIO()
                    torch.save({'config': cfg, 'model': model.state_dict()}, stream)
                    stream.seek(0)
                    ck = torch.load(stream, weights_only=True)
                    restored = model_from_config(ck['config']).eval()
                    restored.load_state_dict(ck['model'])
                    torch.testing.assert_close(output, restored(x))

    def test_no_revin_bypasses_normalizer_and_old_config_defaults(self):
        model = model_from_config(self.config(revin=False)).eval()
        with patch.object(model.revin_layer, 'forward', side_effect=AssertionError('must bypass')):
            model(torch.randn(2, 3, 129))
        # Older encoder checkpoints omit the new fields and remain mean + RevIN.
        older = model_from_config(self.config()).eval()
        explicit = model_from_config(self.config(revin=True, pooling='mean')).eval()
        explicit.load_state_dict(older.state_dict(), strict=True)
        x = torch.randn(2, 3, 129)
        torch.testing.assert_close(older(x), explicit(x))

    def test_spectral_drift_derivative_units_and_snv(self):
        n = 1105
        # Known linear spectrum: derivative after gain and slope is analytical.
        x = torch.zeros(3, n)
        x[0] = 1 + .002 * torch.arange(n)
        out = apply_spectral_drift(x, 1.05, .03, .025)
        expected = .002 * 1.05 + 2 * .025 / (n - 1)
        torch.testing.assert_close(out[1], torch.full((n,), expected), atol=2e-6, rtol=0.)
        torch.testing.assert_close(out[2], (out[0]-out[0].mean())/(out[0].std(unbiased=False)+1e-6))
        self.assertTrue(torch.isfinite(out).all())
        with self.assertRaises(ValueError):
            SpectraDataset('unused.h5', [], augmentation_strength=-1)

    def test_evaluation_accepts_classifier_logits(self):
        model = model_from_config(self.config()).eval()
        loader = DataLoader(TensorDataset(torch.randn(7, 3, 129), torch.arange(7), torch.zeros(7)), batch_size=3)
        result = evaluate(model, loader, 'cpu', [str(i) for i in range(7)])
        self.assertEqual(sum(map(sum, result['cm'])), 7)

    def test_launcher_plan_only(self):
        import os
        before = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT)
        text = subprocess.check_output(['bash', 'star-bio/launch_8_compare.sh'], cwd=ROOT,
                                      env={**os.environ, 'DRY_RUN': '1', 'MODE': 'formal'}, text=True)
        lines = [shlex.split(line) for line in text.splitlines()]
        self.assertEqual(len(lines), 8)
        def opt(cmd, key): return cmd[cmd.index(key)+1]
        self.assertEqual([opt(cmd, '--gpu-id') for cmd in lines], list(map(str, range(8))))
        self.assertEqual([opt(cmd, '--batch-size') for cmd in lines], ['4096','1024','256','64','256','256','256','256'])
        self.assertTrue(all(opt(cmd, '--d-model') == '128' for cmd in lines))
        self.assertIn('--no-revin', lines[4])
        self.assertEqual(opt(lines[5], '--pooling'), 'segments')
        self.assertEqual(opt(lines[6], '--pooling'), 'attention')
        self.assertIn('--batch-shift', lines[7])
        self.assertTrue(all('--no-batch-shift' in cmd for cmd in lines[:7]))
        self.assertEqual(before, subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT))


if __name__ == '__main__':
    unittest.main()
