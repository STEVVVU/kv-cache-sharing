import copy
import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from kvshare import Sharing, cka, calibrate, validate_map, unique_cache_bytes, trainable_consumers, distill_loss, pareto


class SharingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        config = Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8)
        config._attn_implementation = 'eager'
        self.model = Qwen3ForCausalLM(config).eval()
        self.ids = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])

    def test_cka_invariances_and_topology(self):
        x = torch.randn(20, 8)
        q = torch.linalg.qr(torch.randn(8, 8)).Q
        self.assertAlmostEqual(cka(x, 3*x @ q + 7), 1.0, places=6)
        self.assertEqual(cka(torch.ones_like(x), x), 0)
        for mapping in ({2: 0}, {1: 0, 2: 1}, {4: 3}, {-1: -2}):
            with self.assertRaises(ValueError):
                validate_map(mapping, 4)
        mapping, history = calibrate(self.model, {'input_ids': self.ids}, 2)
        validate_map(mapping, 4)
        self.assertEqual(len(history), 2)

    def test_no_share_parity(self):
        with torch.no_grad():
            expected = self.model(self.ids).logits
            route = Sharing(self.model)
            actual = self.model(self.ids).logits
            route.close()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cache_alias_and_prefill_decode(self):
        with torch.no_grad():
            baseline = self.model(self.ids, use_cache=True)
            route = Sharing(self.model, {1: 0, 3: 2})
            calls = []
            hooks = [getattr(self.model.model.layers[i].self_attn, name).register_forward_hook(lambda *_: calls.append(1))
                     for i in (1, 3) for name in ('k_proj', 'v_proj')]
            full = self.model(self.ids, use_cache=True)
            self.assertIs(full.past_key_values.layers[0], full.past_key_values.layers[1])
            self.assertEqual(unique_cache_bytes(full.past_key_values)*2, unique_cache_bytes(baseline.past_key_values))
            prefix = self.model(self.ids[:, :3], use_cache=True)
            decode = self.model(self.ids[:, 3:], past_key_values=prefix.past_key_values, use_cache=True)
            torch.testing.assert_close(decode.logits[:, -1], full.logits[:, -1], rtol=1e-5, atol=1e-6)
            self.assertEqual(decode.past_key_values.get_seq_length(), 4)
            separate = self.model(self.ids[:1], use_cache=True)
            torch.testing.assert_close(separate.logits, full.logits[:1], rtol=1e-5, atol=1e-6)
            changed = self.ids.clone(); changed[:, -1] = 9
            future = self.model(changed).logits
            torch.testing.assert_close(future[:, :-1], full.logits[:, :-1], rtol=1e-5, atol=1e-6)
            self.assertEqual(calls, [])
            for hook in hooks: hook.remove()
            route.close()

    def test_distillation_freeze_and_update(self):
        teacher = copy.deepcopy(self.model).requires_grad_(False)
        mapping = {1: 0}
        route = Sharing(self.model, mapping)
        parameters = trainable_consumers(self.model, mapping)
        frozen = {n: p.clone() for n, p in self.model.named_parameters() if not p.requires_grad}
        before = parameters[0].clone()
        optimizer = torch.optim.SGD(parameters, lr=0.05)
        with torch.no_grad(): target = teacher(self.ids, output_hidden_states=True)
        prediction = self.model(self.ids, output_hidden_states=True, use_cache=False)
        loss = distill_loss(prediction, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward(); optimizer.step()
        self.assertGreater(float((before - parameters[0]).abs().max().detach()), 0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        for name, param in self.model.named_parameters():
            if name in frozen: torch.testing.assert_close(param, frozen[name], rtol=0, atol=0)
            if param.requires_grad:
                self.assertTrue(name.startswith('model.layers.1.'))
                self.assertFalse(any(x in name for x in ('k_proj', 'v_proj', 'o_proj', 'k_norm')))
        route.close()

    def test_left_padding_and_cache_isolation(self):
        route = Sharing(self.model, {1: 0})
        ids = self.ids.clone(); ids[0, :2] = 0
        mask = torch.ones_like(ids); mask[0, :2] = 0
        with torch.no_grad():
            batch = self.model(ids, attention_mask=mask, use_cache=True)
            single = self.model(ids[:1, 2:], use_cache=True)
            torch.testing.assert_close(batch.logits[0, 2:], single.logits[0], rtol=1e-5, atol=1e-6)
            self.assertIsNot(batch.past_key_values.layers[0], single.past_key_values.layers[0])
        route.close()

    def test_window_and_pareto(self):
        with self.assertRaises(ValueError): Sharing(self.model, {1: 0}, 0)
        route = Sharing(self.model, {1: 0}, 2)
        with torch.no_grad():
            full = self.model(self.ids, use_cache=True)
            prefix = self.model(self.ids[:, :3], use_cache=True)
            decode = self.model(self.ids[:, 3:], past_key_values=prefix.past_key_values, use_cache=True)
            torch.testing.assert_close(full.logits[:, -1], decode.logits[:, -1], rtol=1e-5, atol=1e-6)
        route.close()
        rows = pareto([dict(unique_cache_bytes=10, teacher_kl=.1), dict(unique_cache_bytes=20, teacher_kl=.2)])
        self.assertTrue(rows[0]['pareto']); self.assertFalse(rows[1]['pareto'])


if __name__ == '__main__': unittest.main()
