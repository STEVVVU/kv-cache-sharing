"""Original KVAlign-style experiment, adapted to the pinned Transformers Qwen3 API."""
import argparse
import copy
import json
from pathlib import Path
import time
import types

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward

MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'


def cka(x, y):
    x, y = x.double(), y.double()
    x, y = x - x.mean(0), y - y.mean(0)
    # Gram form avoids a feature-by-feature matrix for wide KV heads.
    a, b = x @ x.T, y @ y.T
    denominator = a.norm() * b.norm()
    return float((a * b).sum() / denominator) if denominator > 0 else 0.0


def validate_map(mapping, layers):
    if any(not isinstance(c, int) or not isinstance(p, int) or c != p + 1 or not 0 <= p < c < layers for c, p in mapping.items()):
        raise ValueError('sharing requires adjacent consumer -> previous producer')
    if set(mapping) & set(mapping.values()) or len(set(mapping.values())) != len(mapping):
        raise ValueError('chained sharing and producer fan-out are forbidden')


class Sharing:
    """Per-model attention routing, preserving native cache and tensor storage aliases."""
    def __init__(self, model, mapping=None, window=None):
        self.model, self.mapping, self.window = model, dict(mapping or {}), window
        validate_map(self.mapping, len(model.model.layers))
        if window is not None and (not isinstance(window, int) or window < 1):
            raise ValueError('consumer window must be a positive integer')
        self.states = {}
        self.hook = model.register_forward_pre_hook(lambda *_: self.states.clear())
        self.originals = []
        for layer in model.model.layers:
            attn = layer.self_attn
            self.originals.append(attn.forward)
            attn.forward = types.MethodType(self._forward(), attn)

    def _forward(self):
        route = self

        def forward(attn, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
            shape = hidden_states.shape[:-1]
            heads = (*shape, -1, attn.head_dim)
            q = attn.q_norm(attn.q_proj(hidden_states).view(heads)).transpose(1, 2)
            cos, sin = position_embeddings
            i = attn.layer_idx
            if i in route.mapping:
                p = route.mapping[i]
                k, v = route.states[p]
                q, _ = apply_rotary_pos_emb(q, k[..., -shape[-1]:, :], cos, sin)
                if past_key_values is not None:
                    past_key_values.layers[i] = past_key_values.layers[p]
            else:
                k = attn.k_norm(attn.k_proj(hidden_states).view(heads)).transpose(1, 2)
                v = attn.v_proj(hidden_states).view(heads).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                if past_key_values is not None:
                    k, v = past_key_values.update(k, v, i)
            route.states[i] = (k, v)
            mask = attention_mask
            if route.window is not None and i in route.mapping:
                # Original ablation: mask consumer history without truncating producer storage.
                nq, nk = q.shape[-2], k.shape[-2]
                row = torch.arange(nk - nq, nk, device=q.device)[:, None]
                col = torch.arange(nk, device=q.device)[None, :]
                local = torch.zeros((nq, nk), device=q.device, dtype=q.dtype)
                local.masked_fill_(col < row - route.window + 1, torch.finfo(q.dtype).min)
                mask = local if mask is None else mask + local
            out, weights = eager_attention_forward(attn, q, k, v, mask, scaling=attn.scaling,
                                                  dropout=attn.attention_dropout if attn.training else 0.0)
            return attn.o_proj(out.reshape(*shape, -1).contiguous()), weights
        return forward

    def close(self):
        for layer, original in zip(self.model.model.layers, self.originals):
            layer.self_attn.forward = original
        self.hook.remove()
        self.states.clear()


def unique_cache_bytes(cache):
    storages = {}
    for layer in cache.layers:
        for tensor in (layer.keys, layer.values):
            if tensor is not None:
                storage = tensor.untyped_storage()
                storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def calibrate(model, inputs, count):
    if count < 0 or count > len(model.model.layers) // 2:
        raise ValueError('pair count outside topology capacity')
    mapping, history = {}, []
    route = Sharing(model)
    try:
        for _ in range(count):
            # Re-run current shared model after every choice to measure induced state drift.
            with torch.no_grad():
                model(**inputs, use_cache=False)
            scores = []
            used = set(mapping) | set(mapping.values())
            for c in range(1, len(model.model.layers)):
                p = c - 1
                if c in used or p in used:
                    continue
                values = []
                for x, y in zip(route.states[c], route.states[p]):
                    x = x.transpose(1, 2).reshape(-1, x.shape[1] * x.shape[-1])
                    y = y.transpose(1, 2).reshape(-1, y.shape[1] * y.shape[-1])
                    valid = inputs.get('attention_mask', torch.ones(x.shape[0], device=x.device)).reshape(-1).bool()
                    values.append(cka(x[valid], y[valid]))
                scores.append((min(values), c, p))
            score, c, p = max(scores)
            mapping[c] = p
            route.mapping = mapping.copy()
            history.append(dict(consumer=c, producer=p, cka=score))
        return mapping, history
    finally:
        route.close()


def trainable_consumers(model, mapping):
    for param in model.parameters():
        param.requires_grad_(False)
    for c in mapping:
        layer = model.model.layers[c]
        for module in (layer.self_attn.q_proj, layer.self_attn.q_norm, layer.input_layernorm,
                       layer.post_attention_layernorm, layer.mlp):
            for param in module.parameters():
                param.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def distill_loss(student, teacher, temperature=2.0, alpha=0.1, hidden_layer=-1, mask=None):
    if temperature <= 0 or alpha < 0:
        raise ValueError('invalid loss hyperparameters')
    s, t = student.logits.float() / temperature, teacher.logits.float() / temperature
    kl = F.kl_div(F.log_softmax(s, -1), F.softmax(t, -1), reduction='none').sum(-1)
    mse = (student.hidden_states[hidden_layer].float() - teacher.hidden_states[hidden_layer].float()).square().mean(-1)
    valid = torch.ones_like(kl) if mask is None else mask.to(kl)
    return ((temperature ** 2 * kl + alpha * mse) * valid).sum() / valid.sum().clamp_min(1)


def metrics(model, inputs, reference):
    with torch.no_grad():
        result = model(**inputs, use_cache=True)
    logits, teacher = result.logits.float(), reference.float()
    valid = inputs['attention_mask'].bool()
    kl = F.kl_div(logits.log_softmax(-1), teacher.softmax(-1), reduction='none').sum(-1)[valid].mean()
    nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), inputs['input_ids'][:, 1:].reshape(-1), reduction='none')
    nll = nll[(valid[:, 1:] & valid[:, :-1]).reshape(-1)].mean()
    return dict(teacher_kl=float(kl), next_token_nll=float(nll), token_agreement=float((logits.argmax(-1)[valid] == teacher.argmax(-1)[valid]).float().mean()),
                unique_cache_bytes=unique_cache_bytes(result.past_key_values))


def pareto(rows):
    for row in rows:
        row['pareto'] = not any(other['unique_cache_bytes'] <= row['unique_cache_bytes'] and other['teacher_kl'] <= row['teacher_kl']
                                and (other['unique_cache_bytes'] < row['unique_cache_bytes'] or other['teacher_kl'] < row['teacher_kl'])
                                for other in rows)
    return rows


def experiment(device, output, steps=4, pairs=2):
    started = time.monotonic()
    torch.manual_seed(7)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
             trust_remote_code=False, use_safetensors=True, dtype=dtype, attn_implementation='eager').to(device).eval()
    teacher.requires_grad_(False)
    student = copy.deepcopy(teacher)
    fixture = json.loads(Path(__file__).with_name('fixtures.json').read_text(encoding='utf-8'))
    def tokens(texts):
        return tokenizer(texts, padding=True, truncation=True, max_length=48, return_tensors='pt').to(device)
    calibration = tokens(fixture['calibration'])
    heldout = tokens(fixture['heldout'])
    with torch.no_grad():
        ref = teacher(**heldout).logits.detach()
        baseline_generation = teacher.generate(**tokens([fixture['generation']]), max_new_tokens=12, do_sample=False)
    mapping, history = calibrate(student, calibration, pairs)
    rows = [dict(policy='independent', **metrics(student, heldout, ref))]
    for name, policy, window in [('uniform', {2*i+1: 2*i for i in range(pairs)}, None), ('cka', mapping, None), ('cka-window-8', mapping, 8)]:
        route = Sharing(student, policy, window)
        try:
            rows.append(dict(policy=name, **metrics(student, heldout, ref)))
        finally:
            route.close()
    route = Sharing(student, mapping)
    parameters = trainable_consumers(student, mapping)
    if not parameters:
        raise ValueError('distillation requires at least one sharing pair')
    before = parameters[0].detach().clone()
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0)
    losses = []
    hidden_layer = min(max(mapping) + 1, len(student.model.layers) - 1)
    try:
        student.train()
        for step in range(steps):
            if time.monotonic() - started > 540:
                raise TimeoutError('bounded experiment exceeded 540 seconds')
            batch = tokens([fixture['train'][step % len(fixture['train'])]])
            with torch.no_grad():
                target = teacher(**batch, output_hidden_states=True, use_cache=False)
            optimizer.zero_grad(set_to_none=True)
            prediction = student(**batch, output_hidden_states=True, use_cache=False)
            loss = distill_loss(prediction, target, hidden_layer=hidden_layer, mask=batch['attention_mask'])
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite distillation loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            del prediction, target, loss
        student.eval()
        rows.append(dict(policy='cka-distilled', **metrics(student, heldout, ref)))
        with torch.no_grad():
            generated = student.generate(**tokens([fixture['generation']]), max_new_tokens=12, do_sample=False)
        delta = float((parameters[0].detach().float() - before.float()).abs().max())
        assert delta > 0 and all(p.grad is None for p in teacher.parameters())
        checkpoint = Path('checkpoints/consumer.safetensors')
        checkpoint.parent.mkdir(exist_ok=True)
        from safetensors.torch import save_file
        save_file({n: p.detach().contiguous().cpu() for n, p in student.named_parameters() if p.requires_grad}, str(checkpoint))
        report = dict(model=MODEL, revision=REVISION, device=device, dtype=str(dtype), mapping=mapping, calibration=history,
                      steps=steps, hidden_layer=hidden_layer, losses=losses, max_parameter_delta=delta, teacher_frozen=True,
                      trainable_parameters=sum(p.numel() for p in parameters), policies=pareto(rows),
                      teacher_generation=tokenizer.decode(baseline_generation[0], skip_special_tokens=True),
                      student_generation=tokenizer.decode(generated[0], skip_special_tokens=True),
                      elapsed_seconds=time.monotonic()-started,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        route.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['experiment'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', default='evidence/experiment.json')
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--pairs', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.steps <= 20 or not 1 <= args.pairs <= 4:
        parser.error('steps must be 1..20 and pairs 1..4 for the bounded experiment')
    experiment(args.device, args.output, args.steps, args.pairs)
