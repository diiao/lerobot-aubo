from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.relative_xyz import (
    RAW_TCP_KEY, compute_relative_xyz_stats, decode_relative_actions,
    encode_relative_actions, shift_xyz,
)


def config():
    return SmolVLAConfig(
        device="cpu", action_representation="relative_tcp_xyz", n_action_steps=1,
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(13,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
        relative_action_stats={"mean": [.2]*8, "std": [.4]*8, "min": [-1.]*8, "max": [1.]*8},
    )


def test_nonzero_distinct_batch_origins_whole_chunk_and_only_xyz():
    a = torch.arange(2*50*8, dtype=torch.float64).reshape(2, 50, 8)/1000
    origin = torch.tensor([[.4, -.2, .1], [-.3, .7, -.9]], dtype=torch.float64)
    original = a.clone()
    d = shift_xyz(a, origin)
    torch.testing.assert_close(d[..., 1:4], a[..., 1:4] - origin[:, None])
    assert torch.equal(d[..., [0,4,5,6,7]], a[..., [0,4,5,6,7]])
    torch.testing.assert_close(shift_xyz(d, origin, inverse=True), a)
    assert torch.equal(a, original)
    # Wrong/missing sample axes must fail instead of broadcasting another sample's TCP.
    with pytest.raises(ValueError):
        shift_xyz(a, origin[:1])


def test_stats_exclude_validation_and_terminal_padding_and_keep_other_scales():
    s = np.zeros((6,13), dtype=np.float32)
    s[:,6:9] = np.arange(6)[:,None]
    a = np.arange(48, dtype=np.float32).reshape(6,8)
    ep = np.array([0,0,0,1,1,10]); f = np.array([0,1,2,0,1,0])
    stats, info = compute_relative_xyz_stats(s,a,ep,f,train_episodes=[0,1],chunk_size=3)
    expected = np.stack([a[t+k,1:4]-s[t,6:9] for start,end in [(0,3),(3,5)]
                         for t in range(start,end) for k in range(min(3,end-t))]).astype(np.float64)
    assert info['valid_xyz_pairs'] == 9
    np.testing.assert_array_equal(stats['action']['mean'][1:4], expected.mean(0).astype(np.float32))
    np.testing.assert_array_equal(stats['action']['std'][1:4], expected.std(0).astype(np.float32))
    np.testing.assert_array_equal(stats['action']['mean'][[0,4,5,6,7]], a[:5].astype(np.float64).mean(0).astype(np.float32)[[0,4,5,6,7]])
    s[-1] = np.nan; a[-1] = np.nan
    other, _ = compute_relative_xyz_stats(s,a,ep,f,train_episodes=[0,1],chunk_size=3)
    for key in stats:
        for moment in stats[key]:
            np.testing.assert_array_equal(stats[key][moment], other[key][moment])


def test_saved_processor_config_and_absolute_output_roundtrip(tmp_path, monkeypatch):
    # Tokenizer is irrelevant to XYZ: bypass only language, with no downloads.
    from lerobot.policies.smolvla import processor_smolvla as module
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch, policy_action_to_transition, transition_to_policy_action,
    )
    monkeypatch.setattr(module, "TokenizerProcessorStep", lambda **kw: module.SmolVLANewLineProcessor())
    cfg = config()
    stats = {"observation.state": {"mean": torch.ones(13), "std": torch.ones(13)*2},
             "action": {k: torch.tensor(v) for k,v in cfg.relative_action_stats.items()}}
    pre, post = module.make_smolvla_pre_post_processors(cfg, stats)
    raw = {"observation.state": torch.arange(26).reshape(2,13).float()/20,
           "action": torch.arange(48).reshape(2,3,8).float()/50}
    cooked = pre(raw)
    assert torch.equal(cooked['action'], raw['action']), "must not normalize actions twice"
    assert torch.equal(cooked[RAW_TCP_KEY], raw['observation.state'][:,6:9])
    assert not torch.equal(cooked['observation.state'], raw['observation.state'])
    latent = encode_relative_actions(cooked['action'], cooked[RAW_TCP_KEY], cfg.relative_action_stats)
    before = post(decode_relative_actions(latent, cooked[RAW_TCP_KEY], cfg.relative_action_stats))
    torch.testing.assert_close(before, raw['action'])
    cfg.save_pretrained(tmp_path); pre.save_pretrained(tmp_path); post.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    loaded.validate_action_representation()
    assert loaded.action_representation == cfg.action_representation
    assert loaded.relative_action_stats == cfg.relative_action_stats
    pre2 = PolicyProcessorPipeline.from_pretrained(tmp_path, 'policy_preprocessor.json',
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post2 = PolicyProcessorPipeline.from_pretrained(tmp_path, 'policy_postprocessor.json',
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    cooked2 = pre2(raw)
    after = post2(decode_relative_actions(latent,cooked2[RAW_TCP_KEY],loaded.relative_action_stats))
    assert torch.equal(before,after)
    # A fresh process must resolve the new processor without prior registration.
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c", """
import sys
from lerobot.processor import PolicyProcessorPipeline
pipeline = PolicyProcessorPipeline.from_pretrained(sys.argv[1], 'policy_preprocessor.json')
assert any(type(step).__name__ == 'SmolVLARawTCPProcessor' for step in pipeline.steps)
""", str(tmp_path)], check=True, capture_output=True, text=True)
    # The unbatched adapter path preserves exactly the same physical origin.
    single = pre2({'observation.state': raw['observation.state'][1]})
    assert torch.equal(single[RAW_TCP_KEY], raw['observation.state'][1:2,6:9])
    assert SmolVLAConfig().action_representation == 'absolute'
    # A legacy preprocessor must not silently run a relative checkpoint.
    with pytest.raises(KeyError):
        decode_relative_actions(latent, raw[RAW_TCP_KEY], loaded.relative_action_stats)


def test_policy_entrypoints_convert_before_padding_and_return_absolute():
    pytest.importorskip('transformers')
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    cfg = config()
    raw = torch.arange(48).reshape(2,3,8).float()/50
    origin = torch.tensor([[.4,-.2,.1],[-.3,.7,-.9]])
    batch = {'action':raw, RAW_TCP_KEY:origin,
             'observation.language.tokens':None, 'observation.language.attention_mask':None}
    fake = SimpleNamespace(config=cfg, _queues={}, prepare_images=lambda b:(None,None), prepare_state=lambda b:None)
    padded = SmolVLAPolicy.prepare_action(fake,batch)
    assert padded.shape == (2,3,32) and torch.count_nonzero(padded[...,8:]) == 0
    fake.model = SimpleNamespace(sample_actions=lambda *args,**kw:padded)
    result = SmolVLAPolicy._get_action_chunk(fake,batch)
    torch.testing.assert_close(result,raw)


@pytest.mark.parametrize('field,value', [('action_representation','typo'), ('relative_action_stats',None),
    ('adapt_to_pi_aloha',True), ('n_obs_steps',2)])
def test_incompatible_relative_configuration_rejected(field,value):
    cfg=config(); setattr(cfg,field,value)
    with pytest.raises(ValueError): cfg.validate_action_representation()
