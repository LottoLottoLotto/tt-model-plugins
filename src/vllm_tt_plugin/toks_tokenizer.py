# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt encoding (text -> token ids) through toks, Actual Computer's tokenizer (exactly hf tokenizers' ids).

``VLLM_TT_TOKS``:
  ``1`` (default)  encode prompts with toks;
  ``0``            pure HF (nothing is patched);
  ``verify``       encode every eligible prompt with both, raise on any difference, return HF's result.

Only the string -> ids path of an HF fast tokenizer (``TokenizersBackend``) changes: ``encode(str)`` and
``__call__(str | list[str])`` with the arguments vLLM's renderer passes (``add_special_tokens``, ``truncation`` with
``max_length``, nothing else). The chat template, decoding, incremental detokenization, vocab lookups and every other
call stay HF's. A call with any argument outside that set (pairs, padding, tensors, offsets, overflow, special-token
masks, pretokenized input, token type ids, ``split_special_tokens``, ...) goes to HF unchanged, and so does a call
whose truncation would actually cut tokens: toks never approximates HF.

toks is built from the HF backend's own runtime serialization (``backend_tokenizer.to_str()``), not from
``tokenizer.json``: transformers may rebuild the backend differently from the file (Qwen2Tokenizer installs its own
pre-tokenizer regex and the extra added tokens of tokenizer_config.json), and the ids must be the ones HF produces at
run time. The serialization's truncation / padding are cleared (HF sets those per call). The built tokenizer must match
HF on a probe set (every added token, template on/off) before it is used. A tokenizer toks refuses, a probe difference
or any other error keeps HF for that tokenizer and is logged (``verify`` raises instead).
"""

import hashlib
import inspect
import json
import os
import threading

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

ENV = "VLLM_TT_TOKS"
_MODES = {"1": "on", "0": "off", "verify": "verify"}

# Keywords of HF's __call__ / encode that leave input_ids and attention_mask as computed here only at this value.
_NEUTRAL = {
    "text_target": None,
    "text_pair_target": None,
    "is_split_into_words": False,
    "pad_to_multiple_of": None,
    "return_overflowing_tokens": False,
    "return_special_tokens_mask": False,
    "return_offsets_mapping": False,
    "return_length": False,
    "tokenizer_kwargs": None,
}

# The HF signatures (transformers 5.12.1) the toks path and its HF fallback mirror.
_ENCODE_PARAMS = ("self", "text", "text_pair", "add_special_tokens", "padding", "truncation", "max_length", "stride",
                  "padding_side", "return_tensors", "kwargs")
_CALL_PARAMS = ("self", "text", "text_pair", "text_target", "text_pair_target", "add_special_tokens", "padding",
                "truncation", "max_length", "stride", "is_split_into_words", "pad_to_multiple_of", "padding_side",
                "return_tensors", "return_token_type_ids", "return_attention_mask", "return_overflowing_tokens",
                "return_special_tokens_mask", "return_offsets_mapping", "return_length", "verbose",
                "tokenizer_kwargs", "kwargs")

# Probe texts checked against HF before toks is used (every added token is probed as well).
_PROBES = (
    "",
    " ",
    "Hello world",
    "  leading and trailing spaces  ",
    "line one\nline two\r\n\tindented\n\n\n",
    "It's we're they've I'm you'll he'd ISN'T",
    "数字 12345 和 3.14159, 中文文本。日本語のテキスト、한국어 텍스트.",
    "emoji 😀👍🏽👨‍👩‍👧‍👦 flags 🇫🇷🇯🇵 zwj\u200d zwsp\u200b nbsp\u00a0",
    "combining: e\u0301 a\u0308 \u0915\u093f\u0928\u094d\u0926\u0940 \u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22 "
    "\u05e2\u05b4\u05d1\u05b0\u05e8\u05b4\u05d9\u05ea",
    "def f(x):\n    return {'a': [1, 2, 3]}  # comment\n\n\nclass A:\n\tpass\n",
    "<think>\n\n</think>\n\n<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
    "\x00\x01\x7f \ufffd \U0010ffff \ud7ff",
)


def toks_mode() -> str:
    value = os.environ.get(ENV, "1").strip().lower()
    if value not in _MODES:
        raise ValueError(f"{ENV} must be 1, 0 or verify, not {value!r}")
    return _MODES[value]


class _Engine:
    """A toks tokenizer bound to one HF backend state."""

    def __init__(self, hf, mode):
        import toks

        backend = hf.backend_tokenizer
        source = backend.to_str()
        if backend.truncation is not None or backend.padding is not None:  # HF sets these per call: clear them
            spec = json.loads(source)
            spec["truncation"] = None
            spec["padding"] = None
            source = json.dumps(spec)
        self.mode = mode
        self.sha256 = hashlib.sha256(source.encode()).hexdigest()
        self.toks = toks.Tokenizer.from_str(source)
        # Added-token count: tokens added to the HF backend later (add_tokens / add_special_tokens) send calls to HF.
        # (get_vocab_size(with_added_tokens=True) would rebuild the whole vocabulary on every call, ~36 ms.)
        self.n_added = len(backend.get_added_tokens_decoder())
        self.n_special = hf.num_special_tokens_to_add(pair=False)
        if self.toks.num_special_tokens_to_add() != self.n_special:
            raise ValueError(f"special tokens added: toks {self.toks.num_special_tokens_to_add()}, HF {self.n_special}")
        added = [t.content for _, t in sorted(hf.added_tokens_decoder.items())]
        probes = list(_PROBES) + added + [f"a{t}b {t}\n{t}{t} " for t in added]
        for text in probes:
            for add_special_tokens in (True, False):
                ours = self.toks.encode(text, add_special_tokens=add_special_tokens)
                ref = hf.encode(text, add_special_tokens=add_special_tokens)
                if ours != ref:
                    raise ValueError(f"probe {text!r} (add_special_tokens={add_special_tokens}): toks {ours} HF {ref}")
        self.n_probes = 2 * len(probes)
        self.version = toks.__version__
        self.tier = self.toks.info().get("tier")
        self._reasons = set()
        self._lock = threading.Lock()
        self.verified = 0

    def fallback(self, reason):
        if reason not in self._reasons:
            with self._lock:
                if reason not in self._reasons:
                    self._reasons.add(reason)
                    logger.info("toks: HF encodes calls with %s (logged once per reason)", reason)


def _plan(tok, engine, call, text, text_pair, add_special_tokens, padding, truncation, max_length, stride,
          padding_side, return_tensors, kwargs):
    """(texts, batched, limit, warn_length, attention_mask) for the toks path, or a reason string for HF."""
    if engine.n_added != len(tok.backend_tokenizer.get_added_tokens_decoder()):
        return "tokens added after toks was built"
    if isinstance(text, str):
        texts, batched = [text], False
    elif (call and isinstance(text, (list, tuple)) and text and all(isinstance(t, str) for t in text)):
        texts, batched = list(text), True
    else:
        return "non-string input"
    if text_pair is not None:
        return "text_pair"
    if type(add_special_tokens) is not bool:
        return "a non-bool add_special_tokens"
    if stride != 0 or padding_side is not None or return_tensors is not None:
        return "stride / padding_side / return_tensors"
    if not (padding is False or (isinstance(padding, str) and padding == "do_not_pad")):
        return "padding"
    verbose = True
    attention_mask = None
    for key, value in kwargs.items():
        if key in _NEUTRAL:
            if value is not _NEUTRAL[key]:
                return key
        elif key == "verbose":
            verbose = value
        elif key == "return_attention_mask":
            attention_mask = value
        elif key == "return_token_type_ids":
            if value:
                return "return_token_type_ids"
        elif key == "split_special_tokens":
            if value is not None and value is not False:
                return "split_special_tokens"
        else:
            return f"keyword {key!r}"
    if tok.split_special_tokens:
        return "split_special_tokens"
    if call:
        if tok._in_target_context_manager:
            return "the target context manager"
        if kwargs.get("return_token_type_ids") is None and "token_type_ids" in tok.model_input_names:
            return "token_type_ids"
        if attention_mask is None:
            attention_mask = "attention_mask" in tok.model_input_names
    try:
        padding_strategy, strategy, limit, _ = tok._get_padding_truncation_strategies(
            padding=padding, truncation=truncation, max_length=max_length, verbose=verbose)
    except Exception:
        return "invalid padding / truncation arguments"
    if padding_strategy != "do_not_pad":
        return "padding"
    if strategy == "do_not_truncate":
        cap = None
    elif strategy == "longest_first" and isinstance(limit, int) and limit >= engine.n_special:
        cap = limit
    else:
        return f"truncation {strategy}"
    return texts, batched, cap, limit, verbose, attention_mask


def _result(ids_list, batched, attention_mask):
    from transformers.tokenization_utils_base import BatchEncoding

    data = {"input_ids": ids_list}
    if attention_mask:
        data["attention_mask"] = [[1] * len(ids) for ids in ids_list]
    if not batched:
        data = {k: v[0] for k, v in data.items()}
    return BatchEncoding(data, n_sequences=1)


def _encode_texts(engine, texts, add_special_tokens, cap):
    if len(texts) == 1:
        ids_list = [engine.toks.encode(texts[0], add_special_tokens=add_special_tokens)]
    else:
        ids_list = engine.toks.encode_batch(texts, add_special_tokens=add_special_tokens)
    if cap is not None and any(len(ids) > cap for ids in ids_list):
        return None
    return ids_list


def _check(engine, what, ours, ref):
    if ours != ref:
        detail = ""
        a = ours.get("input_ids") if isinstance(ours, dict) else ours
        b = ref.get("input_ids") if isinstance(ref, dict) else ref
        if isinstance(a, list) and isinstance(b, list):
            n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            detail = f" (lengths {len(a)} / {len(b)}, first difference at {n}: toks {a[n:n + 8]} HF {b[n:n + 8]})"
        logger.error("%s=verify: toks and HF differ for %s%s", ENV, what, detail)
        raise RuntimeError(f"{ENV}=verify: toks and HF encodings differ for {what}{detail}")
    engine.verified += 1
    logger.info("%s=verify: toks == HF for %s (%d verified)", ENV, what, engine.verified)


def wrap_tokenizer(tokenizer, mode=None):
    """Return ``tokenizer`` with its prompt encoding served by toks (a copy whose class subclasses the original),
    or ``tokenizer`` itself when the mode is off or toks cannot serve it exactly."""
    mode = mode or toks_mode()
    if mode == "off":
        return tokenizer
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        from transformers.tokenization_utils_tokenizers import TokenizersBackend

        cls = type(tokenizer)
        if not isinstance(tokenizer, TokenizersBackend):
            raise TypeError(f"{cls.__name__} is not an HF fast (tokenizers) tokenizer")
        # The toks path mirrors these exact implementations; an override anywhere in the MRO keeps HF.
        for name, owner in (("encode", PreTrainedTokenizerBase), ("__call__", PreTrainedTokenizerBase),
                            ("_get_padding_truncation_strategies", PreTrainedTokenizerBase),
                            ("_encode_plus", TokenizersBackend), ("set_truncation_and_padding", TokenizersBackend),
                            ("_convert_encoding", TokenizersBackend)):
            if getattr(cls, name) is not getattr(owner, name):
                raise TypeError(f"{cls.__name__} overrides {name}")
        if hasattr(tokenizer, "_switch_to_input_mode") or hasattr(tokenizer, "_switch_to_target_mode"):
            raise TypeError(f"{cls.__name__} has input / target modes")
        for fn, params in ((PreTrainedTokenizerBase.encode, _ENCODE_PARAMS),
                           (PreTrainedTokenizerBase.__call__, _CALL_PARAMS)):
            if tuple(inspect.signature(fn).parameters) != params:
                raise TypeError(f"transformers' {fn.__name__} signature differs from the one mirrored here")
        engine = _Engine(tokenizer, mode)
    except Exception as e:
        if mode == "verify":
            raise
        logger.warning("toks: prompts keep HF encoding for %s: %s: %s", type(tokenizer).__name__, type(e).__name__, e)
        return tokenizer

    base = type(tokenizer)
    inner = tokenizer

    class ToksTokenizer(base):  # type: ignore[misc, valid-type]
        def encode(self, text, text_pair=None, add_special_tokens=True, padding=False, truncation=None,
                   max_length=None, stride=0, padding_side=None, return_tensors=None, **kwargs):
            plan = _plan(self, engine, False, text, text_pair, add_special_tokens, padding, truncation, max_length,
                         stride, padding_side, return_tensors, kwargs)
            ids = None
            if isinstance(plan, str):
                engine.fallback(plan)
            else:
                texts, _, cap, warn_length, verbose, _ = plan
                ids_list = _encode_texts(engine, texts, add_special_tokens, cap)
                if ids_list is None:
                    engine.fallback("truncation that cuts tokens")
                else:
                    ids = ids_list[0]
                    if engine.mode != "verify":
                        self._eventual_warn_about_too_long_sequence(ids, warn_length, verbose)
                        return ids
            ref = super().encode(text, text_pair=text_pair, add_special_tokens=add_special_tokens, padding=padding,
                                 truncation=truncation, max_length=max_length, stride=stride,
                                 padding_side=padding_side, return_tensors=return_tensors, **kwargs)
            if ids is not None:
                _check(engine, f"encode ({len(ref)} ids, add_special_tokens={add_special_tokens}, "
                               f"truncation={truncation!r}, max_length={max_length})", ids, ref)
            return ref

        def __call__(self, text=None, text_pair=None, text_target=None, text_pair_target=None,
                     add_special_tokens=True, padding=False, truncation=None, max_length=None, stride=0,
                     is_split_into_words=False, pad_to_multiple_of=None, padding_side=None, return_tensors=None,
                     return_token_type_ids=None, return_attention_mask=None, return_overflowing_tokens=False,
                     return_special_tokens_mask=False, return_offsets_mapping=False, return_length=False,
                     verbose=True, tokenizer_kwargs=None, **kwargs):
            extra = dict(text_target=text_target, text_pair_target=text_pair_target,
                         is_split_into_words=is_split_into_words, pad_to_multiple_of=pad_to_multiple_of,
                         return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask,
                         return_overflowing_tokens=return_overflowing_tokens,
                         return_special_tokens_mask=return_special_tokens_mask,
                         return_offsets_mapping=return_offsets_mapping, return_length=return_length,
                         verbose=verbose, tokenizer_kwargs=tokenizer_kwargs, **kwargs)
            plan = _plan(self, engine, True, text, text_pair, add_special_tokens, padding, truncation, max_length,
                         stride, padding_side, return_tensors, extra)
            out = None
            if isinstance(plan, str):
                engine.fallback(plan)
            else:
                texts, batched, cap, warn_length, verbose_, attention_mask = plan
                ids_list = _encode_texts(engine, texts, add_special_tokens, cap)
                if ids_list is None:
                    engine.fallback("truncation that cuts tokens")
                else:
                    out = _result(ids_list, batched, attention_mask)
                    if engine.mode != "verify":
                        for ids in ids_list:
                            self._eventual_warn_about_too_long_sequence(ids, warn_length, verbose_)
                        return out
            ref = super().__call__(text=text, text_pair=text_pair, add_special_tokens=add_special_tokens,
                                   padding=padding, truncation=truncation, max_length=max_length, stride=stride,
                                   padding_side=padding_side, return_tensors=return_tensors, **extra)
            if out is not None:
                n = len(ref["input_ids"])
                _check(engine, f"__call__ ({'batch of ' if batched else ''}{n} "
                               f"{'texts' if batched else 'ids'}, add_special_tokens={add_special_tokens}, "
                               f"truncation={truncation!r}, max_length={max_length})", dict(out), dict(ref))
            return ref

        def __reduce__(self):
            return wrap_tokenizer, (inner, mode)

    ToksTokenizer.__name__ = ToksTokenizer.__qualname__ = f"Toks{base.__name__}"
    ToksTokenizer._toks_engine = engine
    # A shallow copy sharing the HF backend, as vLLM's get_cached_tokenizer makes (copy.copy would go through the
    # cached tokenizer's __reduce__, a full rebuild).
    wrapped = object.__new__(ToksTokenizer)
    wrapped.__dict__.update(tokenizer.__dict__)
    logger.info("toks %s prompt encoding %s for %s (tier %s, runtime tokenizer sha256 %s, %d probes identical to HF)",
                engine.version, "VERIFIED against HF" if mode == "verify" else "on", base.__name__, engine.tier,
                engine.sha256[:16], engine.n_probes)
    return wrapped


def install() -> None:
    """Serve the prompt encoding of every tokenizer vLLM loads through ``CachedHfTokenizer`` with toks, in the
    processes that tokenize prompts (the API server / frontend). The engine core process (``EngineCore_DP<n>``) only
    uses its tokenizer for vocabulary lookups (structured output, reasoning end detection): it keeps HF and saves the
    toks engine's memory (~0.2 GB) and build time."""
    import multiprocessing

    import vllm.envs as envs

    mode = toks_mode()
    # Known to vLLM's environment validation (no "Unknown vLLM environment variable" warning / hard failure).
    envs.environment_variables.setdefault(ENV, lambda: os.getenv(ENV, "1"))
    if mode == "off":
        logger.info("%s=0: prompts use HF encoding", ENV)
        return
    if multiprocessing.current_process().name.startswith("EngineCore"):
        logger.debug("toks: engine core process keeps HF (prompts are encoded by the frontend)")
        return
    from vllm.tokenizers.hf import CachedHfTokenizer

    if getattr(CachedHfTokenizer, "_tt_toks_mode", None) is not None:
        return
    original = CachedHfTokenizer.from_pretrained.__func__

    def from_pretrained(cls, *args, **kwargs):
        return wrap_tokenizer(original(cls, *args, **kwargs), mode)

    CachedHfTokenizer.from_pretrained = classmethod(from_pretrained)
    CachedHfTokenizer._tt_toks_mode = mode
