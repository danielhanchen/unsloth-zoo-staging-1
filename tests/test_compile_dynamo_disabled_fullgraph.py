# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""`fullgraph = True` regions when the user has switched Dynamo off.

`TORCH_COMPILE_DISABLE=1` (and `torch._dynamo.config.disable = True`, which is
the same flag by hand) takes torch.compile out of the picture while debugging.
Up to torch 2.13 a `fullgraph = True` region then ran eagerly. From torch 2.14
torch runs the body and raises afterwards, out of its post-call bookkeeping:

    RuntimeError: torch.compile with fullgraph=True found no compiled frames.
    Skipped frames: ... Dynamo tracing is disabled

Every `torch_compile_with_fallback` region inherits that, and there are many:
`rl_replacements.py` wraps the GRPO loss accumulator in one, and `compiler.py`
writes the decorator into every generated `unsloth_compiled_cache` module. So a
user who disables torch.compile on torch 2.14 loses GRPO training at the first
step, having asked only to turn the compiler off.

The flag is therefore read live, before each call, rather than recovered from
after one: torch has already run the body by the time it raises, so recovering
would run it twice, and entering the compiled callable at all makes torch mark
the code object skipped permanently.

`TORCHDYNAMO_DISABLE=1` is a different switch, handled inside
`torch._dynamo.optimize`, which returns the undecorated function. Nothing here
has to deal with it, and the tests below pin that too.
"""

import pytest
import torch

from unsloth_zoo.temporary_patches import utils as u


@pytest.fixture(autouse = True)
def clear_checkpoint_markers():
    # Process-wide and only a step boundary clears them, so a test that packs
    # under a checkpoint would otherwise decide the next test's mode.
    yield
    u._PACKED_COMPILED_IN_CHECKPOINT = False
    u._COMPILED_OK_LABELS.clear()


@pytest.fixture
def dynamo_off():
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        yield
    finally:
        torch._dynamo.config.disable = prev


def _add_one(x):
    return x + 1


def _counting_fn():
    """A fresh code object per test: torch skip marks attach to the code."""
    calls = []
    src = {}
    exec("def f(x):\n    calls.append(1)\n    return x + 1\n", {"calls": calls}, src)
    return src["f"], calls


def test_decorated_before_disabling_runs_eagerly_and_exactly_once():
    fn, calls = _counting_fn()
    wrapped = u.torch_compile_with_fallback(fullgraph = True)(fn)
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        del calls[:]
        got = wrapped(torch.zeros(3))
    finally:
        torch._dynamo.config.disable = prev
    assert torch.equal(got, torch.ones(3))
    # 2 here is the torch 2.14 double-run: body, then raise, then eager retry.
    assert len(calls) == 1


def test_decorated_while_disabled_runs_eagerly_and_exactly_once(dynamo_off):
    fn, calls = _counting_fn()
    wrapped = u.torch_compile_with_fallback(fullgraph = True)(fn)
    del calls[:]
    got = wrapped(torch.zeros(3))
    assert torch.equal(got, torch.ones(3))
    assert len(calls) == 1


def test_re_enabling_dynamo_restores_compilation():
    # Discarding the compiled callable at decoration time made a flag that was
    # set only during import permanent for the rest of the run.
    fn, _ = _counting_fn()
    wrapped = u.torch_compile_with_fallback(fullgraph = True)(fn)
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        wrapped(torch.zeros(3))
    finally:
        torch._dynamo.config.disable = prev
    assert torch.equal(wrapped(torch.zeros(3)), torch.ones(3))
    state = getattr(wrapped, "_unsloth_fallback_state", {})
    assert not state.get("eager", False)


def test_fullgraph_false_is_untouched(dynamo_off):
    wrapped = u.torch_compile_with_fallback(fullgraph = False)(_add_one)
    assert torch.equal(wrapped(torch.zeros(3)), torch.ones(3))


def test_a_real_graph_break_still_raises():
    # The eager dispatch is keyed on the live flag, not on an error message, so
    # a genuine fullgraph failure is never absorbed.
    def breaks(x):
        print("graph break")
        return x + 1
    wrapped = u.torch_compile_with_fallback(fullgraph = True)(breaks)
    with pytest.raises(Exception):
        wrapped(torch.zeros(3))


def _checkpointed_step(flip_to):
    from torch.utils.checkpoint import checkpoint

    def block(x, w):
        return torch.nn.functional.layer_norm(x @ w, (w.shape[-1],)) * x

    torch.manual_seed(0)
    wrapped = u.torch_compile_with_fallback(fullgraph = True)(block)
    x = torch.randn(4, 8, requires_grad = True)
    w = torch.randn(8, 8, requires_grad = True)
    out = checkpoint(wrapped, x, w, use_reentrant = False)
    if flip_to is not None:
        torch._dynamo.config.disable = flip_to
    out.sum().backward()
    return w.grad


def _checkpointed_wrapper():
    """One wrapper plus the list of body executions, for multi-step tests."""
    calls = []
    src = {}
    exec("def block(x, w):\n"
         "    calls.append(1)\n"
         "    return torch.nn.functional.layer_norm(x @ w, (w.shape[-1],)).sin()\n",
         {"calls": calls, "torch": torch}, src)
    return u.torch_compile_with_fallback(fullgraph = True)(src["block"]), calls


def _run_checkpointed(wrapped):
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(0)
    x = torch.randn(4, 8, requires_grad = True)
    w = torch.randn(8, 8, requires_grad = True)
    checkpoint(wrapped, x, w, use_reentrant = False).sum().backward()
    return w.grad


@pytest.mark.parametrize("starts_disabled", [True, False])
def test_a_mid_step_flip_does_not_abort_a_checkpointed_backward(starts_disabled):
    """A flip between a checkpointed forward and its backward keeps its mode.

    Non-reentrant checkpointing recomputes the forward during backward and
    compares what each pass saved, so recomputing a compiled pack eagerly, or
    an eager pack compiled, raises `CheckpointError` and ends the step.
    """
    prev = torch._dynamo.config.disable
    try:
        torch._dynamo.config.disable = starts_disabled
        want = _checkpointed_step(flip_to = None)
        u._PACKED_COMPILED_IN_CHECKPOINT = False
        u._COMPILED_OK_LABELS.clear()
        torch._dynamo.config.disable = starts_disabled
        got = _checkpointed_step(flip_to = not starts_disabled)
    finally:
        torch._dynamo.config.disable = prev
    assert torch.allclose(want, got, atol = 1e-5)


def test_a_nested_wrapper_is_traceable_after_running_eager():
    # The checkpoint probe reaches a pybind builtin Dynamo refuses to enter, so
    # an outer fullgraph trace of a wrapper that has run eager must not take it.
    def inner(x):
        return x * 2
    wrapped = u.torch_compile_with_fallback(fullgraph = True)(inner)
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        wrapped(torch.zeros(3))
    finally:
        torch._dynamo.config.disable = prev

    def outer(x):
        return wrapped(x) + 1
    assert torch.equal(
        torch.compile(outer, fullgraph = True)(torch.zeros(3)), torch.ones(3),
    )


def test_the_force_eager_stance_is_left_to_torch():
    # torch only raises "found no compiled frames" while the stance is
    # "default" (torch/_dynamo/eval_frame.py), so force_eager needs nothing.
    fn, calls = _counting_fn()
    wrapped = u.torch_compile_with_fallback(fullgraph = True)(fn)
    torch.compiler.set_stance("force_eager")
    try:
        del calls[:]
        got = wrapped(torch.zeros(3))
    finally:
        torch.compiler.set_stance("default")
    assert torch.equal(got, torch.ones(3))
    assert len(calls) == 1


def test_another_regions_pack_does_not_force_this_one_into_the_compiler():
    # The evidence is per wrapper, not a process-wide marker or a shared
    # __qualname__: a wrapper with nothing cached must still go eager.
    fn, calls = _counting_fn()
    other, _ = _counting_fn()
    other.__qualname__ = fn.__qualname__              # a colliding label
    compiled_one = u.torch_compile_with_fallback(fullgraph = True)(other)
    u._PACKED_COMPILED_IN_CHECKPOINT = True           # some other region's pack
    compiled_one(torch.zeros(3))                      # records the shared label
    wrapped = u.torch_compile_with_fallback(fullgraph = True)(fn)
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        del calls[:]
        got = wrapped(torch.zeros(3))
    finally:
        torch._dynamo.config.disable = prev
    assert torch.equal(got, torch.ones(3))
    assert len(calls) == 1


def test_a_checkpointed_wrapper_compiles_again_on_the_next_step():
    """Re-enabling must reach a wrapper that is only ever called under one.

    Every checkpoint forward is inside a region, so deferring on "inside a
    region" rather than "being recomputed" left such a wrapper eager forever.
    """
    prev = torch._dynamo.config.disable
    try:
        torch._dynamo.config.disable = True
        wrapped, calls = _checkpointed_wrapper()
        _run_checkpointed(wrapped)
        torch._dynamo.config.disable = False
        _run_checkpointed(wrapped)
        _run_checkpointed(wrapped)
    finally:
        torch._dynamo.config.disable = prev
    assert wrapped._unsloth_fallback_state["ran_eager_disabled"] is False
    assert wrapped._unsloth_fallback_state["last_call_compiled"] is True


def test_disabling_after_a_completed_checkpointed_step_takes_effect():
    # The compiled evidence must not outlive the recompute it was for: nothing
    # inside this package clears a step marker.
    prev = torch._dynamo.config.disable
    try:
        torch._dynamo.config.disable = False
        wrapped, calls = _checkpointed_wrapper()
        _run_checkpointed(wrapped)
        torch._dynamo.config.disable = True
        del calls[:]
        _run_checkpointed(wrapped)
    finally:
        torch._dynamo.config.disable = prev
    # Forward plus recompute, each exactly once: eager, and never twice.
    assert len(calls) == 2


def test_dynamo_tracing_disabled_reads_the_live_config():
    prev = torch._dynamo.config.disable
    try:
        torch._dynamo.config.disable = True
        assert u.dynamo_tracing_disabled() is True
        torch._dynamo.config.disable = False
        assert u.dynamo_tracing_disabled() is False
    finally:
        torch._dynamo.config.disable = prev
