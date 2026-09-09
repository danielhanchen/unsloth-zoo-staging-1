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

`TORCHDYNAMO_DISABLE=1` is the documented way to take torch.compile out of the
picture while debugging. Up to torch 2.13 a `fullgraph = True` region then ran
eagerly. From torch 2.14 it raises instead:

    RuntimeError: torch.compile with fullgraph=True found no compiled frames.
    Skipped frames: ... Dynamo tracing is disabled

Every `torch_compile_with_fallback` region inherits that, and there are many:
`rl_replacements.py` wraps the GRPO loss accumulator in one, and `compiler.py`
writes the decorator into every generated `unsloth_compiled_cache` module. So a
user who disables Dynamo on torch 2.14 loses GRPO training at the first step,
having asked only to turn the compiler off.

Two arms, because the switch can be thrown on either side of decoration:

  * before, which is what the env var does, caught at decoration time;
  * after, which a test or a notebook does by assigning to the config, caught
    once at call time and latched.
"""

import pytest
import torch

from unsloth_zoo.temporary_patches import utils as u


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


def test_decorating_under_disabled_dynamo_returns_the_plain_function(dynamo_off):
    # Decorated while disabled: never compiled at all, so nothing can raise.
    wrapped = u.torch_compile_with_fallback(fullgraph = True)(_add_one)
    assert wrapped is _add_one
    assert torch.equal(wrapped(torch.zeros(3)), torch.ones(3))


def test_disabling_dynamo_after_decoration_falls_back_to_eager():
    # The env var route cannot reach a wrapper that already exists, which is
    # exactly the case the runtime arm covers.
    wrapped = u.torch_compile_with_fallback(fullgraph = True)(_add_one)
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        got = wrapped(torch.zeros(3))
    finally:
        torch._dynamo.config.disable = prev
    assert torch.equal(got, torch.ones(3))


def test_fullgraph_false_is_untouched(dynamo_off):
    # Dynamo already falls back by itself there, so the guard must not fire.
    wrapped = u.torch_compile_with_fallback(fullgraph = False)(_add_one)
    assert torch.equal(wrapped(torch.zeros(3)), torch.ones(3))


@pytest.mark.parametrize(
    "text,want",
    [
        ("torch.compile with fullgraph=True found no compiled frames. "
         "Skipped frames: Dynamo tracing is disabled", True),
        # A real fullgraph graph break must still raise.
        ("Unsupported: call_function BuiltinVariable(print)", False),
        ("torch.compile with fullgraph=True hit a graph break", False),
        ("found no compiled frames", False),
    ],
)
def test_the_signature_match_is_narrow(text, want):
    assert u._is_fullgraph_without_frames(RuntimeError(text)) is want


def test_dynamo_tracing_disabled_reads_the_live_config():
    prev = torch._dynamo.config.disable
    try:
        torch._dynamo.config.disable = True
        assert u.dynamo_tracing_disabled() is True
        torch._dynamo.config.disable = False
        assert u.dynamo_tracing_disabled() is False
    finally:
        torch._dynamo.config.disable = prev
