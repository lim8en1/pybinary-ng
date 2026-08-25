"""SKILL.md must not drift from the implementation.

A spec that has drifted is worse than no spec: it makes a model confidently
generate code that cannot run. So the agreement is enforced here rather than by
a read-through -- the examples execute, and the documented surface is checked
against ``pybinary.__all__`` in *both* directions.
"""

import pathlib
import re

import pytest

import pybinary

SKILL = pathlib.Path(__file__).resolve().parent.parent / "SKILL.md"
TEXT = SKILL.read_text()

# The prompt budget SKILL.md exists to respect. Chars/3.6 is a deliberate
# over-estimate of the token count, so passing here means passing in practice.
# The design target is 900-1300, and 1300 is the hard cap. The earlier 700 bought
# brevity by leaving out which slots accept an expression, what a Pointer offset
# is relative to, and what describe() prints -- so a model had to read the source
# to write a real format, which costs far more than the lines saved. Failing at
# the cap is deliberate: it forces a scrutiny pass whenever a row is added.
TOKEN_BUDGET = 1300


def _code_blocks():
    return re.findall(r"```python\n(.*?)```", TEXT, re.S)


def test_skill_file_exists_and_has_frontmatter():
    assert TEXT.startswith("---\n")
    head = TEXT.split("---", 2)[1]
    assert "name: pybinary" in head
    assert "description:" in head


def test_it_stays_within_the_token_budget():
    approx = round(len(TEXT) / 3.6)
    assert approx <= TOKEN_BUDGET, (
        f"SKILL.md is ~{approx} tokens, over the {TOKEN_BUDGET} target. "
        f"It is a prompt, not a README -- for each line ask what an agent does "
        f"with it, and whether it repeats something already here."
    )


def test_every_example_runs():
    blocks = _code_blocks()
    assert blocks, "SKILL.md has no python example"
    for block in blocks:
        ns = {}
        # `buf` is illustrative in the example; supply one that fits Chunk.
        block = block.replace(
            "c    = Chunk.unpack(buf)",
            "buf = Chunk(ln=2, kind=b'IHDR', data=b'hi').pack()\n"
            "c    = Chunk.unpack(buf)",
        ).replace(
            "c, e = Chunk.unpack_from(buf, 8)",
            "c, e = Chunk.unpack_from(buf, 0)",
        )
        exec(compile(block, "<SKILL.md>", "exec"), ns)


def _documented_declarations():
    """Left-hand column of the declaration table, split into single spellings."""
    out = []
    for line in TEXT.splitlines():
        if not line.startswith("| `"):
            continue
        cell = line.split("|")[1].strip()
        out.extend(re.findall(r"`([^`]+)`", cell))
    return out


def test_every_documented_declaration_compiles():
    from pybinary import (  # noqa: F401 - names are used by eval below
        Array, Binary, Bits, Bytes, Checksum, Const, If, Padding,
        Pointer, Str, Switch, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64,
    )
    import zlib  # noqa: F401

    class Rec(Binary, endian="<"):
        v: u16

    A = B = C = Rec
    scope = dict(locals())

    checked = 0
    for decl in _documented_declarations():
        if decl.startswith("u8 u16"):
            continue  # the scalar list is prose, covered by its own test
        try:
            value = eval(decl, dict(vars(pybinary)), scope)
        except Exception as exc:  # pragma: no cover - failure is the point
            pytest.fail(f"SKILL.md documents {decl!r}, which does not evaluate: {exc}")
        # Either a marker tuple, or a structure class used directly.
        ok = (isinstance(value, tuple) and len(value) == 2) or (
            isinstance(value, type) and issubclass(value, Binary)
        )
        assert ok, f"SKILL.md documents {decl!r}, which is not a field declaration"
        checked += 1
    assert checked >= 10, "the declaration table looks truncated"


def test_every_scalar_type_is_documented():
    scalars = [n for n in pybinary.__all__ if re.fullmatch(r"[uif]\d+", n)]
    assert scalars
    for name in scalars:
        assert name in TEXT, f"the scalar {name} is missing from SKILL.md"


def test_every_field_type_is_in_the_declaration_table():
    """The check that actually bites.

    Searching the whole file is too weak: a type dropped from the table still
    appears in the example's import line. A field type earns its place by being
    *declarable*, so it has to show up in the declaration column.
    """
    import pybinary.types as t

    table = " ".join(_documented_declarations())
    scalars = {n for n in t.__all__ if re.fullmatch(r"[uif]\d+", n)}
    missing = [
        name for name in t.__all__
        if name not in scalars and not re.search(rf"\b{re.escape(name)}\b", table)
    ]
    assert not missing, (
        f"field types missing from SKILL.md's declaration table: {missing}"
    )


def test_every_scalar_type_is_documented():
    scalars = [n for n in pybinary.__all__ if re.fullmatch(r"[uif]\d+", n)]
    assert scalars
    for name in scalars:
        assert name in TEXT, f"the scalar {name} is missing from SKILL.md"


def test_every_field_type_is_in_the_declaration_table():
    """The check that actually bites.

    Searching the whole file is too weak: a type dropped from the table still
    appears in the example's import line. A field type earns its place by being
    *declarable*, so it has to show up in the declaration column.
    """
    import pybinary.types as t

    table = " ".join(_documented_declarations())
    scalars = {n for n in t.__all__ if re.fullmatch(r"[uif]\d+", n)}
    missing = [
        name for name in t.__all__
        if name not in scalars and not re.search(rf"\b{re.escape(name)}\b", table)
    ]
    assert not missing, (
        f"field types missing from SKILL.md's declaration table: {missing}"
    )




def test_no_public_surface_is_undocumented():
    """Backstop for the rest of the API: errors, Binary itself."""
    # BinaryMeta and FieldSpec are internals a user never declares.
    exempt = {"BinaryMeta", "FieldSpec", "PyBinaryError"}
    missing = [
        name for name in pybinary.__all__
        if name not in exempt and not re.search(rf"\b{re.escape(name)}\b", TEXT)
    ]
    assert not missing, f"public names missing from SKILL.md: {missing}"


def test_nothing_documented_is_absent_from_the_package():
    """The other direction: every pybinary-looking name in the spec resolves."""
    referenced = set(re.findall(r"`([A-Z][A-Za-z]*)\[", TEXT))
    referenced |= set(re.findall(r"\b([A-Z][A-Za-z]*Error)\b", TEXT))
    unknown = [n for n in referenced if not hasattr(pybinary, n)
               and n not in {"A", "B", "C", "Rec", "Chunk"}]
    assert not unknown, f"SKILL.md names things pybinary does not export: {unknown}"


def test_documented_methods_exist():
    for attr in ("unpack", "unpack_from", "pack", "describe"):
        assert hasattr(pybinary.Binary, attr), f"SKILL.md documents {attr}()"

    class T(pybinary.Binary, endian="<"):
        a: pybinary.u8

    assert hasattr(T, "__codec_source__")


def test_the_documented_copy_default_is_the_real_one():
    class T(pybinary.Binary, endian="<"):
        n:    pybinary.u8
        data: pybinary.Bytes["n"]

    raw = T(n=2, data=b"hi").pack()
    assert isinstance(T.unpack(raw).data, bytes)
    assert isinstance(T.unpack(raw, copy=False).data, memoryview)


def test_the_documented_python_floor_matches_pyproject():
    pyproject = (SKILL.parent / "pyproject.toml").read_text()
    floor = re.search(r'requires-python = ">=([\d.]+)"', pyproject).group(1)
    assert f"Python {floor}+" in TEXT, (
        f"SKILL.md must state Python {floor}+ to match pyproject.toml"
    )
