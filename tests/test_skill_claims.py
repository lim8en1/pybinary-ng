"""Every factual claim in SKILL.md, executed.

The sibling ``test_skill_doc`` checks the spec's *surface* -- that names resolve,
declarations compile and the budget holds. This checks the spec's *claims*: each
row's stated meaning and each rule, run against the real implementation. A spec
that compiles but lies is still a spec that makes a model write broken code.
"""

import pathlib
import re
import struct
import sys
import zlib

import pytest

from pybinary import (
    Array, Binary, Bits, BuildError, Bytes, Checksum, Const, If, LayoutError,
    Padding, ParseError, Pointer, Str, Switch,
    f32, f64, i8, i16, i32, i64, u8, u16, u32, u64,
)
import pybinary


def raises(exc_type, fn, match=None):
    with pytest.raises(exc_type, match=match):
        fn()


TEXT = (pathlib.Path(__file__).resolve().parent.parent / "SKILL.md").read_text()
BLOCK = re.search(r"```python\n(.*?)```", TEXT, re.S).group(1)

_ns = {}
exec(BLOCK.split("c    = Chunk.unpack(buf)")[0], _ns)
Chunk = _ns["Chunk"]
BUF = Chunk(ln=2, kind=b"IHDR", data=b"hi").pack()

def test_01_example_block_declarations_execute_verbatim():
    """example block: declarations execute verbatim"""
    assert Chunk.__name__ == "Chunk"


def test_02_example_c_chunk_unpack_buf_raw_c_pack_round_trips():
    """example: c = Chunk.unpack(buf); raw = c.pack() round-trips"""
    g = dict(_ns, buf=BUF)
    exec("c    = Chunk.unpack(buf)\nraw  = c.pack()", g)
    assert g["raw"] == BUF


def test_03_example_unpack_is_strict_rejects_trailing_bytes():
    """example: unpack() is strict -- rejects trailing bytes"""
    raises(ParseError, lambda: Chunk.unpack(BUF + b"\x00"), "trailing")


def test_04_example_unpack_from_buf_off_returns_obj_end():
    """example: unpack_from(buf, off) returns (obj, end)"""
    obj, end = Chunk.unpack_from(b"XXXXXXXX" + BUF, 8)
    assert obj.kind == b"IHDR" and end == 8 + len(BUF)


def test_05_example_comment_endian_accepts_the_four_symbols():
    """example comment: endian accepts "<" ">" "!" "=" """
    # the long aliases work too; the spec lists only the symbols
    for e in ("<", ">", "!", "=", "little", "big", "network", "native"):
        type("E", (Binary,), {"__annotations__": {"a": u16}}, endian=e)


# ----------------------------------------------------------- table rows
def test_06_row_all_ten_scalars_exist_with_the_advertised_widths():
    """row: all ten scalars exist with the advertised widths"""
    want = {"u8": 1, "u16": 2, "u32": 4, "u64": 8, "i8": 1, "i16": 2,
            "i32": 4, "i64": 8, "f32": 4, "f64": 8}
    for name, size in want.items():
        cls = type("S", (Binary,), {"__annotations__": {"a": getattr(pybinary, name)}})
        assert cls.__struct_size__ == size, name


def test_07_row_const_b_blob_literal():
    """row: Const[b'BLOB'] literal"""
    C = type("C", (Binary,), {"__annotations__": {"m": Const[b"BLOB"]}})
    assert C().pack() == b"BLOB"


def test_08_row_const_0x1a2b_uses_the_struct_s_byte_order():
    """row: Const[0x1A2B] uses the struct's byte order"""
    L = type("L", (Binary,), {"__annotations__": {"m": Const[0x1A2B]}}, endian="<")
    B = type("B", (Binary,), {"__annotations__": {"m": Const[0x1A2B]}}, endian=">")
    assert L().pack() == b"\x2b\x1a" and B().pack() == b"\x1a\x2b"


def test_09_row_const_1a_2b_is_hex():
    """row: Const['1a 2b'] is hex"""
    H = type("H", (Binary,), {"__annotations__": {"m": Const["1a 2b"]}})
    assert H().pack() == b"\x1a\x2b"


def test_10_row_bytes_n_bytes_4_bytes_rest_of_buffer():
    """row: Bytes['n'] / Bytes[4] / Bytes[...] = rest of buffer"""
    class T(Binary, endian="<"):
        n: u8
        a: Bytes["n"]
        b: Bytes[2]
        c: Bytes[...]
    t = T.unpack(b"\x01Xyz" + b"rest!")
    assert (t.a, t.b, t.c) == (b"X", b"yz", b"rest!")


def test_11_row_str_n_and_str_4_ascii_utf_8_by_default():
    """row: Str['n'] and Str[4, 'ascii'], utf-8 by default"""
    class T(Binary, endian="<"):
        n: u8
        a: Str["n"]
        b: Str[4, "ascii"]
    raw = b"\x02" + "€".encode() [:0] + "ab".encode() + b"cdef"
    t = T.unpack(b"\x02abcdef")
    assert t.a == "ab" and t.b == "cdef"
    class U(Binary, endian="<"):
        s: Str[3]
    assert U.unpack("€".encode()).s == "€"     # 3 utf-8 bytes


def test_12_row_array_u16_n_and_array_rec_4():
    """row: Array[u16, 'n'] and Array[Rec, 4]"""
    class Rec(Binary, endian="<"):
        v: u8
    class T(Binary, endian="<"):
        n: u8
        a: Array[u16, "n"]
        b: Array[Rec, 4]
    t = T.unpack(b"\x02" + struct.pack("<2H", 7, 8) + b"wxyz")
    assert t.a == (7, 8) and [r.v for r in t.b] == [119, 120, 121, 122]


def test_13_row_padding_3_is_skipped_on_parse_and_zero_filled_on_build():
    """row: Padding[3] is skipped on parse and zero-filled on build"""
    class T(Binary, endian="<"):
        a: u8
        _p: Padding[3]
        b: u8
    assert T(a=1, b=2).pack() == b"\x01\x00\x00\x00\x02"
    assert T.unpack(b"\x01\xff\xff\xff\x02") == T(a=1, b=2)


def test_14_row_bits_are_msb_first():
    """row: Bits are MSB-first"""
    class T(Binary, endian=">"):
        hi: Bits[4]
        lo: Bits[4]
    assert T(hi=4, lo=5).pack() == b"\x45"


def test_15_row_a_bits_run_must_total_whole_bytes():
    """row: a Bits run must total whole bytes"""
    def bad():
        class T(Binary):
            a: Bits[3]
            b: Bits[2]
    raises(LayoutError, bad, "whole number of bytes")


def test_16_row_bits_3_0_is_a_constant_verified_written_not_an_attribute():
    """row: Bits[3, 0] is a constant -- verified, written, not an attribute"""
    class T(Binary, endian=">"):
        a: Bits[5]
        _r: Bits[3, 0]
    t = T(a=1)
    assert t.pack() == b"\x08" and not hasattr(t, "_r")
    raises(ParseError, lambda: T.unpack(b"\x09"))


def test_17_row_if_is_present_only_when_true_else_none():
    """row: If is present only when true, else None"""
    class T(Binary, endian="<"):
        v: u8
        x: If["v > 1", u32]
    assert T.unpack(b"\x01").x is None
    assert T.unpack(b"\x02" + struct.pack("<I", 9)).x == 9


def test_18_row_switch_picks_by_an_earlier_field_fallback_none_no_bytes():
    """row: Switch picks by an earlier field; ... = fallback; None = no bytes"""
    class A(Binary, endian="<"):
        v: u8
    class T(Binary, endian="<"):
        k: u8
        b: Switch["k", {1: A, 2: None, ...: u16}]
    assert T.unpack(b"\x01\x07").b == A(v=7)
    assert T.unpack(b"\x02").b is None
    assert T.unpack(b"\x63\x07\x00").b == 7


def test_19_row_pointer_target_is_at_off_from_this_record_s_start():
    """row: Pointer target is at off from THIS record's start"""
    class T(Binary, endian="<"):
        off: u16
        pad: u8
        p:   Bytes[1]
    class P(Binary, endian="<"):
        off: u16
        pad: u8
        p:   Pointer["off", Bytes[1]]
    # record placed at offset 4 in a larger buffer; offset stays relative
    inner = P(pad=0, p=b"Z")
    out = bytearray(b"JUNK")
    inner.pack_into(out)
    got, _ = P.unpack_from(out, 4)
    assert got.p == b"Z" and got.off == 3


def test_20_row_pointer_leaves_the_cursor_put():
    """row: Pointer leaves the cursor put"""
    class P(Binary, endian="<"):
        off:  u16
        p:    Pointer["off", Bytes[1]]
        tail: u8
    # tail is read immediately after `off`, not after the target
    assert P.unpack(b"\x03\x00\x09Z").tail == 9


def test_21_row_pointer_nullable_offset_0_means_absent():
    """row: Pointer "nullable" -- offset 0 means absent"""
    class P(Binary, endian="<"):
        off:  u16
        tail: u8
        p:    Pointer["off", Bytes[1], "nullable"]
    assert P.unpack(b"\x00\x00\x01").p is None
    assert P(tail=1, p=None).pack() == b"\x00\x00\x01"


def test_22_row_checksum_takes_any_callable_covering_start_here():
    """row: Checksum takes any callable, covering start..here"""
    class T(Binary, endian=">"):
        a:  u8
        b:  Bytes[2]
        ck: Checksum[u16, lambda d: sum(d), "b"]
    t = T(a=9, b=b"\x01\x02")
    assert t.pack() == b"\x09\x01\x02\x00\x03"      # sum covers b only


def test_23_line_const_padding_and_checksum_are_not_attributes():
    """line: Const, Padding and Checksum are not attributes"""
    class T(Binary, endian=">"):
        m:  Const[b"AB"]
        _p: Padding[2]
        d:  Bytes[1]
        ck: Checksum[u16, zlib.crc32, "d"]
    t = T(d=b"x")
    for absent in ("m", "_p", "ck"):
        assert not hasattr(t, absent), absent
    assert t.pack() == b"AB\x00\x00x" + (zlib.crc32(b"x") & 0xFFFF).to_bytes(2, "big")


# --------------------------------------------------------------- rules
def test_24_rule_quoted_references_work():
    """rule: quoted references work"""
    class T(Binary, endian="<"):
        n: u8
        d: Bytes["n"]
    assert T(n=1, d=b"x").pack() == b"\x01x"


def test_25_rule_unquoted_references_hit_module_globals_first():
    """rule: unquoted references hit module globals first"""
    src = "class T(Binary):\n    n: u8\n    d: Bytes[n]\n"
    shadowed = dict(vars(pybinary)); shadowed["n"] = 5
    clean = dict(vars(pybinary))
    if sys.version_info >= (3, 14):
        # unshadowed, the unquoted form resolves to the field and works
        exec(src, clean)
        assert clean["T"].fields()[1].expr == "n"
        # shadowed, pybinary sees the capture and refuses
        raises(LayoutError, lambda: exec(src, shadowed), "resolved to")
    else:
        # 3.13 evaluates eagerly: no global -> NameError at class-body time...
        raises(NameError, lambda: exec(src, clean))
        # ...but a shadowing global is SILENT, baking in a static length of 5.
        exec(src, shadowed)
        assert shadowed["T"].fields()[1].count == 5
        assert shadowed["T"].fields()[1].expr is None


def test_26_rule_work_in_if_conditions():
    """rule: & | ~ work in If conditions"""
    class T(Binary, endian="<"):
        v: u8
        x: If["(v > 1) & (v < 9)", u8]
    assert T.unpack(b"\x05\x07").x == 7
    assert T.unpack(b"\x5a").x is None


def test_27_rule_and_or_not_silently_yield_a_different_condition_documented_trap():
    """rule: and/or/not silently yield a different condition (documented trap)"""
    if sys.version_info < (3, 14):
        return
    g = dict(vars(pybinary))
    exec("class T(Binary):\n    v: u8\n    x: If[v > 1 and v, u8]\n", g)
    assert g["T"].fields()[1].cond == "v"      # the `v > 1` half is gone
    raises(LayoutError, lambda: exec(
        "class U(Binary):\n    v: u8\n    x: If[not v, u8]\n", dict(vars(pybinary))),
        "collapsed to the constant")


def test_28_rule_a_variant_class_defined_after_the_referencing_class_is_rejected():
    """rule: a variant class defined after the referencing class is rejected"""
    if sys.version_info < (3, 14):
        return
    src = ("class T(Binary):\n    k: u8\n    b: Switch['k', {1: Later}]\n"
           "class Later(Binary):\n    v: u8\n")
    raises(LayoutError, lambda: exec(src, dict(vars(pybinary))))


def test_29_rule_pack_fills_a_length_referenced_by_exactly_one_field_by_bare_name():
    """rule: pack() fills a length referenced by exactly one field by bare name"""
    class T(Binary, endian="<"):
        ln: u16
        s:  Str["ln"]
    assert T(s="hello").ln == 5


def test_30_rule_a_length_referenced_by_two_fields_stays_required():
    """rule: a length referenced by two fields stays required"""
    class T(Binary, endian="<"):
        n: u8
        a: Bytes["n"]
        b: Bytes["n"]
    raises(TypeError, lambda: T(a=b"x", b=b"y"))


def test_31_rule_pack_fills_pointer_offsets():
    """rule: pack() fills in pointer offsets"""
    class P(Binary, endian="<"):
        n:    u8
        off:  u16
        name: Pointer["off", Str["n"]]
    p = P(n=3, name="abc")
    raw = p.pack()
    assert int.from_bytes(raw[1:3], "little") == 3
    assert p.off == 0                       # untouched


def test_32_rule_describe_prints_offsets_sizes_and_types():
    """rule: describe() prints offsets, sizes and types"""
    class T(Binary, endian="<"):
        a: u8
        b: u16
    rows = [l.split() for l in T.describe().splitlines()]
    assert rows[1] == ["off", "size", "field", "type"]
    assert rows[2] == ["0", "1", "a", "u8"]
    assert rows[3] == ["1", "2", "b", "u16"]


def test_33_errors_layouterror_is_raised_at_class_creation():
    """errors: LayoutError is raised at class creation"""
    raises(LayoutError, lambda: type("B", (Binary,), {"__annotations__": {"a": 1.5}}))


def test_34_errors_parseerror_on_truncation():
    """errors: ParseError on truncation"""
    raises(ParseError, lambda: Chunk.unpack(b"\x00"))


def test_35_errors_parseerror_on_const_mismatch():
    """errors: ParseError on Const mismatch"""
    C = type("C", (Binary,), {"__annotations__": {"m": Const[b"AB"]}})
    raises(ParseError, lambda: C.unpack(b"XY"))


def test_36_errors_parseerror_on_checksum_mismatch():
    """errors: ParseError on checksum mismatch"""
    bad = bytearray(BUF); bad[-1] ^= 0xFF
    raises(ParseError, lambda: Chunk.unpack(bytes(bad)), "checksum")


def test_37_errors_parseerror_on_an_unknown_discriminator():
    """errors: ParseError on an unknown discriminator"""
    class T(Binary, endian="<"):
        k: u8
        b: Switch["k", {1: u8}]
    raises(ParseError, lambda: T.unpack(b"\x09\x00"), "no case for discriminator")


def test_38_errors_parseerror_on_trailing_bytes():
    """errors: ParseError on trailing bytes"""
    raises(ParseError, lambda: Chunk.unpack(BUF + b"\x00"), "trailing")


def test_39_errors_builderror_when_a_declared_length_disagrees():
    """errors: BuildError when a declared length disagrees"""
    class T(Binary, endian="<"):
        a:  u8
        ln: u8
        s:  Str["ln"]
    raises(BuildError, lambda: T(a=1, ln=9, s="hi").pack())


def test_40_errors_builderror_when_a_pointer_offset_disagrees():
    """errors: BuildError when a pointer offset disagrees"""
    class P(Binary, endian="<"):
        off: u16
        pad: u16
        d:   Pointer["off * 2", Bytes[1]]
    raises(BuildError, lambda: P(off=9, pad=0, d=b"x").pack())


def test_41_errors_builderror_when_a_value_is_out_of_range():
    """errors: BuildError when a value is out of range"""
    T = type("T", (Binary,), {"__annotations__": {"a": u8}})
    raises(BuildError, lambda: T(a=300).pack())


