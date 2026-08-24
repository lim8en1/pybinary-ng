"""``Checksum`` -- a computed, verified integer over a span of earlier bytes."""

import struct
import zlib

import pytest

from pybinary import (
    Binary, Bits, Bytes, Checksum, Const, LayoutError, ParseError, Str,
    f64, u8, u16, u32,
)


class Chunk(Binary, endian=">"):
    ln:   u32
    kind: Bytes[4]
    data: Bytes["ln"]
    crc:  Checksum[u32, zlib.crc32, "kind"]


def test_the_checksum_matches_the_reference_implementation():
    raw = Chunk(ln=4, kind=b"IHDR", data=b"abcd").pack()
    assert raw[-4:] == zlib.crc32(b"IHDRabcd").to_bytes(4, "big")


def test_round_trips():
    c = Chunk(ln=4, kind=b"IHDR", data=b"abcd")
    assert Chunk.unpack(c.pack()) == c


def test_a_checksum_is_not_stored():
    c = Chunk(ln=1, kind=b"IDAT", data=b"x")
    assert not hasattr(c, "crc")
    assert "crc" not in repr(c)


def test_a_mismatch_is_a_parse_error_showing_both_values():
    raw = bytearray(Chunk(ln=4, kind=b"IHDR", data=b"abcd").pack())
    raw[-1] ^= 0xFF
    with pytest.raises(ParseError, match=r"checksum is 0x[0-9a-f]+ but the covered "
                                         r"bytes hash to 0x[0-9a-f]+"):
        Chunk.unpack(bytes(raw))


def test_corrupting_the_covered_data_is_caught():
    raw = bytearray(Chunk(ln=4, kind=b"IHDR", data=b"abcd").pack())
    raw[10] ^= 0xFF   # inside `data`
    with pytest.raises(ParseError, match="checksum"):
        Chunk.unpack(bytes(raw))


def test_bytes_before_the_span_are_not_covered():
    # `ln` precedes `kind`, so changing it changes the parse but not the span
    c = Chunk(ln=4, kind=b"IHDR", data=b"abcd")
    assert c.pack()[-4:] == zlib.crc32(b"IHDRabcd").to_bytes(4, "big")


# --------------------------------------------------------------------------
# spans and widths
# --------------------------------------------------------------------------


def test_a_span_starting_in_a_fixed_run():
    class S(Binary, endian="<"):
        a:   u8
        b:   u16
        sum: Checksum[u8, lambda d: sum(d), "b"]

    s = S(a=1, b=0x0203)
    raw = s.pack()
    assert raw == b"\x01\x03\x02" + bytes([(3 + 2) & 0xFF])
    assert S.unpack(raw) == s


def test_the_result_is_masked_to_the_field_width():
    class Narrow(Binary, endian=">"):
        data: Bytes[4]
        ck:   Checksum[u8, zlib.crc32, "data"]

    n = Narrow(data=b"wxyz")
    assert n.pack()[-1] == zlib.crc32(b"wxyz") & 0xFF
    assert Narrow.unpack(n.pack()) == n


@pytest.mark.parametrize("ty, width", [(u8, 1), (u16, 2), (u32, 4)])
def test_every_integer_width_works(ty, width):
    cls = type("C", (Binary,), {"__annotations__": {
        "data": Bytes[2], "ck": Checksum[ty, zlib.crc32, "data"],
    }})
    c = cls(data=b"hi")
    assert len(c.pack()) == 2 + width
    assert cls.unpack(c.pack()) == c


def test_a_checksum_keeps_a_structure_statically_sized():
    class Fixed(Binary, endian=">"):
        a:  u16
        ck: Checksum[u32, zlib.crc32, "a"]

    assert Fixed.__struct_size__ == 6


def test_the_callable_may_receive_a_memoryview():
    seen = []

    def probe(data):
        seen.append(type(data).__name__)
        return 0

    class P(Binary, endian=">"):
        data: Bytes[2]
        ck:   Checksum[u8, probe, "data"]

    raw = P(data=b"hi").pack()
    P.unpack(raw, copy=False)
    assert seen  # the function ran in both modes without assuming bytes


def test_pack_into_a_prepopulated_buffer():
    c = Chunk(ln=2, kind=b"tEXt", data=b"hi")
    out = bytearray(b"JUNK")
    c.pack_into(out)
    assert bytes(out) == b"JUNK" + c.pack()


# --------------------------------------------------------------------------
# a real PNG
# --------------------------------------------------------------------------


class PNG(Binary, endian=">"):
    magic:  Const[0x89504E470D0A1A0A]
    chunks: Bytes[...]


def _chunk(kind, data):
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data)))


def _png():
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + _chunk(b"IEND", b"")
    )


def test_a_real_png_round_trips_byte_for_byte():
    raw = _png()
    f = PNG.unpack(raw)

    chunks, off = [], 0
    while off < len(f.chunks):
        c, off = Chunk.unpack_from(f.chunks, off)
        chunks.append(c)

    assert [c.kind for c in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    rebuilt = PNG(chunks=b"".join(c.pack() for c in chunks)).pack()
    assert rebuilt == raw


def test_a_corrupt_png_chunk_is_rejected():
    raw = bytearray(_png())
    raw[20] ^= 0xFF
    f = PNG.unpack(bytes(raw))
    with pytest.raises(ParseError, match="checksum"):
        Chunk.unpack_from(f.chunks, 0)


def test_a_wrong_png_magic_is_rejected():
    raw = b"\x89PNGxxxx" + _png()[8:]
    with pytest.raises(ParseError, match="magic"):
        PNG.unpack(raw)


# --------------------------------------------------------------------------
# layout errors
# --------------------------------------------------------------------------


def test_checksum_needs_three_arguments():
    with pytest.raises(LayoutError, match="takes an integer type, a function"):
        class T(Binary):
            a:  u8
            ck: Checksum[u32, zlib.crc32]


def test_the_stored_type_must_be_an_integer_scalar():
    with pytest.raises(LayoutError, match="must store an integer scalar"):
        class T(Binary):
            a:  u8
            ck: Checksum[f64, zlib.crc32, "a"]


def test_the_function_must_be_callable():
    with pytest.raises(LayoutError, match="must be callable"):
        class T(Binary):
            a:  u8
            ck: Checksum[u32, 5, "a"]


def test_the_span_must_name_a_declared_field():
    with pytest.raises(LayoutError, match="not a field declared before this one"):
        class T(Binary):
            a:  u8
            ck: Checksum[u32, zlib.crc32, "nope"]


def test_the_span_may_not_start_at_a_bit_field():
    with pytest.raises(LayoutError, match="cannot start at the bit field"):
        class T(Binary):
            a:  Bits[4]
            b:  Bits[4]
            ck: Checksum[u32, zlib.crc32, "a"]
