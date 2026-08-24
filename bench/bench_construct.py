"""Compare pybinary against construct and a hand-written struct baseline.

    uv run bench/bench_construct.py
"""

import struct
import timeit

import construct as C

from pybinary import Array, Binary, Bytes, Const, Switch, f64, i16, u8, u16, u32

N = 20_000

# --------------------------------------------------------------------------
# Case 1: fixed-size record
# --------------------------------------------------------------------------


class PbPoint(Binary, endian="<"):
    x: i16
    y: i16
    z: i16
    w: i16


CPoint = C.Struct(
    "x" / C.Int16sl, "y" / C.Int16sl, "z" / C.Int16sl, "w" / C.Int16sl,
)

POINT_RAW = struct.pack("<4h", 1, -2, 3, -4)
_POINT = struct.Struct("<4h")


# --------------------------------------------------------------------------
# Case 2: variable-length record
# --------------------------------------------------------------------------


class PbPayload(Binary, endian="<"):
    kind: u8
    value: f64


class PbHeader(Binary, endian="<"):
    magic: Const[b"BLOB"]
    version: u16
    n: u32
    items: Bytes[n]
    tags: Array[u16, n]
    body: PbPayload


CHeader = C.Struct(
    "magic" / C.Const(b"BLOB"),
    "version" / C.Int16ul,
    "n" / C.Int32ul,
    "items" / C.Bytes(C.this.n),
    "tags" / C.Array(C.this.n, C.Int16ul),
    "body" / C.Struct("kind" / C.Int8ul, "value" / C.Float64l),
)

ITEMS = b"payload-bytes"
TAGS = tuple(range(len(ITEMS)))
HEADER_RAW = (
    struct.pack("<4sHI", b"BLOB", 3, len(ITEMS))
    + ITEMS
    + struct.pack(f"<{len(TAGS)}H", *TAGS)
    + struct.pack("<Bd", 9, 1.5)
)


def hand_unpack(buf):
    magic, version, n = struct.unpack_from("<4sHI", buf, 0)
    if magic != b"BLOB":
        raise ValueError("bad magic")
    off = 10
    items = buf[off:off + n]
    off += n
    tags = struct.unpack_from(f"<{n}H", buf, off)
    off += n * 2
    kind, value = struct.unpack_from("<Bd", buf, off)
    return version, n, items, tags, kind, value


# --------------------------------------------------------------------------
# Case 3: one large payload, to locate the zero-copy crossover
# --------------------------------------------------------------------------


class PbBlob(Binary, endian="<"):
    n: u32
    data: Bytes[n]


def _blob(size: int) -> bytes:
    return struct.pack("<I", size) + bytes(size)


# --------------------------------------------------------------------------
# Case 4: tagged-union dispatch, the thing construct's Switch is used for
# --------------------------------------------------------------------------


class PbPing(Binary, endian="<"):
    seq: u16


class PbPong(Binary, endian="<"):
    seq: u16
    ok: u8


class PbTagged(Binary, endian="<"):
    kind: u8
    body: Switch["kind", {1: PbPing, 2: PbPong}]


CTagged = C.Struct(
    "kind" / C.Int8ul,
    "body" / C.Switch(C.this.kind, {
        1: C.Struct("seq" / C.Int16ul),
        2: C.Struct("seq" / C.Int16ul, "ok" / C.Int8ul),
    }),
)

TAGGED_RAW = struct.pack("<BHB", 2, 513, 1)


# --------------------------------------------------------------------------


def check():
    """The two libraries must agree before any timing is meaningful."""
    p = PbPoint.unpack(POINT_RAW)
    c = CPoint.parse(POINT_RAW)
    assert (p.x, p.y, p.z, p.w) == (c.x, c.y, c.z, c.w) == (1, -2, 3, -4)
    assert p.pack() == CPoint.build(dict(x=1, y=-2, z=3, w=-4)) == POINT_RAW

    t = PbTagged.unpack(TAGGED_RAW)
    ct = CTagged.parse(TAGGED_RAW)
    assert t.kind == ct.kind == 2
    assert t.body.seq == ct.body.seq and t.body.ok == ct.body.ok
    assert t.pack() == TAGGED_RAW

    h = PbHeader.unpack(HEADER_RAW, copy=True)
    ch = CHeader.parse(HEADER_RAW)
    # construct's Container is a dict subclass, so ch.items would be dict.items.
    assert h.version == ch.version and h.n == ch.n
    assert h.items == ch["items"] == ITEMS
    assert tuple(h.tags) == tuple(ch.tags) == TAGS
    assert (h.body.kind, h.body.value) == (ch.body.kind, ch.body.value) == (9, 1.5)

    built = CHeader.build(dict(
        version=3, n=len(ITEMS), items=ITEMS, tags=list(TAGS),
        body=dict(kind=9, value=1.5),
    ))
    assert h.pack() == built == HEADER_RAW
    print("both libraries agree on parse and build\n")


def report(title, rows):
    base = rows[0][1]
    width = max(len(name) for name, _ in rows)
    print(title)
    print(f"  {'':<{width}}  {'us/rec':>8}  {'vs pybinary':>12}")
    for name, t in rows:
        ratio = f"{t / base:.1f}x" if name != rows[0][0] else "-"
        print(f"  {name:<{width}}  {t * 1e6:>8.3f}  {ratio:>12}")
    print()


def bench(fn):
    return timeit.timeit(fn, number=N) / N


def main():
    check()

    report("fixed-size record, parse", [
        ("pybinary", bench(lambda: PbPoint.unpack(POINT_RAW))),
        ("struct (hand-written)", bench(lambda: _POINT.unpack(POINT_RAW))),
        ("construct", bench(lambda: CPoint.parse(POINT_RAW))),
    ])

    report("variable-length record, parse", [
        ("pybinary (copy)", bench(lambda: PbHeader.unpack(HEADER_RAW))),
        ("pybinary (copy=False)", bench(lambda: PbHeader.unpack(HEADER_RAW, copy=False))),
        ("struct (hand-written)", bench(lambda: hand_unpack(HEADER_RAW))),
        ("construct", bench(lambda: CHeader.parse(HEADER_RAW))),
    ])

    report("tagged union, parse", [
        ("pybinary", bench(lambda: PbTagged.unpack(TAGGED_RAW))),
        ("construct", bench(lambda: CTagged.parse(TAGGED_RAW))),
    ])

    # Zero-copy only pays off once the payload is big enough that skipping the
    # memcpy beats the cost of building memoryview objects.
    rows = []
    for size in (16, 256, 4096, 65536):
        big = _blob(size)
        rows.append((
            f"{size:>6} B payload",
            bench(lambda b=big: PbBlob.unpack(b, copy=False)),
            bench(lambda b=big: PbBlob.unpack(b, copy=True)),
        ))
    width = max(len(r[0]) for r in rows)
    print("zero-copy vs copy, single Bytes field")
    print(f"  {'':<{width}}  {'copy=False':>10}  {'copy (default)':>14}")
    for name, zc, cp in rows:
        print(f"  {name:<{width}}  {zc * 1e6:>9.3f}u  {cp * 1e6:>13.3f}u")
    print()

    pb = PbHeader.unpack(HEADER_RAW, copy=True)
    cargs = dict(version=3, n=len(ITEMS), items=ITEMS, tags=list(TAGS),
                 body=dict(kind=9, value=1.5))
    report("variable-length record, build", [
        ("pybinary", bench(pb.pack)),
        ("construct", bench(lambda: CHeader.build(cargs))),
    ])


if __name__ == "__main__":
    main()
