# pybinary

Declarative binary structures for Python that **compile to specialized codecs**
when the class is created, instead of walking a field tree on every call.

```python
from pybinary import Binary, Const, Bytes, Array, u8, u16, u32, f64

class Payload(Binary, endian="<"):
    kind:  u8
    value: f64

class Header(Binary, endian="<"):
    magic:   Const[b"BLOB"]
    version: u16
    n:       u32
    items:   Bytes[n]          # length taken from an earlier field
    tags:    Array[u16, n]     # count taken from an earlier field
    body:    Payload           # nested structure

h = Header.unpack(buf)
raw = h.pack()
```

Requires Python 3.13+. Unquoted field references like `Bytes[n]` rely on PEP 649
lazy annotations and need 3.14; the quoted form `Bytes["n"]` works everywhere and
is the recommended spelling.

## Why

`construct` has the field vocabulary but interprets a tree of subcon objects on
every parse. `dataclasses-struct` and `ctypes` are fast but only handle
fixed-size layouts, so no length-prefixed fields, no arrays sized by an earlier
field, no nesting. pybinary keeps the declarative surface and generates code:

- runs of consecutive fixed-width fields collapse into **one** `struct.Struct` call
- variable-length fields are the only interpreted steps
- a length expression such as `Bytes[n * 2 + 1]` is **inlined into the generated
  source**, so there is no runtime context dict and no expression evaluation

The generated source is on `cls.__codec_source__`:

```python
>>> print(Header.__codec_source__)
def _unpack_view(_buf, _off):
    _lim = _len(_buf)
    _o = _new(_cls)
    if _off + 10 > _lim: _short(_CLS, 'magic', _off, 10, _lim)
    _k0_0, _o.version, n = _s0.unpack_from(_buf, _off)
    ...
```

## Field types

| Declaration | Meaning |
| --- | --- |
| `u8 u16 u32 u64 i8 i16 i32 i64 f32 f64` | fixed-width numbers |
| `Const[b"BLOB"]` / `Const[0x01312F76]` / `Const["76 2f 31 01"]` | literal, verified on parse, not stored |
| `Bytes[4]` / `Bytes[n]` / `Bytes[n * 2]` / `Bytes["n"]` / `Bytes[...]` | raw bytes; `...` is the rest of the buffer |
| `Str[n]` / `Str[4, "ascii"]` | text, utf-8 by default |
| `Array[u16, n]` / `Array[Point, 4]` | sequence of scalars or structures |
| `Payload` (a `Binary` subclass) | embedded structure |
| `Padding[3]` | skipped bytes, zero-filled on build |
| `Bits[4]` / `Bits[3, 0]` | sub-byte fields, MSB-first |
| `If["ver > 1", u32]` | conditional field, `None` when false |
| `Switch["kind", {1: A, ...: B}]` | variant selected by an earlier field |
| `Pointer["off", Bytes["n"]]` | target stored elsewhere in the record |
| `Checksum[u32, zlib.crc32, "kind"]` | computed and verified over a byte span |

### Magic numbers

`Const` accepts the three ways magic values get written. An integer is encoded in
the structure's byte order, so the constant reads the way the specification
states it rather than the way it lands on the wire:

```python
class Header(Binary, endian="<"):
    magic: Const[0x01312F76]     # -> 76 2f 31 01
    other: Const["76 2f 31 01"]  # hex string, used as-is
    ascii: Const[b"BLOB"]        # bytes, used as-is
```

A `str` is *always* hex — write `b"..."` for an ASCII magic.

### Bitfields

Consecutive `Bits` fields form a run that must total a whole number of bytes, and
are read MSB-first regardless of `endian=` — the order every RFC-style wire
diagram uses. A run collapses into the surrounding `struct` call, so the IPv4
header below still costs one `unpack_from`:

```python
class IPv4(Binary, endian=">"):
    ver:   Bits[4]
    ihl:   Bits[4]
    tos:   u8
    total: u16
```

`Bits[3, 0]` declares a reserved constant: verified on parse, written on build,
not stored. There is no auto-padding — a short run is a `LayoutError`, because
silently widening it would rewrite the wire layout when a field is inserted.

### Conditionals and variants

```python
class Msg(Binary, endian="<"):
    kind:  u8
    extra: If["kind > 1", u32]
    body:  Switch["kind", {1: Ping, 2: Pong, ...: Bytes[4]}]
```

In an `If` condition use `&`, `|` and `~`. Python evaluates `and`/`or`/`not`
eagerly while the annotation is being built, so the left operand is gone before
pybinary can see it — and it is gone from the annotation's source text too, so
there is nothing to detect. A condition that collapses to a plain `bool` *is*
caught, which covers `not x` and the shadowed-global case.

### Pointers and checksums

A `Pointer` offset is counted from the start of its own record, so a record works
wherever it lands; at the top level that is also the file offset. The cursor does
not advance, so the next field reads from just after the pointer.

```python
class Chunk(Binary, endian=">"):
    ln:   u32
    kind: Bytes[4]
    data: Bytes["ln"]
    crc:  Checksum[u32, zlib.crc32, "kind"]   # covers kind..here
```

`pack()` fills in a pointer offset written as a bare field name used by exactly
one pointer — the target is appended after the record and the offset patched in
place. Any other offset expression must agree with where the target lands. Those
computed offsets are left out of `__eq__` so a hand-built record still compares
equal to a parsed one; `pack()` itself never mutates the instance.

`endian=` accepts `"<"`, `">"`, `"!"`, `"="` or `"little"`, `"big"`,
`"network"`, `"native"`, and is inherited by subclasses.

### Inheritance

A subclass appends its fields to the base's, and length references reach across
the boundary. The combined layout is re-segmented, so base and subclass fields
still collapse into shared `struct` calls:

```python
class Head(Binary, endian="<"):
    kind: u8
    n:    u16

class Body(Head):
    data: Bytes[n]      # 'n' is declared in the base class body
    tail: u8
```

Redeclaring an inherited field is a `LayoutError`, as is inheriting fields from
two structures at once.

A nested field's **declared class fixes the wire layout**, so passing a subclass
instance is a `BuildError` rather than a silent encoding of extra bytes the
reader will never decode:

```python
class Holder(Binary):
    h: Head

Holder(h=Body(...)).pack()   # BuildError: expected a Head instance, got Body
```

`Switch` applies the same rule to each variant, so a tagged union cannot silently
encode the wrong arm under a given tag.

### Field references and shadowing

`Bytes[n]` works because PEP 649 evaluates annotations lazily, so `n` is
unresolved at class-body time and arrives as a forward reference. But
annotations resolve against module globals and builtins *first*, so a name that
exists out there wins:

```python
n = 5                       # module-level

class Trap(Binary):
    n: u8
    data: Bytes[n]          # LayoutError - 'n' resolved to the global, not the field
```

pybinary compares the annotation's source text against what actually resolved
and refuses to compile rather than silently encoding a static length of 5. Quote
the reference to force the field:

```python
class Ok(Binary):
    len:  u8
    data: Bytes["len"]      # quoted names always mean the field
```

## Performance

`uv run bench/bench_construct.py`. Python 3.14.6, Apple silicon, µs per record;
the benchmark asserts both libraries produce identical bytes before timing
anything.

| | pybinary | construct | hand-written `struct` |
| --- | --- | --- | --- |
| fixed-size record, parse | **0.15** | 2.03 (13x) | 0.03 |
| variable-length record, parse | **0.44** | 6.27 (14x) | 0.25 |
| tagged union, parse | **0.25** | 2.74 (11x) | — |
| variable-length record, build | **0.33** | 5.60 (17x) | — |

The hand-written column only unpacks tuples; it does not build an object. A
slotted class populated from `struct.unpack` costs 0.077 µs against pybinary's
0.107 µs for the same record, so the generated codec is close to the ceiling for
anything that returns real objects.

## Zero-copy

`unpack(buf, copy=False)` returns `Bytes` fields as **memoryviews over the
caller's buffer** — nothing is copied. That means the parsed object keeps the
source buffer alive, and mutating the buffer changes what the object reports.

```python
h = Header.unpack(buf)               # h.items is bytes, buffer not retained
h = Header.unpack(buf, copy=False)   # h.items is a memoryview into buf
```

Copying is the default because zero-copy is not automatically faster, and a
memoryview surprises anything downstream that expects `bytes` — `.decode()`,
`json`, hashing. Parse time for a single `Bytes` field:

| payload | zero-copy | `copy=True` |
| --- | --- | --- |
| 16 B | 0.272 | **0.211** |
| 256 B | 0.285 | **0.212** |
| 4 KiB | **0.293** | 0.265 |
| 64 KiB | **0.291** | 0.788 |

Zero-copy is flat in payload size; copying is cheaper below roughly 4 KiB
because building the memoryview costs more than a small memcpy. Reach for
`copy=False` on large payloads, and accept that it pins the source buffer.
Structures with no `Bytes` field skip the memoryview entirely in both modes.

## Lengths on build

A length field is filled in automatically when exactly one field references it
by bare name:

```python
class Msg(Binary):
    ln:   u16
    text: Str[ln]

Msg(text="hello").ln        # 5
```

When several fields share a length field it stays required, because deriving it
from one of them could silently contradict the other. Either way `pack()`
validates every declared length and raises `BuildError` on a mismatch, so a
record can never be silently encoded with an inconsistent header.

## Errors

- `LayoutError` — bad declaration, raised when the class is created (forward
  reference, unknown field type, reference to a non-integer field, a reference
  captured by a global or builtin)
- `ParseError` — truncated buffer, `Const` mismatch, trailing bytes
- `BuildError` — declared length disagrees with the value, or a value is out of
  range for its field

## Not yet

Untagged unions, lazy or streaming parse, and `from __future__ import annotations`
(PEP 563 stringifies every annotation; pybinary detects it and tells you to remove
the import).
