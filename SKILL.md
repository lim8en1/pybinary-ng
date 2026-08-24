---
name: pybinary
description: Declarative binary structures for Python that compile to specialized codecs. Use when parsing or building binary formats and wire protocols - headers, chunked formats, TLV, bitfields, checksums.
---

# pybinary

Fields are plain annotations. Python 3.13+.

```python
import zlib
from pybinary import Binary, Bytes, Checksum, u32

class Chunk(Binary, endian=">"):    # "<" ">" "!" "="
    ln:   u32
    kind: Bytes[4]
    data: Bytes["ln"]               # length from an earlier int field
    crc:  Checksum[u32, zlib.crc32, "kind"]

c    = Chunk.unpack(buf)            # strict: rejects trailing bytes
c, e = Chunk.unpack_from(buf, 8)    # one record at an offset -> (obj, end)
raw  = c.pack()
```

| Declaration | Meaning |
|---|---|
| `u8 u16 u32 u64 i8 i16 i32 i64 f32 f64` | fixed-width numbers |
| `Const[b"BLOB"]` `Const[0x1A2B]` `Const["1a 2b"]` | literal; int uses the struct's byte order, str is hex |
| `Bytes["n"]` `Bytes[4]` `Bytes[...]` | raw bytes; `...` = rest of buffer |
| `Str["n"]` `Str[4, "ascii"]` | text, utf-8 default |
| `Array[u16, "n"]` `Array[Rec, 4]` | sequence |
| `Rec` (a Binary subclass) | embedded struct |
| `Padding[3]` | skipped, zero-filled |
| `Bits[4]` `Bits[3, 0]` | sub-byte, MSB-first; a run must total whole bytes. 2-arg = constant |
| `If["ver > 1", u32]` | present only when true, else `None` |
| `Switch["kind", {1: A, 2: B, ...: C}]` | variant by earlier field; `...` = fallback, `None` = no bytes |
| `Pointer["off", Bytes["n"]]` | at `off` from *this record's* start; cursor stays put. `, "nullable"`: 0 = absent |
| `Checksum[u32, zlib.crc32, "kind"]` | any callable, over `kind`..here |

`Const`, `Padding` and `Checksum` are verified on parse and written on build, but are not attributes.

Rules:
- **Quote field references** (`Bytes["n"]`). Unquoted needs 3.14, and hits globals first.
- In `If` conditions use `&` `|` `~`, never `and`/`or`/`not` — Python evaluates those while the
  annotation is built, so pybinary silently sees a different condition.
- Define variant and nested classes *before* the class using them.
- `pack()` fills in a length referenced by exactly one field by bare name, and pointer offsets.
- `Chunk.describe()` tabulates offsets, sizes and types — check a layout against a hex dump.

Errors: `LayoutError` at class creation · `ParseError` (truncation, `Const`/checksum mismatch, unknown
discriminator, trailing bytes) · `BuildError` (length or offset disagrees, value out of range).
