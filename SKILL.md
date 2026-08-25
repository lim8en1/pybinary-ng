---
name: pybinary
description: Declarative binary structures for Python that compile to specialized codecs. Use when parsing or building binary formats and wire protocols - headers, chunked formats, TLV, bitfields, checksums.
---

# pybinary

Fields are plain annotations. Python 3.13+.

Install with: `uv pip install https://github.com/lim8en1/pybinary-ng.git`

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

The slots taking an expression are `Bytes`/`Str` length, `Array` count, `Pointer` offset, `If`
condition and `Switch` tag:
- Names must be **integer** fields (`u*` `i*` `Bits`) declared *earlier in the same class*, or in a
  base. No dotted paths — a nested struct cannot see its parent's fields.
- Grammar: names, int literals, `+ - * // % & | ^ ~ << >>`. No `/`, no calls; `If` adds comparisons.
- `Padding[3]` and `Bits[4]` take a **literal int only**; for a variable gap use `Bytes["off - 8"]`.
- `Bytes[...]`/`Str[...]` run to the end of the buffer. There is no greedy `Array`.

A `Pointer` offset counts from the start of *this record*, so at the top level it is the file
offset. Its target may be a scalar, struct or `Array`, not `If`/`Switch`/`Pointer`/`Checksum`.
`pack()` appends the target and patches the offset only when the offset is a bare name used by one
pointer; any other expression must already agree. A struct holding a pointer may not also use
`Bytes[...]`, and its `unpack()` stops rejecting trailing bytes.

`describe()` checks a layout against a hex dump. `?` = known only at parse time, and
`If`/`Switch`/`Pointer`/`Checksum` rows show just the kind:

```text
Ptr  endian='<'  (variable)
  off  size  field  type
    0     4  off    u32
    4     ?  v      pointer
```

Also: `inst.pack_into(bytearray)` appends · `cls.fields()`, `cls.__struct_size__`,
`cls.__codec_source__` · `unpack(buf, copy=False)` gives `Bytes` as memoryviews over `buf`. A
scalar `Array` parses to a tuple, a struct `Array` to a list. `Str` lengths count **bytes**, and
NULs are kept.

Not available: enums, alignment, seek/tell, untagged unions, streaming, greedy `Array`, and a
runtime-chosen byte order — `endian=` is fixed at class creation, so a format with a byte-order
mark (TIFF `II`/`MM`) needs one class tree per order.

Errors: `LayoutError` at class creation · `ParseError` (truncation, `Const`/checksum mismatch, unknown
discriminator, trailing bytes) · `BuildError` (length or offset disagrees, value out of range).
