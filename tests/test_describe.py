"""``describe()`` — the layout table used to check a structure against a dump."""

from pybinary import (
    Array, Binary, Bytes, Const, Padding, Str,
    f64, u8, u16, u32,
)


class Point(Binary, endian="<"):
    kind: u8
    val:  f64


class Header(Binary, endian="<"):
    magic: Const[b"BLOB"]
    ver:   u16
    n:     u32
    name:  Str[8, "ascii"]
    items: Bytes["n"]
    tags:  Array[u16, "n"]
    pair:  Array[Point, 2]
    body:  Point


def test_fixed_layout_reports_every_offset():
    lines = Point.describe().splitlines()
    assert lines[0] == "Point  endian='<'  (9 bytes)"
    assert lines[1].split() == ["off", "size", "field", "type"]
    assert lines[2].split() == ["0", "1", "kind", "u8"]
    assert lines[3].split() == ["1", "8", "val", "f64"]


def test_offsets_run_until_the_first_variable_field():
    rows = [ln.split() for ln in Header.describe().splitlines()[2:]]
    by_name = {r[2]: r for r in rows}

    assert by_name["magic"][:2] == ["0", "4"]
    assert by_name["ver"][:2] == ["4", "2"]
    assert by_name["n"][:2] == ["6", "4"]
    # a statically sized Str still contributes a known width
    assert by_name["name"][:2] == ["10", "8"]
    # ...and the first variable field is where offsets stop being knowable
    assert by_name["items"] == ["18", "?", "items", "Bytes['n']"]
    assert by_name["tags"][0] == "?"


def test_static_widths_survive_an_unknown_offset():
    by_name = {
        ln.split()[2]: ln.split()
        for ln in Header.describe().splitlines()[2:]
    }
    # position unknown, but the width of a fixed-size struct still is
    assert by_name["body"][:2] == ["?", "9"]
    assert by_name["pair"][:2] == ["?", "18"]


def test_declarations_round_trip_to_their_source_form():
    text = Header.describe()
    for decl in (
        "Const[b'BLOB']", "u16", "u32", "Str[8, 'ascii']",
        "Bytes['n']", "Array[u16, 'n']", "Array[Point, 2]", "Point",
    ):
        assert decl in text, decl


def test_variable_structures_say_so():
    assert Header.describe().startswith("Header  endian='<'  (variable)")


def test_padding_and_utf8_str_render_without_noise():
    class P(Binary, endian=">"):
        a:    u8
        _pad: Padding[3]
        s:    Str[4]
        n:    Point

    text = P.describe()
    assert "Padding[3]" in text
    assert "Str[4]" in text and "utf-8" not in text
    assert "Point" in text
    assert text.startswith("P  endian='>'  (17 bytes)")
