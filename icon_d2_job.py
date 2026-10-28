#!/usr/bin/env python3
"""
Holt Niederschlagsvorhersagen des DWD (Open Data), rechnet sie auf Regen pro Stunde um und schreibt
kleine Dateien für die Regenradar-App.

  --model d2   ICON-D2, 48 Stunden, stündlich, 2 km  ->  icon_d2.bin.gz + latest.json
  --model eu   ICON-EU, 120 Stunden (stündlich bis +78, dann 3-stündlich), 6,25 km
               ->  icon_eu.bin.gz + latest_eu.json   (Format RGN2, siehe unten)

Ist der neueste Lauf schon veröffentlicht (--published-url), passiert nichts.

Datei-Format RGN1 von icon_d2.bin.gz (nach dem Entpacken, alles Big-Endian):
  4 Byte   "RGN1"
  8 Byte   int64   Laufzeit (Epoch-Sekunden, UTC)
  2 Byte   uint16  nx (Spalten)         2 Byte uint16 ny (Zeilen)
  4x4 Byte float32 lat_nord, lon_west   (Mittelpunkt der Zelle links oben), dlat, dlon (Grad, positiv)
  1 Byte   uint8   Anzahl Zeitschritte n
  2 Byte   uint16  Minuten je Schritt (60)
  danach n mal nx*ny Byte: Regen der Stunde, die zu (Laufzeit + k*60 min) endet, k = 1..n,
  in Zehntel mm (0..254), Zeilen von Nord nach Süd, Spalten von West nach Ost. 0 = trocken/außerhalb.

Format RGN2 (icon_eu.bin.gz): wie RGN1, aber nach "n" folgen n Werte uint16 mit den Vorlaufzeiten
(Minuten ab Laufzeit, Ende des jeweiligen Zeitraums) statt eines festen Schritts; die Werte sind
Zehntel mm pro Stunde (bei 3-Stunden-Schritten durch 3 geteilt).

Benötigt numpy; für ICON-EU (CCSDS-Packung) die Bibliothek libaec (apt: libaec0).
"""
import argparse
import bz2
import datetime as dt
import gzip
import json
import os
import struct
import sys
import time
import urllib.request
import ctypes
import ctypes.util
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BASE = "https://opendata.dwd.de/weather/nwp"
# Ausschnitt, den die App anzeigt (West, Süd, Ost, Nord in Grad)
CROP = (-3.9, 43.2, 20.3, 58.0)

MODELS = {
    "d2": dict(
        path="icon-d2/grib/{hh}/tot_prec/icon-d2_germany_regular-lat-lon_single-level_{run}_{lead:03d}_2d_tot_prec.grib2.bz2",
        runs_every=3, leads=list(range(0, 49)), block=2, out="icon_d2.bin.gz", meta="latest.json", fmt=1,
    ),
    "eu": dict(
        # Läufe 00 und 12 UTC reichen bis +120 Stunden; stündlich bis +78, danach alle 3 Stunden
        path="icon-eu/grib/{hh}/tot_prec/icon-eu_europe_regular-lat-lon_single-level_{run}_{lead:03d}_TOT_PREC.grib2.bz2",
        runs_every=12, leads=list(range(0, 79)) + list(range(81, 121, 3)), block=1,
        out="icon_eu.bin.gz", meta="latest_eu.json", fmt=2,
    ),
}
UA = "regenradar-icon-d2-job/1.0 (private project)"


# --- CCSDS/AEC über libaec -------------------------------------------------------------------

class _AecStream(ctypes.Structure):
    _fields_ = [
        ("next_in", ctypes.c_void_p), ("avail_in", ctypes.c_size_t), ("total_in", ctypes.c_size_t),
        ("next_out", ctypes.c_void_p), ("avail_out", ctypes.c_size_t), ("total_out", ctypes.c_size_t),
        ("bits_per_sample", ctypes.c_uint), ("block_size", ctypes.c_uint), ("rsi", ctypes.c_uint),
        ("flags", ctypes.c_uint), ("state", ctypes.c_void_p),
    ]


def aec_decode(raw: bytes, n: int, bits: int, flags: int, block: int, rsi: int) -> np.ndarray:
    """Entpackt n Werte (bits <= 16 oder 24/32) aus einem CCSDS-Strom."""
    path = ctypes.util.find_library("aec")
    if not path:
        raise RuntimeError("libaec nicht gefunden (apt install libaec0)")
    lib = ctypes.CDLL(path)
    nbytes = (bits + 7) // 8
    if nbytes == 3:
        nbytes = 4 if flags & 2 == 0 else 3
    src = ctypes.create_string_buffer(raw, len(raw))
    out = ctypes.create_string_buffer(n * nbytes)
    st = _AecStream()
    st.next_in = ctypes.cast(src, ctypes.c_void_p)
    st.avail_in = len(raw)
    st.next_out = ctypes.cast(out, ctypes.c_void_p)
    st.avail_out = n * nbytes
    st.bits_per_sample, st.block_size, st.rsi, st.flags = bits, block, rsi, flags
    rc = lib.aec_buffer_decode(ctypes.byref(st))
    if rc != 0 or st.total_out != n * nbytes:
        raise ValueError(f"CCSDS-Entpacken fehlgeschlagen (rc={rc}, {st.total_out}/{n * nbytes} Byte)")
    if nbytes == 3:
        b = np.frombuffer(out.raw, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
        return (b[:, 0] << 16 | b[:, 1] << 8 | b[:, 2]).astype(np.float64)
    order = ">" if flags & 4 else "<"
    return np.frombuffer(out.raw, dtype=f"{order}u{nbytes}").astype(np.float64)


# --- GRIB2 (einfache Packung und CCSDS) ---------------------------------------------------------

def _signed(v: int, bits: int) -> int:
    """GRIB speichert Vorzeichen und Betrag getrennt."""
    mag = v & ((1 << (bits - 1)) - 1)
    return -mag if v >> (bits - 1) else mag


def decode_grib(data: bytes) -> dict:
    """Liest eine GRIB2-Datei mit einem Feld (Gitter regulär, Packung 5.0). Gibt Werte als
    float32-Feld [ny, nx] (NaN = ungültig) und die Gitterdaten zurück."""
    if data[:4] != b"GRIB" or data[7] != 2:
        raise ValueError("keine GRIB2-Datei")
    pos, secs = 16, {}
    while data[pos:pos + 4] != b"7777":
        length, num = struct.unpack(">IB", data[pos:pos + 5])
        secs[num] = data[pos:pos + length]
        pos += length
    s1, s3, s5, s6, s7 = (secs[k] for k in (1, 3, 5, 6, 7))
    year, mon, day, hour, minute, sec = struct.unpack(">HBBBBB", s1[12:19])
    run = dt.datetime(year, mon, day, hour, minute, sec, tzinfo=dt.timezone.utc)

    if struct.unpack(">H", s3[12:14])[0] != 0:
        raise ValueError("Gitter ist nicht regulär (Vorlage 3.0)")
    npts_total = struct.unpack(">I", s3[6:10])[0]
    ni, nj = struct.unpack(">II", s3[30:38])
    la1, lo1, _, la2, lo2, di, dj, scan = struct.unpack(">iiBiiIIB", s3[46:72])
    if ni * nj != npts_total:
        raise ValueError("Gitterangaben passen nicht zusammen")
    if scan != 64:
        raise ValueError(f"unerwartete Abtastrichtung {scan}")
    lo1 = lo1 / 1e6
    lo1 = lo1 - 360.0 if lo1 > 180 else lo1
    grid = dict(nx=ni, ny=nj, lat_s=la1 / 1e6, lon_w=lo1, dlat=dj / 1e6, dlon=di / 1e6)

    n_valid = struct.unpack(">I", s5[5:9])[0]
    tmpl5 = struct.unpack(">H", s5[9:11])[0]
    if tmpl5 not in (0, 42):
        raise ValueError(f"Packung 5.{tmpl5} wird nicht unterstützt")
    ref = struct.unpack(">f", s5[11:15])[0]
    e = _signed(struct.unpack(">H", s5[15:17])[0], 16)
    d = _signed(struct.unpack(">H", s5[17:19])[0], 16)
    bits = s5[19]

    if s6[5] == 0:
        mask = np.unpackbits(np.frombuffer(s6[6:], dtype=np.uint8))[:npts_total].astype(bool)
    elif s6[5] == 255:
        mask = np.ones(npts_total, dtype=bool)
    else:
        raise ValueError("unbekannte Bitmaske")
    if int(mask.sum()) != n_valid:
        raise ValueError("Anzahl gültiger Punkte passt nicht zur Bitmaske")

    raw = s7[5:]
    if bits == 0:
        x = np.zeros(n_valid, dtype=np.float64)
    elif tmpl5 == 42:
        x = aec_decode(raw, n_valid, bits, s5[21], s5[22], struct.unpack(">H", s5[23:25])[0])
    elif bits in (8, 16, 32):
        x = np.frombuffer(raw[: n_valid * bits // 8], dtype=f">u{bits // 8}").astype(np.float64)
    else:
        raise ValueError(f"{bits} Bit je Wert wird nicht unterstützt")
    vals = ((ref + x * 2.0 ** e) / 10.0 ** d).astype(np.float32)

    field = np.full(npts_total, np.nan, dtype=np.float32)
    field[mask] = vals
    return dict(run=run, field=field.reshape(nj, ni), **grid)


# --- Umrechnen -------------------------------------------------------------------------------

def crop_indices(grid: dict):
    """Zeilen-/Spaltenbereich (Süd->Nord, West->Ost) im Gitter, der in CROP liegt."""
    west, south, east, north = CROP
    c0 = max(0, int(np.ceil((west - grid["lon_w"]) / grid["dlon"] - 1e-6)))
    c1 = min(grid["nx"] - 1, int(np.floor((east - grid["lon_w"]) / grid["dlon"] + 1e-6)))
    r0 = max(0, int(np.ceil((south - grid["lat_s"]) / grid["dlat"] - 1e-6)))
    r1 = min(grid["ny"] - 1, int(np.floor((north - grid["lat_s"]) / grid["dlat"] + 1e-6)))
    return r0, r1, c0, c1


def to_tenths(diff: np.ndarray, hours: float, block: int) -> np.ndarray:
    """Regenmenge über [hours] Stunden (mm), Süd oben -> mm/h in Zehntel, Nord oben, auf block x block gemittelt."""
    diff = np.maximum(np.nan_to_num(diff, nan=0.0), 0.0) / hours  # Rundungsrauschen weg, mm pro Stunde
    ny, nx = diff.shape
    if block > 1:
        ny2, nx2 = ny // block, nx // block
        diff = diff[: ny2 * block, : nx2 * block].reshape(ny2, block, nx2, block).mean(axis=(1, 3))
    diff = diff[::-1]  # Süd->Nord wird Nord->Süd
    return np.clip(np.rint(diff * 10.0), 0, 254).astype(np.uint8)


def build(model: str, run: dt.datetime, leads: list, fields: list, grid: dict) -> bytes:
    """fields[i] = Summenfeld zur Vorlaufzeit leads[i] (bereits auf CROP zugeschnitten, Süd->Nord).
    Liefert den unkomprimierten Dateiinhalt."""
    cfg = MODELS[model]
    block = cfg["block"]
    steps, kept = [], []
    for i in range(1, len(leads)):
        hours = leads[i] - leads[i - 1]
        steps.append(to_tenths(fields[i] - fields[i - 1], hours, block))
        kept.append(leads[i])
    ny2, nx2 = steps[0].shape
    ny, nx = fields[0].shape
    # Mitte der (block x block)-Zelle links oben; "lat_s"/"lon_w" des zugeschnittenen Gitters
    off = (block - 1) / 2.0
    lat_n_c = grid["lat_s"] + (ny - 1) * grid["dlat"] - (ny - ny2 * block) * grid["dlat"] - off * grid["dlat"]
    lon_w_c = grid["lon_w"] + off * grid["dlon"]
    head = b"RGN1" if cfg["fmt"] == 1 else b"RGN2"
    head += struct.pack(
        ">qHHffffB", int(run.timestamp()), nx2, ny2,
        lat_n_c, lon_w_c, grid["dlat"] * block, grid["dlon"] * block, len(steps),
    )
    if cfg["fmt"] == 1:
        head += struct.pack(">H", 60)
    else:
        head += struct.pack(f">{len(kept)}H", *[m * 60 for m in kept])
    return head + b"".join(s.tobytes() for s in steps)


# --- Download --------------------------------------------------------------------------------

def _get(url: str, tries: int = 4) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


def _exists(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30):
            return True
    except Exception:  # noqa: BLE001
        return False


def url_for(model: str, run: dt.datetime, lead: int) -> str:
    cfg = MODELS[model]
    return f"{BASE}/" + cfg["path"].format(hh=run.strftime("%H"), run=run.strftime("%Y%m%d%H"), lead=lead)


def latest_complete_run(model: str, now: dt.datetime) -> dt.datetime:
    """Neuester Lauf, dessen letzte Datei schon da ist."""
    cfg = MODELS[model]
    every = cfg["runs_every"]
    t = now.replace(minute=0, second=0, microsecond=0)
    t -= dt.timedelta(hours=t.hour % every)
    for _ in range(6):
        if _exists(url_for(model, t, cfg["leads"][-1])):
            return t
        t -= dt.timedelta(hours=every)
    raise RuntimeError(f"kein vollständiger Lauf von {model} gefunden")


def load_run(model: str, run: dt.datetime, local: str | None) -> tuple:
    """Lädt alle Vorlaufzeiten, schneidet auf CROP zu. Gibt (Felder, Gitter) zurück."""
    cfg = MODELS[model]

    def fetch(lead: int) -> bytes:
        if local:
            name = os.path.basename(url_for(model, run, lead))
            with open(os.path.join(local, name), "rb") as f:
                return f.read()
        return _get(url_for(model, run, lead))

    def one(lead: int):
        g = decode_grib(bz2.decompress(fetch(lead)))
        if g["run"] != run:
            raise RuntimeError(f"Stunde {lead}: Laufzeit {g['run']} statt {run}")
        r0, r1, c0, c1 = crop_indices(g)
        sub = np.ascontiguousarray(g["field"][r0:r1 + 1, c0:c1 + 1])
        grid = dict(nx=c1 - c0 + 1, ny=r1 - r0 + 1, lat_s=g["lat_s"] + r0 * g["dlat"],
                    lon_w=g["lon_w"] + c0 * g["dlon"], dlat=g["dlat"], dlon=g["dlon"])
        return sub, grid

    with ThreadPoolExecutor(max_workers=6) as pool:
        res = list(pool.map(one, cfg["leads"]))
    base = res[0][1]
    for _, g in res:
        if g != base:
            raise RuntimeError("Gitter ändert sich zwischen den Zeitschritten")
    return [f for f, _ in res], base


def published_run(url: str | None, meta: str) -> str | None:
    """Laufzeit der schon veröffentlichten Datei (oder None)."""
    if not url:
        return None
    try:
        return json.loads(_get(url.rstrip("/") + "/" + meta, tries=1)).get("run")
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="d2")
    ap.add_argument("--out", default="out")
    ap.add_argument("--local", help="Ordner mit schon geladenen Dateien (Test)")
    ap.add_argument("--run", help="Laufzeit JJJJMMTTHH (Test); sonst der neueste")
    ap.add_argument("--published-url", default=os.environ.get("PUBLISHED_URL"),
                    help="Adresse, unter der die Dateien liegen; ist der Lauf dort schon aktuell, wird nichts getan")
    a = ap.parse_args()
    cfg = MODELS[a.model]

    if a.run:
        run = dt.datetime.strptime(a.run, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
    else:
        run = latest_complete_run(a.model, dt.datetime.now(dt.timezone.utc))
    run_text = run.strftime("%Y-%m-%dT%H:%M:%SZ")
    if not a.local and published_run(a.published_url, cfg["meta"]) == run_text:
        print(f"{a.model}: Lauf {run_text} ist schon veröffentlicht, nichts zu tun")
        return 0

    t0 = time.time()
    fields, grid = load_run(a.model, run, a.local)
    raw = build(a.model, run, cfg["leads"], fields, grid)
    os.makedirs(a.out, exist_ok=True)
    gz = gzip.compress(raw, compresslevel=9, mtime=0)
    tmp = os.path.join(a.out, cfg["out"] + ".tmp")
    with open(tmp, "wb") as f:
        f.write(gz)
    os.replace(tmp, os.path.join(a.out, cfg["out"]))
    meta = dict(
        run=run_text, run_epoch=int(run.timestamp()), steps=len(cfg["leads"]) - 1,
        bytes=len(gz), created=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    with open(os.path.join(a.out, cfg["meta"]), "w") as f:
        json.dump(meta, f)
    print(f"{a.model}: Lauf {run_text}: {len(raw)/1e6:.1f} MB roh, {len(gz)/1e6:.2f} MB gzip, {time.time()-t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
