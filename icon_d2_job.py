#!/usr/bin/env python3
"""
Holt den neuesten ICON-D2-Lauf (Gesamtniederschlag, DWD Open Data), rechnet auf Stundenregen um
und schreibt eine kleine Datei für die Regenradar-App.

Ausgabe in OUT_DIR:
  icon_d2.bin.gz   die Daten (Format siehe unten)
  latest.json      Lauf-Zeit, Größe, Erstellzeit (zum billigen Nachsehen, ob es etwas Neues gibt)

Datei-Format von icon_d2.bin.gz (nach dem Entpacken, alles Big-Endian):
  4 Byte   "RGN1"
  8 Byte   int64   Laufzeit (Epoch-Sekunden, UTC)
  2 Byte   uint16  nx (Spalten)         2 Byte uint16 ny (Zeilen)
  4x4 Byte float32 lat_nord, lon_west   (Mittelpunkt der Zelle links oben), dlat, dlon (Grad, positiv)
  1 Byte   uint8   Anzahl Zeitschritte n
  2 Byte   uint16  Minuten je Schritt (60)
  danach n mal nx*ny Byte: Regen der Stunde, die zu (Laufzeit + k*60 min) endet, k = 1..n,
  in Zehntel mm (0..254), Zeilen von Nord nach Süd, Spalten von West nach Ost. 0 = trocken/außerhalb.

Nur Standardbibliothek + numpy.
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
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BASE = "https://opendata.dwd.de/weather/nwp/icon-d2/grib"
NAME = "icon-d2_germany_regular-lat-lon_single-level_{run}_{lead:03d}_2d_tot_prec.grib2.bz2"
MAX_LEAD = 48
BLOCK = 2  # 2x2 Zellen zusammenfassen: 0,02 Grad (2 km) -> 0,04 Grad (4 km)
UA = "regenradar-icon-d2-job/1.0 (private project)"


# --- GRIB2 (einfache Packung) --------------------------------------------------------------

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
    if struct.unpack(">H", s5[9:11])[0] != 0:
        raise ValueError("Packung ist nicht einfach (Vorlage 5.0)")
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
    elif bits in (8, 16, 32):
        x = np.frombuffer(raw[: n_valid * bits // 8], dtype=f">u{bits // 8}").astype(np.float64)
    else:
        raise ValueError(f"{bits} Bit je Wert wird nicht unterstützt")
    vals = ((ref + x * 2.0 ** e) / 10.0 ** d).astype(np.float32)

    field = np.full(npts_total, np.nan, dtype=np.float32)
    field[mask] = vals
    return dict(run=run, field=field.reshape(nj, ni), **grid)


# --- Umrechnen -------------------------------------------------------------------------------

def hourly_tenths(prev, cur) -> np.ndarray:
    """Stundenregen aus zwei aufeinanderfolgenden Summen, auf BLOCK x BLOCK gemittelt,
    in Zehntel mm, Nord oben."""
    diff = np.nan_to_num(cur - prev, nan=0.0)
    diff = np.maximum(diff, 0.0)  # Rundungsrauschen
    ny, nx = diff.shape
    ny2, nx2 = ny // BLOCK, nx // BLOCK
    small = diff[: ny2 * BLOCK, : nx2 * BLOCK].reshape(ny2, BLOCK, nx2, BLOCK).mean(axis=(1, 3))
    small = small[::-1]  # Süd->Nord wird Nord->Süd
    return np.clip(np.rint(small * 10.0), 0, 254).astype(np.uint8)


def build(run: dt.datetime, fields: list, grid: dict) -> bytes:
    """fields[k] = Summenfeld zur Stunde k (k = 0..n). Liefert den unkomprimierten Dateiinhalt."""
    n = len(fields) - 1
    steps = [hourly_tenths(fields[k - 1], fields[k]) for k in range(1, n + 1)]
    ny2, nx2 = steps[0].shape
    lat_n = grid["lat_s"] + (grid["ny"] - 1) * grid["dlat"]
    # Mitte der 2x2-Zelle liegt eine halbe Ursprungszelle neben dem ersten Punkt
    off = (BLOCK - 1) / 2.0
    lat_n_c = lat_n - (grid["ny"] - ny2 * BLOCK) * grid["dlat"] - off * grid["dlat"]
    lon_w_c = grid["lon_w"] + off * grid["dlon"]
    head = b"RGN1" + struct.pack(
        ">qHHffffBH", int(run.timestamp()), nx2, ny2,
        lat_n_c, lon_w_c, grid["dlat"] * BLOCK, grid["dlon"] * BLOCK, n, 60,
    )
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


def url_for(run: dt.datetime, lead: int) -> str:
    tag = run.strftime("%Y%m%d%H")
    return f"{BASE}/{run.strftime('%H')}/tot_prec/" + NAME.format(run=tag, lead=lead)


def latest_complete_run(now: dt.datetime) -> dt.datetime:
    """Neuester Lauf, dessen letzte Datei (Stunde 48) schon da ist. ICON-D2 läuft alle 3 Stunden."""
    t = now.replace(minute=0, second=0, microsecond=0)
    t -= dt.timedelta(hours=t.hour % 3)
    for _ in range(6):
        if _exists(url_for(t, MAX_LEAD)):
            return t
        t -= dt.timedelta(hours=3)
    raise RuntimeError("kein vollständiger ICON-D2-Lauf gefunden")


def load_run(run: dt.datetime, local: str | None) -> list:
    def fetch(lead: int) -> bytes:
        if local:
            with open(os.path.join(local, NAME.format(run=run.strftime("%Y%m%d%H"), lead=lead)), "rb") as f:
                return f.read()
        return _get(url_for(run, lead))

    def one(lead: int):
        g = decode_grib(bz2.decompress(fetch(lead)))
        if g["run"] != run:
            raise RuntimeError(f"Stunde {lead}: Laufzeit {g['run']} statt {run}")
        return g

    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(one, range(0, MAX_LEAD + 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--local", help="Ordner mit schon geladenen Dateien (Test)")
    ap.add_argument("--run", help="Laufzeit JJJJMMTTHH (Test); sonst der neueste")
    a = ap.parse_args()

    if a.run:
        run = dt.datetime.strptime(a.run, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
    else:
        run = latest_complete_run(dt.datetime.now(dt.timezone.utc))
    t0 = time.time()
    grids = load_run(run, a.local)
    base = grids[0]
    for g in grids:
        for k in ("nx", "ny", "lat_s", "lon_w", "dlat", "dlon"):
            if g[k] != base[k]:
                raise RuntimeError("Gitter ändert sich zwischen den Stunden")
    raw = build(run, [g["field"] for g in grids], base)
    os.makedirs(a.out, exist_ok=True)
    gz = gzip.compress(raw, compresslevel=9, mtime=0)
    with open(os.path.join(a.out, "icon_d2.bin.gz.tmp"), "wb") as f:
        f.write(gz)
    os.replace(os.path.join(a.out, "icon_d2.bin.gz.tmp"), os.path.join(a.out, "icon_d2.bin.gz"))
    meta = dict(
        run=run.strftime("%Y-%m-%dT%H:%M:%SZ"), run_epoch=int(run.timestamp()), steps=MAX_LEAD,
        bytes=len(gz), created=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    with open(os.path.join(a.out, "latest.json"), "w") as f:
        json.dump(meta, f)
    print(f"Lauf {meta['run']}: {len(raw)/1e6:.1f} MB roh, {len(gz)/1e6:.2f} MB gzip, {time.time()-t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
