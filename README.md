# ICON-D2-Job für die Regenradar-App

`icon_d2_job.py` holt den neuesten ICON-D2-Lauf (Gesamtniederschlag) von opendata.dwd.de,
rechnet auf Stundenregen (4 km, Zehntel mm) um und schreibt `icon_d2.bin.gz` + `latest.json`.
Der Workflow `.github/workflows/icon-d2.yml` läuft alle 3 Stunden und lädt beides per (S)FTP
auf das Webhosting. Dateiformat: siehe Kopf von `icon_d2_job.py`.

## Einrichten
1. Neues **öffentliches** GitHub-Repository anlegen, den Inhalt dieses Ordners hochladen.
2. Unter *Settings → Secrets and variables → Actions* anlegen:
   `FTP_PROTOCOL` (`sftp` oder `ftp`), `FTP_HOST`, `FTP_USER`, `FTP_PASSWORD`, `FTP_DIR`.
   Den FTP-Benutzer in KAS auf einen eigenen Ordner beschränken.
3. Unter *Actions* den Workflow „ICON-D2 Regenvorhersage“ einmal von Hand starten ("Run workflow").

## Test ohne Netz
`python icon_d2_job.py --local ORDNER --run 2026102621 --out out`
Datenbasis: Deutscher Wetterdienst (opendata.dwd.de), umgerechnet.
