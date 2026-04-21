"""
fetch_arome.py  (v2)
====================
Genera arome_data.json de ~10-20 MB (no 350 MB).

Diferencias respecto v1:
  - Solo guarda la MEDIA del ensemble (no los 6 miembros raw)
  - Solo guarda puntos DENTRO del polígono Lyngen (no bbox rectangular completo)
  - Arrays planos: variables[var] = [t0_p0, t0_p1, ..., t1_p0, ...]
  - Redondeo a 2 decimales

Uso:
    pip install netCDF4 numpy
    python fetch_arome.py
"""

import json, time
import numpy as np
import netCDF4 as nc
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────
URL = (
    "https://thredds.met.no/thredds/dodsC/aromearcticlatest/latest/"
    "arome_arctic_lagged_6_h_latest_2_5km_latest.nc"
)

# Bbox región amplia Troms/Nordland
LAT_MIN, LAT_MAX = 68.99, 70.49
LON_MIN, LON_MAX = 16.57, 22.61

# Polígono Lyngen Alps [lon, lat]
POLYGON = [
  [18.464,69.695],[18.351,69.687],[18.180,69.704],[18.303,69.624],
  [18.074,69.613],[18.052,69.557],[18.196,69.525],[18.410,69.540],
  [18.629,69.565],[18.719,69.568],[18.799,69.594],[18.805,69.626],
  [18.725,69.636],[18.758,69.674],[18.891,69.692],[19.023,69.669],
  [18.932,69.605],[18.973,69.541],[19.014,69.528],[18.992,69.495],
  [18.980,69.463],[19.045,69.435],[19.088,69.387],[19.163,69.370],
  [19.307,69.375],[19.419,69.327],[19.430,69.253],[19.555,69.218],
  [19.412,69.223],[19.323,69.234],[19.253,69.197],[19.253,69.110],
  [19.382,69.076],[19.630,69.088],[19.873,69.165],[20.013,69.172],
  [20.301,69.235],[20.477,69.282],[20.736,69.352],[20.908,69.455],
  [20.975,69.506],[21.079,69.733],[21.138,69.812],[21.038,69.790],
  [20.961,69.782],[20.895,69.819],[20.936,69.851],[20.844,69.855],
  [20.984,69.925],[20.891,69.922],[20.829,69.889],[20.771,69.875],
  [20.797,69.853],[20.778,69.827],[20.730,69.836],[20.682,69.869],
  [20.738,69.912],[20.564,69.900],[20.533,69.852],[20.549,69.793],
  [20.534,69.763],[20.478,69.751],[20.537,69.724],[20.478,69.656],
  [20.487,69.610],[20.571,69.592],[20.655,69.529],[20.824,69.498],
  [20.639,69.511],[20.563,69.548],[20.519,69.543],[20.439,69.589],
  [20.375,69.540],[20.308,69.450],[20.254,69.407],[20.261,69.379],
  [20.164,69.379],[20.036,69.333],[19.977,69.292],[19.903,69.257],
  [19.918,69.301],[19.970,69.351],[20.102,69.387],[20.150,69.407],
  [20.162,69.471],[20.243,69.518],[20.228,69.570],[20.320,69.594],
  [20.339,69.638],[20.257,69.679],[20.357,69.854],[20.387,69.889],
  [20.332,69.962],[20.269,69.977],[20.210,69.940],[20.180,69.895],
  [20.162,69.948],[20.081,69.872],[19.988,69.771],[19.952,69.757],
  [19.966,69.807],[19.985,69.841],[19.904,69.794],[19.830,69.748],
  [19.826,69.720],[19.790,69.683],[19.871,69.621],[19.753,69.603],
  [19.742,69.561],[19.731,69.525],[19.712,69.491],[19.723,69.473],
  [19.653,69.447],[19.653,69.418],[19.605,69.435],[19.495,69.392],
  [19.672,69.486],[19.631,69.547],[19.701,69.602],[19.734,69.642],
  [19.642,69.673],[19.701,69.743],[19.657,69.759],[19.727,69.786],
  [19.712,69.816],[19.587,69.795],[19.532,69.798],[19.373,69.771],
  [19.295,69.779],[19.166,69.756],[19.123,69.737],[19.083,69.678],
  [19.041,69.665],[18.934,69.714],[19.045,69.768],[19.030,69.816],
  [19.085,69.786],[19.362,69.827],[19.498,69.946],[19.107,69.958],
  [18.738,69.955],[18.727,69.915],[18.919,69.859],[19.037,69.817],
  [18.941,69.830],[18.820,69.845],[18.775,69.806],[18.754,69.771],
  [18.714,69.729],[18.710,69.699],[18.724,69.684],[18.464,69.695],
]

# Variables: nombre_nc → unidad_salida
VARS = {
    "air_temperature_2m":         "°C",
    "wind_speed_of_gust":         "km/h",
    "cloud_area_fraction":        "%",
    "precipitation_amount_acc":   "mm",
    "snowfall_amount_acc":        "mm",
    "air_pressure_at_sea_level":  "hPa",
    "fog_area_fraction":          "%",
    "x_wind_10m":                 "km/h",
    "y_wind_10m":                 "km/h",
}

# Variables en niveles de presión — shape diferente: [time, pressure, ensemble, y, x]
# pressure levels: index 0 = 500 hPa, index 1 = 850 hPa (según DDS)
PRESSURE_VARS = {
    "air_temperature_pl": "°C",   # temperatura en 500 y 850 hPa
}

# ── Helpers ───────────────────────────────────────────────────────────
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def point_in_polygon(lon, lat, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj-xi)*(lat-yi)/(yj-yi)+xi):
            inside = not inside
        j = i
    return inside

def to_list(arr):
    """Convierte numpy array a lista Python, NaN → None."""
    flat = arr.flatten()
    out = []
    for v in flat:
        if np.isnan(v):
            out.append(None)
        else:
            out.append(round(float(v), 2))
    return out

# ── Main ──────────────────────────────────────────────────────────────
def main():
    log("Conectando a OPeNDAP THREDDS...")
    t0 = time.time()
    ds = nc.Dataset(URL)
    log(f"  Conectado en {time.time()-t0:.1f}s")

    # Run time y times
    ref_t    = float(ds.variables["forecast_reference_time"][:])
    run_time = datetime.fromtimestamp(ref_t, tz=timezone.utc).isoformat()
    times_raw = ds.variables["time"][:]
    times_iso = [datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat()
                 for t in times_raw]
    n_times = len(times_iso)
    log(f"  Run: {run_time}  |  Pasos: {n_times}")

    # Coordenadas 2D
    log("  Descargando lat/lon...")
    lat2d = np.array(ds.variables["latitude"][:])
    lon2d = np.array(ds.variables["longitude"][:])

    # Bbox índices — rectángulo simple, sin filtro de polígono
    mask_bbox = ((lat2d >= LAT_MIN) & (lat2d <= LAT_MAX) &
                 (lon2d >= LON_MIN) & (lon2d <= LON_MAX))
    rows, cols = np.where(mask_bbox)
    j_min, j_max = int(rows.min()), int(rows.max())
    i_min, i_max = int(cols.min()), int(cols.max())
    log(f"  Bbox índices: j={j_min}:{j_max}, i={i_min}:{i_max}")

    sub_lat = lat2d[j_min:j_max+1, i_min:i_max+1]
    sub_lon = lon2d[j_min:j_max+1, i_min:i_max+1]
    ny_sub, nx_sub = sub_lat.shape

    # Todos los puntos del bbox (sin máscara de polígono — es rectangular)
    n_pts    = ny_sub * nx_sub
    flat_idx = np.arange(n_pts)   # todos los puntos
    lats_out = [round(float(v), 5) for v in sub_lat.flatten()]
    lons_out = [round(float(v), 5) for v in sub_lon.flatten()]
    log(f"  Puntos en bbox: {ny_sub} × {nx_sub} = {n_pts}")

    # Extraer variables
    result    = {}   # media ensemble
    spread    = {}   # std ensemble
    n_members = len(ds.variables["ensemble_member"][:])

    for nc_name in VARS:
        log(f"  Descargando {nc_name}...")
        t1 = time.time()
        var = ds.variables[nc_name]
        # Shape: [time, height_dim=1, ensemble_member, y, x]
        raw = var[:, 0, :, j_min:j_max+1, i_min:i_max+1]
        # → (n_times, n_members, ny_sub, nx_sub)
        raw = np.array(raw, dtype=np.float32)

        # Fill value → NaN
        fv = getattr(var, "_FillValue", None) or getattr(var, "missing_value", None)
        if fv is not None:
            raw[raw == float(fv)] = np.nan

        # Conversiones
        if nc_name == "air_temperature_2m":
            raw -= 273.15
        elif nc_name == "air_pressure_at_sea_level":
            raw /= 100.0
        elif nc_name in ("x_wind_10m", "y_wind_10m", "wind_speed_of_gust"):
            raw *= 3.6
        elif nc_name in ("cloud_area_fraction", "fog_area_fraction"):
            raw *= 100.0

        # Media y spread ensemble  → (n_times, ny_sub, nx_sub)
        mean_arr   = np.nanmean(raw, axis=1)
        spread_arr = np.nanstd(raw, axis=1)

        # Aplanar spatial y filtrar al polígono → (n_times, n_pts)
        mean_flat   = mean_arr.reshape(n_times, -1)[:, flat_idx]
        spread_flat = spread_arr.reshape(n_times, -1)[:, flat_idx]

        # Serializar: lista plana [t0_p0, t0_p1, ..., t1_p0, ...]
        result[nc_name] = to_list(mean_flat)
        spread[nc_name] = to_list(spread_flat)
        log(f"    OK en {time.time()-t1:.1f}s")

    # Extraer variables en niveles de presión
    # Shape: [time, pressure=2, ensemble, y, x]
    # Guardamos los 2 niveles por separado: _500 y _850
    for nc_name, unit in PRESSURE_VARS.items():
        log(f"  Descargando {nc_name} (niveles de presión)...")
        t1 = time.time()
        var = ds.variables[nc_name]
        # Obtener índices de presión disponibles
        p_levels = ds.variables["pressure"][:]  # e.g. [500., 850.] hPa
        log(f"    Niveles disponibles: {list(p_levels)} hPa")

        for pi, p_hpa in enumerate(p_levels):
            p_hpa = float(p_hpa)
            key = f"{nc_name}_{int(p_hpa)}"
            raw = var[:, pi, :, j_min:j_max+1, i_min:i_max+1]
            raw = np.array(raw, dtype=np.float32)

            fv = getattr(var, "_FillValue", None) or getattr(var, "missing_value", None)
            if fv is not None:
                raw[raw == float(fv)] = np.nan

            # K → °C
            if nc_name == "air_temperature_pl":
                raw -= 273.15

            mean_arr   = np.nanmean(raw, axis=1)   # media ensemble
            mean_flat  = mean_arr.reshape(n_times, -1)[:, flat_idx]
            result[key] = to_list(mean_flat)
            log(f"    {key} OK en {time.time()-t1:.1f}s")

    # Calcular velocidad de viento escalar (U²+V²)^0.5
    log("  Calculando wind_speed_10m...")
    u = np.array([v if v is not None else np.nan for v in result["x_wind_10m"]], dtype=np.float32)
    v = np.array([v if v is not None else np.nan for v in result["y_wind_10m"]], dtype=np.float32)
    ws = np.sqrt(u**2 + v**2)
    result["wind_speed_10m"] = [None if np.isnan(v) else round(float(v), 2) for v in ws]

    ds.close()

    # Cargar elevación del terreno (generada por fetch_dem.py)
    elevation = None
    try:
        with open("arome_dem.json") as f:
            dem = json.load(f)
        if dem.get("n_pts") == n_pts:
            elevation = dem["elevation"]
            log(f"  DEM cargado: {n_pts} puntos, max {max(elevation):.0f} m")
        else:
            log(f"  WARNING: DEM tiene {dem.get('n_pts')} pts, grid tiene {n_pts} — ignorando")
    except FileNotFoundError:
        log("  WARNING: arome_dem.json no encontrado — ISO 0°C sin orografía")
        log("           Ejecuta fetch_dem.py para generarlo")

    # Construir JSON de salida
    output = {
        "run_time":  run_time,
        "generated": datetime.now(tz=timezone.utc).isoformat(),
        "source":    URL.split("/")[-1],
        "grid": {
            "n_pts": n_pts,
            "lats":  lats_out,
            "lons":  lons_out,
            "elevation": elevation,  # metros MSL por punto, None si no disponible
        },
        "times":     times_iso,
        "variables": result,
        "spread":    spread,
    }

    out_file = "arome_data.json"
    log(f"Guardando {out_file}...")
    with open(out_file, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    import os
    size_mb = os.path.getsize(out_file) / 1e6
    log(f"✓ {out_file}  ({size_mb:.1f} MB)")
    log(f"  {n_pts} puntos × {n_times} pasos × {len(VARS)+1} variables")
    log(f"  Bbox: lat {LAT_MIN}–{LAT_MAX}, lon {LON_MIN}–{LON_MAX}")
    log("Listo. Ahora arranca el servidor: python -m http.server 8080")

if __name__ == "__main__":
    main()
