import argparse
import sys
import os
import glob
import pandas as pd
import numpy as np


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



# =========================================
# Parametri di default
# =========================================
VOICING_THR_DEFAULT = 0.55     # soglia voicingFinalUnclipped_sma per vmask (0..1)
MIN_PAUSE_DEFAULT   = 0.20     # s, durata minima per considerare una pausa
LONG_PAUSE_DEFAULT  = 1.5      # s, soglia per LongPauseCount
DT                  = 0.01     # s (10 ms, frame step)


# Colonne richieste dai CSV LLD (estendibile)
REQUIRED_COLS = [
    "voicingFinalUnclipped_sma",         # per vmask e tutte le feature basate sul parlato/silenzio
    "F0final_sma",                       # per DeltaF0Mean / DeltaF0Std
    "pcm_RMSenergy_sma",                 # per DeltaRMSMean / DeltaRMSStd
    "pcm_fftMag_spectralCentroid_sma",   # per CentroidMean / CentroidStd
    "pcm_fftMag_spectralFlux_sma",       # per FluxStd
    "pcm_fftMag_spectralEntropy_sma",    # per EntropyStd
    "mfcc_sma[5]"                        # per MFCC5Mean / MFCC5Std
]

# AGGIUNTO: Lista completa delle feature disponibili
ALL_CUSTOM_FEATURES = [
    "VoicedRatio",
    "SilencePerMinute",
    "PausesPerMinute",
    "PauseMean",
    "PauseMax",
    "PauseVariance",
    "MedianPause",
    "LongPauseCount",
    "TransitionVUV",
    "PhonationRatio",
    "DeltaF0Mean",
    "DeltaF0Std",
    "DeltaRMSMean",
    "DeltaRMSStd",
    "CentroidMean",
    "MFCC5Mean",
    "MFCC5Std",
    "CentroidStd",
    "FluxStd",
    "EntropyStd"
]


# =========================================
# 1) Caricamento dei dati LLD
# =========================================
def load_lld_csv_min(path, usecols):
    """
    Legge dal CSV solo le colonne in 'usecols', provando separatore/decimale diversi.
    Restituisce un DataFrame con le colonne richieste se disponibili.
    """
    attempts = [( ',', '.' ), (';', '.'), (';', ','), (',', ',')]
    last_err = None
    for sep, dec in attempts:
        try:
            df = pd.read_csv(path, sep=sep, decimal=dec, engine='python', usecols=usecols)
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Impossibile leggere {path}: {last_err}")



# =========================================
# 2) Funzioni di supporto per i calcoli
# =========================================
def smooth_prob(prob, win_ms=50, dt=DT):
    """
    Media mobile della probabilità di voicing per stabilizzare il segnale.
    win_ms: finestra (ms). Esempi utili: 30-50 ms.
    """
    win = max(1, int(round((win_ms/1000.0)/dt)))
    if win <= 1:
        return prob
    k = np.ones(win, dtype=float) / win
    return np.convolve(prob, k, mode='same')


def vmask_hysteresis(prob, thr_on, thr_off):
    """
    Maschera booleana con due soglie:
      - thr_on  : per passare in stato voiced (ON)
      - thr_off : per tornare a unvoiced (OFF)
    Richiede thr_on >= thr_off.
    """
    on  = prob >= thr_on
    off = prob <= thr_off
    mask = np.zeros_like(prob, dtype=bool)
    state = False
    for i in range(len(prob)):
        if not state and on[i]:
            state = True
        elif state and off[i]:
            state = False
        mask[i] = state
    return mask


def compute_vmask(v_series, thr, smooth_win_ms=0, use_hysteresis=False, thr_off=None):
    """
    Costruisce il vmask da voicingFinalUnclipped_sma con opzioni:
      - smooth_win_ms: 0 = no smoothing; se >0 applica media mobile (ms).
      - use_hysteresis: True = usa due soglie (thr_on=thr, thr_off=thr_off o thr-0.08).
      - thr: soglia singola se isteresi è False.
    """
    prob = pd.to_numeric(v_series, errors='coerce').fillna(0.0).values


    if smooth_win_ms and smooth_win_ms > 0:
        prob = smooth_prob(prob, win_ms=smooth_win_ms, dt=DT)


    if use_hysteresis:
        if thr_off is None:
            thr_off = max(0.0, min(thr, thr - 0.04))  # isteresi di default: 0.04
        return vmask_hysteresis(prob, thr_on=thr, thr_off=thr_off)
    else:
        return prob > thr


def _run_boundaries(mask):
    """Restituisce due array starts, ends con gli indici dei run consecutivi in mask."""
    n = mask.size
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    changes = np.flatnonzero(np.diff(mask.astype(np.int8)) != 0) + 1
    starts  = np.r_[0, changes]
    ends    = np.r_[changes, n]
    return starts, ends


def pause_durations_seconds(vmask, dt, min_pause):
    """
    Array con le durate (s) delle pause = run non-voiced con durata >= min_pause.
    """
    starts, ends = _run_boundaries(vmask)
    if starts.size == 0:
        return np.array([], dtype=float)
    run_is_voiced = vmask[starts]
    lengths = (ends - starts) * dt
    pauses  = (~run_is_voiced) & (lengths >= min_pause)
    return lengths[pauses]



# =========================================
# 3) Funzioni di calcolo delle feature
# =========================================
def voiced_ratio(vmask):
    """Percentuale di frame voiced sul totale (in %)."""
    n = vmask.size
    return (100.0 * np.sum(vmask) / n) if n > 0 else np.nan


def silence_per_minute(vmask, dt):
    """Secondi di silenzio per minuto = (tempo unvoiced / durata) * 60."""
    total = vmask.size * dt
    if total <= 0:
        return np.nan
    silence_time = np.sum(~vmask) * dt
    return (silence_time / total) * 60.0


def count_pauses_per_minute(vmask, dt, min_pause):
    """Numero di pause/minuto (pause = run non-voiced con durata >= min_pause)."""
    n = vmask.size
    if n == 0:
        return np.nan
    durations = pause_durations_seconds(vmask, dt, min_pause)
    minutes = (n * dt) / 60.0
    return (durations.size / minutes) if minutes > 0 else np.nan


def pause_mean_seconds(vmask, dt, min_pause):
    """Media delle durate delle pause (in secondi)."""
    durs = pause_durations_seconds(vmask, dt, min_pause)
    return float(durs.mean()) if durs.size > 0 else 0.0


def pause_max_seconds(vmask, dt, min_pause):
    """Massima durata di pausa (in secondi)."""
    durs = pause_durations_seconds(vmask, dt, min_pause)
    return float(durs.max()) if durs.size > 0 else 0.0


def pause_variance_seconds(vmask, dt, min_pause):
    """Varianza delle durate delle pause (in secondi^2)."""
    durs = pause_durations_seconds(vmask, dt, min_pause)
    return float(np.var(durs, ddof=0)) if durs.size > 0 else 0.0


def median_pause_seconds(vmask, dt, min_pause):
    """Mediana delle durate delle pause (in secondi)."""
    durs = pause_durations_seconds(vmask, dt, min_pause)
    return float(np.median(durs)) if durs.size > 0 else 0.0


def long_pause_count(vmask, dt, min_pause, long_pause_thr):
    """Numero di pause 'lunghe' (durata >= long_pause_thr) nell'intero audio."""
    durs = pause_durations_seconds(vmask, dt, min_pause)
    return int(np.sum(durs >= long_pause_thr)) if durs.size > 0 else 0


def transition_vuv_per_minute(vmask, dt):
    """
    Numero di transizioni Voiced<->Unvoiced per minuto.
    (proxy di instabilità/segmentazione del parlato)
    """
    n = vmask.size
    if n == 0:
        return np.nan
    transitions = np.sum(np.diff(vmask.astype(np.int8)) != 0)
    minutes = (n * dt) / 60.0
    return transitions / minutes if minutes > 0 else np.nan


def phonation_ratio(vmask, dt):
    """
    Rapporto di fonazione: frazione del tempo totale in stato voiced (0..1).
    """
    n = vmask.size
    if n == 0:
        return np.nan
    return float(np.sum(vmask)) / float(n)


def delta_f0_mean(df):
    """Media delle differenze frame-to-frame di F0 (Hz)."""
    if "F0final_sma" not in df.columns:
        return np.nan
    f0 = pd.to_numeric(df["F0final_sma"], errors="coerce").fillna(0.0).values
    if len(f0) < 2:
        return np.nan
    return np.mean(np.diff(f0))


def delta_f0_std(df):
    """Deviazione standard delle differenze frame-to-frame di F0 (Hz)."""
    if "F0final_sma" not in df.columns:
        return np.nan
    f0 = pd.to_numeric(df["F0final_sma"], errors="coerce").fillna(0.0).values
    if len(f0) < 2:
        return np.nan
    return np.std(np.diff(f0))


def delta_rms_mean(df):
    """Media delle differenze frame-to-frame di RMS energy."""
    if "pcm_RMSenergy_sma" not in df.columns:
        return np.nan
    rms = pd.to_numeric(df["pcm_RMSenergy_sma"], errors="coerce").fillna(0.0).values
    if len(rms) < 2:
        return np.nan
    return np.mean(np.diff(rms))


def delta_rms_std(df):
    """Deviazione standard delle differenze frame-to-frame di RMS energy."""
    if "pcm_RMSenergy_sma" not in df.columns:
        return np.nan
    rms = pd.to_numeric(df["pcm_RMSenergy_sma"], errors="coerce").fillna(0.0).values
    if len(rms) < 2:
        return np.nan
    return np.std(np.diff(rms))


def centroid_mean(df):
    """Media del spectral centroid."""
    if "pcm_fftMag_spectralCentroid_sma" not in df.columns:
        return np.nan
    c = pd.to_numeric(df["pcm_fftMag_spectralCentroid_sma"], errors="coerce").fillna(0.0).values
    return np.mean(c) if len(c) > 0 else np.nan


def centroid_std(df):
    """Deviazione standard del spectral centroid."""
    if "pcm_fftMag_spectralCentroid_sma" not in df.columns:
        return np.nan
    c = pd.to_numeric(df["pcm_fftMag_spectralCentroid_sma"], errors="coerce").fillna(0.0).values
    return np.std(c) if len(c) > 0 else np.nan


def flux_std(df):
    """Deviazione standard del spectral flux."""
    if "pcm_fftMag_spectralFlux_sma" not in df.columns:
        return np.nan
    f = pd.to_numeric(df["pcm_fftMag_spectralFlux_sma"], errors="coerce").fillna(0.0).values
    return np.std(f) if len(f) > 0 else np.nan


def entropy_std(df):
    """Deviazione standard dello spectral entropy."""
    if "pcm_fftMag_spectralEntropy_sma" not in df.columns:
        return np.nan
    e = pd.to_numeric(df["pcm_fftMag_spectralEntropy_sma"], errors="coerce").fillna(0.0).values
    return np.std(e) if len(e) > 0 else np.nan


def mfcc5_mean(df):
    """Media del coefficiente MFCC5."""
    if "mfcc_sma[5]" not in df.columns:
        return np.nan
    m = pd.to_numeric(df["mfcc_sma[5]"], errors="coerce").fillna(0.0).values
    return np.mean(m) if len(m) > 0 else np.nan


def mfcc5_std(df):
    """Deviazione standard del coefficiente MFCC5."""
    if "mfcc_sma[5]" not in df.columns:
        return np.nan
    m = pd.to_numeric(df["mfcc_sma[5]"], errors="coerce").fillna(0.0).values
    return np.std(m) if len(m) > 0 else np.nan


# =========================================
# 4) Funzione di elaborazione file
# =========================================
# MODIFICATO: Aggiunto parametro selected_features
def process_file(path, voicing_thr, min_pause, long_pause_thr, dt,
                 smooth_win_ms, use_hysteresis, selected_features=None):
    """
    Elabora un singolo file LLD ed estrae le feature richieste.
    
    Args:
        selected_features: Lista delle feature da estrarre (None = tutte)
    """
    # Se non specificato, usa tutte le feature
    if selected_features is None:
        selected_features = ALL_CUSTOM_FEATURES
    
    # Lettura
    try:
        df = load_lld_csv_min(path, usecols=REQUIRED_COLS)
    except Exception as e:
        print(f"[ERR read] {path} -> {e}", file=sys.stderr)
        return None

    # Controllo colonne richieste
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"[ERR cols] {path} -> mancano colonne {missing}", file=sys.stderr)
        return None

    # Maschera voiced
    vmask = compute_vmask(df['voicingFinalUnclipped_sma'],
                          thr=voicing_thr,
                          smooth_win_ms=smooth_win_ms,
                          use_hysteresis=use_hysteresis,
                          thr_off=None)

    # Filename e subjectId
    fname = os.path.basename(path)
    base_name = os.path.splitext(fname)[0]
    base_name = base_name.replace('_LLD', '')  # Rimuovi _LLD dal nome


    def _generate_id_from_filename(s: str, base: int = 131, mod: int = 2**31 - 1) -> int:
        h = 0
        for ch in s:
            h = (h * base + ord(ch)) % mod
        return h

    subject_id = str(_generate_id_from_filename(base_name))

    # MODIFICATO: Dizionario completo con tutte le feature
    all_features = {
        "filename": fname,
        "subjectId": subject_id,
        "VoicedRatio": voiced_ratio(vmask),
        "SilencePerMinute": silence_per_minute(vmask, dt),
        "PausesPerMinute": count_pauses_per_minute(vmask, dt, min_pause),
        "PauseMean": pause_mean_seconds(vmask, dt, min_pause),
        "PauseMax": pause_max_seconds(vmask, dt, min_pause),
        "PauseVariance": pause_variance_seconds(vmask, dt, min_pause),
        "MedianPause": median_pause_seconds(vmask, dt, min_pause),
        "LongPauseCount": long_pause_count(vmask, dt, min_pause, long_pause_thr),
        "TransitionVUV": transition_vuv_per_minute(vmask, dt),
        "PhonationRatio": phonation_ratio(vmask, dt),
        "DeltaF0Mean": delta_f0_mean(df),
        "DeltaF0Std": delta_f0_std(df),
        "DeltaRMSMean": delta_rms_mean(df),
        "DeltaRMSStd": delta_rms_std(df),
        "CentroidMean": centroid_mean(df),
        "MFCC5Mean": mfcc5_mean(df),
        "MFCC5Std": mfcc5_std(df),
        "CentroidStd": centroid_std(df),
        "FluxStd": flux_std(df),
        "EntropyStd": entropy_std(df)
    }
    
    # MODIFICATO: Restituisci solo le feature selezionate
    result = {"filename": fname, "subjectId": subject_id}
    for feature in selected_features:
        if feature in all_features:
            result[feature] = all_features[feature]
    
    return result


# =========================================
# 5) Feature Extraction Functions
# =========================================
# MODIFICATO: Aggiunto parametro selected_features
def extract_custom_features(path, voicing_thr=VOICING_THR_DEFAULT, min_pause=MIN_PAUSE_DEFAULT,
                          long_pause_thr=LONG_PAUSE_DEFAULT, smooth_win_ms=50, hysteresis=True,
                          output_path=None, selected_features=None):
    """
    Estrae le feature custom da un file audio o da una cartella contenente file audio
    
    Args:
        path (str): Percorso al file audio o alla cartella contenente i file audio
        voicing_thr (float): Soglia voicing (default 0.55)
        min_pause (float): Durata minima pausa in s (default 0.20)
        long_pause_thr (float): Soglia di durata per LongPauseCount (s, default 1.5)
        smooth_win_ms (int): Smoothing della probabilità di voicing (ms). 0=OFF, default=50 ms
        hysteresis (bool): Abilita isteresi (default True)
        output_path (str): Path del file CSV di output
        selected_features (list): Lista delle feature da estrarre (None = tutte)
    """
    # Se non specificato, usa tutte le feature
    if selected_features is None:
        selected_features = ALL_CUSTOM_FEATURES
    
    # Import LLD generation functions
    try:
        try:
            from feature_extractors.extract_feature_batch_LLD import generate_lld_for_file, generate_lld_in_tree
        except Exception:
            from .extract_feature_batch_LLD import generate_lld_for_file, generate_lld_in_tree
    except Exception:
        generate_lld_for_file = None
        generate_lld_in_tree = None

    files = []
    if os.path.isfile(path):
        base_audio = os.path.basename(path)
        base = os.path.splitext(base_audio)[0]
        expected = os.path.join(os.path.dirname(path), base + "_LLD.csv")
        if os.path.exists(expected):
            files = [expected]
        else:
            if generate_lld_for_file:
                print(f"LLD non trovato per {path}. Provo a generarlo -> {expected}")
                try:
                    res = generate_lld_for_file(path)
                    if res:
                        files = [res]
                except Exception as e:
                    raise RuntimeError(f"Generazione LLD fallita per {path}: {e}")
    else:
        # folder
        pattern = "**/*LLD*.csv"
        root = path
        files = [f for f in glob.glob(os.path.join(root, pattern), recursive=True) if os.path.isfile(f)]
        if not files and generate_lld_in_tree:
            print(f"Nessun LLD trovato. Genero LLD per la cartella...")
            try:
                files = generate_lld_in_tree(root)
            except Exception as e:
                raise RuntimeError(f"Generazione LLD nella cartella fallita: {e}")

    if not files:
        raise ValueError("Nessun file LLD trovato o generato.")

    rows = []
    for fp in files:
        # MODIFICATO: Passa selected_features a process_file
        res = process_file(fp, voicing_thr, min_pause, long_pause_thr, DT,
                         smooth_win_ms, hysteresis, selected_features)
        if res is not None:
            rows.append(res)
            print(f"[OK] {fp}")
        else:
            print(f"[SKIP] {fp}", file=sys.stderr)

    # Output
    if not output_path:
        raise ValueError("output_path must be provided to extract_custom_features")
    out_path = output_path
    df_output = pd.DataFrame(rows)
    df_output.to_csv(out_path, index=False, sep=";", mode='w')
    print(f"\n✅ Fatto. Salvato: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Estrazione feature pause/voicing da CSV LLD (dt=10ms).")
    ap.add_argument("root", help="Cartella o file WAV da processare")
    ap.add_argument("-o", "--out", help="File CSV di output (default: stessa cartella input)")
    ap.add_argument("--voicing-thr", type=float, default=VOICING_THR_DEFAULT, 
                    help="Soglia voicing (default 0.55).")
    ap.add_argument("--min-pause", type=float, default=MIN_PAUSE_DEFAULT, 
                    help="Durata minima pausa in s (default 0.20).")
    ap.add_argument("--long-pause-thr", type=float, default=LONG_PAUSE_DEFAULT,
                    help="Soglia di durata per LongPauseCount (s, default 1.5).")
    ap.add_argument("--smooth-win-ms", nargs='?', const=50, type=int, default=0,
                    help="Smoothing della probabilità di voicing (ms).")
    ap.add_argument("--hysteresis", action="store_true",
                    help="Abilita isteresi.")
    
    args = ap.parse_args()
    
    # Output di default nella stessa cartella dell'input
    if args.out:
        out = args.out
    else:
        if os.path.isfile(args.root):
            out_dir = os.path.dirname(args.root)
        else:
            out_dir = args.root
        out = os.path.join(out_dir, "extracted_features_custom.csv")
    
    print(f"Input: {args.root}")
    print(f"Output: {out}")
    
    extract_custom_features(
        args.root, 
        float(args.voicing_thr), 
        float(args.min_pause), 
        float(args.long_pause_thr),
        args.smooth_win_ms, 
        args.hysteresis, 
        out
    )

if __name__ == "__main__":
    main()
