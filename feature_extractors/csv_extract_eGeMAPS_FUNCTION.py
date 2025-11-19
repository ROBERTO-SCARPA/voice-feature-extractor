import os
import subprocess
import time
import pandas as pd
import arff
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# NUOVO: Carica i path dal JSON della GUI
def load_paths_from_gui_config():
    """Carica i path dal file gui_config.json"""
    config_file = os.path.join(os.path.dirname(__file__), '..', 'UI', 'gui_config.json')
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return {
                    'SMILE_path': config.get('SMILE_path', ''),
                    'eGeMAPS_config_path': config.get('eGeMAPS_config_path', ''),
                    'root_folder_path': config.get('root_folder_path', '')
                }
        except Exception as e:
            print(f"Warning: Could not load gui_config.json: {e}")
    
    return {
        'SMILE_path': '',
        'eGeMAPS_config_path': '',
        'root_folder_path': ''
    }

# Definisci le 48 caratteristiche più pertinenti (versione 2.0 standard)
egemaps_features_v2 = [
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
    "loudness_sma3_amean",
    "loudness_sma3_stddevNorm",
    "loudness_sma3_percentile50.0",
    "jitterLocal_sma3nz_amean",
    "jitterLocal_sma3nz_stddevNorm",
    "shimmerLocaldB_sma3nz_amean",
    "shimmerLocaldB_sma3nz_stddevNorm",
    "HNRdBACF_sma3nz_amean",
    "HNRdBACF_sma3nz_stddevNorm",
    "mfcc1_sma3_stddevNorm",
    "mfcc2_sma3_stddevNorm",
    "mfcc3_sma3_stddevNorm",
    "VoicedSegmentsPerSec",
    "MeanVoicedSegmentLengthSec",
    "loudness_sma3_percentile20.0",
    "mfcc2_sma3_amean",
    "mfcc1_sma3_amean",
    "mfcc3_sma3_amean",
    "loudness_sma3_percentile80.0",
    "loudness_sma3_pctlrange0-2",
    "loudness_sma3_meanRisingSlope",
    "loudness_sma3_meanFallingSlope",
    "loudness_sma3_stddevFallingSlope",
    "spectralFlux_sma3_amean",
    "spectralFlux_sma3_stddevNorm",
    "F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
    "F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope",
    "mfcc4_sma3_amean",
    "mfcc4_sma3_stddevNorm",
    "F1frequency_sma3nz_amean",
    "F1frequency_sma3nz_stddevNorm",
    "F2frequency_sma3nz_amean",
    "F2frequency_sma3nz_stddevNorm",
    "F3frequency_sma3nz_amean",
    "F3frequency_sma3nz_stddevNorm",
    "alphaRatioV_sma3nz_amean",
    "hammarbergIndexV_sma3nz_amean",
    "slopeV0-500_sma3nz_amean",
    "slopeV500-1500_sma3nz_amean",
    "StddevVoicedSegmentLengthSec"
]

def extract_and_save_features(input_audio_file, selected_features=None):
    if selected_features is None:
        selected_features = egemaps_features_v2
    
    # RICARICA I PATH OGNI VOLTA
    config = load_paths_from_gui_config()
    SMILE_path = config['SMILE_path']
    eGeMAPS_config_path = config['eGeMAPS_config_path']

    # Validazione
    if not SMILE_path or not os.path.exists(SMILE_path):
        raise ValueError(f"SMILExtract not configured. Please configure paths in GUI Settings.")
    if not eGeMAPS_config_path or not os.path.exists(eGeMAPS_config_path):
        raise ValueError(f"eGeMAPS config not found: {eGeMAPS_config_path}")
    
    file_name = os.path.basename(input_audio_file)
    base_name = os.path.splitext(file_name)[0]
    arff_file = os.path.join(os.path.dirname(input_audio_file), base_name + "_eGeMAPS_v2.arff")

    features_row = None
    columns = None
    
    try:
        # Setup environment
        smile_dir = os.path.dirname(SMILE_path) if SMILE_path and os.path.isabs(SMILE_path) else None
        env = os.environ.copy()
        if smile_dir:
            env['PATH'] = smile_dir + os.pathsep + env.get('PATH', '')
        
        proc = subprocess.run(
            [SMILE_path, "-C", eGeMAPS_config_path, "-I", input_audio_file, "-O", arff_file],
            cwd=smile_dir,
            env=env,
            capture_output=True,
            text=True
        )
        
        if proc.returncode != 0:
            stderr = proc.stderr or proc.stdout or f"exit code {proc.returncode}"
            raise RuntimeError(f"SMILExtract failed: {stderr}")

        time.sleep(0.2)
        
        # Leggi ARFF
        with open(arff_file, 'r', encoding='utf-8', errors='ignore') as f:
            arff_data = arff.load(f)
        df = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])

        # Genera ID
        def _generate_id_from_filename(s: str, base: int = 131, mod: int = 2**31 - 1) -> int:
            h = 0
            for ch in s:
                h = (h * base + ord(ch)) % mod
            return h

        subject_id = str(_generate_id_from_filename(base_name))
        
        # MODIFICATO: Gestisci "all" per estrarre tutte le feature
        if isinstance(selected_features, str) and selected_features.lower() == "all":
            # Estrai TUTTE le colonne tranne 'name' , 'frameTime' , 'class'
            exclude_cols = ['name', 'frameTime', 'class']
            all_feature_cols = [col for col in df.columns if col not in exclude_cols]
            selected_features = all_feature_cols
            print(f"[INFO] Extracting ALL {len(all_feature_cols)} eGeMAPS features")
        
        features_row = [file_name, subject_id]

        for feature in selected_features:
            if feature in df.columns:
                features_row.append(df[feature].values[0])
            else:
                features_row.append(None)

        columns = ["filename", "subjectId"] + selected_features
        
    except Exception as e:
        print(f"[ERROR] extract_and_save_features: {e}")
        raise
        
    finally:
        # Cleanup ARFF
        time.sleep(0.3)
        for attempt in range(5):
            try:
                if os.path.exists(arff_file):
                    os.remove(arff_file)
                    break
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.5)
            except Exception:
                break
    
    return features_row, columns

def extract_egemaps_features(path, output_path=None, selected_features=None):
    """
    Estrae le feature eGeMAPS da un file audio o da una cartella
    
    Args:
        path (str): Percorso al file audio o alla cartella
        output_path (str): Percorso del file CSV di output
        selected_features (list|str): Lista delle feature da estrarre, "all" per tutte, None = v2 (48)
    """
    if not output_path:
        raise ValueError("output_path must be provided to extract_egemaps_features")
    
    # MODIFICATO: Gestisci "all"
    if selected_features is None:
        selected_features = egemaps_features_v2
    elif isinstance(selected_features, str) and selected_features.lower() == "all":
        # Passa "all" alla funzione extract_and_save_features
        pass
    
    all_rows = []
    columns = None  # Sarà definito dalla prima estrazione
    
    if os.path.isfile(path):
        if path.lower().endswith('.wav'):
            try:
                row, cols = extract_and_save_features(path, selected_features)
                if row:
                    all_rows.append(row)
                    columns = cols
                    print(f"[OK] Extracted {len(row)-2} features from: {row[0]}")
            except Exception as e:
                print(f"[ERROR] Exception caught: {e}")
                import traceback
                traceback.print_exc()
    else:
        for dirpath, _, filenames in os.walk(path):
            wav_files = [f for f in filenames if f.lower().endswith(".wav")]
            for file in wav_files:
                input_file = os.path.join(dirpath, file)
                try:
                    row, cols = extract_and_save_features(input_file, selected_features)
                    if row:
                        all_rows.append(row)
                        if columns is None:
                            columns = cols
                        print(f"[OK] {file}")
                except Exception as e:
                    print(f"[ERROR] {file}: {e}")
    
    # Scrivi tutto alla fine
    time.sleep(0.5)

    if all_rows and columns:
        try:
            df_output = pd.DataFrame(all_rows, columns=columns)
            df_output.to_csv(output_path, index=False, sep=';', mode='w')
            print(f"\n✅ Salvato: {output_path}")
            print(f"   Features estratte: {len(columns)-2}")
            print(f"   File processati: {len(all_rows)}")
        except Exception as e:
            print(f"\n❌ ERRORE scrittura CSV: {e}")
            raise
    else:
        print("\n⚠ Nessuna feature estratta")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Estrai eGeMAPS da file WAV")
    ap.add_argument("root", help="Cartella o file WAV da processare")
    ap.add_argument("-o", "--out", help="Percorso del file CSV di output (default: stessa cartella input)")
    ap.add_argument("--all", action="store_true", help="Estrai tutte le 88 feature invece delle 48 standard")
    args = ap.parse_args()
    
    # Output di default nella stessa cartella dell'input
    if args.out:
        out = args.out
    else:
        if os.path.isfile(args.root):
            out_dir = os.path.dirname(args.root)
        else:
            out_dir = args.root
        out = os.path.join(out_dir, "extracted_features_eGeMAPS.csv")
    
    # Determina feature set
    features = "all" if args.all else None
    
    print(f"Input: {args.root}")
    print(f"Output: {out}")
    print(f"Feature set: {'ALL (88)' if args.all else 'Standard v2 (48)'}")
    extract_egemaps_features(args.root, out, features)
