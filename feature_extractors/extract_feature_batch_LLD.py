import os
import subprocess
import pandas as pd
import json

import os
import sys
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
                    'Compare2016_config_path': config.get('Compare2016_config_path', ''),
                    'root_folder_path': config.get('root_folder_path', '')
                }
        except Exception as e:
            print(f"Warning: Could not load gui_config.json: {e}")
    
    return {
        'SMILE_path': '',
        'Compare2016_config_path': '',
        'root_folder_path': ''
    }

# Carica i path
_config = load_paths_from_gui_config()
SMILE_path = _config['SMILE_path']
Compare2016_config_path = _config['Compare2016_config_path']
root_folder_path = _config['root_folder_path']


def generate_lld_for_file(input_file, smile_path=None, config_path=None):
    """Generate a single _LLD.csv for input_file using openSMILE.

    Returns the path to the generated LLD CSV, or None on failure.
    """
    # RICARICA I PATH OGNI VOLTA
    if smile_path is None or config_path is None:
        config = load_paths_from_gui_config()
        smile_path = smile_path or config['SMILE_path']
        config_path = config_path or config['Compare2016_config_path']
    
    base = os.path.splitext(os.path.basename(input_file))[0]
    lld_csv = os.path.join(os.path.dirname(input_file), base + "_LLD.csv")
    if os.path.exists(lld_csv):
        return lld_csv

    try:
        # Ensure SMILExtract can find its runtime dependencies by using its folder as cwd
        smile_dir = os.path.dirname(smile_path) if smile_path and os.path.isabs(smile_path) else None
        env = os.environ.copy()
        if smile_dir:
            env['PATH'] = smile_dir + os.pathsep + env.get('PATH', '')

        # verify SMILExtract binary exists
        if not smile_path or not os.path.exists(smile_path):
            raise RuntimeError(f"SMILExtract executable not found: {smile_path}. Please configure openSMILE in GUI Settings.")

        # verify config exists
        if not config_path or not os.path.exists(config_path):
            raise RuntimeError(f"Config file not found: {config_path}. Please configure openSMILE in GUI Settings.")

        proc = subprocess.run([
            smile_path,
            "-C", config_path,
            "-I", input_file,
            "-lldcsvoutput", lld_csv,
            "-nologfile", "-loglevel", "2"
        ], cwd=smile_dir if smile_dir else None, env=env, capture_output=True, text=True)

        if proc.returncode != 0:
            stderr = proc.stderr or proc.stdout or f"exit code {proc.returncode}"
            raise RuntimeError(f"SMILExtract failed for {input_file}: {stderr}")

        if os.path.exists(lld_csv):
            try:
                df = pd.read_csv(lld_csv, sep=';')
                print(f"✅ Estratto: {lld_csv} | colonne: {len(df.columns)} | righe: {len(df)}")
            except Exception:
                print(f"⚠️ Estratto ma non leggibile: {lld_csv}")
            return lld_csv
        else:
            raise RuntimeError(f"SMILExtract reported success but {lld_csv} was not created")
    except Exception as e:
        print(f"Errore generazione LLD per {input_file}: {e}")
        raise


def generate_lld_in_tree(root_folder):
    """Walk root_folder and generate LLD CSVs.
    
    Returns a list of generated LLD CSV file paths.
    """
    generated = []
    for dirpath, _, filenames in os.walk(root_folder):
        wav_files = [f for f in filenames if f.lower().endswith('.wav')]
        if not wav_files:
            continue

        for file in wav_files:
            input_file = os.path.join(dirpath, file)
            base = os.path.splitext(file)[0]
            lld_csv = os.path.join(dirpath, base + '_LLD.csv')
            if os.path.exists(lld_csv):
                generated.append(lld_csv)  # Conta anche quelli esistenti
                continue
            try:
                res = generate_lld_for_file(input_file)
                if res:
                    generated.append(res)
            except Exception as e:
                print(f"Errore con {input_file}: {e}")
    return generated


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description="Generate LLD CSV files from WAV files using openSMILE.")
    ap.add_argument("--root", default=root_folder_path, help="Root folder to scan for WAV files.")
    ap.add_argument("--verbose", action="store_true", help="Print verbose messages.")
    args = ap.parse_args()

    gen = generate_lld_in_tree(args.root)
    print(f"Generati {len(gen)} file LLD in {args.root}")
