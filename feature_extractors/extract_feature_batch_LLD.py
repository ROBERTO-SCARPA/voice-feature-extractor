import os
import subprocess
import pandas as pd

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from common.paths import (root_folder_path, SMILE_path, Compare2016_config_path)


def generate_lld_for_file(input_file, smile_path=SMILE_path, config_path=Compare2016_config_path):
    """Generate a single _LLD.csv for input_file using openSMILE.

    Returns the path to the generated LLD CSV, or None on failure.
    """
    base = os.path.splitext(os.path.basename(input_file))[0]
    lld_csv = os.path.join(os.path.dirname(input_file), base + "_LLD.csv")
    if os.path.exists(lld_csv):
        return lld_csv

    try:
        # Use the paths provided or fall back to values from common.paths
        # (smile_path and config_path have defaults from common.paths)
        # Ensure SMILExtract can find its runtime dependencies by using its folder as cwd
        smile_dir = os.path.dirname(smile_path) if smile_path and os.path.isabs(smile_path) else None
        env = os.environ.copy()
        if smile_dir:
            # Prepend smile_dir to PATH so DLLs/loaders can be found on Windows
            env['PATH'] = smile_dir + os.pathsep + env.get('PATH', '')

        # verify SMILExtract binary exists
        if not smile_path or not os.path.exists(smile_path):
            raise RuntimeError(f"SMILExtract executable not found: {smile_path}. Please place SMILExtract.exe in the 'dependencies' folder or set common.paths.SMILE_path")

        # verify config exists
        if not config_path or not os.path.exists(config_path):
            raise RuntimeError(f"Config file not found: {config_path}. Please place Compare2016.conf (or ComParE_2016.conf) in the 'dependencies' folder or set common.paths.Compare2016_config_path")

        proc = subprocess.run([
            smile_path,
            "-C", config_path,
            "-I", input_file,
            "-lldcsvoutput", lld_csv,
            "-nologfile", "-loglevel", "2"
        ], cwd=smile_dir if smile_dir else None, env=env, capture_output=True, text=True)

        if proc.returncode != 0:
            # Surface stderr/stdout for debugging
            stderr = proc.stderr or proc.stdout or f"exit code {proc.returncode}"
            raise RuntimeError(f"SMILExtract failed for {input_file}: {stderr}")

        if os.path.exists(lld_csv):
            # quick read to validate
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
        # Re-raise to let callers handle/display the error
        raise


def generate_lld_in_tree(root_folder):
    """Walk root_folder and generate LLD CSVs according to the folder rules.

    Returns a list of generated LLD CSV file paths.
    """
    """
    Simpler behavior for public use:
    - Walk the directory tree under `root_folder`.
    - For every file that ends with .wav, attempt to generate a corresponding _LLD.csv
      if it does not already exist.

    Returns a list with the paths of generated LLD files.
    """
    generated = []
    for dirpath, _, filenames in os.walk(root_folder):
        wav_files = [f for f in filenames if f.lower().endswith('.wav')]
        if not wav_files:
            continue

        # Process all WAV files found in the folder (no special-name filtering)
        for file in wav_files:
            input_file = os.path.join(dirpath, file)
            base = os.path.splitext(file)[0]
            lld_csv = os.path.join(dirpath, base + '_LLD.csv')
            if os.path.exists(lld_csv):
                # skip already existing LLD files
                continue
            res = generate_lld_for_file(input_file)
            if res:
                generated.append(res)
    return generated


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description="Generate LLD CSV files from WAV files using openSMILE.")
    ap.add_argument("--root", default=root_folder_path, help="Root folder to scan for WAV files.")
    ap.add_argument("--verbose", action="store_true", help="Print verbose messages.")
    args = ap.parse_args()

    gen = generate_lld_in_tree(args.root)
    print(f"Generati {len(gen)} file LLD in {args.root}")
