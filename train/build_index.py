import os
import re
import pandas as pd
import argparse

def generate_id_from_filename(s: str, base: int = 131, mod: int = 2**31 - 1) -> int:
    """
    Genera un ID numerico univoco da una stringa usando hashing polinomiale.
    
    Questa funzione implementa il rolling hash (polinomial rolling hash) per
    generare ID deterministici e univoci da nomi di file. Lo stesso filename
    produce sempre lo stesso ID.
    
    Args:
        s (str): Stringa di input (es. base_name del file o path)
        base (int): Base per il calcolo hash (default 131, primo)
        mod (int): Modulo per mantenere l'hash in un range (default 2^31 - 1)
    
    Returns:
        int: Hash numerico come intero positivo.
    """
    
    h = 0
    for ch in s:
        h = (h * base + ord(ch)) % mod
    return h

def main(root_dir, output_excel):
    if os.path.isdir(output_excel):
        output_excel = os.path.join(output_excel, "dataset_index.xlsx")
    else:
        base, ext = os.path.splitext(output_excel)
        if ext == "":
            output_excel = base + ".xlsx"
    
    # lista per raccogliere i dati
    records = []
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            # Considera solo i file che finiscono con ".wav" e non iniziano con "old_"
            if file.lower().endswith(".wav") and not file.lower().startswith("old_"):
                name = os.path.splitext(file)[0]  # senza estensione
                
                # ---- Estrazione Type ----
                tipo = "Free" if name.startswith("D") else "Read"
            
                # ---- Generazione Subject ID usando hash ----
                # Usa il base_name completo (senza estensione) per generare un ID univoco
                subject_id = str(generate_id_from_filename(name))
                
                # ---- Tipo soggetto (Paziente/Controllo) ----
                if "PAZIENTI" in root.upper():
                    tipo_soggetto = "Paziente"
                elif "CONTROLLI" in root.upper():
                    tipo_soggetto = "Controllo"
                else:
                    tipo_soggetto = "?"
                
                # Aggiungi il record alla lista
                records.append({
                    "FileName": file,
                    "ID": subject_id,
                    "Tipo audio": tipo,
                    "Tipo soggetto": tipo_soggetto
                })
    
    # ---- Salvataggio in Excel ----
    df = pd.DataFrame(records)
    df.to_excel(output_excel, index=False)
    print(f"✅ File Excel creato: {output_excel}")
    print(f"📊 Totale record: {len(records)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crea un file Excel di indice dei file audio nella cartella specificata."
    )
    
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Cartella principale da esplorare."
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Percorso del file Excel di output."
    )
    
    args = parser.parse_args()
    
    main(
        root_dir=args.root,
        output_excel=args.output
    )
