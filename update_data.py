import requests
import json
import os
import sys

API_URL = "https://stats.tennismylife.org/api/data-files"
DATA_DIR = "data 2/tennis_atp"

# The files we want to keep updated: last 4 years + ongoing
TARGET_YEARS = ["2023", "2024", "2025", "2026"]

def download_file(url, dest_path):
    print(f"Descargando {url} -> {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✓ Completado.")
    except Exception as e:
        print(f"✗ Error al descargar {url}: {e}")

def main():
    if not os.path.exists(DATA_DIR):
        print(f"El directorio {DATA_DIR} no existe. Creándolo...")
        os.makedirs(DATA_DIR, exist_ok=True)
        
    print(f"Consultando la API: {API_URL}")
    try:
        r = requests.get(API_URL)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Error al conectar con la API: {e}")
        sys.exit(1)
        
    files = data.get("files", [])
    
    # Process files
    for file_info in files:
        name = file_info.get("name", "")
        url = file_info.get("url", "")
        
        if not name or not url:
            continue
            
        # Check if it's ongoing_tourneys
        if name == "ongoing_tourneys.csv":
            dest = os.path.join(DATA_DIR, "ongoing_tourneys.csv")
            download_file(url, dest)
            continue
            
        # Check if it's one of the target years
        for year in TARGET_YEARS:
            if name == f"{year}.csv":
                # Rename it to atp_matches_YYYY.csv so db_builder reads it automatically
                dest_name = f"atp_matches_{year}.csv"
                dest = os.path.join(DATA_DIR, dest_name)
                download_file(url, dest)
                break
                
    print("\n¡Actualización completada!")
    print("Recuerda ejecutar db_builder.py para regenerar la base de datos con los nuevos archivos.")

if __name__ == "__main__":
    main()
