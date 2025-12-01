import shutil
import subprocess
import os

def check_system():
    print("--- Verificando entorno para análisis de video ---\n")

    # 1. Verificar FFmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"✅ FFmpeg encontrado en: {ffmpeg_path}")
        # Probar versión
        try:
            result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"   Versión detectada: {result.stdout.splitlines()[0]}")
        except Exception as e:
            print(f"   ⚠️ FFmpeg está en el PATH pero falló al ejecutarse: {e}")
    else:
        print("❌ FFmpeg NO encontrado.")
        print("   Solución: Descarga FFmpeg, descomprimelo y agrega la carpeta /bin a tu variable de entorno PATH.")
        print("   [WinError 2] ocurre porque la librería de transcripción intenta llamar a este comando y no lo halla.")

    print("\n------------------------------------------------")

    # 2. Verificar ruta de archivo (Prueba con tu ruta específica)
    test_path = r"C:\Users\david\Downloads\comocultivarchile.mp4"
    if os.path.exists(test_path):
        print(f"✅ El archivo de video es accesible: {test_path}")
    else:
        print(f"⚠️ El archivo de video NO existe en la ruta: {test_path}")
        print("   Verifica que el nombre sea correcto y que Python tenga permisos de lectura.")

if __name__ == "__main__":
    check_system()