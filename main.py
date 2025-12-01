import os
import logging
import whisper
import ollama
import json
import requests
import uuid
import re
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- CONFIGURACIÓN DE LOGS ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELOS ---
print("⏳ Cargando modelo Whisper (Audio/Video)...")
try:
    whisper_model = whisper.load_model("tiny")
    print("✅ Whisper cargado.")
except Exception as e:
    print(f"❌ Error cargando Whisper: {e}")

# Asegúrate de tener estos modelos con: `ollama pull llama3.2:1b` y `ollama pull llava`
MODELO_TEXTO = 'llama3.2:1b'
MODELO_VISION = 'llava'

TEMP_DIR = "temp_downloads"
os.makedirs(TEMP_DIR, exist_ok=True)


# --- CLASES ---
class TipoEvidencia(str, Enum):
    AUDIO = 'Audio'
    VIDEO = 'Video'
    FOTO = 'Foto'
    TEXTO = 'Texto'


class AnalysisRequest(BaseModel):
    tipo: TipoEvidencia
    file_path: str | None = None
    url: str | None = None
    text_content: str | None = None


# --- UTILIDADES ---

def download_file(url: str) -> str:
    """Descarga el archivo temporalmente."""
    try:
        ext = url.split('.')[-1] if '.' in url.split('/')[-1] else 'tmp'
        if len(ext) > 4: ext = 'tmp'
        filename = f"{uuid.uuid4()}.{ext}"
        local_path = os.path.join(TEMP_DIR, filename)

        logging.info(f"⬇️ Descargando desde: {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_path
    except Exception as e:
        raise Exception(f"Error descarga: {str(e)}")


def clean_and_parse_json(text_raw):
    """
    Intenta limpiar y parsear la respuesta de la IA.
    Maneja bloques Markdown (```json) y errores de formato.
    """
    if not text_raw:
        return None

    # 1. Limpieza de bloques de código Markdown
    if "```json" in text_raw:
        text_raw = text_raw.split("```json")[1].split("```")[0]
    elif "```" in text_raw:
        text_raw = text_raw.split("```")[1].split("```")[0]

    # 2. Intentar parseo directo
    try:
        return json.loads(text_raw)
    except:
        pass

    # 3. Intentar búsqueda con Regex (busca el primer objeto JSON válido)
    try:
        match = re.search(r'\{.*\}', text_raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass

    # 4. Fallback si no es JSON válido
    logging.warning(f"⚠️ No se pudo parsear JSON. Texto crudo: {text_raw[:50]}...")
    return {
        "tema": "Error de Formato IA",
        "resumen": text_raw[:200] + "...",  # Devolvemos el texto plano como resumen
        "pasos_clave": ["Revisión manual requerida"]
    }


def get_ollama_json(prompt: str, model: str, images: list = None):
    """Consulta a Ollama forzando salida JSON."""
    messages = [{'role': 'user', 'content': prompt}]
    kwargs = {
        'model': model,
        'messages': messages,
        'format': 'json',  # Forzamos modo JSON nativo de Ollama
        'options': {'temperature': 0.1}  # Baja temperatura para ser más determinista
    }
    if images:
        messages[0]['images'] = images

    logging.info(f"🧠 Consultando Ollama ({model})...")
    try:
        response = ollama.chat(**kwargs)
        content = response['message']['content']
        logging.info(f"🤖 Respuesta IA recibida.")
        return clean_and_parse_json(content)
    except Exception as e:
        logging.error(f"❌ Error Ollama: {e}")
        return {"tema": "Error Ollama", "resumen": str(e), "pasos_clave": []}


# --- PROCESADORES ESPECÍFICOS ---

async def process_audio_video(file_path: str):
    logging.info(f"🎙️ Transcribiendo multimedia...")
    # Whisper corre localmente
    result = whisper_model.transcribe(file_path, fp16=False)
    text = result["text"]

    prompt = f"""
    Actúa como ingeniero agrónomo. Analiza la siguiente transcripción de un reporte de plaga.

    REGLAS:
    1. Responde SOLO en JSON.
    2. Si la transcripción es ruido o irrelevante, pon "Información insuficiente".

    JSON ESPERADO: {{"tema": "Titulo", "pasos_clave": ["Paso 1", "Paso 2"], "resumen": "..."}}

    Transcripción: "{text}"
    """
    return get_ollama_json(prompt, MODELO_TEXTO), text


async def process_image(file_path: str):
    logging.info(f"👁️ Analizando imagen...")
    prompt = """
    Analiza esta imagen agrícola. Busca plagas, daños en hojas o insectos.
    Responde en Español y SOLO en JSON.
    JSON: {"tema": "Nombre de la plaga o cultivo", "pasos_clave": ["Síntoma visual 1", "Síntoma visual 2"], "resumen": "Descripción breve del daño."}
    """
    return get_ollama_json(prompt, MODELO_VISION, images=[file_path]), "Análisis Visual de Imagen"


async def process_text_content(content: str):
    logging.info(f"📄 Analizando texto ({len(content)} chars)...")

    # --- PROMPT DEFENSIVO MEJORADO ---
    prompt = f"""
    Eres un experto agrónomo asistente. Analiza el reporte de un agricultor.

    TEXTO DEL REPORTE: "{content[:4000]}"

    INSTRUCCIONES:
    1. Genera un JSON válido.
    2. Si el texto es muy corto (ej: "apareció", "no sé", "gusano"), vago o grosero:
       - tema: "Información Insuficiente"
       - resumen: "El usuario no proporcionó detalles técnicos."
       - pasos_clave: ["Solicitar más fotos", "Contactar al usuario"]
    3. Si hay información útil, extrae el problema real.

    FORMATO JSON:
    {{
        "tema": "Título corto (ej: Gusano Cogollero)",
        "pasos_clave": ["Lista de síntomas o acciones recomendadas"],
        "resumen": "Resumen técnico de 1 párrafo."
    }}
    """
    return get_ollama_json(prompt, MODELO_TEXTO), content


# --- ENDPOINT PRINCIPAL ---

@app.post("/analizar")
async def analyze_evidence(request: AnalysisRequest):
    logging.info(f"📨 Request: Tipo={request.tipo}, TieneTexto={'Si' if request.text_content else 'No'}")

    file_path = None
    is_temp = False

    try:
        # --- 1. GESTIÓN DE ENTRADA ---
        if request.tipo == TipoEvidencia.TEXTO and request.text_content:
            pass  # Todo bien
        elif request.url:
            file_path = download_file(request.url)
            is_temp = True
        elif request.file_path and os.path.exists(request.file_path):
            file_path = request.file_path
        else:
            raise HTTPException(400, "Falta url, file_path o text_content.")

        # --- 2. PROCESAMIENTO ---
        summary = {}
        transcription = ""

        if request.tipo in [TipoEvidencia.VIDEO, TipoEvidencia.AUDIO]:
            if not file_path: raise HTTPException(400, "Audio/Video requiere archivo")
            summary, transcription = await process_audio_video(file_path)

        elif request.tipo == TipoEvidencia.FOTO:
            if not file_path: raise HTTPException(400, "Foto requiere archivo")
            summary, transcription = await process_image(file_path)

        elif request.tipo == TipoEvidencia.TEXTO:
            content = request.text_content if request.text_content else ""

            # Si no hay contenido directo, intentamos leer el archivo descargado
            if not content and file_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()

            # --- VALIDACIÓN DE LONGITUD (Short-circuit) ---
            # Si el usuario escribe algo trivial, no gastamos tiempo de GPU/CPU
            clean_content = content.strip()
            if len(clean_content) < 10:
                logging.info("⚠️ Texto demasiado corto. Saltando análisis IA.")
                summary = {
                    "tema": "Reporte Breve",
                    "resumen": f"El usuario reportó: '{clean_content}'. Se requiere seguimiento manual.",
                    "pasos_clave": ["Contactar al productor para más detalles"]
                }
                transcription = content
            else:
                summary, transcription = await process_text_content(content)

        return {
            "status": "success",
            "summary": summary,
            "transcription": transcription
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"💥 Error interno: {e}", exc_info=True)
        # Devolver 500 pero con detalle para debug
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Limpieza de archivos temporales
        if is_temp and file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"🗑️ Temp eliminado: {file_path}")
            except:
                pass


if __name__ == "__main__":
    import uvicorn

    # Ejecutar en puerto 8001 para no chocar con Laravel/Vue
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)