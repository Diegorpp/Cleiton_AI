import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
import whisper
from gtts import gTTS
import os
from playsound import playsound
from dotenv import load_dotenv


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

def gravar_audio(arquivo='entrada.wav', duracao=5, taxa=44100):
    print("Gravando...")
    audio = sd.rec(int(duracao * taxa), samplerate=taxa, channels=1, dtype='int16')
    sd.wait()
    write(arquivo, taxa, audio)
    print("Gravação salva.")

def transcrever(arquivo='entrada.wav'):
    modelo = whisper.load_model("base")
    resultado = modelo.transcribe(arquivo, language='pt')
    print("Transcrição:", resultado["text"])
    return resultado["text"]

def responder(texto):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "Você é um assistente simpático e direto ao ponto."},
            {"role": "user", "content": texto}
        ]
    )
    resposta = response.choices[0].message.content.strip()
    print("Resposta:", resposta)
    return resposta

def falar(texto, arquivo='resposta.mp3'):
    tts = gTTS(text=texto, lang='pt')
    tts.save(arquivo)
    playsound(arquivo)
    os.remove(arquivo)

# ==== Execução ====
gravar_audio()
texto = transcrever()
resposta = responder(texto)
falar(resposta)


# import speech_recognition as sr

# def ouvir_microfone():
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Diga algo...")
#         audio = r.listen(source)
#     try:
#         texto = r.recognize_google(audio, language='pt-BR')
#         print(f"Você disse: {texto}")
#         return texto
#     except sr.UnknownValueError:
#         print("Não entendi o que você disse.")
#     except sr.RequestError:
#         print("Erro ao conectar com o serviço de reconhecimento.")
