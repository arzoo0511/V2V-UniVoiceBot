import asyncio
import os
import shutil
import subprocess
import sys
import time
from typing import Optional
import requests
from dotenv import load_dotenv
from deepgram import (DeepgramClient, DeepgramClientOptions, LiveOptions,LiveTranscriptionEvents, Microphone)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,MessagesPlaceholder,SystemMessagePromptTemplate)
from langchain_groq import ChatGroq

load_dotenv()

ffmpeg_bin = r"C:\Users\ICG0148\Downloads\ffmpeg-7.1.1-full_build (1)\ffmpeg-7.1.1-full_build\bin"
if ffmpeg_bin not in os.environ["PATH"]:
    os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]
print("ffplay path:", shutil.which("ffplay"))

class Config:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        self.llm_model = "llama3-8b-8192"
        self.stt_model = "nova-2"
        self.tts_model = "aura-helios-en"
        self.Bot_prompt_file = "Bot_prompt.txt"
        
        self._validate()

    def _validate(self):
        if not self.groq_api_key:
            raise ValueError("ERROR: GROQ_API_KEY is not set in the environment.")
        if not self.deepgram_api_key:
            raise ValueError("ERROR: DEEPGRAM_API_KEY is not set in the environment.")
        if not os.path.exists(self.Bot_prompt_file):
            raise FileNotFoundError(f"ERROR: Bot prompt file not found at '{self.Bot_prompt_file}'")

        ffmpeg_bin = r"C:\Users\ICG0148\Downloads\ffmpeg-7.1.1-full_build (1)\ffmpeg-7.1.1-full_build\bin"
        if ffmpeg_bin not in os.environ["PATH"]:
            os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]
        if not self._is_installed("ffmpeg"):
            raise RuntimeError("ERROR: ffmpeg is not installed or not found in PATH.")
    
        if not self._is_installed("ffplay"):
            raise RuntimeError("ERROR: ffplay is not installed or not found in PATH.")

    @staticmethod
    def _is_installed(lib_name: str) -> bool:
        return shutil.which(lib_name) is not None
          
class LiveTranscriber:
    def __init__(self, config: Config):
        client_config = DeepgramClientOptions(options={"keepalive": "true"})
        self.client = DeepgramClient(config.deepgram_api_key, client_config)
        self.stt_model = config.stt_model
        self.transcript_future: Optional[asyncio.Future] = None

    async def listen(self) -> str:
        self.transcript_future = asyncio.Future()        
        connection = self.client.listen.asynclive.v("1") 
        connection.on(LiveTranscriptionEvents.Transcript, self._on_message)
        connection.on(LiveTranscriptionEvents.Error, self._on_error)

        options = LiveOptions(
            model=self.stt_model,
            language="en-US",
            punctuate=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300, 
            smart_format=True,
        )
        await connection.start(options)
        
        microphone = Microphone(connection.send)
        microphone.start()

        try:
            final_transcript = await self.transcript_future
            return final_transcript
        finally:
            microphone.finish()
            await connection.finish()

    async def _on_message(self, _, result, **kwargs):
        if result.is_final and result.channel.alternatives[0].transcript.strip():
            transcript = result.channel.alternatives[0].transcript
            if self.transcript_future and not self.transcript_future.done():
                self.transcript_future.set_result(transcript)

    async def _on_error(self, _, error, **kwargs):
        print(f"\nSTT Error: {error}\n")
        if self.transcript_future and not self.transcript_future.done():
            self.transcript_future.set_exception(Exception(f"STT Error: {error}"))

class LLMProcessor:
    def __init__(self, config: Config):
        llm = ChatGroq(temperature=0, model_name=config.llm_model, groq_api_key=config.groq_api_key)
        
        with open(config.Bot_prompt_file, 'r') as f:
            Bot_prompt = f.read().strip()
        
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(Bot_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)        
        self.conversation_chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

    def generate_response(self, user_text: str) -> str:
        start_time = time.time()
        response = self.conversation_chain.invoke({"text": user_text})
        end_time = time.time()
        elapsed_ms = int((end_time - start_time) * 1000)
        
        ai_response = response.get('text', 'I am not sure how to respond to that.')
        print(f"LLM ({elapsed_ms}ms): {ai_response}")
        return ai_response

class SpeechSynthesizer:
    def __init__(self, config: Config):
        self.api_key = config.deepgram_api_key
        self.model_name = config.tts_model
        self.api_url = f"https://api.deepgram.com/v1/speak?model={self.model_name}&encoding=linear16&sample_rate=24000"

    def speak(self, text: str):
        headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"}
        payload = {"text": text}
        
        player_command = [r"C:\Users\ICG0148\Downloads\ffmpeg-7.1.1-full_build (1)\ffmpeg-7.1.1-full_build\bin\ffplay.exe", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        request_start_time = time.time()
        
        try:
            with requests.post(self.api_url, stream=True, headers=headers, json=payload, timeout=20) as response:
                response.raise_for_status()
                first_byte_received = False
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        if not first_byte_received:
                            ttfb = int((time.time() - request_start_time) * 1000)
                            print(f"TTS TTFB: {ttfb}ms\n")
                            first_byte_received = True
                        player_process.stdin.write(chunk)
                        player_process.stdin.flush()
        except requests.exceptions.RequestException as e:
            print(f"TTS Request Error: {e}")
        finally:
            if player_process.stdin:
                player_process.stdin.close()
            player_process.wait()

class VoiceAssistant:
    TERMINATION_PHRASE = "goodbye"
    def __init__(self, config: Config):
        self.transcriber = LiveTranscriber(config)
        self.llm_processor = LLMProcessor(config)
        self.synthesizer = SpeechSynthesizer(config)

    async def run(self):
        print("--- Voice Assistant Activated ---")
        print(f"Say '{self.TERMINATION_PHRASE}' to end the conversation.")
        
        while True:
            try:
                print("\nListening...")
                user_text = await self.transcriber.listen()                
                if not user_text:
                    continue
                    
                print(f"Human: {user_text}")

                if self.TERMINATION_PHRASE in user_text.lower().strip():
                    print("Termination phrase detected. Shutting down.")
                    goodbye_message = "Goodbye! Have a great day."
                    print(f"AI: {goodbye_message}")
                    self.synthesizer.speak(goodbye_message)
                    break

                ai_response = self.llm_processor.generate_response(user_text)
                self.synthesizer.speak(ai_response)
                
            except Exception as e:
                print(f"An error occurred in the main loop: {e}")
                print("Restarting listening loop...")
                await asyncio.sleep(1)

async def main():
    try:
        config = Config()
        assistant = VoiceAssistant(config)
        await assistant.run()
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n--- Assistant Deactivated by User ---")
    except Exception as e:
        print(f"An unexpected fatal error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())