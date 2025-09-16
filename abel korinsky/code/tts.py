# tts.py
import asyncio
import edge_tts
import tempfile
import os
import subprocess
import platform
import time

class TextToSpeech:
    async def speak_async(self, text):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            file_path = fp.name

        print(f"Generating TTS audio to: {file_path}")
        # Generate TTS audio
        communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
        await communicate.save(file_path)
        print("TTS audio generation complete.")

        # Play audio and wait for it to finish
        try:
            current_os = platform.system()

            if current_os == "Windows":
                # Use pygame as primary method (most reliable for MP3)
                try:
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(file_path)
                    pygame.mixer.music.play()
                    print("Playing audio on Windows using pygame...")
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    pygame.mixer.quit()
                    print("Audio playback completed")
                except ImportError:
                    print("pygame not available. Trying alternative methods...")
                    # Fallback to system commands
                    try:
                        abs_path = os.path.abspath(file_path)
                        # Use cmd /c start /wait for proper blocking behavior
                        player_command = ["cmd", "/c", "start", "/wait", '""', abs_path]
                        print(f"Playing audio on Windows (start): {player_command}")
                        subprocess.run(player_command, check=True)
                    except Exception as e2:
                        print(f"Start command failed: {e2}")
                        raise Exception("No suitable audio player found on Windows")
                except Exception as e:
                    print(f"pygame playback failed: {e}")
                    # Fallback to system commands
                    try:
                        abs_path = os.path.abspath(file_path)
                        player_command = ["cmd", "/c", "start", "/wait", '""', abs_path]
                        print(f"Playing audio on Windows (start): {player_command}")
                        subprocess.run(player_command, check=True)
                    except Exception as e2:
                        print(f"Start command failed: {e2}")
                        raise Exception("Audio playback failed with all methods")
                                
            elif current_os == "Darwin":
                # macOS
                player_command = ["afplay", file_path]
                print(f"Playing audio on macOS: {player_command}")
                subprocess.run(player_command, check=True)
            else:
                # Linux (requires mpg123, paplay, or similar)
                try:
                    player_command = ["mpg123", "-q", file_path]
                    print(f"Playing audio on Linux (mpg123): {player_command}")
                    subprocess.run(player_command, check=True)
                except FileNotFoundError:
                    try:
                        player_command = ["paplay", file_path]
                        print(f"Playing audio on Linux (paplay): {player_command}")
                        subprocess.run(player_command, check=True)
                    except FileNotFoundError:
                        print("Error: Neither mpg123 nor paplay found. Please install one of them (e.g., sudo apt install mpg123).")
                        return

        except FileNotFoundError as fnfe:
            print(f"Audio playback command not found: {fnfe}. Please ensure the audio player is installed and in your PATH.")
        except subprocess.CalledProcessError as cpe:
            print(f"Audio playback process failed with exit code {cpe.returncode}. Stderr: {cpe.stderr.decode() if cpe.stderr else 'N/A'}")
        except Exception as e:
            print(f"Audio playback failed unexpectedly: {e}")
        finally:
            # Clean up the temporary file after playback (or error)
            if os.path.exists(file_path):
                print(f"Deleting temporary audio file: {file_path}")
                try:
                    os.remove(file_path)
                except PermissionError:
                    # Sometimes the file is still locked by the media player
                    print(f"Could not delete {file_path} immediately, it may still be in use")
                    # You might want to implement a delayed cleanup here

    def speak(self, text):
        try:
            asyncio.run(self.speak_async(text))
        except KeyboardInterrupt:
            print("\n[Interrupted] Speech was canceled.")
        except Exception as e:
            print(f"Error in TextToSpeech.speak: {e}")


# Recommended: Clean TextToSpeech class using pygame
class TextToSpeechClean:
    def __init__(self):
        try:
            import pygame
            pygame.mixer.init()
            self.pygame_available = True
            print("pygame initialized for audio playback")
        except ImportError:
            self.pygame_available = False
            print("pygame not available. Install with: pip install pygame")
    
    async def speak_async(self, text):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            file_path = fp.name

        print(f"Generating TTS audio to: {file_path}")
        communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
        await communicate.save(file_path)
        print("TTS audio generation complete.")

        if self.pygame_available:
            try:
                import pygame
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                print("Playing audio...")
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                print("Audio playback completed")
            except Exception as e:
                print(f"pygame playback failed: {e}")
        else:
            print("No audio playback available - pygame not installed")
        
        # Clean up
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted temporary audio file: {file_path}")
            except PermissionError:
                print(f"Could not delete {file_path} immediately")

    def speak(self, text):
        try:
            asyncio.run(self.speak_async(text))
        except KeyboardInterrupt:
            print("\n[Interrupted] Speech was canceled.")
        except Exception as e:
            print(f"Error in TextToSpeech.speak: {e}")