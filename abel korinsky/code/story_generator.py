# story_generator.py
from jinja2 import Template
import requests
from typing import Dict

class StoryGenerator:
    def __init__(self, model: str = "phi3.5", host: str = "http://localhost:11434"):
        self.model = model
        self.api_url = f"{host}/api/generate"
        
        # Updated prompt template for speculative stories with sustainability focus
        self.template = Template("""
Create a speculative character observation about this person's connection to sustainable living:

Age: {{ age }}
Gender: {{ gender }}
Emotion: {{ emotion }}
Clothing: {{ clothing_list }}
Style: {{ style }}

Write exactly 1 paragraph (35-50 words). Use "this person" instead of names. Focus on their potential lifestyle choices related to sustainability (e.g., eco-conscious habits, minimalism, or lack thereof). Keep it neutral, creative, and complete.

Example: "This person likely embraces sustainable living, choosing minimalist clothing that suggests a preference for quality over quantity, possibly supporting eco-friendly brands and a low-waste lifestyle."

Story:""")
    
    def _check_ollama_connection(self) -> bool:
        """Verify Ollama server is running"""
        try:
            response = requests.get(f"{self.api_url.replace('/generate', '/tags')}", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def build_prompt(self, attributes: Dict) -> str:
        """Build the prompt from person attributes"""
        clothing_items = attributes.get('clothing', [])
        clothing_list = ', '.join(clothing_items) if clothing_items else 'casual attire'
        
        prompt_data = {
            'age': attributes.get('age', 'unknown'),
            'gender': attributes.get('gender', 'unknown'),
            'emotion': attributes.get('emotion', 'neutral'),
            'clothing_list': clothing_list,
            'style': attributes.get('style', 'casual')
        }
        
        return self.template.render(**prompt_data)

    def generate_story(self, attributes: Dict, max_retries: int = 3) -> str:
        """Generate a story based on person attributes with retries for incomplete stories"""
        if not self._check_ollama_connection():
            return self._fallback_story("Ollama server not available")

        prompt = self.build_prompt(attributes)
        print(f"Generating story with prompt: {prompt[:100]}...")

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.8,  # Slightly higher for creativity
                            "top_p": 0.9,
                            "num_predict": 100,  # Increased to ensure complete stories
                            "stop": ["\n\n", "Example:", "Note:"]  # Stop tokens
                        }
                    }
                )
                
                response.raise_for_status()
                data = response.json()
                story = data.get("response", "").strip()

                # Clean up the response
                story = self._clean_story(story)
                
                # Relaxed word count check to allow slightly longer stories
                if story and len(story.split()) >= 30:
                    print(f"Generated story (attempt {attempt + 1}): {story}")
                    return story
                else:
                    print(f"Story too short (attempt {attempt + 1}): {story}")
                    if attempt < max_retries - 1:
                        print("Retrying story generation...")
                        continue
                    return self._fallback_story("Story length invalid after retries")

            except requests.exceptions.RequestException as e:
                print(f"Request error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    print("Retrying story generation...")
                    continue
                return self._fallback_story("Request failed")
            except Exception as e:
                print(f"Unexpected error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    print("Retrying story generation...")
                    continue
                return self._fallback_story("Unexpected error")

        return self._fallback_story("Max retries reached")

    def _clean_story(self, story: str) -> str:
        """Clean and format the generated story"""
        # Remove common unwanted phrases
        unwanted_phrases = [
            "Story:", "Character observation:", "Brief description:",
            "This is a", "Here is a", "The person appears to be"
        ]
        
        for phrase in unwanted_phrases:
            story = story.replace(phrase, "").strip()
        
        # Ensure it starts with "this person"
        if not story.lower().startswith("this person"):
            if story.lower().startswith("a person") or story.lower().startswith("the person"):
                story = "This person" + story[8:]
            else:
                story = "This person " + story.lower()
        
        # Remove extra whitespace and ensure proper capitalization
        story = ' '.join(story.split())
        if story:
            story = story[0].upper() + story[1:] if len(story) > 1 else story.upper()
        
        return story

    def _fallback_story(self, reason: str) -> str:
        """Generate a fallback story when AI generation fails"""
        print(f"Using fallback story: {reason}")
        
        fallback_stories = [
            "This person likely embraces sustainable living, choosing minimalist clothing that suggests a preference for eco-friendly brands and a low-waste lifestyle.",
            "This person might prioritize sustainability, their casual style hinting at a preference for thrifted or durable clothing, reflecting an eco-conscious approach.",
            "This person could be committed to a green lifestyle, their practical attire suggesting a focus on functionality and sustainable fashion choices.",
            "This person seems to value simplicity, their style indicating a possible preference for sustainable materials and mindful consumption habits.",
            "This person may lead an eco-friendly life, their clothing choices reflecting a commitment to reducing environmental impact through thoughtful purchases."
        ]
        
        import random
        return random.choice(fallback_stories)