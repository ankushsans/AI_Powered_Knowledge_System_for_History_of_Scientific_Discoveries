import logging
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import pytesseract
import wikipediaapi
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import face_recognition
import numpy as np
import os
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
# Load contexts from dataset.pkl
with open("dataset_files/dataset4.pkl", "rb") as f:
    documents = pickle.load(f)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class MultimodalAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize LLaMA-3.2-1B-Instruct
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.qa_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to(self.device)

        # Initialize MiniLM for embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        self.contexts = documents
        self.context_embeddings = self.embedder.encode(self.contexts, convert_to_tensor=True, device=self.device)

        # Load known faces (add your own images for real use)
        self.known_faces = {
            "Albert Einstein": face_recognition.face_encodings(self.preprocess_image(Image.open("face_recgn_imgs/einstein.jpg")))[0],
            "Alexander Fleming": face_recognition.face_encodings(self.preprocess_image(Image.open("face_recgn_imgs/fleming.jpg")))[0]
            # Add more known faces as needed
        }

    def preprocess_image(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)

    def find_relevant_context(self, query):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True, device=self.device)
        similarities = util.pytorch_cos_sim(query_embedding, self.context_embeddings)[0]
        best_idx = similarities.argmax().item()
        if similarities[best_idx] < 0.9:  # Fallback to Wikipedia if similarity is low
            wiki = wikipediaapi.Wikipedia(
        user_agent='MyProjectName (merlin@example.com)',
        language='en'
        )
            page = wiki.page(query.split()[-1])
            if page.exists():
                return page.summary
        return self.contexts[best_idx]

    def generate_answer(self, question, context, reasoning="cot"):
        if reasoning == "cot":
            prompt = f"Reason step-by-step to answer: {question}\nContext: {context}\nAnswer:"
        elif reasoning == "tot":
            prompt = f"Explore multiple reasoning paths and provide the best answer for: {question}\nContext: {context}\nAnswer:"
        elif reasoning == "got":
            prompt = f"Connect related concepts to answer: {question}\nContext: {context}\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.qa_model.generate(
            inputs["input_ids"], attention_mask=inputs["attention_mask"],  max_new_tokens=200, num_beams=4, do_sample=True, top_p=0.95, temperature=0.9
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()

    def refine_answer(self, question, answer):
        if len(answer) < 20:
            logger.info("Answer too short, refining...")
            context = self.find_relevant_context(question)
            return self.generate_answer(f"Please elaborate on: {question}", context)
        return answer

    def process_text_input(self, text):
        context = self.find_relevant_context(text)
        answer = self.generate_answer(text, context, reasoning="cot")  # Default to Chain of Thought
        #return answer
        return self.refine_answer(text, answer)

    def recognize_face(self, image):
        image_np = self.preprocess_image(image)
        face_encodings = face_recognition.face_encodings(image_np)
        if not face_encodings:
            return None
        for name, known_encoding in self.known_faces.items():
            if face_recognition.compare_faces([known_encoding], face_encodings[0])[0]:
                return name
        return None

    def process_image_input(self, image_file, question=None):
        image = Image.open(image_file)
        person = self.recognize_face(image)
        if person:
            further_info = f"Who is {person}"
            further_info_answer = self.process_text_input(further_info)
            if question:
                relevant_question = f"Regarding {person}, {question}"
                answer = self.process_text_input(relevant_question)
                return f"Recognized: {person}. {answer}\nMore info: {further_info_answer}"
            return f"Recognized: {person}. Info: {further_info_answer}"
        text = pytesseract.image_to_string(image)
        if text.strip() and question:
            return self.process_text_input(question)
        return "No recognizable content or person found."


    def process_voice_input(self, audio_data):
        try:
            # Save the uploaded audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_webm:
                temp_webm.write(audio_data)
                temp_webm_path = temp_webm.name
            # Convert to WAV
            audio = AudioSegment.from_file(temp_webm_path, format="webm")
            temp_wav_path = temp_webm_path.replace('.webm', '.wav')
            audio.export(temp_wav_path, format="wav")
            # Transcribe
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_wav_path) as source:
                audio_record = recognizer.record(source)
                text = recognizer.recognize_google(audio_record)
                logger.info(f"Question: {text}")
            # Process the transcribed text
            answer = self.process_text_input(text)
            # Clean up temporary files
            os.remove(temp_webm_path)
            os.remove(temp_wav_path)
            return answer
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError as e:
            return f"Speech recognition service error: {e}"
        except Exception as e:
            logger.error(f"Error processing voice: {e}")
            return "Error processing voice input"
# Flask app setup
app = Flask(__name__)
CORS(app)
agent = MultimodalAgent()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_text', methods=['POST'])
def upload_text():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    answer = agent.process_text_input(text)
    return jsonify({'result': answer})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image_file = request.files['image']
    question = request.form.get('question', '')
    result = agent.process_image_input(image_file, question)
    return jsonify({'result': result})

@app.route('/upload_voice', methods=['POST'])
def upload_voice():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']
    audio_data = audio_file.read()
    result = agent.process_voice_input(audio_data)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
