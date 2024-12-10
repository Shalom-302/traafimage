import tempfile
from fastapi import FastAPI, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
import requests
from langchain.schema import Document
from PIL import Image
import pytesseract
from io import BytesIO
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from map import get_mapreduce_chain
load_dotenv()
# os.getenv("")
genai.configure(api_key="")

app = FastAPI()

# Définir le modèle de données pour la route POST
class ImageURL(BaseModel):
    url: str

def download_image_to_temp(image_url: str) -> str:
    try:
        # Télécharger l'image depuis l'URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # Vérifier les erreurs HTTP

        # Créer un fichier temporaire
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors du téléchargement de l'image : {e}")


# Fonction pour extraire le texte de l'image
def get_image_text(image_path: str) -> str:
    try:
        # Charger l'image avec PIL
        img = Image.open(image_path)

        # Extraire le texte avec Tesseract
        text = pytesseract.image_to_string(img, lang="fr")
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'extraction du texte : {e}")


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs

# Fonction pour créer un index vectoriel à partir des chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Extraire le contenu textuel des Documents
    text_data = [doc.page_content for doc in text_chunks]
    vector_store = FAISS.from_texts(text_data, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Route POST qui prend une URL d'image et renvoie un JSON avec les informations extraites
@app.post("/extractinfo")

async def extract_info(image_url: ImageURL):
    """
    APPRENDRE FASTAPI
    """
    try:
        print("Début du traitement pour URL :", image_url.url)

        # Étape 1: Télécharger l'image
        image_path = download_image_to_temp(image_url.url)
        print("Image téléchargée :", image_path)

        # Étape 2: Extraire le texte
        extracted_text = get_image_text(image_path)
        print("Texte extrait :", extracted_text)

        # Étape 3: Diviser en chunks
        text_chunks = get_text_chunks(extracted_text)
        print("Chunks créés :", len(text_chunks))

        # Étape 4: Créer un vecteur store
        vector_store = get_vector_store(text_chunks)
        print("Vector Store créé :", vector_store)

        # Étape 5: Utiliser Gemini
        chain = get_mapreduce_chain()
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        fixed_question = (
            "Retourne-moi un JSON des informations extraites de l'image. "
            "Fusionne-les et organise-les sous forme d'un JSON structuré contenant les informations suivantes : "
            "{ 'hashtag': hashtag, 'location': location, 'description': description, 'date': date }"
        )

        # Recherche des documents similaires
        docs = new_db.similarity_search(fixed_question)
        chain = get_mapreduce_chain()
        result = chain.invoke({"input_documents":docs, "question": fixed_question })
        print("Résultat du modèle :", result)
        if isinstance(result, str):
            # Si la réponse est sous forme de texte, il faut peut-être l'analyser et la structurer
            structured_result = {
                "hashtag": "exemple_hashtag",
                "location": "exemple_location",
                "description": "exemple_description",
                "date": "exemple_date"
            }
            return {"result": structured_result}

        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur durant le processus : {e}")
