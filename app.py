from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from BertSentimentClassifier import BERTSentimentClassifier
from transformers import AutoTokenizer
import torch

N_ClASSES = 2
MAX_LEN = 200
DATASET_PATH = 'data/Caracteristicas_Empleo_Falso.xlsx'
PRE_TRAINED_MODEL = 'bert-base-cased'
MODEL_PATH = 'models/trabajos_fraudulentos_bert.pth'
TOKERNIZER_PATH = 'trabajos_fraudulentos_bert_tokenizer'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #en caso de tener GPU, usar esto para habilitarlo
#device = 'cpu'

# cargar el model
model = BERTSentimentClassifier(N_ClASSES, PRE_TRAINED_MODEL)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
print("modelo cargado de forma exitosa")

#cargar el tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKERNIZER_PATH)
print("tokenizer cargado de forma exitosa")

#creacion de la API

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],      # Permite peticiones de cualquier origen o pagina
    allow_credentials = True,   
    allow_methods = ["*"],      # permite todos los metodos como GET O POST
    allow_headers = ["*"]        # Permite todos los headers
)

'''from BertSentimentClassifier import BERTSentimentClassifier

# cargar el modelo 
modelo = BERTSentimentClassifier(N_ClASSES)
modelo.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
modelo.to('cpu')
print('Modelo cargado con exito')

#cargar el tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKERNIZER_PATH)
print('Tokenizer cargo con exito')
'''
from prueba_modelo import predecir_trabajo_fraudulento
@app.get('/predecir/')
async def predecir(text: str, prob: bool):

    prediccion = predecir_trabajo_fraudulento(text, obtener_prob=prob)
    return {'Sentimiento': prediccion}

#estructura del json
from pydantic import BaseModel
class SentimentRequest(BaseModel):
    text: str
    prob: bool
@app.post('/predict/')
async def predecir(request: SentimentRequest):
    test_text = request.text  # Obtener el texto del cuerpo
    prediction_prob = request.prob
    prediccion = predecir_trabajo_fraudulento(test_text, obtener_prob=prediction_prob) 
    return {'Sentimiento': prediccion}

import uvicorn
import os
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Usa el puerto asignado por Render
    uvicorn.run(app, host="0.0.0.0", port=port)