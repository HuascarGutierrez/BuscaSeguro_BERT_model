from BertSentimentClassifier import BERTSentimentClassifier
from transformers import AutoTokenizer
import torch
import numpy as np

RANDOM_SEED = 42 
N_ClASSES = 2
MAX_LEN = 200
DATASET_PATH = 'data/Caracteristicas_Empleo_Falso.xlsx'
PRE_TRAINED_MODEL = 'bert-base-cased'
MODEL_PATH = 'models/trabajos_fraudulentos_bert.pth'
TOKERNIZER_PATH = 'trabajos_fraudulentos_bert_tokenizer'
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #en caso de tener GPU, usar esto para habilitarlo
device = 'cpu'
np.random.seed(RANDOM_SEED) #establece la semilla para que tenga el mismo valor en cada ejecucion

# cargar el model
model = BERTSentimentClassifier(N_ClASSES, PRE_TRAINED_MODEL)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
print("modelo cargado de forma exitosa")

#cargar el tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKERNIZER_PATH)
print("tokenizer cargado de forma exitosa")

def predecir_trabajo_fraudulento(text, model=model, tokenizer=tokenizer, device=device, obtener_prob=False):
    encoding = tokenizer.encode_plus(
            text,
            max_length=MAX_LEN,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids = input_ids, attention_mask = attention_mask)
        # en caso de activar obtener_prob realizar las acciones dentro del if y retorna la probabilidad
        if(obtener_prob):
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prob_fraude = probabilities[0][1].item()  # Probabilidad de que sea fraudulento
            return prob_fraude
        prediction = torch.argmax(outputs, dim=1).item()
    return prediction

# Ejemplo
#test_text = 'Restaurante requiere contratar los servicios de ayudantes de cocina  Ref: 70514349'
# para este caso da 0 para Seguro y 1 para Fraudulento
#print(f'\ntipo de trabajo: {predecir_trabajo_fraudulento(test_text, model, tokenizer, device)}')

# si quieres saber la probabilidad, entonces agregas la condicion obtener_prob=True
#print(f'\ntipo de trabajo: {predecir_trabajo_fraudulento(test_text, model, tokenizer, device, obtener_prob=True)}')
