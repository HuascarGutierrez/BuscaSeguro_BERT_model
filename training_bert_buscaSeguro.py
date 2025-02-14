from transformers import AutoTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# inicializacion de las VARIABLES USADAS
# en caso de hacer algun cambio, hacerlo en estas variables
RANDOM_SEED = 42 #
N_ClASSES = 2
MAX_LEN = 200
BATCH_SIZE = 16
# en caso de obtener un nuevo dataset, cambiar la siguiente variable
DATASET_PATH = 'data/Caracteristicas_Empleo_Falso.xlsx'
EPOCHS = 5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #en caso de tener GPU, usar esto para habilitarlo
#device = 'cpu'

np.random.seed(RANDOM_SEED) #establece la semilla para que tenga el mismo valor en cada ejecucion

dataframe = pd.read_excel(DATASET_PATH)
#capturamos la columna Anuncio y Fraudulento para su entrenamiento
dataframe = dataframe[['Anuncio','Fraudulento']]
#print(dataframe.head()) #con esto revisas los 5 primeros ejemplos de anuncios
#PRE_TRAINED_MODEL = 'dccuchile/bert-base-spanish-wwm-cased'
PRE_TRAINED_MODEL = 'bert-base-cased'
#creacion del tokenizer
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL)
'''#CON EL SIGUIENTE EJEMPLO PUEDES REVISAR COMO FUNCIONA EL tokenizer
sample_text = 'Como me gusta comer papas'
tokens = tokenizer.tokenize(sample_text)
token_ids = tokenizer.convert_tokens_to_ids(tokens=tokens)
print('Frase: ',sample_text)
print('Tokens: ', tokens)
print('Tokens numericos: ',token_ids)
# codificacion para inteoducir a BERT
encoding = tokenizer.encode_plus(
    sample_text,
    max_length=10,
    truncation=True,
    add_special_tokens=True,
    return_token_type_ids=False,
    padding='max_length',
    return_attention_mask=True,
    return_tensors='pt'
)
print(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))
print(encoding['input_ids'][0])
print(encoding['attention_mask'][0])
'''

#CREACION DEL DATASET
class trabajoFraudulentoDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        label = self.labels[item]
        encoding = tokenizer.encode_plus(
                review,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'  
        )
        return {
            'review': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# DATA LOADER
def data_loader(df, tokenizer, max_len, batch_size):
    dataset = trabajoFraudulentoDataset(
            reviews=df.Anuncio.to_numpy(),
            labels=df.Fraudulento.to_numpy(),
            tokenizer=tokenizer,
            max_length=MAX_LEN
    )
    return DataLoader(dataset, batch_size = BATCH_SIZE, num_workers = 0)

# PARA UNA BUENA PRACTICA, dividir los datos para entrenamiento y test   
#df_train, df_test = train_test_split(Dataframe, test_size=0.2, random_state= RANDOM_SEED)

# en este caso, hay pocos datos, asi que se usara todo con el siguiente comando
df_train = dataframe
df_test = dataframe

train_data_loader = data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
# MODELO 
class BERTSentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BERTSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL)
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Intentamos usar pooler_output, pero si es None, tomamos el CLS token de last_hidden_state
        cls_output = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]

        drop_output = self.drop(cls_output)
        output = self.linear(drop_output)
        return output
model = BERTSentimentClassifier(N_ClASSES)
model = model.to(device)
#model = model.to(device)
print(model)

# RE-ENTREMIENTO DEL MODELO, PARA QUE SE ADECUE A LOS DATOS DE LOS TRABAJOS FRAUDULENTOS
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False) # lr = learning rate
#nota. Mostrara un error con AdamW si es que actualizaste las librerias
# posible solucion. implementar y ajustar torch.optim.AdamW
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps= total_steps,
)
loss_fn = nn.CrossEntropyLoss().to(device)

# funcion de entrenamiento
def train_model(model, data_loader, loss_fn, optimizar, device, scheduler, n_examles):
    model = model.train()
    losses = []
    correct_predictions = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask= attention_mask)
        _, preds = torch.max(outputs, dim=1)
        #outputs_prob = torch.amax(outputs, dim=1)
        #preds = (outputs_prob>=0.5).float()
        loss = loss_fn(outputs, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double()/n_examles, np.mean(losses)

# funcion evaluacion o test
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            _, preds = torch.max(outputs, dim=1)
            #outputs_prob = torch.amax(outputs, dim=1)
            #preds = (outputs_prob >= 0.5).float()
            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

        return correct_predictions.double()/n_examples, np.mean(losses)

# Entrenamiento 
for epoch in range(EPOCHS):
    print('Epoch {} de {}'.format(epoch+1, EPOCHS))
    print('_____________________')
    train_acc, train_loss = train_model(
            model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train)
    )
    test_acc, test_loss = eval_model(
            model, test_data_loader, loss_fn, device, len(df_test)
    )
    print('Entrenamiento: Loss: {}, accuracy: {}'.format(train_loss, train_acc))
    print('Validacion: Loss: {}, accuracy: {}'.format(test_loss, test_acc))
    print('')

MODEL_PATH = "models/trabajos_fraudulentos_bert.pth"
TOKENIZER_PATH = "trabajos_fraudulentos_bert_tokenizer"

torch.save(model.state_dict(),MODEL_PATH)
tokenizer.save_pretrained(TOKENIZER_PATH)
print("Modelo y tokenizer guardados exitosamente.")     