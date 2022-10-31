# KoBert-Project
!pip install mxnet
!pip install gluonnlp pandas tqdm
!pip install sentencepiece
!pip install transformers==3.0.2
!pip install torch

!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master

# torch
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm.notebook import tqdm

#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#GPU 사용
device = torch.device("cuda:0")

#BERT 모델, Vocabulary 불러오기 필수
bertmodel, vocab = get_pytorch_kobert_model()


# KoBERT에 입력될 데이터셋 정리
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))  

# 모델 정의
class BERTClassifier(nn.Module): ## 클래스를 상속
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

# Setting parameters
max_len = 128  #기존 max_len 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 20
max_grad_norm = 1
log_interval = 200  #기존 log_interval = 100
learning_rate =  5e-5

## 학습 모델 로드
PATH = '/content/drive/MyDrive/Colab Notebooks/KoBERT Model'
model = torch.load(PATH + 'KoBERT_Project_9_128.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load(PATH + 'model_state_dict_project_9_128.pt'))  # state_dict를 불러 온 후, 모델에 저장

#토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def new_softmax(a) : 
    c = np.max(a) # 최댓값
    exp_a = np.exp(a-c) # 각각의 원소에 최댓값을 뺀 값에 exp를 취한다. (이를 통해 overflow 방지)
    sum_exp_a = np.sum(exp_a)
    y = (exp_a / sum_exp_a) * 100
    return np.round(y, 3)

acc_result=[]

def predict(predict_sentence):

  data = [predict_sentence, '0']
  dataset_another = [data]

  another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
  test_dataloader = torch.utils.data.DataLoader(another_test, batch_size = batch_size, num_workers = 2)

  model.eval()
  
  
  


  
  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader): 
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)

    valid_length = valid_length
    label = label.long().to(device)

    out=model(token_ids, valid_length, segment_ids)

    test_eval=[]

    
    
    for i in out: 
      
    
      logits=i
      logits = logits.detach().cpu().numpy()

      if np.argmax(logits) == 0:
                test_eval.append("부정")
      elif np.argmax(logits) == 1:
                test_eval.append("긍정")
      
      
      

      
      
      print(">> 입력하신 내용에서 " + test_eval[0] + "이 느껴집니다.")
      
      
  return acc_result.append(np.argmax(logits))
  
## 입력 파일 추출
import pandas as pd
import io


input_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Valid_data/로얄캐닌 유리너리 SO .csv')


input_data = input_data.drop(['Unnamed: 0'], axis = 1)
input_data = input_data.drop(['name'], axis = 1)
input_data.columns=['review', 'star']

review = input_data.loc[:, 'review']

star = input_data.loc[:, 'star']
star = star.replace(1,0)
star = star.replace(2,0)
star = star.replace(3,np.nan)
star = star.replace(4,1)

input_data = input_data.replace(1,0)
input_data = input_data.replace(2,0)
input_data = input_data.replace(3,np.nan)
input_data = input_data.replace(4,1)

import re

input_data['review'] = input_data['review'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

input_data['review'] = input_data['review'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
input_data['review'].replace('', np.nan, inplace=True)
print(input_data.isnull().sum())
input_data.loc[input_data.review.isnull()][:5]
input_data = input_data.dropna(how = 'any')
star = star.dropna(how = 'any')
review = review.str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣]","")
review = review.dropna(how = 'any')

acc_test = []
count_total = 0

for i in review:
  print(i)
  predict(i)
  
count_o = 0
count_x = 0

count = 0

for rate in star:

  
  if (acc_result[count] == rate):
     acc_test.append('O')
     count_o += 1
  else:
     acc_test.append('X')
     count_x += 1
  count +=1

total = count_o + count_x

print('acc_result = ', acc_result)

print('acc_test = ', acc_test)
print('count_O =', count_o, 'count_X =', count_x, 'total = ', total)
# 0 = 부정, 1 = 긍정

print('acc = ' ,count_o/total *100)
