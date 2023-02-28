import json
from sentence_transformers import SentenceTransformer
import faiss
class FaissSearch():
    
    def __init__(self):
        self.data = []
        self.load_data()
        self.sentence_embeddings,self.model=self.transform_model()
        
        
    def load_data(self):
        with open('../ipc_id.json', 'r',encoding='utf-8') as f:
            new_data=json.load(f)
            for i in range(0,len(new_data)):
                self.data.append(new_data[str(i)]["section_desc"])
            # print(self.data)
            
    def transform_model(self):
        print("downloading data for model.....")
        model=SentenceTransformer('bert-base-nli-mean-tokens')
        sentence_embeddings = model.encode(self.data)
        
        print("Shape :",sentence_embeddings.shape)
        return sentence_embeddings,model
        
    def find_index(self,to_search:str):
        se_shape=self.sentence_embeddings.shape[1]
        index = faiss.IndexFlatL2(se_shape)
        index.add(self.sentence_embeddings)
        k=4
        xq = self.model.encode([to_search])
        D, I = index.search(xq, k)  # search
        print(I)
        print(D)
        
        
        
            
    
   
    
                
                
                
                
f=FaissSearch()
f.find_index("Kill")
