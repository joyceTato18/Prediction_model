from keras.models import load_model
import numpy as np

def _load_model():
    """
    Loads and returns the pretrained model
    """
    model = load_model("model5_new.h5")
    print("Model loaded")
    return model

def prepare_data(data):
    # resize the input image and preprocess it
    
    # Itérer sur les lignes
    print(data)
    tab = []
    for i in data:
        tab.append(i)
    es = [ele for ele in tab if ele != '\n']
    es = [ele for ele in tab if ele != '\r']
    res=[]
    del tab[2:6]
    print(tab)
    tab= np.array(tab)
    res=tab.reshape(-1,1,3)
    print(res)
    return res
    


def predict(res, model):
    # We keep the 2 classes with the highest confidence score
    results = model.predict(res)
    results = float(results)
    
    return results