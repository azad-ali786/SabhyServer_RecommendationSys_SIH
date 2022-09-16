import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert = TFDistilBertModel.from_pretrained("distilbert-base-uncased")

texts = """
Machine learning is transforming the world around us. To become successful, you\ufffdd better know what kinds of problems can be solved with machine learning, and how they can be solved. Don\ufffdt know where to start? The answer is one button away. During this course you will: - Identify practical problems which can be solved with machine learning - Build, tune and apply linear models with Spark MLLib - Understand methods of text processing - Fit decision trees and boost them with ensemble learning - Construct your own recommender system. As a practical assignment, you will - build and apply linear models for classification and regression tasks; - learn how to work with texts; - automatically construct decision trees and improve their performance with ensemble learning; - finally, you will build your own recommender system! With these skills, you will be able to tackle many practical machine learning tasks. We provide the tools, you choose the place of application to make this world of machines more intelligent. Special thanks to: - Prof. Mikhail Roytberg, APT dept., MIPT, who was the initial reviewer of the project, the supervisor and mentor of half of the BigData team. He was the one, who helped to get this show on the road. - Oleg Sukhoroslov (PhD, Senior Researcher at IITP RAS), who has been teaching MapReduce, Hadoop and friends since 2008. Now he is leading the infrastructure team. - Oleg Ivchenko (PhD student APT dept., MIPT), Pavel Akhtyamov (MSc. student at APT dept., MIPT) and Vladimir Kuznetsov (Assistant at P.G. Demidov Yaroslavl State University), superbrains who have developed and now maintain the infrastructure used for practical assignments in this course. - Asya Roitberg, Eugene Baulin, Marina Sudarikova. These people never sleep to babysit this course day and night, to make your learning experience productive, smooth and exciting.

"""
encoded_dict = {'Ethics':0,'Machine Learning':1,'Deep Learning':2,
                'Artificial Intelligence':3,'DSA':4,'Business':5,
                'Science':6,'Cryptography':7,'CS Fundamentals':8,
                'Web Dev':9,'App Dev':10,'Technology':11,'Others':12}

model = tf.keras.models.load_model('ML\my_model.h5', custom_objects={'TFDistilBertModel': TFDistilBertModel})

def predict_topic(Description):
    x_val = tokenizer(
    text=Description,
    add_special_tokens=True,
    max_length=100,
    truncation=True,
    padding='max_length', 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True) 
    validation = model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
    results = []
    validation[0].sort()
    for key , value in zip(encoded_dict.keys(),validation[0][-3:]):
        results.append(key)
    return results

print(predict_topic(texts))