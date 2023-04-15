import pandas as pd
import numpy as np
import pickle
import time

class PredictService():
    def __init__(self, pathToModelHard=None, pathToModelSoft=None):
        self.pathToModel = pathToModelHard,
        self.pathToModel = pathToModelSoft,
        if pathToModelHard:
            self.modelHard = pickle.load(open(pathToModelHard, 'rb')),
        if pathToModelSoft:
            self.modelSoft = pickle.load(open(pathToModelSoft, 'rb'))


    def predict(self, data):
        data.item = data.item.astype('int16')
        data['item_food_category_article_or_ingredient'] = data['item'].astype(str) + '_' + data['food_category'].astype(str)+'_'+data['article_or_ingredient'].astype(str)
        d=data.loc[:,['waste','item_food_category_article_or_ingredient']]
        d = pd.get_dummies(d, columns=['item_food_category_article_or_ingredient'])
        print(self.modelHard)
        start_time = time.time()
        data['Has Anomlay HardAnomaly'] = self.modelHard[0].predict(d)
        print(self.modelHard[0].predict(d))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Elapsed time: ', elapsed_time)
        
        # # Obtener la lista de columnas One-Hot Encoded
        # col_names = d.filter(like='item_food_category_article_or_ingredient_').columns.tolist()

        # # Revertir el One-Hot Encoding
        # d['item_food_category_article_or_ingredient_decoded'] = d[col_names].idxmax(axis=1).str.replace('item_food_category_article_or_ingredient_', '')

        # # Eliminar las columnas One-Hot Encoded
        # d = d.drop(columns=col_names)
        # d = d.rename(columns={'item_food_category_article_or_ingredient_decoded': 'item_food_category_article_or_ingredient'})
        #data_out = d.merge(data[['foo']], left_index=True, right_index=True)
        return data
    def predictJson(self, data):
        data.item = data.item.astype('int16')
        data['item_food_category_article_or_ingredient'] = data['item'].astype(str) + '_' + data['food_category'].astype(str)+'_'+data['article_or_ingredient'].astype(str)
        d=data.loc[:,['waste','item_food_category_article_or_ingredient']]
        d = pd.get_dummies(d, columns=['item_food_category_article_or_ingredient'])
        print(self.modelHard)
        start_time = time.time()
        data['Has Anomlay HardAnomaly'] = self.modelHard[0].predict(d)
        print(self.modelHard[0].predict(d))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Elapsed time: ', elapsed_time)
        
        # # Obtener la lista de columnas One-Hot Encoded
        # col_names = d.filter(like='item_food_category_article_or_ingredient_').columns.tolist()

        # # Revertir el One-Hot Encoding
        # d['item_food_category_article_or_ingredient_decoded'] = d[col_names].idxmax(axis=1).str.replace('item_food_category_article_or_ingredient_', '')

        # # Eliminar las columnas One-Hot Encoded
        # d = d.drop(columns=col_names)
        # d = d.rename(columns={'item_food_category_article_or_ingredient_decoded': 'item_food_category_article_or_ingredient'})
        #data_out = d.merge(data[['foo']], left_index=True, right_index=True)
        return data, data.to_json()