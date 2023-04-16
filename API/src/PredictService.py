import pandas as pd
import numpy as np
import pickle
import time

class PredictService():
    def __init__(self, pathToModelHard=None, pathToModelSoft=None, pathToVariables=None):
        self.pathToModel = pathToModelHard,
        self.pathToModel = pathToModelSoft,
        self.pathToVariables= pathToVariables
        if pathToModelHard:
            self.modelHard = pickle.load(open(pathToModelHard, 'rb')),
        if pathToModelSoft:
            self.modelSoft = pickle.load(open(pathToModelSoft, 'rb')),
        if pathToVariables:
            self.variables = pd.read_csv(pathToVariables, header=None,sep=';').iloc[0].tolist()


    def predict(self, data):
        data.item = data.item.astype('int16')
        data['item_food_category_article_or_ingredient'] = data['item'].astype(str) + '_' + data['food_category'].astype(str)+'_'+data['article_or_ingredient'].astype(str)
        d=data.loc[:,['waste','item_food_category_article_or_ingredient']]
        d = pd.get_dummies(d, columns=['item_food_category_article_or_ingredient'])
        start_time = time.time()
        data['Has Anomlay HardAnomaly'] = self.modelHard[0].predict(d)
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
    def predictJson(self, json):
        data = pd.json_normalize(json)
        data.item = data.item.astype('int16')
        data['item_food_category_article_or_ingredient'] = data['item'].astype(str) + '_' + data['food_category'].astype(str)+'_'+data['article_or_ingredient'].astype(str)
        d=data.loc[:,['waste','item_food_category_article_or_ingredient']]
        for columna_nueva in self.variables:
            if columna_nueva not in d.columns:
                d[columna_nueva] = 0
        # Crear un nuevo DataFrame con las columnas originales

        change_col='item_food_category_article_or_ingredient_'+data['item'].astype(str) + '_' + data['food_category'].astype(str)+'_'+data['article_or_ingredient'].astype(str)
        d[change_col]=1
        d = d.drop(columns='item_food_category_article_or_ingredient')
        start_time = time.time()
        data['Has Anomlay HardAnomaly'] = self.modelHard[0].predict(d)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Elapsed time: ', elapsed_time)
        print(data)
        return data.to_json(orient='records')