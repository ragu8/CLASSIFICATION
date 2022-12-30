
import os         
import cv2
import netron
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot    
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
import sklearn.model_selection as model_selection
import dataframe_image as dfi
import visualkeras
from keras.utils import plot_model

def load_data(dataset):
    class_names = []
    images = []
    labels = []  
    for folder in os.listdir(dataset):
        class_names.append(folder)    
    class_names_label = {class_name:i for i, class_name in enumerate(class_names)} 
    print("Loading Dataset {}".format(dataset))
    for folder in os.listdir(dataset):
        label = class_names_label[folder]
        for file in tqdm(os.listdir(os.path.join(dataset, folder))):
            img_path = os.path.join(os.path.join(dataset, folder), file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            IMAGE_SIZE = (150, 150)
            image = cv2.resize(image, IMAGE_SIZE)
            images.append(image)
            labels.append(label)
    return images, labels , class_names





def Performance_Metrics(actual, predictions , class_names):
    confusion_matrix_df = pd.DataFrame(confusion_matrix(actual,predictions),index = class_names, columns =class_names)

    FP = abs(confusion_matrix_df.sum(axis=0) - np.diag(confusion_matrix_df)) 
    FN = abs(confusion_matrix_df.sum(axis=1) - np.diag(confusion_matrix_df))
    TP = np.diag(confusion_matrix_df)
    TN = abs(confusion_matrix_df.sum() - (FP + FN + TP))
    NPV = TN/(TN+FN)
    FDR = FP/(TP+FP)
    FNR = FP/(FP+TN)
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    Error_rate = (FP + FN) / (TP + FP + FN + TN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN )
    F_measure = (2 * Recall * Precision) / (Recall + Precision)
    FPR = FN/(TP+FN)
    Specificity = TN/(TN+FP)
    
    
    dict = {}
    dict['class_name'] = class_names
    dict['TP'] = TP
    dict['FP'] = FP
    dict['TN'] = TN
    dict['FN'] = FN
    measures1 = pd.DataFrame(dict)

    dict1 = {}
    dict1['class_name'] = class_names 
    dict1['Accuracy'] = Accuracy *100
    dict1['Precision'] = Precision *100
    dict1['Recall'] = Recall*100
    dict1['F_measure'] = F_measure*100
    dict1['Error_rate'] = Error_rate*100
    dict1['Specificity'] = Specificity*100
    dict1['FNR'] = FNR*100
    dict1['FPR'] = FPR*100
    
    measures2 = pd.DataFrame(dict1)
    
    measures3 = pd.merge(measures1,measures2)                            
    new_row = {'class_name': 'Average', 'TP':np.round(np.average(TP),2), 
               'FP': np.round(np.average(FP),2), 'TN': np.round( np.average(TN),2), 
               'FN': np.round(np.average(FN),2), 'Accuracy': np.round(np.average(Accuracy)*100,2), 'Precision': np.round(np.average(Precision)*100,2),
               'Recall': np.round(np.average(Recall)*100,2), 'F_measure': np.round(np.average(F_measure)*100,2), 
               'Error_rate': np.round( np.average(Error_rate)*100,2), 'Specificity':  np.round( np.average(Specificity)*100,2), 
               'FNR': np.round( np.average(FNR)*100,2), 'FPR': np.round(np.average(FPR)*100,2)}

    measures3.loc['Average'] = new_row

    measures3.to_csv("measures.csv", index=False,float_format='%.2f')
    measures3 =pd.read_csv("measures.csv")
    measures3.TP = measures3.TP.astype(int)
    measures3.FP = measures3.FP.astype(int)
    measures3.TN = measures3.TN.astype(int)
    measures3.FN = measures3.FN.astype(int)
    measures3.iloc[-1,1:5] = ''
   #dfi.export(measures3, "Measures.png")
    return measures3
    
def Confusion_matrix(actual, predictions , class_names):
    confusion_matrix_df = pd.DataFrame(confusion_matrix(actual,predictions),index = class_names, columns =class_names)
    plt.figure(figsize=(15,15))
    sns.heatmap(confusion_matrix_df, annot=True,cmap = pyplot.cm.YlOrBr)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.show()
    
def gui_model(model):
    return visualkeras.layered_view(model, legend=True)
    
    
    
def filters(model):
    
    for layer in model.layers:
        if 'conv' not in layer.name:
            continue
 
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape)
        
        n_filters, ix = 6, 1
        for i in range(n_filters):
            f = filters[:, :, :, i]

            for j in range(3):

                ax = pyplot.subplot(n_filters, 3, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                pyplot.imshow(f[:, :, j], cmap='gray')
                ix += 1

            pyplot.show()
            
def feature_map():
    pass
      

def model_blocks(model):
    return plot_model(model, to_file='model.jpg')
