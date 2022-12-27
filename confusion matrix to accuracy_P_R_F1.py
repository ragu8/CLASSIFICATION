import pandas as pd


# Read the Excel file into a DataFrame
df = pd.read_excel('confusion_matrix.xlsx', sheet_name='Sheet1')

# Convert the DataFrame to a NumPy array
confusion_matrix = df.to_numpy()

print(confusion_matrix)

num_classes = confusion_matrix.shape[0]
TP = [0] * num_classes
TN = [0] * num_classes
FP = [0] * num_classes
FN = [0] * num_classes

for i in range(num_classes):
    TP[i] = confusion_matrix[i, i]
    TN[i] = sum([confusion_matrix[j, k] for j in range(num_classes) for k in range(num_classes) if j != i and k != i])
    FP[i] = sum([confusion_matrix[j, i] for j in range(num_classes)]) - TP[i]
    FN[i] = sum([confusion_matrix[i, j] for j in range(num_classes)]) - TP[i]



accuracy = [0] * num_classes
raccuracy = [0] * num_classes
precision= [0] * num_classes
recall = [0] * num_classes
f1_score = [0] * num_classes
for i in range(num_classes):
    accuracy [i]= (TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i])* 100
    precision [i]= TP[i] / (TP[i] + FP[i]) * 100
    recall [i]= TP[i] / (TP[i] + FN[i]) * 100
    f1_score [i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) 

import numpy as np

def round_4f(x):
    return  round(x,2)

# Vectorize the function
format_4f_vec = np.vectorize(round_4f)

print('Class \t TP \t TN \t FP \t FN \t accuracy \t precision \t recall \t f1_score')
for i in range(num_classes):
    accuracy = format_4f_vec(accuracy)
    precision = format_4f_vec(precision)
    recall = format_4f_vec(recall)
    f1_score = format_4f_vec(f1_score)
    print( i,'\t',TP[i], '\t ', TN[i], '\t ', FP[i], '\t ', FN[i],  '\t', "{:.2f}".format(accuracy[i]), '\t \t', "{:.2f}".format(precision[i]), '\t \t', "{:.2f}".format(recall[i]), '\t\t ', "{:.2f}".format(f1_score[i]))
 
 
AVG_accuracy = round(np.average(accuracy),2)   
AVG_precision = round(np.average(precision),2)   
AVG_recall = round(np.average(recall),2)
AVG_f1_score = round(np.average(f1_score),2)

print( '\n Average :\t\t\t\t',"{:.2f}".format(AVG_accuracy), '\t \t', "{:.2f}".format(AVG_precision), '\t \t', "{:.2f}".format(AVG_recall) , '\t\t ', "{:.2f}".format(AVG_f1_score))
    
    
    
    
