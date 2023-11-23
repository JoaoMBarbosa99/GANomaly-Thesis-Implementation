import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, accuracy_score, ConfusionMatrixDisplay, PrecisionRecallDisplay, fbeta_score
from sklearn.metrics import recall_score, f1_score, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

"Script that loads train and test data to calculate metrics() and merge the results on a single table"

"This algorithm also plots graphs and saves images"

#Load train csv
traind = pd.read_csv('output/train_results.csv')
epoch_types = traind['epoch'].unique()
total_epochs = max(epoch_types+1)

traind_metrics = pd.DataFrame(columns = ["Epoch","Avg pred_r", "Avg pred_f","Err_d","Err_g"]) #Create dataframe for train metrics results
np_traind = traind.to_numpy() #Convert traind to an array
n_epochs = len(epoch_types) #Determines the number of epochs
n_images = len(traind) #Determines the number of images seen by the model
train_normal_img = len(np_traind[np.where(np_traind[:,1]==0)]) #Determines the number of images per epoch

#Create test metrics dataframe
testd_metrics = pd.DataFrame(columns =["Abnormal Images", "Normal Images", #Create dataframe for test metrics results
                       "Epoch", "Threshold", "TP", "FP",
                        "TN", "FN", "TPR", "FPR", "ACC", "Precision",
                        "Recall", "F1", "F2", "AUC","AUPR"]) 
#Load test csv
testd = pd.read_csv('output/test_results.csv')

# Images on each test epoch
normal_img = len(testd[(testd['gt_labels']== 0) & (testd['epoch']== 0)]) #Number of normal used images for testing
abnormal_img = len(testd[(testd['gt_labels']== 1) & (testd['epoch']== 0)]) #Number of abnormal images used for testing

#Save Data
#Create Main Folder
directory ='Results_N{}_N{}_A{}_E{}'.format(train_normal_img,normal_img,abnormal_img,total_epochs)
parent_dir ='' #Used for JupyterNotebook
#parent_dir = "/output/"  #Used for Colab
path = os.path.join(parent_dir, directory)
os.mkdir(directory)
print("Directory '% s' created" % directory)
 
#Train Plots (train sub folder)
train_dir = 'train'
train_plots_dir = 'plots'
path = '{}/'.format(directory)
new_train_dir = os.path.join(path,train_dir)
os.mkdir(new_train_dir)
new_plot_dir = os.path.join(new_train_dir,train_plots_dir)
os.mkdir(new_plot_dir)

#Test Plots (test sub folder)
test_dir = 'test'
test_plots_dir = 'plots' 
new_test_dir = os.path.join(path,test_dir)
os.mkdir(new_test_dir)
new_plot_dir = os.path.join(new_test_dir,test_plots_dir)
os.mkdir(new_plot_dir)

# Create array for err_d
err_d_real = [] 
err_d_fake = [] 
err_d= [] 
l_adv = []
l_con = []
l_enc = []
l_g = []

#weights defined on ganomaly article and as hyperparameter input
w_enc= 1 
w_con= 50
w_adv= 1

for i in epoch_types: #for cycle to get results for each epoch 
    epoch = i

    n_epochs = len(epoch_types) #Determines the number of epochs
    n_images = len(traind) #Determines the number of images seen by the model
    train_normal_img = len(np_traind[np.where(np_traind[:,1]==epoch)]) #Determines the number of images per epoch

    # Predictions of real an fake images
    x_images = np.arange(1,train_normal_img) 
    epoch_preds = traind[traind['epoch']==i]
    pred_real = epoch_preds['pred_real']
    pred_fake = epoch_preds['pred_fake']

    average_pred_real = sum(epoch_preds['pred_real'])/len(epoch_preds)
    average_pred_fake = sum(epoch_preds['pred_fake'])/len(epoch_preds)
    #print(average_pred_real, average_pred_fake)
    
    # Evolution of err_d_real and err_d_fake + Plot
    #err_d_real = traind['err_d_real'] & traind['epoch']==i
    #err_d_fake = traind['err_d_fake'].unique()
    np_epoch = np_traind[np.where(np_traind[:,1]==i)]      #Separate array by epochs
    err_d_real_array = np_epoch[:,6]                       #Get err_d_real column for epoch i
    err_d_fake_array = np_epoch[:,7]                       #Get err_d_fake column for epoch i
    err_d_real_avg = np.average(err_d_real_array)          #Average batches real error for epoch i 
    err_d_fake_avg = np.average(err_d_fake_array)          #Average batches fake error for epoch i
    err_d_real.insert(epoch, err_d_real_avg)
    err_d_fake.insert(epoch, err_d_fake_avg)
    err_d_avg = (err_d_real_avg + err_d_fake_avg)/2
    err_d.insert(epoch,err_d_avg)
    
    #Loop Losses (adv,con,enc)
    l_adv.insert(epoch, np.average(np_epoch[:,8]))
    l_con.insert(epoch, np.average(np_epoch[:,9]))
    l_enc.insert(epoch, np.average(np_epoch[:,10]))
     #Calculation of the total loss
    l_g_loss = w_adv * np.average(np_epoch[:,8]) + w_con * np.average(np_epoch[:,9]) + w_enc * np.average(np_epoch[:,10])
    l_g.insert(epoch, l_g_loss) #Generator Objective Function

    #Merge data
    new_data = pd.DataFrame({"Epoch":i,"Avg pred_r": average_pred_real, "Avg pred_f": average_pred_fake, "Err_d":err_d_avg, "Err_g":l_g_loss},index=[epoch])
    traind_metrics = pd.concat([traind_metrics,new_data])
    
#Plot for discriminator's predictios evolution
avg_pred_real = traind_metrics['Avg pred_r']
avg_pred_fake = traind_metrics['Avg pred_f']
plt.plot(epoch_types, avg_pred_real, label = 'pred_real(1)', color = 'r')
plt.plot(epoch_types, avg_pred_fake, label = 'pred_fake(0)', color = 'c')
plt.title('Discriminators Pred Evolution')
plt.ylabel('Prediction')
plt.xlabel('Epoch')
plt.legend(loc = 'lower right')
de_file = '{}/train/plots/DiscEvol.png'.format(directory)
plt.savefig(de_file,dpi=300)
plt.show()

#Dricriminator Accuracy
#D = Correct_preds / Total_preds

# Evolution of err_d_real and err_d_fake + Plot
plt.plot(epoch_types, err_d_real, label = 'err_d_real', color = 'blue')
plt.plot(epoch_types, err_d_fake, label = 'err_d_fake', color = 'r')
plt.plot(epoch_types, err_d, label = 'err_d', color = 'black')
plt.title('Discriminator Cross Entropy Evolution')
plt.ylabel('Discrimiantor Error')
plt.xlabel('Epoch')
plt.legend(loc = 'best')
DCE_file = '{}/train/plots/DCE.png'.format(directory)
plt.savefig(DCE_file,dpi=300)
plt.show()

# Evolution of losses (generator)
    #Plot Losses
plt.plot(epoch_types, l_enc, label = 'l_enc', color = 'b')
plt.plot(epoch_types, l_con, label = 'l_con', color = 'm')
plt.plot(epoch_types, l_adv, label = 'l_adv', color = 'g')
plt.plot(epoch_types, l_g, label = 'l_g', color = 'black')
plt.title('Generator Losses Evolution')
plt.ylabel('Losses')
plt.xlabel('Epoch')
plt.legend(loc = 'upper right')
ge_file = '{}/train/plots/GenEvol.png'.format(directory)
plt.savefig(ge_file,dpi=300)
plt.show()

#For loop for test results of each epoch
for i in epoch_types:
    epoch = i
    
    #Actual and Predictions
    array = testd.to_numpy()
    array = array[np.where(array[:,1]==epoch)] #split array by epoch
    actual = array[:,3] #get column of ground truth
    predicted = array[:,2] #get column of ground truth
    
    #ROC (plot) & AUC - calculate 1 plot with lines for each epoch -------------------------------------
    "A good AUROC is "
    "tpr Proportion of abnormal samples correctly classified" 
    "fpr Proportion of not abnormal samples that were incorrectly classified"
    
        #Returns fpr and tpr for diferent thresholds
    fpr, tpr, roc_thresholds = metrics.roc_curve(actual,predicted) 
        #ROC value
    roc_auc = metrics.auc(fpr, tpr)
        #Plot
    plt.title('Receiver Operating Characteristic epoch:%i' % epoch)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    roc_file = '{}/test/plots/ROC{}.png'.format(directory,epoch)
    plt.savefig(roc_file,dpi=300)
    plt.show()
    #roc_auc = auc(fpr, tpr)
    
    #PRC (plot) & AUPRC --------------------------------------------------------------------------------
    "Maximizing Recall: Appropriate when false negatives is the focus"
    
        #Returns precision and recall for diferent thresholds
    precision, recall, pr_thresholds = precision_recall_curve(actual, predicted)
        #Plot
    PrecisionRecallDisplay.from_predictions(actual, predicted, name = 'AUPR')
    plt.title('Precision-Recall Curve epoch:%i' % epoch)
    plt.legend(loc="lower right")
    pr_file = '{}/test/plots/PrecRec{}.png'.format(directory,epoch)
    plt.savefig(pr_file,dpi=300)
    plt.show()
    aupr = average_precision_score(actual, predicted)
    
        #f2_score (recall twice as important as precision)
    #f2_score = fbeta_score(actual, predicted, average='binary', beta=2)
    numerator = 5 
    denom = ((4/precision)+(1/recall))
            #Use np.divide because precision_recall_curve sometimes picks threshsolds 
            #that output precision and recall 0, resulting in error. This method when denominator is not 0.
    f2_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0)) 
            #Get max f2_Score
    f2_score = np.max(f2_scores)
            #Get threshold out of max f2
    optimal_threshold = pr_thresholds[np.argmax(f2_score)]
    
        #Optimal Threshold (from f2_score)
    optimal_threshold = pr_thresholds[np.argmax(f2_scores)]
    #optimal_threshold = threshol[optimal_idx] #According to AUROC
    
        #Actual and Predictions
    array = testd.to_numpy()
    array = array[np.where(array[:,1]==epoch)] #split table by epoch
    actual = array[:,3] #get column of ground truth
    predicted = array[:,2] #get column of ground truth
    predicted = np.where(predicted > optimal_threshold, 1 ,0) #if prediction are bigger than optimal_threshold they turn 1, else 0
    #print(predicted)
    
        # TP, TN, FP, FN
    tp = len(testd[(testd['gt_labels']== 1) & (testd['epoch']== epoch) & (testd['anomaly_score']>optimal_threshold)]) #True Positives
    tn = len(testd[(testd['gt_labels']== 0) & (testd['epoch']== epoch) & (testd['anomaly_score']<optimal_threshold)]) #True Negatives
    fp = len(testd[(testd['gt_labels']== 0) & (testd['epoch']== epoch) & (testd['anomaly_score']>optimal_threshold)]) #False Positive
    fn = len(testd[(testd['gt_labels']== 1) & (testd['epoch']== epoch) & (testd['anomaly_score']<optimal_threshold)]) #False Negative
    
        # Acuraccy (acc)
    acc = accuracy_score(actual, predicted)
    
        #Precision
    p_score = precision_score(actual, predicted, average='binary')

    #Recall 
    r_score = recall_score(actual, predicted, average='binary')

        #f1 score for threshold set 
    f_score = f1_score(actual, predicted, average='binary')
    
    
        #anomalyscoredistribution (plot) 
    p0 = testd[(testd['gt_labels']== 0) & (testd['epoch']== epoch)]
    p0 = p0.to_numpy()
    p0 = p0[:,2] # Gt 0 predictions
    p1 = testd[(testd['gt_labels']== 1) & (testd['epoch']== epoch)]
    p1 = p1.to_numpy()
    p1 = p1[:,2] # Gt 1 predictions

    with plt.style.context("bmh"):
        plt.hist(p0, alpha=.5, label="Normal")
        plt.hist(p1, alpha=.5, label="Abnormal")
        plt.axvline(x=optimal_threshold, color='black', linestyle='dashed', label="$t_r$="+str(optimal_threshold))
        plt.legend()
        plt.title("Data Distribution epoch:%i" % epoch)
        dd_file = '{}/test/plots/DataDist{}.png'.format(directory,epoch)
        plt.savefig(dd_file,dpi=300)
        plt.show()

        # Confusion Matrix with optimal threshold
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, 
                                                display_labels = ["Normal", "Abnormal"])
    n_file = '{}/test/plots/CM_epoch{}.png'.format(directory,epoch)
    cm_display.plot()
    plt.savefig(n_file,dpi=300)
    plt.show()
    
        #TPR and FPR for the optimal threshold
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
        #save on the data frame -----------------------------------------------------------------------------
    new_data = pd.DataFrame({"Abnormal Images":abnormal_img, "Normal Images": normal_img,
                       "Epoch": epoch, "Threshold":optimal_threshold, "TP": tp, "FP": fp,
                        "TN": tn, "FN": fn, "TPR": tpr, "FPR": fpr, "ACC": acc, "Precision": p_score,
                        "Recall": r_score, "F1": f_score, "F2": f2_score, "AUC": roc_auc,"AUPR": aupr}, index = [epoch])
    
    testd_metrics = pd.concat([testd_metrics, new_data])

#Test Metrics Evolution (Plot)
plt.plot(testd_metrics['Epoch'], testd_metrics['Threshold'], label = 'Threshold', color = 'black')
plt.plot(testd_metrics['Epoch'], testd_metrics['TPR'], label = 'TPR', color = 'green')
plt.plot(testd_metrics['Epoch'], testd_metrics['FPR'], label = 'FPR', color = 'red' )
plt.plot(testd_metrics['Epoch'], testd_metrics['ACC'], label = 'Acc', color = 'orange' )
plt.plot(testd_metrics['Epoch'], testd_metrics['Precision'], label = 'Precision', color = 'blue')
plt.plot(testd_metrics['Epoch'], testd_metrics['Recall'], label = 'Recall', color = 'yellow')
plt.plot(testd_metrics['Epoch'], testd_metrics['F1'], label = 'F1', color = 'cyan' )
plt.plot(testd_metrics['Epoch'], testd_metrics['F2'], label = 'F2', color = 'olive' )
plt.plot(testd_metrics['Epoch'], testd_metrics['AUC'], label = 'AUROC', color = 'm' )
plt.plot(testd_metrics['Epoch'], testd_metrics['AUPR'], label = 'AUPR', color = 'purple' )
plt.xlim([0,epoch*1.1]) #Math to add size for labels off the graph lines
plt.ylim([0, 1])
plt.title('Metrics Evolution')
plt.ylabel('Metrics')
plt.xlabel('Epoch')
plt.legend(loc = 'best')
te_file = '{}/test/plots/TestEvol{}.png'.format(directory,epoch) 
plt.savefig(te_file,dpi=300)
plt.show()

#os.chdir(directory)
print(traind_metrics)
print(testd_metrics)
#Save to XLSX & CSV
traind_metrics.to_excel('traind_metrics.xlsx')
traind_metrics.to_csv('traind_metrics.csv') 
testd_metrics.to_excel('testd_metrics.xlsx') 
testd_metrics.to_csv('testd_metrics.csv') 