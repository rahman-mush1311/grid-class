import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score,roc_curve,ConfusionMatrixDisplay

TRUE_LABELS = "true_labels"
PREDICTED_LABELS= "predicted_labels"
LOG_PDFS='log_pdfs'
DEAD='d'
ALIVE='a'

class ModelEvaluation:
    def __init__(self, data, window=2):
    
        self.data = data
        self.window_size = window
        self.filtered_thresholds = []
        self.best_accuracy_threshold = None
        self.best_precision_threshold = None
        self.best_classify = -float('inf')
        self.best_precision = -float('inf')
    
    def combine_data(self,other_data):
    
        for obj_id, values in other_data.items():
            if obj_id in self.data:
                # Merge log values if object already exists
                self.data[obj_id][LOG_PDFS].extend(values[LOG_PDFS])
            else:
                # Add new object
                self.data[obj_id] = values
    
    def run_thresholding(self):
    
        self.get_thresholds_from_roc()
        
        for threshold in self.filtered_thresholds:
            predictions = []
            true_labels = []
            
            for obj_id, values in self.data.items():
                cls = DEAD
                log_values = values[LOG_PDFS]
                for i in range(len(log_values) - self.window_size + 1):
                    w = log_values[i:i+self.window_size]
                    if all(p <= threshold for p in w):
                        cls = ALIVE
                predictions.append(1 if cls == 'a' else 0)
                true_labels.append(1 if values[TRUE_LABELS] == 'a' else 0)
            if len(set(true_labels)) < 2:
                print(f"Warning: Only one class present in true_labels: {set(true_labels)}. Skipping evaluation.")
                #continue
            
            cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions)
            recall = recall_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions)
            classify = cm[0, 0] + cm[1, 1]  # True positives + True negatives
            
            if classify > self.best_classify:
                self.best_classify = classify
                self.best_accuracy_threshold = threshold
            if self.best_precision < precision:
                self.best_precision_threshold = threshold
                self.best_precision = precision
        
        print(f"Optimal Threshold: {self.best_accuracy_threshold}, Maximum Classification: {self.best_classify}, Precision: {self.best_precision}, Threshold for Best Precision: {self.best_precision_threshold}")
        return 
    
    def get_thresholds_from_roc(self):
    
        true_labels=[]
        log_pdf_values=[]
        for obj_data in self.data.values():
            log_pdf_values.extend(obj_data[LOG_PDFS])
            true_labels.extend([1 if obj_data[TRUE_LABELS] == ALIVE else 0] * len(obj_data[LOG_PDFS]))
    
        true_labels=np.array(true_labels)
        log_pdf_values=np.array(log_pdf_values)
        print(true_labels.shape,log_pdf_values.shape)
        fpr, tpr, roc_thresholds = roc_curve(true_labels, log_pdf_values)
        #print(min(roc_thresholds),max(roc_thresholds))
    
        for i in range(1, len(roc_thresholds)):
            if round(tpr[i],2) > round(tpr[i - 1],2) or round(fpr[i],2)<round(fpr[i-1],2):
                self.filtered_thresholds.append(roc_thresholds[i])
        
        '''
        plt.figure(figsize=(10, 6))
        # Plot the ROC curve
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.show()
        '''
        return 
    
    def predict_probabilities_dictionary_update(self):
    
        #self.run_thresholding()
        
        for obj_id in self.data:
            cls = DEAD
            for i in range(len(self.data[obj_id][LOG_PDFS]) - self.window_size + 1):
                w = self.data[obj_id][LOG_PDFS][i:i+self.window_size]
                if all([p <= self.best_accuracy_threshold for p in w]):
                    cls = ALIVE

        # Update the dictionary with predicted and true labels
            self.data[obj_id] = {
                LOG_PDFS: self.data[obj_id][LOG_PDFS],  # Original log PDF values
                TRUE_LABELS: self.data[obj_id][TRUE_LABELS],
                PREDICTED_LABELS: cls
            }
        
        true_labels = [self.data[obj_id][TRUE_LABELS] for obj_id in self.data]
        predicted_labels = [self.data[obj_id][PREDICTED_LABELS] for obj_id in self.data]

        # Create the confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=[DEAD, ALIVE])
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, pos_label='a', average='binary')
        recall = recall_score(true_labels, predicted_labels, pos_label='a', average='binary')
        precision = precision_score(true_labels, predicted_labels, pos_label='a', average='binary')
        print(f"{accuracy:<10.3f}{f1:<10.3f}{recall:<10.3f}{precision:<10.3f}")
        
         # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Dead (0)", "Alive (1)"])
        disp.plot(cmap="Blues")
        disp.ax_.set_title(f"  Confusion Matrix Using ")
        disp.ax_.set_xlabel("Predicted Labels")
        disp.ax_.set_ylabel("True Labels")
    
        metrics_text = (
            f"Accuracy: {accuracy:.3f}\n"
            f"F1-Score: {f1:.3f}\n"
            f"Recall: {recall:.3f}\n"
            f"Precision: {precision:.3f}\n"
            f"Threshold: {self.best_accuracy_threshold:.3f}"
        )
        disp.ax_.legend(
            handles=[
                plt.Line2D([], [], color='white', label=metrics_text)
            ],
            loc='lower right',
            fontsize=10,
            frameon=False
        )
    
        plt.show()
    
    def find_the_misclassified_obj(self):
   
        misclassified_ids = [ obj_id for obj_id, details in self.data.items()
        if details[TRUE_LABELS] != details[PREDICTED_LABELS] 
        ]
        print(len(misclassified_ids))
        print(misclassified_ids)
        
        return 
        
      
    
   