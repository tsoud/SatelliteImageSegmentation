'''
Author: Tamer Abousoud
---
Metrics for judging prediction quality using:
- Intersection over Union (IoU)
- Precision and Recall
'''

import numpy as np
import tifffile as tiff
from collections import defaultdict
import pandas as pd

from img_utils import normalize
from predict_from_patches import calculate_npatches, predict_patch

class prediction_metrics(object):
    '''
    Determine useful segemenatation metrics from an overall image prediction.
    '''
    def __init__(self, gt_mask, pred_mask):
        '''
        gt_mask: mask image with ground truth labels
        pred_mask: predicted mask image
        ---
        Stores metrics and lists of coordinates of true positive, false positive 
        and false negative predictions for each label class.
        '''
        
        from collections import defaultdict
        # make gt_mask and pred_mask into binary
        self.gt_mask = np.where(gt_mask == 255, 1, 0)
        self.pred_mask = np.where(pred_mask >= 0.5, 1, 0)
        
        self.n_classes = self.gt_mask.shape[0]  # pylint: disable=unsubscriptable-object
        self.metrics = defaultdict(dict)

        for i in range(self.n_classes):
            # find ground truth and predicted positives
            gt_pos = set(zip(*np.where(self.gt_mask[i,:,:] == 1)))
            pred_pos = set(zip(*np.where(self.pred_mask[i,:,:] == 1)))
            # get true positives, false positives and false negatives
            TP = sorted(list(gt_pos & pred_pos))
            FP = sorted(list(pred_pos - gt_pos))
            FN = sorted(list(gt_pos - pred_pos))
            # calculate mean IoU, precision, recall
            # prevent division by zero if denominators are zero
            IoU = len(TP)/(len(TP) + len(FP) + len(FN)) if (len(TP) + len(FP) + len(FN)) > 0 else 0
            precision = len(TP)/(len(TP) + len(FP)) if (len(TP) + len(FP)) > 0 else 0
            recall = len(TP)/(len(TP) + len(FN)) if (len(TP) + len(FN)) > 0 else 0

            self.metrics['Class' + str(i)] = {'TP': TP, 
                                              'FP': FP, 
                                              'FN': FN, 
                                              'IoU': IoU, 
                                              'Precision': precision, 
                                              'Recall': recall }


    def scores(self):
        '''
        Returns intersection-over-union (IoU), Precision and Recall
        scores for each label class.
        '''
        scores = defaultdict(dict)

        for Class in self.metrics.keys():
            scores[Class] = {score: self.metrics.get(Class).get(score) \
                             for score in ('IoU', 'Precision', 'Recall')}

        return pd.DataFrame(scores).T


    def show_prediction_errors(self, predicted_class, values='TP'):
        '''
        Display image of prediction errors for the given class.
        ---
        predicted_class: the class for which to display errors
        values: 'TP', 'FP' or 'FN' to plot true positive, false positive 
                 or false negative labeled pixels
        '''

        show_map = np.zeros_like(self.gt_mask[:3,:,:])
        errors = self.metrics.get(predicted_class).get(values)
        for i, j in errors:
            show_map[0, i, j] = 255  # show predicted pixels in red
            
        tiff.imshow(show_map)


# --------------------------------------------------------------------------- #


# Use this function to aggregate scores over all images for a model

def overall_scores(model, img_list, img_dir, mask_dir, patch_size, overlap):
    '''
    Get overall scores across all training images.
    ---
    model: trained model to use for prediction
    img_list: list of training images
    patch_size: patch size for `predict_patch()` function
    overlap: overlap for `predict_patch()` function
    '''
    
    overall_results = dict(IoU = {}, Precision = {}, Recall = {})
    scores_df = pd.DataFrame()
    
    for im in img_list:
        img = tiff.imread(f'{img_dir}{im}')  # read mband image
        img = normalize(img)
        img = img.transpose([1,2,0])  # re-order to channels last
        mask = tiff.imread(f'{mask_dir}{im}')  # original mask
        # predict labels
        predicted_mask = predict_patch(img, model, patch_size=patch_size, overlap=overlap).transpose([2,0,1])
        df = prediction_metrics(mask, predicted_mask).scores()
        scores_df = pd.concat([scores_df, df])
    
    # get scores for each class
    classes = scores_df.index.unique()
    scores = scores_df.to_dict('series')
    
    # return average metric score for each class
    # use `np.compress()` to return non-zero values only
    # assuming zero value for class means it is not in image
    for cl in classes:
        iou = scores.get('IoU')[cl].to_numpy()
        overall_results['IoU'][cl] = np.mean(np.compress(iou > 0, iou))
        prc = scores.get('Precision')[cl].to_numpy()
        overall_results['Precision'][cl] = np.mean(np.compress(prc > 0, prc))
        rcl = scores.get('Recall')[cl].to_numpy()
        overall_results['Recall'][cl] = np.mean(np.compress(rcl > 0, rcl))
        
    return overall_results

