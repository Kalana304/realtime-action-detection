###################################################################################################################################################
# STEP-1: loop over the videos present in the predicited Tubes
# STEP-2: for each video get the GT Tubes
# STEP-3: Compute the spatio-temporal overlap bwtween GT tube and predicited tubes
# STEP-4: then label tp 1 or fp 0 to each predicted tube
# STEP-5: Compute PR and AP for each class using scores, tp and fp in allscore
###################################################################################################################################################

import numpy as np 
from collections import defaultdict 
from compute_spatio_temporal_iou import compute_spatio_temporal_iou
from xVOCap import voc_ap
from tqdm import tqdm
import sys

def getActionName(str):
    indx = str.split('/')
    action = indx[0]
    vidID = indx[1]

    return action, vidID

def sort_detection(dt_tubes):
    sorted_tubes = dt_tubes

    if len(dt_tubes['class']) != 0:
        num_detection = len(dt_tubes['class'])
        scores = np.array(dt_tubes['score'])
        indexs = np.argsort(scores)[::-1]

        sorted_tubes['class'] = list(np.array(dt_tubes['class'])[indexs])
        sorted_tubes['score'] = list(np.array(dt_tubes['score'])[indexs])
        sorted_tubes['framenr']['fnr'] = list(np.array(dt_tubes['framenr']['fnr'])[indexs])
        sorted_tubes['boxes']['bxs'] = list(np.array(dt_tubes['boxes']['bxs'])[indexs])
        
    return sorted_tubes

def get_PR_curve(annot, xmldata, testlist, actions, iou_th):
    print("Running PR curve function")
    num_vid = len(testlist[:300])    # original code line

    #-------------------------------------------------#
    #dtNames = list(xmldata.keys())
    #num_vid = len(dtNames)
    #-------------------------------------------------#

    num_actions = len(actions)
    AP = np.zeros(num_actions)
    averageIoU = np.zeros(num_actions)

    cc = np.zeros(num_actions).astype(np.int)
    allscore = {}
    for a in range(num_actions):
        allscore[a] = np.zeros((10000,2))
    
    total_num_gt_tubes = np.zeros(num_actions)

    preds = np.zeros(num_vid) - 1
    gts = np.zeros(num_vid)
    annotNames = list(annot.keys())
    #dtNames = xmldata["videoName"]
    #details = xmldata["details"]
    progress = tqdm(ncols=100, total=num_vid, desc="video", leave = True, position=0, file=sys.stdout)
    print("STEP-01")
    for vid in range(num_vid):
        #print("{:03d}".format(vid))
        progress.update(1)
        maxscore = -10000
        action, _ = getActionName(testlist[vid])
        action_id = annot[testlist[vid]]['label']
        #------------------------------------------------------#
        #action, _ = getActionName(dtNames[vid])
        #action_id = annot[dtNames[vid]]['label']
        #gtVidInd = np.where(np.array(annotNames) == testlist[vid])
        #drVidInd = np.where(np.array(dtNames) == testlist[vid])
        #------------------------------------------------------#

        dt_tubes = sort_detection(xmldata[testlist[vid]])
        gt_tubes = annot[testlist[vid]]['annotations']
        #--------------------------------------------------------#
        #dt_tubes = sort_detection(xmldata[dtNames[vid]])
        #gt_tubes = annot[dtNames[vid]]['annotations']
        #--------------------------------------------------------#
        num_detection = len(dt_tubes['class'])
        num_gt_tubes = len(gt_tubes)

        for gtind in range(num_gt_tubes):
            action_id = gt_tubes[gtind]['label']
            total_num_gt_tubes[action_id] += 1
        
        gts[vid] = action_id
        dt_labels = np.array(dt_tubes['class']).astype(np.int)
        covered_gt_tubes = np.zeros(num_gt_tubes)
        #import pdb; pdb.set_trace()
        for dtind in range(num_detection):
            dt_fnr = dt_tubes['framenr']['fnr'][dtind]
            dt_bb = dt_tubes['boxes']['bxs'][dtind]
            dt_label = dt_labels[dtind]
            
            if dt_tubes['score'][dtind] > maxscore:
                preds[vid] = dt_label
                maxscore = dt_tubes['score'][dtind]
            cc[dt_label] = cc[dt_label] + 1

            ioumax = -1*np.inf; maxgtind = 0

            for gtind in range(num_gt_tubes):
                action_id = gt_tubes[gtind]['label']
                if ((not covered_gt_tubes[gtind]) and (dt_label == action_id)):
                    #import pdb; pdb.set_trace()
                    gt_fnr = np.arange(gt_tubes[gtind]['sf']-1, gt_tubes[gtind]['ef']) # check ISSUE
                    gt_bb = gt_tubes[gtind]['boxes']
                    iou = compute_spatio_temporal_iou(gt_fnr, gt_bb, dt_fnr, dt_bb)
                    if iou > ioumax:
                        ioumax = iou
                        maxgtind = gtind
            
            if ioumax > iou_th:
                covered_gt_tubes[maxgtind] = 1 
                #import pdb; pdb.set_trace()
                allscore[dt_label][cc[dt_label]-1,0] = dt_tubes["score"][dtind]
                allscore[dt_label][cc[dt_label]-1,1] = 1
            else:
                #if vid == 287:
                #    import pdb; pdb.set_trace()
                allscore[dt_label][(cc[dt_label]-1)%10000,0] = dt_tubes["score"][dtind]
                allscore[dt_label][(cc[dt_label]-1)%10000,1] = 0
                
    for a in range(num_actions):
        #import pdb; pdb.set_trace()
        allscore[a] = allscore[a][0:cc[a],:]
        scores = np.array(allscore[a][:,0])
        labels = np.array(allscore[a][:,1])

        si = np.argsort(scores)[::-1]
        labels = labels[si]
        fp = np.cumsum(labels == 0)
        tp = np.cumsum(labels == 1)

        if len(tp) > 0:
            averageIoU[a] = (averageIoU[a]+1e-6)/(tp[-1]+1e-5)

        recall = tp/(total_num_gt_tubes[a]+1e-16)
        precision = tp/(fp+tp)

        AP[a] = voc_ap(recall, precision)

    acc = np.mean(preds == gts)
    AP = np.nan_to_num(AP)
    mAP = np.mean(AP)

    averageIoU = np.nan_to_num(averageIoU)
    mAIoU = np.mean(averageIoU)

    return mAP, mAIoU, acc, AP
        