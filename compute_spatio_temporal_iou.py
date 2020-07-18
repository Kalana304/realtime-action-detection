import numpy as np 

def rectint(bounds1, bounds2):
    xmin_1, xmin_2 = float(bounds1[0]),float(bounds2[0])
    ymin_1, ymin_2 = float(bounds1[1]), float(bounds2[1])
    xmax_1, xmax_2 = xmin_1 + float(bounds1[2]), xmin_2 + float(bounds2[2])
    ymax_1, ymax_2 = ymin_1 + float(bounds1[3]), ymin_2 + float(bounds2[3])

    dx = min(xmax_1, xmax_2) - max(xmin_1,xmin_2)
    dy = min(ymax_1, ymax_2) - max(ymin_1,ymin_2)

    if dx > 0 and dy > 0:
        return dx*dy 
    else:
        return 0

def inters_union(bounds1, bounds2):
    inters = rectint(bounds1,bounds2)
    ar1 = float(bounds1[2])*float(bounds1[3])
    ar2 = float(bounds2[2])*float(bounds2[3])
    union = ar1 + ar2 - inters

    return inters / (union + 1e-10)

def compute_spatio_temporal_iou(gt_fnr, gt_bb, dt_fnr, dt_bb):
    tgb = gt_fnr[0] # time gt begins
    tge = gt_fnr[-1] # time gt ends
    tdb = dt_fnr[0] # time dt begins
    tde = dt_fnr[-1] # time dt ends

    T_i = max(0, min(tge,tde)-max(tgb,tdb)) # temporal intersection

    if T_i > 0:
        T_i += 1

        T_u = max(tge,tde) - min(tgb,tdb) # temporal union
        T_iou = T_i/T_u # temporal IoU
        int_fnr = np.arange(max(tgb,tdb),min(tge,tde)) # intersect frame numbers

        # find the ind of the intersected frames in the detected frames
        int_find_dt = np.where(np.in1d(dt_fnr, int_fnr))[0]
        int_find_gt = np.where(np.in1d(gt_fnr, int_fnr))[0] # check ISSUE

        assert (len(int_find_dt) == len(int_find_gt))

        iou = np.zeros((len(int_find_dt),1))
        #import pdb; pdb.set_trace()
        for i in range(len(int_find_dt)):
            if int_find_gt[i] < 1:
                pf = 0
            else:
                pf = i 
            
            gt_bound = gt_bb[int_find_gt[pf],:]
            dt_bound = dt_bb[int_find_dt[pf],:]+1

            iou[i] = inters_union(np.array(gt_bound), np.array(dt_bound))
        st_iou = T_iou*np.mean(iou)
    else:
        st_iou = 0
    
    return st_iou

