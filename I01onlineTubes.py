import os
import pickle
import torch
import _init_path
import json 

#from utils.nms_wrapper import nms
from box_utils import nms
from get_PR_curve import get_PR_curve 

def initDatasetOpts(data_root, baseDir, dataset, imgType, model_type, listid, iteration_num, iouthresh, costtype, gap, sample_size):
    opts = {}
    opts['imgType'] = imgType
    opts['costtype'] = costtype
    opts['gap'] = gap
    opts['n_samples'] = sample_size
    opts['baseDir'] = baseDir
    opts['dataset'] = dataset
    opts['iouThresh'] = iouthresh
    opts['weight'] = iteration_num
    opts['listid'] = listid

    opts['vidList'] = os.path.join(data_root, dataset, 'splitfiles', 'testlist{}.txt'.format(listid))

    if dataset == 'ucf24':
        opts['actions'] = ['Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling',
                           'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing',
                           'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing',
                           'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping',
                           'VolleyballSpiking', 'WalkingWithDog']
    elif dataset == 'JHMDB':
        opts['actions'] = ['brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 'jump',
                           'kick_ball', 'pick', 'pour', 'pullup', 'push', 'run', 'shoot_ball', 'shoot_bow',
                           'shoot_gun', 'sit', 'stand', 'swing_baseball', 'throw', 'walk', 'wave']
    elif dataset == 'LIRIS':
        opts['actions'] = ['discussion', 'give_object_to_person', 'put_take_obj_into_from_box_desk',
                           'enter_leave_room_no_unlocking', 'try_enter_room_unsuccessfully', 'unlock_enter_leave_room',
                           'leave_baggage_unattended', 'handshaking', 'typing_on_keyboard', 'telephone_conversation']
    else:
        raise NotImplementedError
    opts['imgDir'] = os.path.join(data_root, dataset, imgType + '-images')
    opts['detDir'] = os.path.join(baseDir, dataset, 'detections',
                                  model_type + '-' + imgType + '-' + listid + '-' + '{iter:06}'.format(
                                      iter=iteration_num))
    # Downloaded from https://github.com/gurkirt/corrected-UCF101-Annots
    opts['annotFile'] = os.path.join(data_root, dataset, 'splitfiles', 'pyannot.pkl')
    opts['actPathDir'] = os.path.join(baseDir, dataset, 'actionPaths/python',
                                      '{}-{}-{}-{:06d}-{}-{:d}-{:04d}'.format(model_type, imgType, listid,
                                                                              iteration_num, costtype, gap,
                                                                              int(100 * iouthresh)))
    opts['tubeDir'] = os.path.join(baseDir, dataset, 'actionTubes/python',
                                   '{}-{}-{}-{:06d}-{}-{:d}-{:04d}'.format(model_type, imgType, listid, iteration_num,
                                                                           costtype, gap, int(iouthresh * 100)))
    if os.path.exists(os.path.join(opts['detDir'])):
        if not os.path.exists(opts['actPathDir']):
            print("Creating {}\n".format(opts['actPathDir']))
            os.mkdir(opts['actPathDir'])
        if not os.path.exists(opts['tubeDir']):
            print("Creating {}\n".format(opts['tubeDir']))
            os.mkdir(opts['tubeDir'])
        if dataset == 'ucf24' or dataset == 'JHMDB':
            create_dirs(opts['actPathDir'], opts['actions'])

    return opts


def create_dirs(basedirs, actions):
    for action in actions:
        save_name_action = os.path.join(basedirs, action)
        if not os.path.exists(save_name_action):
            print("Creating {}".format(save_name_action))
            os.makedirs(save_name_action)


def I01onlineTubes():
    # TODO Make arguments
    data_root = '../data/'
    save_root = '../data/'
    sample_size = 300
    iteration_num_rgb = 120000
    iteration_num_flow = 120000

    complete_list = [['ucf24', '01', 'rgb', iteration_num_rgb, 'score'],
                     ['ucf24', '01', 'brox', iteration_num_flow, 'score'],
                     ['ucf24', '01', 'fastOF', iteration_num_flow, 'score']]

    count = 0
    gap = 3
    model_type = 'CONV'
    alldopts = {}
    #for setind in range(len(complete_list)):
    for setind in range(1):
        dataset, listid, imgType, iteration, costtype = complete_list[setind]
        iouthresh = 0.1
        # generate directory sturcture based on the options
        dopts = initDatasetOpts(data_root, save_root, dataset, imgType, model_type, listid, iteration, iouthresh,
                                costtype, gap, sample_size)
        if os.path.exists(dopts['detDir']):
            alldopts[count] = dopts
            count += 1

    temp_results = {}

    # For each option, build tubes and evaluate them
    for index in range(count):
        opts = alldopts[index]
        if os.path.exists(opts['detDir']):
            print(
                'Video List {:02d} :: {}\nAnnotFile :: {}\nImage Dir :: {}\nDetection Dir :: {}\nActionpath Dir :: {}\nTube Dir :: {}\nSamples :: {}\n' \
                .format(index, opts['vidList'], opts['annotFile'], opts['imgDir'], opts['detDir'], opts['actPathDir'],
                        opts['tubeDir'], opts['n_samples']))
            # Build action paths given frame level detections
            #actionPaths(opts)
            # Perform temproal labelling and evaluate; results saved in results cell
            result_cell = gettubes(opts)
            if index not in list(temp_results.keys()):
                temp_results[index+1] = {} 
            
            temp_results[index+1]['Results'] = ["mAP", "mAIoU", "acc", "AP"] + result_cell.tolist()
            temp_results[index+1]['Opts'] = opts
    
    save_dir = os.path.join(save_root,'results/python')
    os.makedirs(save_dir, exist_ok = True)

    with open(f"{save_dir}/online_tubes_results_CONV.json", 'w') as f:
        json.dump(temp_results,f, indent = 3)


def gettubes(dopts):
    '''
    Facade function for smoothing tubes and evaluating them
    :param dopts:
    :return:
    '''
    numActions = len(dopts['actions'])
    results = np.zeros((2,6))
    counter=0
    class_aps = {}
    # save file name to save result for eah option type
    saveName = os.path.join('{}'.format(dopts['tubeDir']),'tubes-results.txt')
    if not os.path.exists(saveName):
        with open(dopts['annotFile'], 'rb') as annot_file:
            annot = pickle.load(annot_file)
            testvideos = getVideoNames(dopts['vidList'])
            '''
            for  alpha = 3 
                fprintf('alpha %03d ',alpha);
                tubesSaveName = sprintf('%stubes-alpha%04d.mat',dopts.tubeDir,uint16(alpha*100));
                if ~exist(tubesSaveName,'file')
            '''
            alpha=3
            print("alpha {:03d}".format(alpha))
            tubesSaveName = os.path.join(dopts['tubeDir'], "tubes-alpha{:04d}.pkl".format(100*alpha))
            if not os.path.exists(tubesSaveName):
                # read action paths
                actionpaths = readALLactionPaths(dopts['vidList'],dopts['actPathDir'],1)
                # perform temporal trimming
                smoothedtubes = parActionPathSmoother(actionpaths,alpha*np.ones((numActions,1)),numActions)
                # pickle the tubes
                with open(tubesSaveName, "wb") as write_file:
                    pickle.dump(smoothedtubes, write_file)
            else:
                with open(tubesSaveName, "rb") as read_file:
                    smoothedtubes = pickle.load(read_file)
                print("Finished reading pickled tube-info file")
            min_num_frames = 8
            kthresh = 0.0
            topk = 40
            
            xmldataSavename = os.path.join('{}/results'.format(dopts['baseDir']),'xmldata-{:04d}.pkl'.format(dopts['n_samples']))

            os.makedirs(os.path.join('{}/results'.format(dopts['baseDir'])), exist_ok=True)
               
            if not os.path.exists(xmldataSavename):
                print("Running convert2eval function")
                xmldata = convert2eval(smoothedtubes, min_num_frames,
                                   kthresh*np.ones((numActions,1)), topk, np.array(testvideos),dopts['n_samples'])
            
                with open(xmldataSavename,'wb') as write_file:
                    pickle.dump(xmldata,write_file)
            else:
                print("Found xmld file: reading the pickled file")
                with open(xmldataSavename,'rb') as read_file:
                    xmldata = pickle.load(read_file)

            print("Making results file")
            for iou_th in [0.2,0.5]:
                tmAP, tmIoU, tacc, AP = get_PR_curve(annot,xmldata,testvideos,dopts["actions"], iou_th)
                print('iou_th: {:.2f} | tmAP: {:.3f} | tacc: {:.3f}'.format(iou_th, tmAP, tacc))
                results[counter,:] = [iou_th, alpha,alpha,tmIoU,tmAP,tacc]
                class_aps[counter] = AP
                counter += 1
            #import pdb; pdb.set_trace()
    return results


def convert2eval(final_tubes,min_num_frames,kthresh,topk,vids,n_samples):
    xmld = {}
    for v in range(min(n_samples,len(vids))):
        struct = {} 
        print(f'{v}: {vids[v]}')

        action_indexes = np.where(np.array(final_tubes['video_id'])==vids[v])
        actionscore = np.array(final_tubes['dpActionScore'])[action_indexes]
        path_scores = np.array(final_tubes['path_scores'])[action_indexes]

        ts = np.array(final_tubes['ts'])[action_indexes]
        starts = np.array(final_tubes['starts'])[action_indexes]
        te = np.array(final_tubes['te'])[action_indexes]

        act_nr = 0
        scores = []
        nr = []
        class_list = []
        frame_nr = {'fnr': []}
        boxes = {'bxs': []}

        for a in range(len(ts)):
            act_ts = ts[a]
            act_te = te[a]
            act_path_scores = np.array(path_scores[a])
            act_scores = np.sort(act_path_scores[act_ts:act_te])[::-1]
            topk_mean = np.mean(act_scores[0:min(topk,len(act_scores))])
            bxs = np.array(final_tubes['path_boxes'])[action_indexes[0][a]][act_ts:act_te]
            bxs = np.array(bxs)
            bxs = np.hstack((bxs[:,:2], bxs[:,2:4]-bxs[:,:2]))
            label = final_tubes['label'][action_indexes[0][a]]
            if (topk_mean > kthresh[int(label)]) and ( (act_te-act_ts) > min_num_frames):
                scores.append(topk_mean)
                nr.append(act_nr)
                class_list.append(label)
                start_end_array = np.array([i for i in range(act_ts, act_te)]) + starts[a]
                frame_nr['fnr'].append(start_end_array)
                boxes['bxs'].append(bxs)
                act_nr += 1
        struct['score'] = scores
        struct['nr'] = nr
        struct['class'] = class_list
        struct['framenr'] = frame_nr
        struct['boxes'] = boxes
        xmld[vids[v]] = struct
    print("Returning xmld file")
    return xmld

def parActionPathSmoother(actionpaths,alpha,num_action):
    alltubes = []

    final_tubes = {'starts':[],'ts':[],'te':[],'label':[],'path_total_score':[],
    'dpActionScore':[],'dpPathScore':[],
    'path_boxes':[],'path_scores':[],'video_id':[]}
    # TODO parallize across actionpaths
    for t in range(len(actionpaths)):
        video_id = actionpaths[t]['video_id']
        alltubes.append(actionPathSmoother4oneVideo(actionpaths[t]['paths'],alpha,num_action,video_id))
        vid_tubes = alltubes[t]
        for k in range(len(vid_tubes['ts'])):
            final_tubes['starts'].append(vid_tubes['starts'][k])
            final_tubes['ts'].append(vid_tubes['ts'][k])
            final_tubes['video_id'].append(vid_tubes['video_id'][k])
            final_tubes['te'].append(vid_tubes['te'][k])
            final_tubes['dpActionScore'].append(vid_tubes['dpActionScore'][k])
            final_tubes['label'].append(vid_tubes['label'][k])
            final_tubes['dpPathScore'].append(vid_tubes['dpPathScore'][k])
            final_tubes['path_total_score'].append(vid_tubes['path_total_score'][k])
            final_tubes['path_boxes'].append(vid_tubes['path_boxes'][k])
            final_tubes['path_scores'].append(vid_tubes['path_scores'][k])
    return final_tubes


def actionPathSmoother4oneVideo(video_paths,alpha,num_action,video_id):
    action_count = 0
    final_tubes = {'starts':[],'ts':[],'te':[],'label':[],'path_total_score':[],
                       'dpActionScore':[],'dpPathScore':[],'vid':[],
                       'path_boxes':[],'path_scores':[],'video_id':[]}
    if len(video_paths)>0:
        for a in range(num_action):
            action_paths = video_paths[a]
            num_act_paths = getPathCount(action_paths)
            for p in range(num_act_paths):
                M = action_paths[p]['allScores'][:,:num_action]
                M = M.T
                M+=20.0
                pred_path,time,D = dpEM_max(M,alpha[a])
                Ts, Te, Scores, Label, DpPathScore = extract_action(pred_path,time,D,a)
                for k in range(len(Ts)):
                    final_tubes['starts'].append(action_paths[p]['start'])
                    final_tubes['ts'].append(Ts[k])
                    final_tubes['video_id'].append(video_id)
                    final_tubes['te'].append(Te[k])
                    final_tubes['dpActionScore'].append(Scores[k])
                    final_tubes['label'].append(Label[k])
                    final_tubes['dpPathScore'].append(DpPathScore[k])
                    final_tubes['path_total_score'].append(np.mean(action_paths[p]['scores']))
                    final_tubes['path_boxes'].append(action_paths[p]['boxes'])
                    final_tubes['path_scores'].append(action_paths[p]['scores'])
                    action_count+=1
    return final_tubes

def extract_action(p,q,D,action):
    '''
    :param p:
    :param q:
    :param D:
    :param action:
    :return:
    '''
    indexes = np.where(p==action)[0]
    if not (len(indexes) > 0):
        ts = np.array([])
        te = np.array([])
        scores = np.array([])
        label = np.array([])
        total_score = np.array([])
    else:
        A = np.append(indexes, indexes[-1]+1)
        B = np.append(indexes[0]-2, indexes)
        indexes_diff = A-B
        ts = np.where(indexes_diff>1)[0]
        if len(ts)>1:
            te = np.append(ts[1:]-1, len(indexes)-1)
        else:
            te = len(indexes)-1
        ts = indexes[ts]
        te = np.array([indexes[te]])
        scores = (D[action,q[te]] - D[action,q[ts]])/(te-ts)
        label = np.ones((len(ts), 1))*action
        total_score = np.ones((len(ts),1))*D[p[-1],q[-1]]/len(p)
    return ts.flatten(),te.flatten(),scores.flatten(),label.flatten(),total_score.flatten()

def dpEM_max(M, alpha):
    r,c = M.shape
    # costs
    D = np.zeros((r, c+1)) # add an extra column
    D[:,0] = 0 # put the maximum cost
    D[:, 1:(c+1)] = M
    phi = np.zeros((r,c))
    # Note:
    # the outer loop (j) is to visit one by one each frames
    # the inner loop (i) is to get the max score for each action label (e.g. 24)
    # the v1 term is to add a penalty by subtracting alpha from the
    # data term for all other class labels other than i; for ith class label
    # it adds zero penalty;
    #  The mask (v[i]=0) will return a logical array consists of 24 elements, in the ith
    # location (or ith action) it is 0 and 1 for all other locations/actions.
    # For the ith action in the action loop,  we get a max value which we add to the
    # data term D(i,j). This way, the 24 max values for 24 different action
    # labels are stored to the jth column (or for the jth frame): D(0,j), D(1,j),...,D(23,j),
    for j in range(1, c+1):#frames
        for i in range(r): #action labels
            v1 = np.ones(r)*alpha
            v1[i]=0
            values = D[:, j-1]-v1
            tb = np.argmax(values)
            dmax = np.max(values)
            D[i,j] = D[i,j]+dmax
            phi[i,j-1] = tb
    D = D[:,1:]
    # best of the last column
    q = c-1 # last frame index
    values = D[:, c-1]
    p = np.argmax(values) # action label with max score in last frame

    i = int(p) # index of max element in last column of D,
    j = int(q) # frame indices
    ps = np.zeros(c)
    qs = np.zeros(c)
    ps[q]=p
    qs[q]=j
    while j>0: # loop over frames in a video
        tb = phi[i,j] # i -> index of max element in frame above, j-> last frame index or last column of D
        j -= 1
        ps[j] = tb
        qs[j] = j
        i=int(tb) # index of max element in frame j

    return ps.astype(np.int),qs.astype(np.int),D

def readALLactionPaths(videolist,actionPathDir,step):
    '''
    :param videolist:
    :param actionPathDir:
    :param step:
    :return:
    '''
    videos = getVideoNames(videolist)
    NumVideos = len(videos)
    count=0
    actionpath = []
    for vid in range(0, NumVideos, step):
        videoID = videos[vid]
        pathsSaveName = os.path.join(actionPathDir,'{}-actionPaths.pkl'.format(videoID))
        try:
            with open(pathsSaveName, "rb") as file:
                allpaths = pickle.load(file)
        except IOError as e:
            print('Action path does not exist please generate action path {}'.format(pathsSaveName))
        #TODO Path counts per video_id currently don't match matlab version
        action_element = {'video_id': videoID, 'paths': allpaths}
        actionpath.append(action_element)
        count+=1
    return actionpath

def actionPaths(dopts):
    detresultpath = dopts['detDir']
    costtype = dopts['costtype']
    gap = dopts['gap']
    videolist = dopts['vidList']
    actions = dopts['actions']
    saveName = dopts['actPathDir']
    iouth = dopts['iouThresh']
    numActions = len(actions)
    nms_thresh = 0.45
    videos = getVideoNames(videolist)
    NumVideos = len(videos)
    for i_vid, videoID in enumerate(videos):
        pathsSaveName = os.path.join(saveName, "{}-actionPaths.pkl".format(videoID))
        videoDetDir = os.path.join(detresultpath, videoID)
        if not os.path.exists(pathsSaveName):

            print('Computing tubes for video [{:d} out of {:d}] video ID = {}\n'.format(i_vid + 1, NumVideos, videoID))

            ## loop over all the frames of the video
            print('Reading detections ')
            frames = readDetections(videoDetDir)

            print('Done reading detections')
            print('Generating action paths ...........')
            # TODO parallelize loop over classes
            # loop over all action class and generate paths for each class
            allpaths = []
            for i_act in range(0, numActions):
                allpaths.append(genActionPaths(frames, i_act, nms_thresh, iouth, costtype, gap))
            print("Results saved in ::: {} for {:d} classes".format(pathsSaveName, len(allpaths)))
            with open(pathsSaveName, 'wb') as f:
                pickle.dump(allpaths, f)
    print("Done computing action paths.")
    return None

def genActionPaths(frames, action_index, nms_thresh, iouth, costtype, gap):
    '''
    :param frames:
    :param action_index:
    :param nms_thresh:
    :param iouth:
    :param costtype:
    :param gap:
    :return:
    '''
    action_frames = []
    for frame_index in range(len(frames)):
        boxes, scores, allscores = dofilter(frames, action_index, frame_index, nms_thresh)
        action_frames.append({'boxes': None, 'scores': None, 'allScores': None})
        action_frames[frame_index]['boxes'] = boxes
        action_frames[frame_index]['scores'] = scores
        action_frames[frame_index]['allScores'] = allscores
    paths = incremental_linking(action_frames, iouth, costtype, gap, gap)
    return paths


def incremental_linking(frames, iouth, costtype, jumpgap, threshgap):
    num_frames = len(frames);
    ## online path building

    live_paths = []  # Stores live paths
    dead_paths = []  # Store the paths that have been terminated
    dp_count = 0
    for t in range(num_frames):
        num_box = frames[t]['boxes'].shape[0]
        # if first frame, start paths
        if t == 0:
            # Start a path for each box in first frame
            for b in range(num_box):
                live_paths.append({'boxes': [], 'scores': [], 'allScores': None,
                                   'pathScore': None, 'foundAt': [], 'count': 1, 'lastfound': 0})
                live_paths[b]['boxes'].append(frames[t]['boxes'][b, :])  # bth box x0,y0,x1,y1 at frame t
                live_paths[b]['scores'].append(frames[t]['scores'][b])  # action score of bth box at frame t
                live_paths[b]['allScores'] = frames[t]['allScores'][b, :].reshape(1,-1)  # scores for all action for bth box at frame t
                live_paths[b]['pathScore'] = frames[t]['scores'][b]  # current path score at frame t
                live_paths[b]['foundAt'].append(0)  # frame box was found in
                live_paths[b]['count'] = 1  # current box count for bth box tube
                live_paths[b]['lastfound'] = 0  # diff between current frame and last frame in bth path
        else:
            # Link each path to detections at frame t
            lp_count = getPathCount(live_paths)  # total paths at time t
            edge_scores = np.zeros((lp_count, num_box))  # (path count) x (number of boxes in frame t)
            for lp in range(lp_count):  # for each path, get linking (IoU) score with detections at frame t
                edge_scores[lp, :] = score_of_edge(live_paths[lp], frames[t], iouth, costtype)

            dead_count = 0
            covered_boxes = np.zeros(num_box)
            path_order_score = np.zeros((1, lp_count))
            for lp in range(lp_count):
                # Check whether path has gone stale
                if live_paths[lp]['lastfound'] < jumpgap:
                    # IoU scores for path lp
                    box_to_lp_score = edge_scores[lp, :]
                    if np.sum(box_to_lp_score) > 0.0:  # check if there's at least one match to detection in this frame
                        maxInd = np.argmax(box_to_lp_score)
                        m_score = np.max(box_to_lp_score)
                        live_paths[lp]['count'] = live_paths[lp]['count'] + 1
                        #lpc = live_paths[lp]['count']
                        # Add detection to live path lp
                        live_paths[lp]['boxes'].append(frames[t]['boxes'][maxInd, :])
                        live_paths[lp]['scores'].append(frames[t]['scores'][maxInd])
                        live_paths[lp]['allScores'] = \
                            np.vstack((live_paths[lp]['allScores'], frames[t]['allScores'][maxInd, :].reshape(1, -1)))
                        # Keep running sum of the path lp
                        live_paths[lp]['pathScore'] += m_score
                        # Record the frame at which the detections were added to path lp
                        live_paths[lp]['foundAt'].append(t)
                        # Reset when we last added to path lp
                        live_paths[lp]['lastfound'] = 0
                        # Squash detection since it's been assigned
                        edge_scores[:, maxInd] = 0
                        covered_boxes[maxInd] = 1
                    else:
                        # if we have no match of this path with a detection at frame t, record the miss
                        live_paths[lp]['lastfound'] += 1
                    scores = sorted(live_paths[lp]['scores'])
                    num_sc = len(scores)
                    path_order_score[:,lp] = np.mean(np.asarray(scores[int(max(0, num_sc - jumpgap-1)):num_sc]))
                else:
                    # If the path is stale, increment the dead_count
                    dead_count += 1

            # Sort the path based on score of the boxes and terminate dead path
            live_paths, dead_paths, dp_count = sort_live_paths(live_paths, path_order_score, dead_paths, dp_count,
                                                               jumpgap)
            lp_count = getPathCount(live_paths)

            # start new paths using boxes that are not assigned
            if np.sum(covered_boxes) < num_box:
                for b in range(num_box):
                    if not covered_boxes.flatten()[b]:
                        live_paths.append({'boxes': [], 'scores': [], 'allScores': None,
                                           'pathScore': None, 'foundAt': [], 'count': 1, 'lastfound': 0})
                        live_paths[lp_count]['boxes'].append(frames[t]['boxes'][b, :])  # bth box x0,y0,x1,y1 at frame t
                        live_paths[lp_count]['scores'].append(
                            frames[t]['scores'][b])  # action score of bth box at frame t
                        live_paths[lp_count]['allScores'] = frames[t]['allScores'][b, :].reshape(1,
                                                                                                 -1)  # scores for all action for bth box at frame t
                        live_paths[lp_count]['pathScore'] = frames[t]['scores'][b]  # current path score at frame t
                        live_paths[lp_count]['foundAt'].append(t)  # frame box was found in
                        live_paths[lp_count]['count'] = 1  # current box count for bth box tube
                        live_paths[lp_count]['lastfound'] = 0  # last frame box was found
                        lp_count += 1
        #print(t)
        #for i in range(len(live_paths)): print(live_paths[i]['pathScore'])
        #for i in range(len(live_paths)): print(live_paths[i]['scores'])

    live_paths = fill_gaps(live_paths, threshgap)
    dead_paths = fill_gaps(dead_paths, threshgap)
    lp_count = getPathCount(live_paths)
    lp = lp_count
    if len(dead_paths) > 0 and 'boxes' in dead_paths[0]:
        for dp in range(len(dead_paths)):
            live_paths.append({'boxes': None, 'scores': None, 'allScores': None,
                               'pathScore': None, 'foundAt': None, 'count': None, 'lastfound': None,
                               'start':None, 'end':None})
            live_paths[lp]['start'] = dead_paths[dp]['start']
            live_paths[lp]['end'] = dead_paths[dp]['end']
            live_paths[lp]['boxes'] = dead_paths[dp]['boxes']
            live_paths[lp]['scores'] = dead_paths[dp]['scores']
            live_paths[lp]['allScores'] = dead_paths[dp]['allScores']
            live_paths[lp]['pathScore'] = dead_paths[dp]['pathScore']
            live_paths[lp]['foundAt'] = dead_paths[dp]['foundAt']
            live_paths[lp]['count'] = dead_paths[dp]['count']
            live_paths[lp]['lastfound'] = dead_paths[dp]['lastfound']
            lp += 1
    return live_paths


def fill_gaps(paths, gap):
    gap_filled_paths = []
    if len(paths)>0 and 'boxes' in paths[0]:
        g_count = 0
        for lp in range(getPathCount(paths)):
            if len(paths[lp]['foundAt']) > gap:
                gap_filled_path_dict = {'start': None, 'end': None,
                                        'pathScore': None, 'foundAt': None,
                                        'count': None, 'lastfound': None,
                                        'boxes': [], 'scores': [], 'allScores': None}
                gap_filled_paths.append(gap_filled_path_dict)
                gap_filled_paths[g_count]['start'] = paths[lp]['foundAt'][0]
                gap_filled_paths[g_count]['end'] = paths[lp]['foundAt'][-1]
                gap_filled_paths[g_count]['pathScore'] = paths[lp]['pathScore']
                gap_filled_paths[g_count]['foundAt'] = paths[lp]['foundAt']
                gap_filled_paths[g_count]['count'] = paths[lp]['count']
                gap_filled_paths[g_count]['lastfound'] = paths[lp]['lastfound']
                count = 0
                i = 0
                while i < len(paths[lp]['scores']):
                    diff_found = paths[lp]['foundAt'][i] - paths[lp]['foundAt'][max(i, 0)]
                    if count == 0 or diff_found == 0:
                        gap_filled_paths[g_count]['boxes'].append(paths[lp]['boxes'][i])
                        gap_filled_paths[g_count]['scores'].append(paths[lp]['scores'][i])
                        if count == 0:
                            gap_filled_paths[g_count]['allScores'] = paths[lp]['allScores'][i, :].reshape(1, -1)
                        else:
                            gap_filled_paths[g_count]['allScores'] = \
                                np.concatenate((gap_filled_paths[g_count]['allScores'],
                                                paths[lp]['allScores'][i, :].reshape(1, -1)), axis=0)
                        i += 1
                        count += 1
                    else:
                        for d in range(diff_found):
                            gap_filled_paths[g_count]['boxes'].append(paths[lp]['boxes'][i])
                            gap_filled_paths[g_count]['scores'].append(paths[lp]['scores'][i])
                            assert gap_filled_paths[g_count]['allScores'].shape[
                                       0] > 1, 'allScores shape dim==0 must be >1'
                            gap_filled_paths[g_count]['allScores'] = \
                                np.concatenate((gap_filled_paths[g_count]['allScores'],
                                                paths[lp]['allScores'][i, :].reshape(1, -1)), axis=0)
                            count += 1
                        i += 1
                g_count += 1
    return gap_filled_paths


def sort_live_paths(live_paths, path_order_score, dead_paths, dp_count, gap):
    inds = path_order_score.flatten().argsort()[::-1]
    sorted_live_paths = []
    lpc = 0
    for lp in range(getPathCount(live_paths)):
        olp = inds[lp]
        if live_paths[olp]['lastfound'] < gap:
            sorted_live_paths.append({'boxes': None, 'scores': None, 'allScores': None,
                                      'pathScore': None, 'foundAt': None, 'count': None, 'lastfound': None})
            sorted_live_paths[lpc]['boxes'] = live_paths[olp]['boxes']
            sorted_live_paths[lpc]['scores'] = live_paths[olp]['scores']
            sorted_live_paths[lpc]['allScores'] = live_paths[olp]['allScores']
            sorted_live_paths[lpc]['pathScore'] = live_paths[olp]['pathScore']
            sorted_live_paths[lpc]['foundAt'] = live_paths[olp]['foundAt']
            sorted_live_paths[lpc]['count'] = live_paths[olp]['count']
            sorted_live_paths[lpc]['lastfound'] = live_paths[olp]['lastfound']
            lpc += 1
        else:
            dead_paths.append({'boxes': None, 'scores': None, 'allScores': None,
                               'pathScore': None, 'foundAt': None, 'count': None, 'lastfound': None})
            dead_paths[dp_count]['boxes'] = live_paths[olp]['boxes']
            dead_paths[dp_count]['scores'] = live_paths[olp]['scores']
            dead_paths[dp_count]['allScores'] = live_paths[olp]['allScores']
            dead_paths[dp_count]['pathScore'] = live_paths[olp]['pathScore']
            dead_paths[dp_count]['foundAt'] = live_paths[olp]['foundAt']
            dead_paths[dp_count]['count'] = live_paths[olp]['count']
            dead_paths[dp_count]['lastfound'] = live_paths[olp]['lastfound']
            dp_count = dp_count + 1
    return sorted_live_paths, dead_paths, dp_count


from utils.cython_bbox import bbox_overlaps


def score_of_edge(v1, v2, iouth, costtype):
    # Number of detections at frame t
    N2 = v2['boxes'].shape[0]
    score = np.zeros((1, N2))
    iou = bbox_overlaps(np.ascontiguousarray(v2['boxes'], dtype=np.float),
                        np.ascontiguousarray(v1['boxes'][-1].reshape(1, -1), dtype=np.float))
    for i in range(0, N2):
        if iou.item(i) >= iouth:
            scores2 = v2['scores'][i]
            scores1 = v1['scores'][-1]
            # if len(v1['allScores'].shape)<2:
            #    v1['allScores'] = v1['allScores'].reshape(1,-1)
            score_similarity = np.sqrt(
                np.sum(((v1['allScores'][-1, :].reshape(1, -1) - v2['allScores'][i, :].reshape(1, -1)) ** 2)))
            if costtype == 'score':
                score[:, i] = scores2
            elif costtype == 'scrSim':
                score[:, i] = 1.0 - score_similarity
            elif costtype == 'scrMinusSim':
                score[:, i] = scores2 + (1. - score_similarity)
    return score


def getPathCount(live_paths):
    if len(live_paths)>0 and 'boxes' in live_paths[0]:
        lp_count = len(live_paths)
    else:
        lp_count = 0
    return lp_count


def dofilter(frames, action_index, frame_index, nms_thresh):
    # filter out least likely detections for actions
    scores = frames[frame_index]['scores'][:, action_index]
    pick = np.where(scores > 0.001)
    scores = scores[pick]
    boxes = frames[frame_index]['boxes'][pick, :].squeeze(0)
    allscores = frames[frame_index]['scores'][pick, :].squeeze(0)
    # sort in descending order
    pick = np.argsort(scores)[::-1]
    # pick at most 50
    to_pick = min(50, len(pick))
    pick = pick[:to_pick]
    scores = scores[pick]
    boxes = boxes[pick, :]
    allscores = allscores[pick, :]
    # Perform nms on picked boxes
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
    if len(boxes)==0 or len(scores) == 0 or len(allscores)==0:
        return boxes, scores, allscores
    pick, counts = nms(torch.from_numpy(boxes), torch.from_numpy(scores), nms_thresh)  # idsn - ids after nms
    pick = pick[:counts]
    #pick = nms(dets, nms_thresh)
    pick = pick[:counts].cpu().numpy()
    boxes = boxes[pick, :]
    scores = scores[pick]
    allscores = allscores[pick, :]
    return boxes, scores, allscores


from scipy.io import loadmat
import numpy as np


def readDetections(detectionDir):
    detectionList = sorted(os.listdir(detectionDir), reverse=False)
    numframes = len(detectionList)
    scores = 0
    loc = 0
    frames = []
    for f in range(0, numframes):
        filename = os.path.join(detectionDir, detectionList[f])
        # Load loc and scores
        # TODO: remove mat format
        mat_file = loadmat(filename)
        loc = mat_file['loc']
        loc[:, 0] = loc[:, 0] * 320
        loc[:, 1] = loc[:, 1] * 240
        loc[:, 2] = loc[:, 2] * 320
        loc[:, 3] = loc[:, 3] * 240
        loc[np.where(loc[:, 0] < 0.0), 0] = 0.0
        loc[np.where(loc[:, 1] < 0.0), 1] = 0.0
        loc[np.where(loc[:, 2] > 319.0), 2] = 319.0
        loc[np.where(loc[:, 3] > 239.0), 3] = 239.0
        # loc+=1.0
        frames.append({'boxes': None, 'scores': None})
        frames[f]['boxes'] = loc
        scores = mat_file['scores']
        frames[f]['scores'] = np.concatenate((scores[:, 1:], scores[:, 0].reshape(-1, 1)), axis=1)
    return frames


def getVideoNames(split_file):
    '''
    Read videos from txt
    :param split_file: txt file with video name per line
    :return: list of video names (str)
    '''
    print('Get both list {}'.format(split_file))
    # Read each line
    videos = []
    with open(split_file, 'r') as f:
        for file_name in f.readlines():
            videos.append(file_name.rstrip('\n'))
    return videos


'''        
fid = fopen(split_file,'r');
data = textscan(fid, '%s');
videos  = cell(1);
count = 0;
for i=1:length(data{1})
filename = cell2mat(data{1}(i,1));
count = count +1;
videos{count} = filename;
%     videos(i).vid = str2num(cell2mat(data{1}(i,1)));
end
end
'''

if __name__ == '__main__':
    I01onlineTubes()
