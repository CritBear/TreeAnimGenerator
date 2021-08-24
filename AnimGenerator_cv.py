import numpy as np
import cv2
import time
import math
from matplotlib import pyplot as plt
from rdp import rdp

import PhysicsSimulator as ps




class Branch:
    NUM_OF_INSTANCE = 0
    
    def __init__(self, parent = None):
        self.id = Branch.NUM_OF_INSTANCE
        Branch.NUM_OF_INSTANCE += 1
        
        self.parent = parent
        self.nodes = []
        self.begin_node = None
        self.end_node = None

        self.isConnectedToRoot = False
        self.meshes = [] # list( list(x, y), w) : image position index, weight

    def push(self, node_ref):
        node = node_ref.copy()
        if self.begin_node is None:
            self.begin_node = node
            
        self.nodes.append(node)
        self.end_node = node

    def get_length(self):
        return math.sqrt(pow(self.begin_node[0] - self.end_node[0], 2) + \
                         pow(self.begin_node[1] - self.end_node[1], 2))

    def get_angle(self):
        x = self.end_node[0] - self.begin_node[0]
        y = self.end_node[1] - self.begin_node[1]
        return math.atan2(y, x)

    def get_center(self):
        return ((self.begin_node[0] + self.end_node[0]) / 2,
                (self.begin_node[1] + self.end_node[1]) / 2)

    def test(self):
        result = rdp(self.nodes, epsilon = 2.0)
        print(self.nodes)
        print(result)

def mouse_callback(event, x, y, flags, param):
    global train_data, img
    preview_img = np.full((256, 256, 3), (img[y][x][0], img[y][x][1], img[y][x][2]), dtype=np.uint8)
    cv2.imshow('preview', preview_img)
    if event == cv2.EVENT_LBUTTONDOWN:
        train_data += str(img[y][x][0] / 100) + ' '
        train_data += str(img[y][x][1] / 100) + ' '
        train_data += str(img[y][x][2] / 100) + ' 0\n'
        print(img[y][x][0], ' ', img[y][x][1], ' ', img[y][x][2], ' 0')
    elif event == cv2.EVENT_RBUTTONDOWN:
        train_data += str(img[y][x][0] / 100) + ' '
        train_data += str(img[y][x][1] / 100) + ' '
        train_data += str(img[y][x][2] / 100) + ' 1\n'
        print(img[y][x][0], ' ', img[y][x][1], ' ', img[y][x][2], ' 1')

def make_train_data():
    file = open('tree_classify_train_data3.txt', 'w')
    file.write(train_data)
    file.close()


def make_simulate_data(branches, img_w, img_h):
    
    # dictionary[ id : tuple( pos_x, pos_y ), tuple( size_x, size_y ) ]
    bones = {}

    # list[ tuple( boneA_id, boneB_id ), reference_angle ]
    junctions = [] 

    if len(branches) > 0:
        # -1 : static body
        bones[-1] = [(img_w / 2, 0), (100, 1), 0]
        search_stack = [branches[0]]
        branches[0].isConnectedToRoot = True
        bones[branches[0].id] = [branches[0].get_center(), (1, branches[0].get_length()), branches[0].get_angle()]
        junctions.append([(-1, branches[0].id), 0]) #branches[0].get_angle()
        
        while len(search_stack) > 0:
            cur_branch = search_stack.pop()
            for branch in branches:
                if branch.parent is None:
                    continue
                if branch.parent == cur_branch.id:
                    search_stack.append(branch)
                    branch.isConnectedToRoot = True
                    bones[branch.id] = [(branch.get_center()[0], branch.get_center()[1]) , (1, branch.get_length()), branch.get_angle()]
                    junctions.append([(cur_branch.id, branch.id), branch.get_angle() - cur_branch.get_angle()]) #0.5


    #for i in junctions:
    #    print(i[0], i[1] / math.pi * 180)
    return bones, junctions

def init():
    # Step 1 - Load image
    try:
        imgfile = 'TreeImage/tree01.png'
        global img
        img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
        #img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
        img_h , img_w, img_c = img.shape
        print(img_h, img_w, img_c)
    except:
        print('Fail to load image')
        return


    # Step 2 - Mask by alpha-channel
    ret, alpha_mask = cv2.threshold(img[:,:,3], 1, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_and(img, img, mask=alpha_mask)


    # Step 3 - Make data to train classfication model


    # Step 4 - Make whole data to classify
    chunk_size = 1

    '''
    file = open('tree_classify_data.txt', 'w')
    
    row_idx = int(chunk_size / 2)
    for row in img[row_idx : img_h-(img_h%chunk_size) : chunk_size]:
        
        col_idx = int(chunk_size / 2)
        for pixel in row[col_idx : img_w-(img_w%chunk_size) : chunk_size]:

            if np.any(pixel[:3] > 0):
                file.writelines([
                    str(pixel[0]/100), ' ',
                    str(pixel[1]/100), ' ',
                    str(pixel[2]/100), ' ',
                    str(row_idx), ' ',
                    str(col_idx), '\n'])
                
            col_idx += chunk_size

        row_idx += chunk_size

    file.close()
    '''
    
    # Step 5 - Mask by classified data
    classify_data = np.loadtxt('tree_classify_result.txt', unpack=True, dtype='uint32')
    classify_data = np.transpose(classify_data)

    branch_mask = np.zeros(shape=(img_h, img_w), dtype='uint8')

    half = int(chunk_size / 2)
    for data in classify_data:
        if data[0] == 1:
            branch_mask[(data[1]-half) : (data[1]+half+1), (data[2]-half) : (data[2]+half+1)] = 1

    #print(branch_mask[22])
    branch_img = cv2.bitwise_and(img, img, mask=branch_mask)
    
    
    # Step 6 - Pre-processing
    gray = cv2.cvtColor(branch_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 50, 7, 21)
    ret, binary = cv2.threshold(gray, 50, 255, 0)
    

    # Step 7 - Find center node
    result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    index_vertical = img_h - 1
    index_begin = 0
    index_center = 0

    nodes = []
    
    for row in binary[-1:0:-10]:
        index = 10
        while index < img_w:
            if row[index] != 0:
                #print('Case 1 - ', index_vertical, ' ', index)
                index -= 9
                while index < img_w:
                    if row[index] != 0:
                        index_begin = index
                        cv2.line(result, (index, index_vertical), (index, index_vertical), (0, 255, 0), 3)
                        while index < img_w - 1:
                            index += 1
                            if row[index] == 0:
                                cv2.line(result, (index, index_vertical), (index, index_vertical), (0, 255, 0), 3)
                                break
                        if index - index_begin >= 5:
                            index_center = int((index + index_begin) / 2)
                            nodes.append([index_center, index_vertical])
                            cv2.line(result, (index_center, index_vertical), (index_center, index_vertical), (0, 0, 255), 3)
                        break
                    else:
                        index += 1
            else:
                index += 10

        index_vertical -= 10

    branches = []
    
    for node in nodes:
        if len(branches) == 0:
            branches.append(Branch())
            branches[0].push(node)
        else:
            isNodeIncluded = False
            substitute = None
            for branch in branches:
                if branch.end_node[1] != node[1] + 10:
                    continue
                
                isConnected = True
                x_offset = (node[0] - branch.end_node[0]) / 10
                for y_offset in range(1, 10):
                    if binary[branch.end_node[1] - y_offset][round(branch.end_node[0] + (x_offset * y_offset))] == 0:
                        isConnected = False
                        break

                if isConnected:
                    rdp_result = rdp(branch.nodes + [node], epsilon = 5.0)
                    if len(rdp_result) == 2:
                        branch.push(node)
                        isNodeIncluded = True
                        break
                    else:
                        if substitute is None:
                            substitute = branch
                        else:
                            if abs(substitute.end_node[0] - node[0]) < abs(branch.end_node[0] - node[0]):
                                substitute = branch

            if isNodeIncluded:
                continue
            
            if substitute is not None:
                new_branch = Branch(substitute.id)
                new_branch.push(substitute.end_node)
                new_branch.push(node)
                branches.append(new_branch)
            else:
                new_branch = Branch()
                new_branch.push(node)
                branches.append(new_branch)
            
                    
                
    # show bone
    img_bone = np.zeros((img_h, img_w, 3))
    for branch in branches:
        #print(branch.nodes)
        cv2.line(img_bone, (branch.begin_node[0], branch.begin_node[1]), (branch.begin_node[0], branch.begin_node[1]), (0, 0, 255), 6)
        cv2.line(img_bone, (branch.end_node[0], branch.end_node[1]), (branch.end_node[0], branch.end_node[1]), (0, 0, 255), 6)
        cv2.line(img_bone, (branch.begin_node[0], branch.begin_node[1]), (branch.end_node[0], branch.end_node[1]), (0, 0, 255), 3)

        if branch.parent is not None:
            parent = branches[branch.parent]
            cv2.line(img_bone, (parent.end_node[0], parent.end_node[1]), (branch.begin_node[0], branch.begin_node[1]), (0, 255, 0), 6)
 

    '''
    
    row_idx = int(chunk_size / 2)
    for row in img[row_idx : img_h-(img_h%chunk_size) : chunk_size]:
        
        col_idx = int(chunk_size / 2)
        for pixel in row[col_idx : img_w-(img_w%chunk_size) : chunk_size]:

            if np.any([:3] > 0):
                file.writelines([
                    str(pixel[0]/100), ' ',
                    str(pixel[1]/100), ' ',
                    str(pixel[2]/100), ' ',
                    str(row_idx), ' ',
                    str(col_idx), '\n'])
                
            col_idx += chunk_size

        row_idx += chunk_size
    '''



    bones, junctions = make_simulate_data(branches, img_w, img_h)
    
    


    
    chunk_size = 5
    half = int(chunk_size / 2)
    # E : branch's end node, S : branch's begin node, P : image index
    vec_se = np.zeros(2)
    vec_sp = np.zeros(2)
    vec_ep = np.zeros(2)

    img_mesh = np.zeros((img_h, img_w, 3))
    
    for row in range(half, img_h-(img_h%chunk_size), chunk_size):
        for col in range(half, img_w-(img_w%chunk_size), chunk_size):
            if np.any(img[row-half : row+half+1, col-half : col+half+1, :3]) > 0:
                img_mesh[row-half : row+half+1, col-half : col+half+1, :] = 255
                
                nearest1_branch = None
                nearest1_distance = img_h + img_w
                nearest2_branch = None
                nearest2_distance = img_h + img_w
                nearest3_branch = None
                nearest3_distance = img_h + img_w
                for branch in branches:
                    if not branch.isConnectedToRoot:
                        continue
                    vec_se[0] = branch.end_node[0] - branch.begin_node[0]
                    vec_se[1] = branch.end_node[1] - branch.begin_node[1]
                    vec_sp[0] = col - branch.begin_node[0]
                    vec_sp[1] = row - branch.begin_node[1]
                    vec_ep[0] = col - branch.end_node[0]
                    vec_ep[1] = row - branch.end_node[1]

                    if np.dot(vec_sp, vec_se) * np.dot(vec_ep, vec_se) <= 0:
                        distance = abs(np.cross(vec_sp, vec_se)) / branch.get_length()
                    else:
                        distance_sp = math.sqrt(pow(branch.begin_node[0] - col, 2) + \
                                                pow(branch.begin_node[1] - row, 2))
                        distance_ep = math.sqrt(pow(branch.end_node[0] - col, 2) + \
                                                pow(branch.end_node[1] - row, 2))
                        distance = min(distance_sp, distance_ep)

                    if nearest1_distance > distance:
                        nearest1_distance = distance
                        nearest1_branch = branch
                    elif nearest2_distance > distance:
                        nearest2_distance = distance
                        nearest2_branch = branch
                    elif nearest3_distance > distance:
                        nearest3_distance = distance
                        nearest3_branch = branch

                distance_sum = nearest1_distance + nearest2_distance + nearest3_distance
                nearest1_branch.meshes.append([[col, row], 1 - (nearest1_distance / distance_sum)])
                #nearest2_branch.meshes.append([[col, row], 1 - (nearest2_distance / distance_sum)])
                #nearest3_branch.meshes.append([[col, row], 1 - (nearest3_distance / distance_sum)])
                        

    for branch in branches:
        if not branch.isConnectedToRoot:
            continue

        for mesh in branch.meshes:
            cv2.line(img_mesh,
                     (mesh[0][0], mesh[0][1]),
                     (int((branch.begin_node[0] + branch.end_node[0])/2), int((branch.begin_node[1] + branch.end_node[1])/2)),
                     (255, 0, 0),
                     1)
        
        cv2.line(img_mesh, (branch.begin_node[0], branch.begin_node[1]), (branch.begin_node[0], branch.begin_node[1]), (0, 0, 255), 6)
        cv2.line(img_mesh, (branch.end_node[0], branch.end_node[1]), (branch.end_node[0], branch.end_node[1]), (0, 0, 255), 6)
        cv2.line(img_mesh, (branch.begin_node[0], branch.begin_node[1]), (branch.end_node[0], branch.end_node[1]), (0, 0, 255), 3)

        if branch.parent is not None:
            parent = branches[branch.parent]
            cv2.line(img_mesh, (parent.end_node[0], parent.end_node[1]), (branch.begin_node[0], branch.begin_node[1]), (0, 255, 0), 6)

        
    
    
    # Show result
    #cv2.namedWindow('result')
    #result_img = np.concatenate((img, branch_img), axis = 1)
    #cv2.imshow('result', result_img)
    #cv2.imshow('gray', gray)
    #cv2.imshow('binary', binary)
    #cv2.imshow('inter-result', result)
    cv2.imshow('bone', img_bone)
    cv2.imshow('mesh', img_mesh)


    simulator = ps.Simulator(img_w, img_h)
    simulator.set_world_sequentially(bones, junctions)
    #simulator.init_world()
    simulator.run_sequentially()

    '''

    cv2.namedWindow('result')
    cv2.namedWindow('preview')
    cv2.imshow('result', img)
    global train_data
    train_data = ''
    cv2.setMouseCallback('result', mouse_callback)
    '''
    
    #cv2.waitKey()
    cv2.destroyAllWindows()

    #make_train_data()
    
    print('animation generator done')

    
init()
