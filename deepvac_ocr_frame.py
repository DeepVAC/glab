import numpy as np
import statistics
import math
import cv2

def takeLUPlacex(elem):
    elem = elem[0]
    return elem.most_right_xy[0]

def takeLUPlacey(elem):
    elem = elem[0]
    return elem.most_bottom_xy[1]

def takeRDPlacex(elem): 
    elem = elem[0]
    return elem.most_left_xy[0]

def takeRDPlacey(elem):
    elem = elem[0]
    return elem.most_top_xy[1]

def takex(elem):
    return elem[0]

def takey(elem):
    return elem[1]

class AggressiveBox:
    def __init__(self, rect, shape, real_angle, credit_by_score=True, credit_by_shape=True):
        self.min_h_ratio = 0.75
        self.max_h_ratio = 1.3
        self.box_min_wh_ratio = 2

        self.ori_rect = rect
        self.rect = rect
        self.init4Points()
        #水平化操作放到这里
        self.rect2reg = rect
        self.shape = shape
        self.credit = credit_by_shape
        self.candidate_box_list_left_up = []
        self.candidate_box_list_right_down = []
        self.real_angle = real_angle
        self.reviseBox()
        self.scaleBox()

    def init4Points(self):
        boxes = list(cv2.boxPoints(self.ori_rect))
        boxes.sort(key=takex)
        self.most_left_xy = boxes[0]
        self.most_right_xy = boxes[3]
        boxes.sort(key=takey)
        self.most_top_xy = boxes[0]
        self.most_bottom_xy = boxes[3]

    def reviseBox(self):
        pass

    def scaleBox(self):
        max_scale = max(self.shape[:2])
        if self.rect[1][1] >= self.rect[1][0]:
            self.rect = (self.rect[0],(self.rect[1][0],self.rect[1][1]+2*max_scale), self.rect[2])
            self.scaleAxis = 'h'
        else:
            self.rect = (self.rect[0],(self.rect[1][0]+2*max_scale,self.rect[1][1]), self.rect[2])
            self.scaleAxis = 'w'

    def getRect(self):
        return self.ori_rect
    
    def ratio(self, rect):
        # 求两个box重合度
        inter_pts = cv2.rotatedRectangleIntersection(self.rect,rect)[1]
        if inter_pts is None:
            return 0, None

        inter_area = cv2.contourArea(inter_pts)
        inter_ratio = inter_area/(rect[1][0]*rect[1][1])
        contex = cv2.convexHull(inter_pts)
        return (inter_ratio, contex)

    def addCandidateBox2Merge(self, rect, rect_id, merge_ratio, contex):
        if self.real_angle > 45 and self.real_angle < 135:
            if rect.getRect()[0][1] < self.ori_rect[0][1]:
                self.candidate_box_list_left_up.append((rect, rect_id, merge_ratio, contex))
            else:
                self.candidate_box_list_right_down.append((rect, rect_id, merge_ratio, contex))
        else:
            if rect.getRect()[0][0] < self.ori_rect[0][0]: 
                self.candidate_box_list_left_up.append((rect, rect_id, merge_ratio, contex))
            else:
                self.candidate_box_list_right_down.append((rect, rect_id, merge_ratio, contex))
    
    def sortCandidateBox(self):
        #按照r->l排序
        if self.real_angle > 45 and self.real_angle < 135:
            self.candidate_box_list_left_up.sort(key=takeLUPlacey, reverse=True)
            self.candidate_box_list_right_down.sort(key=takeRDPlacey)
        else:
            self.candidate_box_list_left_up.sort(key=takeLUPlacex, reverse=True)
            self.candidate_box_list_right_down.sort(key=takeRDPlacex)

        #按照l -> r 排序

    
    def isMerge(self, rect):
        similiar = isSimilarAngle(self.real_angle, rect[0].real_angle)
        #print(self.real_angle, rect[0].real_angle)
        #print(similiar)
        rect = rect[0].getRect()
        x = self.ori_rect[0][0] - rect[0][0]
        y = self.ori_rect[0][1] - rect[0][1]
        distance = math.sqrt((x**2)+(y**2))
        w = (max(self.ori_rect[1]) + max(rect[1]))/2
        h = min(self.ori_rect[1]) + min(rect[1])
        if distance - w > h:
            return False
        #print('#################',max(rect[1]) / min(rect[1]))
        if max(rect[1]) / min(rect[1]) <= self.box_min_wh_ratio:
            return True
        h_ratio = min(rect[1])/min(self.ori_rect[1])
        #print('###############################h_ratio:',h_ratio)
        return h_ratio>=self.min_h_ratio and h_ratio<=self.max_h_ratio and similiar

    def isTruncated(self, rect):
        pass

    def merge(self, rect):
        contex = rect[3]
        rect_box_1 = cv2.boxPoints(self.ori_rect).tolist()
        if contex is not None:
            flat_list = []
            for sublist in contex:
                for item in sublist:
                    flat_list.append(item)
            rect_box_1.extend(flat_list)
        self.ori_rect = cv2.minAreaRect(np.array(rect_box_1).astype(np.int32))

    def mergeLeftOrUp(self):
        result_ids = []
        while True:
            merge_count = 0
            for idx,rect in enumerate(self.candidate_box_list_left_up):
                if rect is None:
                    continue

                if self.isMerge(rect):
                    self.merge(rect)
                    self.candidate_box_list_left_up[idx] = None
                    result_ids.append(rect[1])
                    merge_count+=1
                else:
                    break
            if merge_count == 0:
                break
        return result_ids 

    def mergeRightOrBottom(self):
        result_ids = []
        while True:
            merge_count = 0
            for idx,rect in enumerate(self.candidate_box_list_right_down): 
                if rect is None:
                    continue

                if self.isMerge(rect):
                    self.merge(rect)
                    self.candidate_box_list_right_down[idx] = None
                    result_ids.append(rect[1])
                    merge_count+=1
                else:
                    break
            if merge_count == 0:
                break
        return result_ids

    def mergeRects(self):
        merge_ids = []
        merge_ids.extend(self.mergeLeftOrUp())
        merge_ids.extend(self.mergeRightOrBottom())
        return merge_ids

similar_min_angle = 15

def isSimilarAngle(angle1, angle2):
    return True if abs(angle1 - angle2) < similar_min_angle  or 180 - abs(angle1 - angle2)<similar_min_angle  else False

def orderedByRatio(rect):
    rect = rect
    return max(rect[1][0]/rect[1][1], rect[1][1]/rect[1][0])

class DeepvacOcrFrame:
    def __init__(self, img, rect_list, is_oneway=False):
        self.merge_ratio = 0.7
        self.similar_box_ratio = 0.95

        self.img  = img
        self.shape = self.img.shape
        self.median_angle = 0
        self.rect_list = rect_list
        self.is_oneway = is_oneway
        self.sortBoxByRatio()
        self.initDominantAngle()
        self.aggresive_box_list = [self.createAggressiveBox(rect) for rect in self.rect_list]

    def sortBoxByRatio(self):
        self.rect_list.sort(key=orderedByRatio, reverse=True)

    def initDominantAngle(self):
        self.real_angle_list = []
        for i in range(len(self.rect_list)):
            rect = self.rect_list[i]
            real_angle = abs(rect[2] - 90) if rect[1][0] < rect[1][1] else abs(rect[2])
            real_angle = 0 if real_angle==180 else real_angle
            self.real_angle_list.append(real_angle)
        self.total_box_num = len(self.real_angle_list)

        #角度的中位数
        median_angle = statistics.median(self.real_angle_list)
        self.median_angle = median_angle if abs(median_angle) > 2 else 0

        self.similar_box_num = 0
        for x in self.real_angle_list:
            if isSimilarAngle(x, median_angle):
                self.similar_box_num+=1
        

        if self.is_oneway:
            return 

        if self.similar_box_num == self.total_box_num:
            self.is_oneway = True
            return

        similar_box_ratio = self.similar_box_num * 1.0 / self.total_box_num
        if similar_box_ratio > self.similar_box_ratio:
            self.is_oneway = True
            return

    def createAggressiveBox(self, rect):
        credit_by_shape = False
        #
        real_angle = abs(rect[2] - 90) if rect[1][0] < rect[1][1] else abs(rect[2])
        real_angle = 0 if real_angle==180 else real_angle 
        if max(rect[1][0]/rect[1][1], rect[1][1]/rect[1][0]) >= 3:
            credit_by_shape = True
        elif isSimilarAngle(real_angle, self.median_angle):
            credit_by_shape = True
        else:
            credit_by_shape = False

        return AggressiveBox(rect, self.shape, real_angle, True, credit_by_shape)
    
    def aggressive4mergePeer(self, aggressive_rect, offset):
        merge_candidate_ids = []
        for i,rect in enumerate(self.aggresive_box_list[offset+1:]):
            if rect is None:
                continue

            merge_ratio, convex = aggressive_rect.ratio(rect.getRect())
            if merge_ratio >self.merge_ratio:
                aggressive_rect.addCandidateBox2Merge(rect, i+offset+1, merge_ratio, convex)
        
        aggressive_rect.sortCandidateBox()
        #print(aggressive_rect.candidate_box_list_left_up)
        merged_ids = aggressive_rect.mergeRects()
        #print('merged_id:',merged_ids)
        for merged_id in merged_ids:
            self.aggresive_box_list[merged_id] = None
        
        return aggressive_rect

    def __call__(self):
        '''
        for i,rect in enumerate(self.aggresive_box_list):
            boxes = cv2.boxPoints(rect.getRect())
            cv2.polylines(self.img,[np.array(boxes).astype(np.int32)],True,(255,0,0),1)
            x_min, y_min = np.min(boxes, axis=0)
            cv2.putText(self.img, str(i), (x_min, y_min), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
        save_name = 'vis/'+self.img_name.split('.')[0]+'_ori.jpg'
        cv2.imwrite(save_name, self.img)
        '''

        new_rect_list = []
        for i,rect in enumerate(self.aggresive_box_list):
            #print('rect:',i)
            if rect is None:
                #print('\n\n')
                continue
            new_rect_list.append(self.aggressive4mergePeer(rect, i))
            #print('\n\n')
        return new_rect_list
