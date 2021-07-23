import cv2
import numpy as np
import math as m
import splines



class contourFinder():
    def __init__(
            self,
            image = None):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(self.image, 127, 255, 0)
        self.P, h_1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.C, h_2 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        self.big_curve = 1.57 #Points over 120 degrees will be included nomatter what
        area = cv2.contourArea(self.P[0])

        self.initMask = self.mask(self.P)

        for i in range (1, len(self.P), +1):
            area = area-cv2.contourArea(self.P[i])
        if len(str(area))>4:
            self.longest_distance = 10*len(str(int(area)))-30
        else:
            self.longest_distance = 10
        self.shortest_distance=5

    def borderMask(self, contourList):
        li = []
        lis = []
        for i in contourList:
            if i[0]<=len(self.initMask)-1 and i[1]<=len(self.initMask[0])-1:
                li.append([int(i[0]),int(i[1])])

        for j in li:
            #color = [150, 10, 10]
            color = [50, 10, 10]
            c = np.array(color)
            if np.array_equal(self.initMask[int(j[0])][int(j[1])], c):
                lis.append([j[0],j[1]])

        for layer in self.P:
             for point in layer:
                 for cpoint in li:
                    if [point[0][0], point[0][1]]==cpoint and [point[0][0], point[0][1]] not in lis:
                        lis.append([point[0][1], point[0][0]])

        return lis

    def numPoints(self, arr):
        num=0
        for i in range(len(arr)):
            num+=len(arr[i])
        return num

    # def polygonise(self, pts: list) -> list:
    #     # axes gives us the varying columns (we can ignore z for this)
    #     axes = {i: set(c) for i, c in enumerate(zip(*pts)) if len(set(c)) != 1}
    #     a, b = axes.keys()
    #
    #     # construct dicts of axis-columns
    #     pts_by_axis = {axis: {col: [] for col in cols} for axis, cols in axes.items()}
    #     for pt in pts:
    #         for axis in axes:
    #             pts_by_axis[axis][pt[axis]].append(pt)
    #
    #     # each axis-column (sorted by the other dimension)
    #     # will now be paired off into edges, and constructed into axis-edge-dicts
    #     edges = {}
    #     for axis in axes:
    #         other = a if axis != a else b
    #         for col, pt_list in pts_by_axis[axis].items():
    #             pt_list.sort(key=lambda p: p[other])
    #             for i in range(0, len(pt_list), 2):
    #                 v1 = tuple(pt_list[i])
    #                 v2 = tuple(pt_list[i + 1])
    #                 edges[tuple((axis, v1))] = other, v2
    #                 edges[tuple((axis, v2))] = other, v1

    # def toPolygon(self, contourList):
    #     list = []
    #     for i in contourList:
    #         list.append([i[0], i[1], 0])
    #     result = self.polygonise(list)
    #
    #     #print(contourList)
    #     return result

    def visualizePoints(self, window, iou, contourList): # make sure to comment out the drawsContours method
        for layer in range(len(contourList)):
            for index in range (len(contourList[layer])):
                window[contourList[layer][index][0][1]][contourList[layer][index][0][0]] = (255,255,255)
        cv2.imshow("P:"+str(self.numPoints(contourList))+", IoU:"+str(iou), window)
        cv2.waitKey(0)

    def mask (self, contourList):
        window =np.zeros_like(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        cv2.drawContours(window, contourList, -1, (50,10,10), 2)
        cv2.fillPoly(window, contourList, (150,10,10))

        return window

    def composite(self, r1, t1, r2, t2):
        r=m.sqrt(r1**2+2*r1*r2*m.cos(t2-t1)+r2**2)
        t=t1+m.atan((r2*m.sin(t2-t1))/(r1+r2*m.cos(t2-t1)))
        return r,t

    def IoU(self, newMask):
        m = self.mask(self.P)
        intersection = np.logical_and(m, newMask)
        union = np.logical_or(m, newMask)
        iou_score = np.sum(intersection)/np.sum(union)
        return iou_score

    def optimize(self):
        curve=0
        score=0
        for curve in range (30, 80, 1):
            arr = self.filter(curve/100, self.image, n=5)
            arr_spline = self.splineArray(arr)
            arr_mask = self.mask(arr_spline)
            iou = self.IoU(arr_mask)
            point_score = (self.numPoints(self.C)-self.numPoints(arr))/self.numPoints(self.C)
            if iou+point_score > score:
                score = iou+point_score
                opt_arr=arr
                opt_spline = arr_spline
                opt_mask = arr_mask
        #opt_spline being the contour list that borders the splined mask
        return opt_mask, self.IoU(self.mask(opt_spline)), opt_arr

    def filter(self, min_curve, I, n=5):
        # Compute gradients
        GX = cv2.Scharr(I, cv2.CV_32F, 1, 0, scale=1)
        GY = cv2.Scharr(I, cv2.CV_32F, 0, 1, scale=1)
        GX = GX + 0.0001  # Avoid div by zero
        x0=-1
        y0=-1
        new_contour = []
        for contour in self.C:
            contour = contour.squeeze()
            measure = []
            N = len(contour)
            com_r=0
            com_t=0
            counter=0
            been=False
            if N<4:
                N=0
            for i in range(N):
                #print(N)
                x1, y1 = contour[i]
                x2, y2 = contour[(i + n) % N]

                # Angle between gradient vectors (gx1, gy1) and (gx2, gy2)
                gx1 = GX[y1, x1]
                gy1 = GY[y1, x1]
                gx2 = GX[y2, x2]
                gy2 = GY[y2, x2]
                cos_angle = gx1 * gx2 + gy1 * gy2
                cos_angle /= (np.linalg.norm((gx1, gy1)) * np.linalg.norm((gx2, gy2)))
                angle = np.arccos(cos_angle)
                if cos_angle < 0:
                    angle = np.pi - angle
                x1, y1 = contour[((2*i + n) // 2) % N]  # Get the middle point between i and (i + n)
                # Add point if curvature is greater than approx 30 degrees
                counter+=1
                if ((angle>min_curve or com_t>min_curve or counter>self.longest_distance) and counter>self.shortest_distance) or angle>self.big_curve:
                    measure.append([[x1, y1]])
                    com_r=0
                    com_t=0
                    counter=0
                elif been==False and m.isnan(angle)==False:
                    measure.append([[x1, y1]])
                    com_r=m.sqrt(m.pow(y1-y0,2)+m.pow(x1-x0,2))
                    com_t=angle
                    been=True
                    counter=0
                elif i!=0 and m.isnan(angle)==False:
                    new_r=m.sqrt(m.pow(y1-y0,2)+m.pow(x1-x0,2))
                    com_r, com_t = self.composite(com_r,com_t,new_r,angle)


                x0=x1
                y0=y1

            if len(measure)>0:
                new_contour.append(np.array(measure))
        return new_contour


    def splineArray(self, contourList, dots_per_second=15):
        #Convert array
        #final = []

        #for j in range(len(contourList)):
            #points = [0]*(len(contourList[j]))

            #for i in range(len(contourList[j])):
                #points[i] = (contourList[j][i][0][0],contourList[j][i][0][1])

        #Create Spline Object
        spline = splines.CatmullRom(contourList, endconditions='closed')

        #Create new contour
        total_duration = spline.grid[-1] - spline.grid[0]
        dots = int(total_duration * dots_per_second) + 1
        times = spline.grid[0] + np.arange(dots) / dots_per_second

        #Evaluate final contour
        final = spline.evaluate(times)

            #Revert format conversion
            #point = []
            #for i in range(len(points)):
                #point.append([[round(points[i][0]), round(points[i][1])]])
            #final.append(np.array(point))


        return final

#if __name__ == '__main__':
    #Window1
    #M = mask(P)
    #visualizePoints(M, IoU(M), P)
    #where p is the original set of contour points

    #New Window 2
    #m, iou, L = optimize(image)
    #visualizePoints(m, iou, L)
    #where L is the final set of contour points

    #cv2.destroyAllWindows()


# Optimizing through curvature
#https://stackoverflow.com/questions/22029548/is-it-possible-in-opencv-to-plot-local-curvature-as-a-heat-map-representing-an-o
