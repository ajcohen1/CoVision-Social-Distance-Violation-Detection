class movingAverage:
    def __init__(self, windowSize):
        self.windowSize = windowSize
        self.id_points_dic = {}
        self.id_avg_dic = {}
        self.frame = 0

    def updatePoints(self, setOfPoints, pointIds):
        frameDic = {idz : point.tolist() for point, idz in zip(setOfPoints, pointIds)}
        print("Points into update points: ", setOfPoints)
        print("Resulting Dictionary", frameDic)

        #remove all the current points in class dic that arent in frame
        print("Normal List", self.id_points_dic.items())

        if(len(frameDic) > 0):
            deletedIds = [id for id in self.id_points_dic if id not in frameDic]
            for id in deletedIds:
                del(self.id_points_dic[id])

        #add all new points to global dic and update current points
        for id, centerCoord in frameDic.items():
            if(id in self.id_points_dic):
                x_coords = self.id_points_dic[id][0]
                y_coords = self.id_points_dic[id][1]
                x_coords.append(centerCoord[0])
                y_coords.append(centerCoord[1])
                if(len(self.id_points_dic[id][0]) > self.windowSize):
                    x_coords.pop(0)
                    self.id_points_dic[id][0] = x_coords
                    y_coords.pop(0)
                    self.id_points_dic[id][1] = y_coords
            else:
                self.id_points_dic[id] = [[centerCoord[0]] , [centerCoord[1]]]

    def getCurrentAverage(self):
        return [(id, sum(coords[0])/self.windowSize, sum(coords[1])/self.windowSize) for id, coords in self.id_points_dic.items()
                 if len(coords[0]) == self.windowSize and len(coords[1]) == self.windowSize]

