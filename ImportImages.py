class Import_Imgs(object):
    def __init__(self, path):
        self.path = path
        self.faceList = []
        self.identity = []

    def loadImage(self, filepath, indx, gender):
        # Read image from file and return as grayscale numpy array.
        img = ndimage.imread(filepath, flatten=True)

        # If grayscale image matrix shape is (n by m), when flattened the array length is (n*m)
        thisImg = img.flatten('C')

        # Append each flatten face and that face's identity & gender to the respective lists
        self.faceList.append(thisImg)
        self.identity.append((indx,gender))

    def createMatrix(self):
        # labels only required to iterate through AT&T face directory.  Remove for production
        label = [1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,39,4,40,5,6,7,8,9]
        gender = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0]
        i = 0

        for directname, directnames, filenames in os.walk(self.path):
            for subdirectname in directnames:
                subject_path = os.path.join(directname, subdirectname)
                for filename in os.listdir(subject_path):
                    # Pass each image's path and its numeric identity label
                    self.loadImage(os.path.join(subject_path, filename), label[i], gender[i])
                i +=1

        # After all face images are flattened and appended to face list as arrays,
        # reshape each face array in the list to size (1 by (n*m))
        for item in self.faceList:
            item = np.asarray(item).reshape(1,-1)

        # Stack the list of arrays vertically to make a single array
        origFaceMat = np.vstack(self.faceList)
        self.identity = np.asarray(self.identity)

        # Mark obsolete objects as none to free memory
        self.faceList = None

        return origFaceMat, self.identity
