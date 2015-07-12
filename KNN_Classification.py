class KNN_Classification(object):
    def __init__(self, knownFaces, newFaces, knownIdentity):
        self.knownFaces = knownFaces
        self.knownIdentity = knownIdentity.reshape(-1,1).flatten()
        self.newFaces = newFaces
        self.identityPrediction = []

    def euclid_distance(self, xi, face):
        return np.sqrt(np.sum(np.power((xi-face),2)))

    def classifier(self):
        for xi in self.newFaces:
            distances = []
            #xi = xi.reshape(-1,1)
            for face in self.knownFaces:
                this_distance = self.euclid_distance(xi, face)
                distances.append(this_distance)

            distances = np.asarray(distances)
            index_order = np.argsort(distances)
            sorted_labels = self.knownIdentity[index_order]
            sorted_labels = sorted_labels[:3][1]
            print "sorted_labels", sorted_labels
            topMatches = np.argwhere(np.bincount(sorted_labels) == np.amax(np.bincount(sorted_labels)))
            topMatches = random.choice(topMatches)
            self.identityPrediction.append(topMatches.flatten().tolist())

        return self.identityPrediction
