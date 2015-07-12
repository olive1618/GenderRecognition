class Validation(object):
    def __init__(self, origFaceMat, identity):
        self.origFaceMat = origFaceMat
        self.identity = identity

    def split_data(self):
        # Randomly split data into training and test set.  40% of data held for test.
        faceTrain, faceTest, identityTrain, identityTest = cross_validation.train_test_split(\
            self.origFaceMat, self.identity, test_size=0.2, random_state=0)

        return faceTrain, faceTest, identityTrain, identityTest

    def measure_accuracy(self, identityPrediction, identityTest):
        correct = 0
        total = 0

        for test, predict in zip (identityTest, identityPrediction):
            if test in predict:
                correct += 1
                total += 1
            else:
                total += 1

        return correct, total

    def output_result(self, identityPrediction, identityTest):
        correct, total = self.measure_accuracy(identityPrediction, identityTest.flatten().tolist())
        print "correct", correct, "total", total
        return float(correct) / float(total)
