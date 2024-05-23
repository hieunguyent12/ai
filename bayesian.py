class Node:
    def __init__(self, numOcc):
        self.numOcc = numOcc
        self.probability = None

class ProbDist:
    def __init__(self):
        self.elements = {}

    def add_item(self, name, value):
        self.elements[name] = Node(value)

    def normalize(self):
        sum = 0
        for node in self.elements.values():
            sum += node.numOcc

        for node in self.elements.values():
            node.probability = node.numOcc/sum

    def printProb(self):
        for key, node in self.elements.items():
            print(f"{key}: {node.probability}", end=", ")


    # wsdf
    def checkTotalProb(self):
        sum = 0
        for node in self.elements.values():
            sum += node.probability

        print(f"Sum: {sum}")

        error = 0.0001
        lowerBound = 1 - error
        upperBound = 1 + error
        # check if close to one
        return lowerBound < sum < upperBound

    def getProb(self):
        eleList = []
        for value in self.elements.values():
            eleList.append(value.probability)
        return tuple(eleList)

    def show_approx(self):
        for key, node in self.elements.items():
            print(f"{key}: {round(node.probability, 3)}", end=", ")

p = ProbDist()
p.add_item("Cat", 50)
p.add_item("Dog", 114)
p.add_item("Rabbit", 64)
print(p.checkTotalProb())

# New item after intital grouping
# p.add_item("Fish", 204)
# p.normalize()
# p.printProb()
# print(p.checkTotalProb())

print(p.getProb())
p.show_approx()
