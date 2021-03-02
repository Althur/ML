from sklearn import tree

fruit_features = [[140, 1], [130, 1], [150, 0], [170, 0]] # [[wieght, texture]], e.g. [[150g, Bumpy]]
fruit_labels = [0, 0, 1, 1,]  # e.g. [apple, apple, orange, orange]

car_features = [[300, 2], [450, 2], [200, 8], [150, 9]] #[[Horsepower, Seats]]
car_labels = [0, 0, 1, 1] # e.g. [sports car, sports car, minivan, minivan]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(car_features, car_labels)  # input features and labels

unknown_fruit = [[150, 0]]
unknown_car = [[150, 6]]

print(clf.predict(unknown_car))