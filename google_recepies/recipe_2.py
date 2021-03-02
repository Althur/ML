
# I couldn't fix issue with creating a visual graph in pdf file. 

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

feature_names = iris.feature_names
target_names = iris.target_names

data = iris.data 
#[x][0]-> sepal length [cm]
#[x][1]-> sepal width [cm]
#[x][2]-> petal length [cm]
#[x][3]-> petal width [cm]
target = iris.target # labels , 

#print(feature_names)
#print(target_names)
#print(data)


test_idx = [0, 50, 100]
# training data 
train_target = np.delete(target, test_idx)
train_data = np.delete(data, test_idx, axis = 0)

# testing data 
test_target = target[test_idx]
test_data = data[test_idx]

# classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

prediction = clf.predict(test_data)
print(prediction) 
print(test_target)


# Viz Code
#import StringIO
from io import StringIO
#import pydot
import graphviz
dot_data = StringIO()
tree.export_graphviz(clf,
                        out_file=dot_data,
                        feature_names=feature_names,
                        class_names=target_names,
                        filled=True, rounded=True, 
                        impurity=False)
graph = graphviz.Source(dot_data.getvalue())
graph.render("iris.pdf", view=True)
#graph.write_pdf("iris.pdf") 
#print(graph[0]) # for first versions tested
"""import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf,
                        out_file=dot_data,
                        feature_names=feature_names,
                        class_names=target_names,
                        filled=True, rounded=True, 
                        impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("iris.pdf")"""