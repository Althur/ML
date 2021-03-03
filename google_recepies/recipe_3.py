
import numpy as np
import matplotlib.pyplot as plt

# creating "random" population

greyhounds = 500
labs = 500

grey_height = 28+4+np.random.randn(greyhounds)
lab_height = 24+4+np.random.randn(labs)



plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()

"""
Avoid useless features: e.g. Eye-Color, assuming that the probability of any dog having any eye color doesnt dhave any relation with its breed.

Best to use features that are independent. 
    e.g. Of features that are not independente -> Height in inches and Height in centimeters

    Hide/Avoid Redundant features 

Easy to understand features are best. 

"""