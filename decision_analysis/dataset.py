from matplotlib import pyplot
import hickle as hkl
import numpy as np
from sklearn.model_selection import train_test_split

def generate_train_test(num_classes, num_class_samples, test_size, mean, sigma):
    data = []
    for class_id in range(num_classes):
        class_data = np.random.normal(mean[class_id], sigma, num_class_samples)
        data.extend(class_data)
     
    y = [0 for i in range(num_class_samples)]
    for i in range(1,num_classes):
        y.extend([i for j in range(num_class_samples)])

    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=test_size)
    
    data = {'xtrain': x_train, 'xtest': x_test,'ytrain': y_train,'ytest':y_test}

    # save dataset
    hkl.dump(data,f'data/{num_classes}_{num_class_samples}_{test_size}_{str(mean[:num_classes])}_{sigma}.hkl')

