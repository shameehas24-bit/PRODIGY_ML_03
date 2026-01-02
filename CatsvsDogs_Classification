from sklearn import svm
import numpy as np

# Mock flattened image pixels (10 samples, 100 features each)
X = np.random.rand(10, 100) 
y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # 0 = Cat, 1 = Dog

clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# Test prediction
test_data = np.random.rand(1, 100)
result = clf.predict(test_data)
print(f"Classification Result: {'Dog' if result[0] == 1 else 'Cat'}")
