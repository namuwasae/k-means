from sklearn.datasets import load_iris
import pandas as pd


# target 변수를 제외한 데이터 프레임
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])

iris_df.head(3)

