# MỘT VÀI THỨ CẦN BIẾT TRONG QUÁ TRÌNH KHẢO SÁT DỮ LIỆU VÀ TẠO MÔ HÌNH ML/DL

- Thường thì một mô hình sẽ gồm 5 bước:
    - Thu thập data.
    - Phân tích data.
    - Lọc data.
    - Train data.
    - Kiểm tra độ chính xác.

## 1. Thu thập data
- Bước này thì chúng ta sẽ lấy data trên kaggle nhé.
- Đọc file: Tương tự như R hồi học thực hành xác suất thống kê.
- Ví dụ:

```python
import pandas as pd
titanic_data = pd.read_csv("Titanic_data.csv")
```

## 2. Phân tích data
- Biểu diễn biểu đồ theo thuộc tính. 
- Xuất thông tin của data.
- Ví dụ:

```python
titanic_data.info()
```
- output:
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
```

## 3. Lọc data
- Kiểm tra xem bộ nào có thuộc tính là null.
- Nếu tồn tại thuộc tính null nhiều thì ta sẽ loại bỏ thuộc tính.
- Nếu bộ nào không có số liệu rõ ràng thì cũng bỏ luôn.

## 4. Train data
- Tách ra cái nào là X_train, X_test, y_train, y_test.
- X_train, y_train: là các bộ được dùng để train model
- X_test, y_test: là các bộ được dùng để test model
- Cách phân ra X_train, X_test, y_train, y_test:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
```

- Sau khi tách được X_train, y_train, X_test, y_test thì ta train model (tùy vào trường hợp mà model khác nhau). 
- Một ví dụ cho LogisticRegression (gửi file sau):

```python
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

```

## 5. Test data
- Dựa vào xác suất để tính xem ta dự đoán đúng bao nhiêu hoặc hàm loss như thế nào (Tùy vào bài toán). Cái này giống cách mình chấm điểm cho mô hình của mình á.

- Ví dụ cho Logistic Regression:

```python
from sklearn.metrics import classification_report
classification_report(y_test, predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

```