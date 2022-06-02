# Здесь должен быть твой код
import pandas as pd
# Функция для разбиения исходного набора данных на выборки для обучения и тестирования моделей 
from sklearn.model_selection import train_test_split
# Класс для стандартизации показателей
from sklearn.preprocessing import StandardScaler
# Класс для создания и обучения модели
from sklearn.neighbors import KNeighborsClassifier
# Функции для оценки точности работы модели
from sklearn.metrics import confusion_matrix, accuracy_score
df = pd.read_csv("titanic.csv")
sc = StandardScaler()

# заменяем пустые значения на порт S
df['Embarked'].fillna('S', inplace = True)

# удаляем ненужные 
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],
axis = 1, inplace = True)

# делаем возроста для каждого класса каюты
age_1 = df[df['Pclass'] == 1]['Age'].median()
age_2 = df[df['Pclass'] == 2]['Age'].median()
age_3 = df[df['Pclass'] == 3]['Age'].median()

# делаем 3 фиктивных переменных
df[list(pd.get_dummies(df['Embarked']).columns)] = pd.get_dummies(df['Embarked'])

# удаляем ненужный 
df.drop('Embarked', axis = 1, inplace = True)

# метод для создания возраста для разных классов кают
def fill_age(row):
    if pd.isnull(row['Age']):
        if row['Pclass'] == 1:
            return age_1
        if row['Pclass'] == 2:
            return age_2
        return age_3
    return row['Age']
df['Age'] = df.apply(fill_age, axis = 1)

# делаем пол (мужской на 1, женский на 0)
def fill_sex(sex):
    if sex == 'male':
        return 1
    return 0
df['Sex'] = df['Sex'].apply(fill_sex)

X = df.drop('Survived', axis = 1)#данные о пассажирах
y = df['Survived']#целевая переменная

# разбиение данных на тестовые и обучающие
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# 
X_train = sc.fit_transform(X_train)
# 
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

n = accuracy_score(y_test, y_pred) * 100
print(n)

# print(df.info())

# print(df.pivot_table(index = 'Survived',
#                 columns = 'Pclass',
#             values = 'Age',
#         aggfunc = 'median'))

#print(df['Embarked'].value_counts())

#print(df.groupby('Pclass')['Age'].median())

