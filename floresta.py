from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar a base de dados
iris = load_iris()
breastcancer = load_breast_cancer()
digits = load_digits()
Xiris, yiris = iris.data, iris.target
Xbreastcancer, ybreastcancer = breastcancer.data, breastcancer.target
Xdigits, ydigits = digits.data, digits.target

# Pré-processamento: normalização dos dados
scaler = StandardScaler()
Xiris_normalized = scaler.fit_transform(Xiris)
Xbreastcancer_normalized = scaler.fit_transform(Xbreastcancer)
Xdigits_normalized = scaler.fit_transform(Xdigits)

# Divisão dos conjuntos de treinamento e teste (80% treinamento, 20% teste)
Xiris_train, Xiris_test, yiris_train, yiris_test = train_test_split(Xiris_normalized, yiris, test_size=0.2,
                                                                    random_state=42)
Xbreastcancer_train, Xbreastcancer_test, ybreastcancer_train, ybreastcancer_test = train_test_split(
    Xbreastcancer_normalized, ybreastcancer, test_size=0.2, random_state=42)
Xdigits_train, Xdigits_test, ydigits_train, ydigits_test = train_test_split(Xdigits_normalized, ydigits, test_size=0.2,
                                                                            random_state=42)

# Verificar o tamanho dos conjuntos de treinamento e teste
print("Tamanho dos conjuntos de treinamento Iris:", Xiris_train.shape[0])
print("Tamanho dos conjuntos de teste Iris:", Xiris_test.shape[0])
print("Tamanho dos conjuntos de treinamento BreastCancer:", Xbreastcancer_train.shape[0])
print("Tamanho dos conjuntos de teste BreastCancer:", Xbreastcancer_test.shape[0])
print("Tamanho dos conjuntos de treinamento Digits:", Xdigits_train.shape[0])
print("Tamanho dos conjuntos de teste Digits:", Xdigits_test.shape[0])
