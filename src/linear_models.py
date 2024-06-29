import pandas as pd
import numpy as np


class LinearRegression():
    """
        ----------------------------------------------------------------------------------
            Класс реализует классический и стохастические градиентный спуск 
            для линейной регресии.
        ----------------------------------------------------------------------------------
    """
    
    def __repr__(self):
        return f"""LinearRegression class: 
        n_iter={self.n_iter}, 
        learning_rate={self.learning_rate},
        metric={"mse" if self.metric_name is None  else self.metric_name},
        reg={self.reg}, 
        l1_coef={self.l1_coef}, 
        l2_coef={self.l2_coef}, 
        sgd_sample={1 if self.sgd_sample is None else self.sgd_sample}, 
        random_state={self.random_state}"""
    
    
    def __init__(self,n_iter = 100, learning_rate: float = 0.1, metric: int = None, 
                 reg:str = None, l1_coef: float = 0, l2_coef: float = 0,
                 sgd_sample: float = None, random_state: int = 42):
        """
        ----------------------------------------------------------------------------------
            При инициализации принимает набор параметров:
            n_iter: default = 100
                 Кол-во итераций
                 
            learning_rate: default = 0.1
                Cкорость обучения. Можете принимать как число, так и функцию
                для расчета коэфициента обучения
                
            metric: {mae, rmse, mape, r2} , default = None
                Метрика качества
                
            reg: {l1, l2, elasticnet}, default = None
                Регуляризация алгоритма
                
            l1_coef: default = 0
                Коэффициент для Lasso
                
            l2_coef:d default = 0
                Коэффициент для Rigde
                
            sgd_sample: default = None
                Параметр определяет размер выборки для каждой итерации при стохастическом
                градиентном спуске. Если параметр от 0 до 1, то интерпретируется как доля 
                от всей выборки. Если параметр больше 1, то округляется до целого и 
                интерпретируется как кол-во строк в выборке.
                
            random_state: default = 42
                Параметр для случайности
        ------------------------------------------------------------------------------------   
            Содержит основные методы:
            fit:
                Обучение модели
                
            predict:
                Построение прогноза
                
            get_best_score:
                Получение финальной метрики качества
                на обучающих данных
                
            get_coef:
                Получить веса признаков
        -------------------------------------------------------------------------------------
                
        """
        self.n_iter = n_iter
        self.learning_rate = learning_rate 
        self.metric_name = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
    

        
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False, intercept: bool = True):
        
        """
        -------------------------------------------------------------------------------------
            Метод класса для нахождения весов в модели 
            X:
                Матрица признаков для обучения
                
            y:
                Вектор таргета
                
            verbose: default = False
                Параметр отвечает за вывод информации об обучении в output,
                принимает число int, интерпретируемое как часто (после скольки
                итераций) выводить информацию
                
            intercept: {True, False},  default = True
                Параметр отвечает за смещение. Если itersept == True,
                то создается константный единичный столбец в признаках
        -------------------------------------------------------------------------------------         
        """
        random.seed(self.random_state) # Установка случайности
        
        n = X.shape[0] # Определение размера выборки
        
        if isinstance(X, pd.DataFrame): # Валидация матрицы признаков
            X = X.values
       
        if intercept == True: # Добавление интерсепта
            X_ = np.concatenate((np.ones(n).reshape((-1,1)),X), axis = 1)
        else:
            X_ = X
            
        w = np.ones(X_.shape[1]) # Инициализация вектора весов
            
        for i in range(1, self.n_iter + 1): # Градиентный спуск
            
            X_sample, y_sample = self.__X_y_sampling(X_,y) # Создание выборки для итерации
            n_sample = X_sample.shape[0] # Определение размера выборки 
            y_pred = X_sample @ w # Оценка таргета на i-ой итерации
            self.__calc_metrics(y_sample, y_pred)  # Расчет метрик
            self.__output_metrics(verbose, i)  # Вывод метрик
            gradient = 2/n_sample * (y_pred - y_sample) @ X_sample + self.__regulation(w)  # Нахождение градиента
            w = w - self.__calc_learning_rate_value(i) * gradient  # Пересчет вектора весов 
        
        self.w = w  # Определение финального вектора весов
        y_pred = X_ @ w  # Финальная оценка таргета на всех данных
        self.__calc_metrics(y, y_pred)  # Финальный расчет метрик
        return self
    
    
    def __X_y_sampling(self, X: np.array, y: np.array):
        """
            Приватный метод для сэмплирования данных при обучении.
            Если не заданы параметры для SGD, то возвращает 
            всю выборку
        """
        if self.sgd_sample is None:
            return X,y
        else:
            if self.sgd_sample <= 1:
                count_rows = round(X.shape[0] * self.sgd_sample)
            else:
                count_rows = self.sgd_sample 
            indx = random.sample(range(X.shape[0]), count_rows)
            return pd.DataFrame(X).iloc[indx].values, pd.Series(y).iloc[indx].values

    

    def __calc_learning_rate_value(self, i: int)->float:
        """
            Расчитывает коэфициент обучения для итерации
        """
        if isinstance(self.learning_rate, type(lambda x:x)) == True:
                learning_rate_value = self.learning_rate(i)
        else:
            learning_rate_value = self.learning_rate
        return learning_rate_value
        
        
    def __regulation(self,w):
        """
            Приватный метод для создания коэффициента регуляризации
        """
        l1 = self.l1_coef * np.sign(w)
        l2 = self.l2_coef * 2 * w
        if self.reg == 'l1': return l1
        elif self.reg == 'l2': return l2
        elif self.reg == 'elasticnet': return l1 + l2
        else: return 0
            
        
    def __calc_metrics(self, y_true: np.array, y_pred: np.array):
        """
        -------------------------------------------------------------------------------------
            Приватный метод класса для расчета метрик качества
            на каждой итерации обучения.
            
            y_true:
                Истинное значение таргета
                
            y_pred:
                Оценка моделью таргета
        -------------------------------------------------------------------------------------
        """
        
        self.mse = ((y_true - y_pred)**2).mean()       
        if self.metric_name == 'mae':
            self.metric_score = abs(y_true - y_pred).mean()    
            
        elif self.metric_name == 'mse':
            self.metric_score = self.mse       
            
        elif self.metric_name == 'rmse':
            self.metric_score = np.sqrt(self.mse)     
            
        elif self.metric_name == 'mape':
            self.metric_score = (abs((y_true - y_pred)/y_true)).mean() *100  
            
        elif self.metric_name == 'r2':
            self.metric_score = 1 - ((y_true - y_pred)**2).sum()/((y_true - y_true.mean())**2).sum()
            
        else:
            self.metric_score = None
            
        
    def __output_metrics(self,verbose:int, i: int):
        """
            Приватный метод отвечающий за вывод в output
            информации об обучение на неообходимых итерациях
        """
        if verbose != False and i % verbose== 0:
            if i == 0: 
                i = 'start'
            if self.metric_name is None: 
                print(f"{i}|loss:{round(self.mse,5)}")
            else:
                print(f"{i}|loss:{round(self.mse,5)}|{self.metric_name}: {round(self.metric_score,5)}")
                
       
    def get_best_score(self):
        """
            Метод для получения финального score
        """
        if self.metric_score is not None: 
            return self.metric_score
        else: 
            return self.mse 
    
    
    def predict(self,X: pd.DataFrame):
        """
            Метод строит прогноз
            X:
                Матрица признаков 
        """
        n = X.shape[0]
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_ = np.concatenate((np.ones(n).reshape((-1,1)),X), axis = 1)
        return X_ @ self.w
        
    def get_coef(self):
        """
            Метод возвращает веса признаков
        """
        return self.w[1:]