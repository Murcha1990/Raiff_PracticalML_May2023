import datetime as dt
from airflow.models import DAG
from airflow.operators.python import PythonOperator

import pandas as pd

args = {
    'owner': 'airflow',  # Информация о владельце DAG
    'start_date': dt.datetime(2020, 12, 23),  # Время начала выполнения пайплайна
    'retries': 1,  # Количество повторений в случае неудач
    'retry_delay': dt.timedelta(minutes=1),  # Пауза между повторами
}

def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/evgpat/edu_stepik_practical_ml/main/datasets/churn_clients.csv")

    X = data.drop('churn', axis=1)  # матрица объект-признак
    y = data[['churn']]  # целевая переменная

    X.to_csv("/data/features.csv", index=False)
    y.to_csv("/data/target.csv", index=False)

def process_data():
    X = pd.read_csv("/data/features.csv")
    X = X.drop(['state', 'areacode', 'voicemailplan', 'internationalplan'], axis=1)
    X.to_csv("/data/features.csv", index=False)


def make_prediction():
    X = pd.read_csv("/data/features.csv")

    X['prediction'] = X['customerservicecalls'].apply(lambda x: 1 if x > 3 else 0)

    X[['prediction']].to_csv("/data/prediction.csv", index=False)

def calculate_accuracy():
    target = pd.read_csv("/data/target.csv")
    prediction = pd.read_csv("/data/prediction.csv")

    print('accuracy:', (target['churn'].values == prediction['prediction'].values).mean())

def calculate_recall():
    target = pd.read_csv("/data/target.csv")
    prediction = pd.read_csv("/data/prediction.csv")

    target = target['churn'].values
    prediction = prediction['prediction'].values

    a, b = 0, 0
    for i in range(len(target)):
        if target[i] == 1:
            b += 1
            if prediction[i] == target[i]:
                a += 1

    print('recall:', a*1./b)

dag = DAG(
    dag_id='ml_pipeline',  # Имя DAG
    schedule_interval=None,  # Периодичность запуска, например, "00 15 * * *"
    default_args=args,  # Базовые аргументы
)

load_data = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
        dag=dag,
)

process_data = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
        dag=dag,
)

make_prediction = PythonOperator(
        task_id='make_prediction',
        python_callable=make_prediction,
        dag=dag,
)

calculate_accuracy = PythonOperator(
        task_id='calculate_accuracy',
        python_callable=calculate_accuracy,
        dag=dag,
)

calculate_recall = PythonOperator(
        task_id='calculate_recall',
        python_callable=calculate_recall,
        dag=dag,
)

load_data >> process_data >> make_prediction >> [calculate_accuracy, calculate_recall]