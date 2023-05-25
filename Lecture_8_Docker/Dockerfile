# Начинаем с официального образа Python версии 3.10
FROM python:3.10

# Внутри контейнера будем работать в директории /code
WORKDIR /code

# Копируем файл с зависимости в рабочую директорию
COPY ./requirements.txt /code/requirements.txt

# Устанавливаем Python-зависимости с помощью менеджера пакетов pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Копируем код приложения в рабочую директорию
# Важно сделать это ближе к концу Docker-файла,
# потому что код приложения может часто меняться
COPY ./app /code/app

# Запускаем наше приложение внутри контейнера
CMD ["uvicorn", "app.churn_api:app", "--host", "127.0.0.1", "--port", "80"]
