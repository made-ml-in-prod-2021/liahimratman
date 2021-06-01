***Обучение / использование модели:***
-----------

Переходим в homework1:
`cd homework1`

***Обучение***

-----------

Config1
`python train_pipeline.py --config-path=configs/train_config.yaml`

Config2
`python train_pipeline.py --config-path=configs/train_config2.yaml`

В результате обучения в директории configs создается evaluation_config.yaml, который можно использовать для предикта.
Все, что нужно, это добавить в него input_data_path, соответствующий пути входных данных, для которых нужно сделать предсказания.

***Предсказание***

-----------

`python predict_pipeline.py --config-path=configs/evaluation_config.yaml`


-----------

***FastApi и запуск через Docker:***
-----------


Аналогично, переходим в online_inference:

`cd online_inference`

-----------

Используется FastApi, если без докера, то поднять можно так:

`uvicorn api:app --port 8000`

-----------

Собрать образ самостоятельно можно с помощью:

`docker build -t frantotti/ml_project_made:v1 .`

Также, образ запушен на докер хаб, стянуть его оттуда можно командой:

`docker pull frantotti/ml_project_made:v1`

Запускаем докер контейнер:

`docker run -p 8000:80 frantotti/ml_project_made:v1`

-----------

Всё, теперь к модели можно стучаться локально на localhost:8000

При этом по эндпоинту localhost:8000/is_ready можно получить информацию о готовности модели к использованию
А, соответственно, на эндпоинт localhost:8000/predict можно слать post-запросы с вашими входными данными.

Также потестить работу модели можно с помощью:

`python online_inference/make_request.py`
