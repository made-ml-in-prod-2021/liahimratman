Обучение / использование модели:
-----------


Переходим в homework1:
`cd homework1`

Обучение
-----------
Config1
`python train_pipeline.py --config-path=configs/train_config.yaml`

Config2
`python train_pipeline.py --config-path=configs/train_config2.yaml`

В результате обучения в директории configs создается evaluation_config.yaml, который можно использовать для предикта.
Все, что нужно, это добавить в него input_data_path, соответствующий пути входных данных, для которых нужно сделать предсказания.

Предсказание
-----------
`python predict_pipeline.py --config-path=configs/evaluation_config.yaml`
