﻿# pipes_counter
в папке ml лежат файлы связанные с обучением нейронной сети и эксперементами с детекцией

```bash
# изображения труб
pipes_counter\ml\cap
# изображения труб после нейросети
pipes_counter\ml\runs\detect\predict
```

в остальных папках расположены файлы, необходимые для работы клиент-серверного приложения

Для запуска системы необходим Python 3.10
## 0 Загрузка 
```bash
git clone https://github.com/ilyatmk/pipes_counter
cd pipes_counter
```
## 1 Создание виртуального окружения:
```bash
# cоздание виртуального окружения
python -m venv pipes_counter
# активация виртуального окружения
pipes_counter/Scripts/activate
# установка необходимых зависимостей
pip install -r requirements.txt
```

## 2 Запуск сервера

```bash
uvicorn main:app --reload
```

Для запуска демоверсии перейдите в браузер по ссылке http://127.0.0.1:8000/

*в демоверсии используются предобработанные нейросетью видео с выделенными трубами. фото обрабатываются в реальном времени
## Примеры
![Image alt](https://raw.githubusercontent.com/ilyatmk/pipes_counter/main/ml/runs/detect/new/5.jpg)
![img](https://raw.githubusercontent.com/ilyatmk/pipes_counter/main/ml/runs/detect/new/7.jpg)

