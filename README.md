# STEP to DXF Converter

Веб-приложение для конвертации STEP файлов в DXF чертежи.

## Возможности

- **3 ортогональные проекции**: вид спереди, сверху, сбоку
- **Изометрический вид**: показывает только видимые контуры (silhouette)
- **Автоматические размерные линии**: габариты на каждом виде
- **Спецификация деталей**: список всех элементов с количеством
- **Вращающийся индикатор прогресса**: показывает статус конвертации

## Технологии

- **Backend**: Python 3.12, FastAPI
- **Frontend**: HTML/CSS/JavaScript (vanilla)
- **Обработка STEP**: trimesh, cascadio, numpy, scipy
- **Генерация DXF**: ezdxf

## Установка

### Требования

- Python 3.12+
- pip

### Установка зависимостей

```bash
cd /var/www/step
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn python-multipart ezdxf numpy trimesh scipy cascadio
```

### Запуск

```bash
source venv/bin/activate
uvicorn app:app --host 127.0.0.1 --port 8002
```

### Systemd сервис

Создайте файл `/etc/systemd/system/step-converter.service`:

```ini
[Unit]
Description=STEP to DXF Converter
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/var/www/step
Environment="PATH=/var/www/step/venv/bin"
ExecStart=/var/www/step/venv/bin/uvicorn app:app --host 127.0.0.1 --port 8002
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Запуск:

```bash
systemctl daemon-reload
systemctl enable step-converter
systemctl start step-converter
```

### Nginx прокси

```nginx
location /step/ {
    proxy_pass http://localhost:8002/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 300;
    client_max_body_size 50M;
}
```

## Использование

1. Откройте веб-интерфейс
2. Загрузите STEP файл (drag & drop или выбор файла)
3. Нажмите "Скачать DXF"
4. Получите DXF файл с чертежом

## Алгоритм

1. **Загрузка STEP**: trimesh загружает файл, cascadio парсит геометрию
2. **Извлечение геометрии**: все детали объединяются, вычисляются габариты
3. **Silhouette edges**: для каждого вида определяются только видимые контуры
4. **Изометрия**: проецируется под углом 30°, показываются только границы видимых/невидимых граней
5. **Генерация DXF**: ezdxf создаёт файл с проекциями, размерами и спецификацией

## Лицензия

MIT

## Автор

Создано с помощью Claude AI
