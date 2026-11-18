# LaunchRadar

Открытый прототип triage обратной связи перед выпуском устройства. Текущая версия запускается на небольшой синтетической выборке и показывает три шага:

1. keyword baseline назначает темы, тональность аспекта, severity и safety flag;
2. похожие формулировки собираются в один issue cluster;
3. synthetic pilot/holdout comparison формирует advisory-решение.

Сравнение cohort не является причинным выводом. Safety-сигнал не конкурирует с частыми UX-жалобами и получает `HOLD`.

## Запуск

```bash
python3 -m launch_radar
python3 -m launch_radar serve
python3 -m launch_radar dmv
python3 -m unittest discover -s tests
```

Интерфейс открывается на [http://127.0.0.1:8765](http://127.0.0.1:8765). Разбор хранится в памяти процесса, пока сервер запущен.

Отчёты California DMV за 2023 год лежат отдельно от синтетического выпуска. Файл не коммитится:

```bash
mkdir -p data/raw
curl -fsSL -o data/raw/dmv-2023-disengagements.csv \
  https://www.dmv.ca.gov/portal/file/2023-autonomous-vehicle-disengagement-reports-csv/
python3 -m launch_radar dmv
```

Зависимостей нет, нужен Python 3.11+.
