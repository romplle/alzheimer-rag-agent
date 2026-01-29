# Alzheimer RAG-Agent

Прототип RAG-агента для помощи исследователям в поиске потенциальных терапевтических мишеней при болезни Альцгеймера, основанный на анализе научных статей. Система использует векторное хранилище для ретрива и генеративную модель для ответов, с простым веб-интерфейсом на Streamlit.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.14-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53-red?logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple)
![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM-orange)

## Оглавление

1. [Интерфейс](#интерфейс)
2. [Часть 1. Подготовка данных](#часть-1-подготовка-данных)
3. [Часть 2. Реализация RAG-системы](#часть-2-реализация-rag-системы)
4. [Часть 3. Интерфейс](#часть-3-интерфейс)
5. [Часть 4. Идеи расширения](#часть-4-идеи-расширения)
6. [Запуск проекта](#запуск-проекта)
7. [Структура проекта](#структура-проекта)
8. [Результаты оценки](#результаты-оценки)
9. [Направления для дальнейшего развития](#направления-для-дальнейшего-развития)

## Интерфейс

Интерфейс реализован как веб-приложение на Streamlit, позволяющее вводить запросы, получать сгенерированные ответы и просматривать источники (имя файла и релевантность). 

![Скриншот интерфейса](docs/Screenshot.jpg)

## Часть 1: Подготовка данных

Для отбора статей я запросил у ChatGPT список наиболее релевантных и актуальных публикаций по запросам «Alzheimer's disease targets», «Alzheimer therapeutic targets», «Alzheimer drug targets». Из них я отобрал 25 штук (доступных в открытом доступе) и скачал их вручную.

Извлечение текста выполнено с помощью библиотеки `pymupdf4llm`, которая конвертирует PDF в Markdown с сохранением структуры. Я извлек только ключевые разделы: Abstract, Introduction и Conclusion/Discussion, чтобы сосредоточиться на сути исследований и избежать шума. Кастомная очистка данных включала:
- Удаление ссылок на фигуры и таблицы (регулярные выражения для паттернов "Figure X:", "Fig. X." и "Table X:").
- Слияние разорванных абзацев (проверка на продолжение предложений без заглавных букв).
- Удаление лишних заголовков, изображений и многократных переносов строк.

## Часть 2: Реализация RAG системы

RAG-пайплайн построен на LlamaIndex. Ключевые компоненты:

- **Эмбеддинги**: `Qwen/Qwen3-Embedding-0.6B` (HuggingFace). Выбрана за компактность (0.6B параметров) и лучшую производительность среди моделей схожего размера. Перспектива: переход на Qwen3-Embedding-4B / 8B при наличии ресурсов.
- **LLM**: `arcee-ai/trinity-large-preview:free` через OpenRouter. Самая крупная бесплатная модель на OpenRouter.
- **Reranker**: `BAAI/bge-reranker-base` (top_n=7). Выбран за компактность (0.3B параметров), хорошую производительность и интеграцию с LlamaIndex. Перспектива: `bge-reranker-large` + увеличение top_n и similarity_top_k.
- **Чанкинг**: SentenceSplitter с chunk_size=1000, overlap=100. Баланс между контекстом и производительностью.
- **Vector store**: Chroma для хранения эмбеддингов. Выбрано за простоту, скорость и интеграцию с LlamaIndex.
- **Query engine**: similarity_top_k=20 → rerank → response_mode='compact' + custom QA_PROMPT.

Метрики оценки: Использованы `FaithfulnessEvaluator` и `RelevancyEvaluator` из LlamaIndex (той же LLM). Оценка на 3 примерах вопросов из задания показала идеальные scores (1.0), что подтверждает качество.

## Часть 3: Интерфейс

Интерфейс реализован как Streamlit-приложение (`app.py`). Обязательные поля:
- Ввод запроса: поле ввода вопроса.
- Отображение ответа: блок с сгенерированным текстом.
- Источники в блоках с названием файла, similarity score и небольшим отрывком (первые 250 символов).

## Часть 4: Идеи расширения

Решение можно расширить на мультимодальные данные:
- **Модальности**: Изображения и таблицы.
- **Реализация**: 
  - Для изображений: Интегрировать мультимодальную модель в LlamaIndex, извлекать с помощью pymupdf (write_images=True) и эмбеддить.
  - Для таблиц: Использовать Tabula-py для извлечения, затем эмбеддить как текст.

Такие расширения существенно повысят полноту и ценность ответов для исследователей, но потребуют значительных вычислительных ресурсов.

## Запуск проекта

1. Клонируйте репозиторий: `git clone <repo_url>`.
2. Установите зависимости: `pip install -r requirements.txt`.
3. Создайте `api_key.py` файл с `OPENROUTER_API_KEY=your_key`.
4. Подготовьте данные: Скачайте PDF в `data/papers` (список ниже). Запустите `python data_load.py` для извлечения текстов в `data/extracted_texts`.
5. Создайте/загрузите индекс: Запустите `python rag.py` (автоматически создаст persistent index в `data/index_storage` если пусто).
6. Запустите приложение: `streamlit run app.py`.
7. Оценка: `python evaluation.py` (выводит метрики в консоль).

## Структура проекта

```
├── app.py                # Основное Streamlit-приложение
├── data_load.py          # Скрипт для извлечения и очистки текстов из PDF
├── evaluation.py         # Скрипт для оценки RAG
├── prompt.py             # Кастомный промпт для QA
├── rag.py                # RAG-пайплайн: настройка моделей, индекса и query engine
├── api_key.py            # Хранение API-ключа
├── data/
│   ├── papers/           # PDF-статьи
│   ├── extracted_texts/  # Извлеченные и очищенные Markdown-тексты
│   └── index_storage/    # Chroma индекс
├── docs/                 # Скриншоты
├── README.md
└── requirements.txt      # Зависимости
```

## Результаты оценки

Оценка проведена на 3 примерах вопросов из задания. Метрики: Faithfulness и Relevancy.

| Вопрос | Faithfulness | Relevancy | Длина ответа (символы) |
|--------|--------------|-----------|-------------------------|
| What are potential targets for Alzheimer's disease treatment? | 1.0 | 1.0 | 864 |
| Are the targets druggable with small molecules, biologics, or other modalities? | 1.0 | 1.0 | 2062 |
| What additional studies are needed to advance these targets? | 1.0 | 1.0 | 3225 |

Выводы: Идеальные scores указывают на высокую точность; пайплайн хорошо справляется с заданием. Однако, оценка субъективна (на той же LLM) и на малой выборке.

## Список использованных статей

| Название | Год |
|----------|-----|
| Alzheimer disease | 2021 |
| Alzheimer’s Disease: Novel Targets and Investigational Drugs for Disease Modification | 2023 |
| Alzheimer’s disease master regulators analysis: search for potential molecular targets and drug repositioning candidates | 2018 |
| Amyloid β-based therapy for Alzheimer’s disease: challenges, successes and future | 2023 |
| Amyloid-β-targeted therapies for Alzheimer’s disease: currently and in the future | 2023 |
| APP dyshomeostasis in the pathogenesis of Alzheimer’s disease implications for current drug targets | 2024 |
| Beyond expectations: investigating nilotinib’s potential in attenuating neurodegeneration in Alzheimer’s disease | 2025 |
| Comprehensive Review on Alzheimer’s Disease: Causes and Treatment | 2020 |
| Current and emerging therapeutic targets of alzheimer's disease for the design of multi-target directed ligands | 2019 |
| Current and future therapeutic strategies for Alzheimer’s disease: an overview of drug development bottlenecks | 2023 |
| Current Research Therapeutic Strategies for Alzheimer’s Disease Treatment | 2025 |
| Current therapeutics for Alzheimer’s disease and clinical trials | 2024 |
| Diverse Molecular Targets for Therapeutic Strategies in Alzheimer’s Disease | 2014 |
| Genetic insights into drug targets for alzheimer’s disease: integrative multiomics analysis | 2025 |
| New Pathways Identify Novel Drug Targets for the Prevention and Treatment of Alzheimer’s Disease | 2023 |
| Non-pharmacological treatment of Alzheimer’s disease: an update | 2025 |
| Pharmacotherapeutic targets in Alzheimerʼs disease | 2008 |
| Protein kinases as therapeutic targets for Alzheimer’s disease: a brief review | 2024 |
| Recent Advances in Drug Development for Alzheimer’s Disease: A Comprehensive Review | 2025 |
| Review of Pharmacotherapeutic Targets in Alzheimer’s Disease and Its Management Using Traditional Medicinal Plants | 2024 |
| Targeting the Synergistic Interaction of Pathologies in Alzheimer's Disease: Rationale and Prospects for Combination Therapy | 2025 |
| Tau and neuroinflammation in Alzheimer’s disease: interplay mechanisms and clinical translation | 2023 |
| Therapeutic Advances in Targeting the Amyloid-β Pathway for Alzheimer’s Disease | 2025 |
| Therapeutic Targets for Alzheimer’s Disease: Amyloid Vs. Non Amyloid. Where Does Consensus Lie Today? An CTAD Task Force Report | 2023 |
| Treatment of Alzheimer’s Disease: Beyond Symptomatic Therapies | 2023 |

## Направления для дальнейшего развития

- Улучшение очистки данных.
- Расширение корпуса статей.
- Использование более мощных embedding и reranker моделей.
- Добавление логирования и мониторинга качества ответов.
- Внедрение мультимодальных возможностей.
