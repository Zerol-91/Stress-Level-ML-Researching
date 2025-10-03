 
Pipeline: собрать данные → очистить → генерация фичей → обучение → оценка → развёртывание

Документация модели: что делает, какие ограничения, какие данные использовались.

# Собрать данные

## [Student Stress Monitoring Datasets](https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets?select=Stress_Dataset.csv)

Датасетик на 7.5/10

Вес: 107кб
Кол-во наблюдений: 844

В датасете очень много людей со стрессом. Дай бог 10% пишут о его отсутствии.
Минус датасета - это классификатор стресса. Он либо есть, либо его нет.

Колонки: 
- *Gender*
- *Age* 
- Have you recently experienced stress in your life?,
- Have you noticed a rapid heartbeat or palpitations?,
- Have you been dealing with anxiety or tension recently?,
- *Do you face any sleep problems or difficulties falling asleep?*
- Have you been dealing with anxiety or tension recently?.
- Have you been getting headaches more often than usual?
- *Do you get irritated easily?*
- *Do you have trouble concentrating on your academic tasks?*
- Have you been feeling sadness or low mood?,
- Have you been experiencing any illness or health issues?,
- *Do you often feel lonely or isolated?*
- *Do you feel overwhelmed with your academic workload?*
- "Are you in competition with your peers, and does it affect you?",
- Do you find that your relationship often causes you stress?,
- *Are you facing any difficulties with your professors or instructors?*
- *Is your working environment unpleasant or stressful?*
- Do you struggle to find time for relaxation and leisure activities?,
- *Is your hostel or home environment causing you difficulties?*
- Do you lack confidence in your academic performance?,
- Do you lack confidence in your choice of academic subjects?,
- Academic and extracurricular activities conflicting for you?
- *Do you attend classes regularly?,
- Have you gained/lost weight?,
- Which type of stress do you primarily experience?

Курсив - то, что мне понравилось. Неподчеркнутые тоже есть хорошие, но курсивные я бы хотел взять в итог.
## [Student Mental Health Survey](https://www.kaggle.com/datasets/abdullahashfaqvirk/student-mental-health-survey)

Датасет на 7/10. 

Кол-во наблюдений - 88

Тут хорошо, что уровень стресса в баллах от 1 до 5 как я понял. Но стресс не напрямую. Здесь описана депрессия + тревога.



- *gender,
- *age,
- university,
- degree_level,
- degree_major,
- academic_year,
- cgpa,
- residential_status,
- campus_discrimination,
- *sports_engagement,
- *average_sleep,
- *study_satisfaction,
- *academic_workload
- *academic_pressure,
- *financial_concerns,
- social_relationships,
- *depression,
- *anxiety,
- *isolation,
- *future_insecurity,
- stress_relief_activities


## [Student stress factors](https://www.kaggle.com/datasets/samyakb/student-stress-factors)

Датасет на 6/10, тут слишком мало фичей. Это главный минус.

520 наблюдений

Тут все в баллах от 1 до 5.

- *Kindly Rate your Sleep Quality 😴*
- How many times a week do you suffer headaches 🤕?,
- *How would you rate you academic performance 👩‍🎓?,
- *how would you rate your study load?,
- *How many times a week you practice extracurricular activities 🎾?,
- *How would you rate your stress levels?

## [Academic Stress Level](https://www.kaggle.com/datasets/ayeshaimran123/academic-stress-level-maintenance-dataset)

Датасет 4/10 потому что предложил влад, а еще тут нечего взять нормального. Мне не нравятся фичи. Можно 2-3 выделить, но очень мало это.

141 наблюдение

- Timestamp,
- Your Academic Stage,
- Peer pressure,
- Academic pressure from your home,
- Study Environment,
- What coping strategy you use as a student?,
- "Do you have any bad habits like smoking, drinking on a daily basis?",
- What would you rate the academic  competition in your student life,
- Rate your academic stress index 


## [Social Anxiety Dataset](https://www.kaggle.com/datasets/natezhang123/social-anxiety-dataset)

Датасет на 8.5/10 

11.000 наблюдений!!

- Age,
- *Gender,
- Occupation,
- *Sleep Hours,
- *Physical Activity (hrs/week),
- *Caffeine Intake (mg/day),
- *Alcohol Consumption (drinks/week),
- *Smoking,
- Family History of Anxiety,
- **Stress Level (1-10),
- Heart Rate (bpm),
- Breathing Rate (breaths/min),
- Sweating Level (1-5),
- *Dizziness,
- Medication,
- Therapy Sessions (per month),
- *Recent Major Life Event,
- *Diet Quality (1-10),
- *Anxiety Level (1-10)* (А стоит ли оставлять эту переменную? Они страшно коррелировать будут)

Очень хорошие данные.

## [General Social Survey](https://www.kaggle.com/datasets/norc/general-social-survey?select=gss.csv)

Данные 9/10

Здесь 10.000 (не наблюдений) **колонок**

Вес 2ГБ

Минус - сложные данные. ОЧЕНЬ много надо убрать.

## [Behavioral Risk Factor Surveillance System](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system)

Данные 9-9.5/10

Данные по 0.5ГБ, разделенные по годам 2011-2015

По разным годам разное кол-во колонок: от 279 до 453

# [Mind-O-Meter](https://www.kaggle.com/datasets/namisanajahraisa/mind-o-meter)

### `balanced_mental_health_dataset.csv`

- **Description**: This CSV file contains demographic, behavioral, and mental health-related data points collected from a diverse set of individuals.
- **Rows**: 5000 (each row represents an individual entry).
- **Columns**: 17 features (detailed descriptions below).
- **File Size**: 558.26 kB

|**Feature Name**|**Type**|**Description**|
|---|---|---|
|`Age`|Integer|Age of the individual.|
|`Gender`|Categorical|Gender of the individual (e.g., Male, Female, Non-binary).|
|`Ethnicity`|Categorical|Ethnic background of the individual (e.g., Asian, Hispanic, African American).|
|`EducationLevel`|Categorical|Highest level of education attained (e.g., High School, Bachelor’s Degree, Doctorate).|
|`EmploymentStatus`|Categorical|Current employment status (e.g., Employed, Unemployed, Student).|
|`DepressionScore_PHQ9`|Integer|Depression score based on the PHQ-9 scale (range: 0–27).|
|`AnxietyScore_GAD7`|Integer|Anxiety score based on the GAD-7 scale (range: 0–21).|
|`StressLevel`|Categorical|Self-reported stress level (e.g., Low, Moderate, High).|
|`SleepHours`|Float|Average hours of sleep per night.|
|`AppetiteChange`|Categorical|Reported changes in appetite (e.g., Stable, Increased).|
|`SubstanceUse`|Categorical|Reports of substance use (Yes/No).|
|`PhysicalActivity`|Categorical|Level of physical activity (e.g., Low, Moderate, High).|
|`SocialInteractions`|Categorical|Quality of social interactions (e.g., Poor, Limited, Good).|
|`LivingSituation`|Categorical|Current living situation (e.g., Alone, With family, With roommates).|
|`SupportSystems`|Categorical|Strength of support systems (e.g., Weak, Moderate, Strong).|
|`TraumaticEvents`|Categorical|History of traumatic events (Yes/No).|
|`NeedsProfessionalHelp`|Categorical|Whether the individual needs professional mental health help (Yes/No).|

Для быстрого бейзлайна очень неплохо.

# [Anxiety and Indicators - Clean Dataset](https://www.kaggle.com/datasets/noristalaban/anxiety-and-indicators-clean-dataset)

This dataset was provided by a user on the platform. After a careful analysis, I cleaned the data, removing some variables that lacked explanation. Additionally, I added a Wellness Score column, which could be useful for your analyses or predictive models. Overall, after examining the dataset, the data appear fairly realistic and very interesting, as they collect information of diverse nature. Enjoy your work

**Age** – Participant’s age (years).

**Occupation** – Profession or job role.

**Sleep Hours (hrs/day)** – Average daily hours of sleep

**Physical activity (hrs/week)** – Weekly hours of physical activity.

**Caffeine dosage day (mg/day)** – Daily caffeine intake.

**Alcohol drinks week (drinks/week)** – Number of alcoholic drinks consumed per week.

**Smoking** – Indicates if the person smokes (0 = No, 1 = Yes).

**Family history anx** – Family history of anxiety (0 = No, 1 = Yes).

**Stress level (1–10)** – Perceived stress level.

**Sweating level (1–5)** – Perceived sweating level.

**Dizziness** – Presence of dizziness or lightheadedness (0 = No, 1 = Yes).

**Medication** – Medication usage (0 = No, 1 = Yes).

**Therapy sessions month (sessions/month)** – Number of therapy sessions per month.

**Recent life event** – Recent significant life events (0 = No, 1 = Yes).

**Anxiety level (1–10)** – Perceived anxiety level.

**Espresso Day** – Derived from caffeine dosage day (1 espresso ≈ 60 mg of caffeine)

**Wellness score (0–10)** – Wellness score calculated from sleep, physical activity, stress, anxiety, caffeine, and alcohol.
# Построить Baseline и Запустить Google Form опрос
## Проблемы

- Вопрос. Можно ли обучить модель на этих данных и дообучить на данных с новыми признаками (новыми колонками). Надо ли, чтобы в новом обучающем датасете были те же данные что в первом обучающем. Могут ли они быть совершенно другими? Тогда в чем смысл? может быть сделать две модели с усредненным предсказанием и проверить ее действие только на тестовой выборке (среди студентов).

- Я думаю взять два датасета и смешать их:
	  Student Stress Monitoring Datasets (Тут только *Да-Нет* по стрессу, посмотри на колонки еще раз.)
	  Social Anxiety Dataset (тут правда target - *Anxiety level*)

В остальных мало наблюдений. На крайний можно взять **Student Mental Health Survey** (80-90 наблюдений), так как здесь стресс в баллах от 1 до 5. + фичи интересные.

Может быть обучить одним фичам на одном датасете и дополнить на другом?

---

Данные по Social Anxiety Dataset на самом деле очень... необычные. Там например не везде пол указан

## Анализ собранных данных (kaggle datasets)

### Social Anxiety Dataset

Возраст... Возможно, после построения бейзлайна и сбора данных на реальных студентах стоит выбрать только определенный диапазон.

Часы сна обязательно стоит оставить.
Физ. активность тоже. Про кофе просто интересно будет, оставляем. 

По поводу питья: (ответ копилота) 
Что означает "drinks/week"?

В контексте большинства медицинских и социальных исследований **"drinks"** — это **стандартные порции алкоголя**. То есть:

- **1 drink** = примерно **14 г чистого алкоголя**, что соответствует:
    
    - 350 мл пива (5%)
        
    - 150 мл вина (12%)
        
    - 45 мл крепкого алкоголя (40%)
        

Так что **"5" в колонке — это 5 стандартных порций алкоголя в неделю**, не 5 бутылок, не 5 вечеринок, а именно 5 порций. Это может быть, например, бокал вина за ужином 5 раз в неделю.

---

Пол мне не нравится в данных. Есть мужской, женский и... other. Я для бейзлайна напишу мужчина 1, женщина 0, другой - 0.5. Как будто для модели будет лучше, если признак равный "другой" никак не повлияет на это.

---

Профессии. Для бейзлайна пока не знаю. Может быть стоит убрать пока. В итоговом датасете вообще мб стоит сделать человек либо работает, либо нет.

---

Теперь наши данные состоят только из чисел, либо номинативных переменных "да", "нет"
Но даже так, надо переработать. Мне казалось, что обрабатывает случ. лес. Но мб только TesotFlow так может.


---
![[Pasted image 20250927211248.png]]
Это можно бы и оставить.
![[Pasted image 20250927211357.png]]
А вот это мб уберу. Но не для бейзлайна.

---
### Бейзлайн
![[Pasted image 20250927220632.png]]
Я так понял слишком много я просил выводить на обучении, так что у меня колаб вылетал.

![[Pasted image 20250927220715.png]]
![[Pasted image 20250927220747.png]]
Вот такой результат)

Я заменил все данные y_train на 5 и получил вот что после предсказания на настоящих тестовых данных:
![[Pasted image 20250927222422.png]]
Модель моя по итогу чуть чуть лучше чем случайность (ЭТО)

---

Меня ЧатГПТ успокоил:
##### 💡 Интерпретация

- **Ошибка 2.35 на шкале 1–10** — это **довольно грубое предсказание** (пользователь на практике может получить прогноз ±2–3 балла от реального уровня стресса).
    
- **R² ~0.1** говорит, что модель **почти не лучше среднего предсказания** (например, «всегда предсказываем среднее значение стресса»).
    
- Такая картина типична для **baseline**, особенно если:
    
    - данные шумные (стресс сложно предсказать по наблюдаемым факторам);
        
    - мало информативных признаков;
        
    - не проведён серьёзный feature engineering.
---

#### 🔧 Что можно улучшить

1. **Feature Engineering**
    
    - Создать новые признаки (отношение сна к времени в телефоне, спорт/учёба и т.д.).
        
    - Попробовать нелинейные взаимодействия (например, квадратичные признаки).
        
2. **Отбор/важность признаков**
    
    - Посмотреть feature importance (RandomForest/XGBoost).
        
    - Убрать слабые и добавить более релевантные.
        
3. **Более сложные модели**
    
    - Gradient Boosting (XGBoost, LightGBM, CatBoost).
        
    - Neural networks (если много данных).
        
4. **Гиперпараметры**
    
    - Увеличить глубину/число деревьев или оптимизировать learning_rate в бустинге.
        
5. **Данные**
    
    - Возможно, существующие фичи действительно слабо связаны со стрессом. Тогда нужен **новый сбор данных** или более точные метрики (например, субъективные анкеты + объективные показатели).

### Теперь фичи смотрим!

```python
import matplotlib.pyplot as plt

num_cols = [
    'Age','Sleep Hours','Physical Activity (hrs/week)',
    'Caffeine Intake (mg/day)','Alcohol Consumption (drinks/week)',
    'Heart Rate (bpm)','Breathing Rate (breaths/min)',
    'Sweating Level (1-5)','Therapy Sessions (per month)',
    'Diet Quality (1-10)'
]
plt.figure(figsize=(15,10))
for i, col in enumerate(num_cols[:6]):  # первые 6 для примера
    plt.subplot(2,3,i+1)
    sns.histplot(
        data=Big_stress_data, x=col, hue='Anxiety Level (1-10)',
        bins=30, palette='coolwarm', element='step', stat='density'
    )
    plt.title(col)
plt.tight_layout()
plt.show()
```
Код для графиков ниже

![[Pasted image 20250927224511.png]]

![[Pasted image 20250927224633.png]]

#### 🔧 Что можно улучшить

1. **Feature Engineering**
    
    - Создать новые признаки (отношение сна к времени в телефоне, спорт/учёба и т.д.).
        
    - Попробовать нелинейные взаимодействия (например, квадратичные признаки).
        
2. **Отбор/важность признаков**
    
    - Посмотреть feature importance (RandomForest/XGBoost).
        
    - Убрать слабые и добавить более релевантные.
        
3. **Более сложные модели**
    
    - Gradient Boosting (XGBoost, LightGBM, CatBoost).
        
    - Neural networks (если много данных).
        
4. **Гиперпараметры**
    
    - Увеличить глубину/число деревьев или оптимизировать learning_rate в бустинге.
        
5. **Данные**
    
    - Возможно, существующие фичи действительно слабо связаны со стрессом. Тогда нужен **новый сбор данных** или более точные метрики (например, субъективные анкеты + объективные показатели).

---
## Baseline по Social Anxiety Dataset

Напомню, здесь сетка параметров подобрана не идеально. Использовался Random Grid Search.

![[Pasted image 20250927225357.png]]
Anxiety Level Стал оказался на удивление очень предсказуемым

***Делаю на этом моменте первый комит***
- **Скачать из Colab:** Получить файлы проекта локально (на компьютер).
    
- **Настроить GitHub Desktop:** Подтянуть пустой репозиторий с GitHub.
    
- **Добавить файлы:** Положить скачанные из Colab файлы в папку репозитория.
    
- **Закоммитить и запушить:** Сделать коммит в Desktop и опубликовать на GitHub.

[[Как сделать комит из колаба в гитхаб через десктоп]]

---

## Вперед, по новой

Теперь я беру еще один датасет про тревожность. 
Это была Behavioral Risk Factor Surveillance System

![[Pasted image 20251001143404.png]]

Отправил нейросети просмотреть признаки.
Для большого датасета есть 2 варианта отбора:
1) через эвристики
2) Вручную удалять

Кратко что надо делать дальше:
1) удаление признаков из секции идентификаторов
2) Собрать набор обязательных фич. То что напрямую связано с тревогой
3) Переводи ordinal variables в порядковые числа; one-hot для небольших категорий; агрегируй даты/месяцы если хочешь сезонность
4) Создать пары/отношения 
5) Можно (но не обязательно) проверить переменные на мультиколлинеарность (Мультиколлинеарность - наличие линейной зависимости между объясняющими переменными регрессионной модели.) (+Случайный лес устойчив к мультиколлинеарности в плане предсказания, но плохо будет при интерпретации)
6) Построй бейзлайн

#### Экскурс по (вероятным) переменным для добавления 
 1. `ADPLEASR` — Days had little interest or pleasure doing things

- **Тип/шкала:** Num — за последние 2 недели: 1–14 (days) / 88 = None / 77/99.
    
- **Почему важно:** потеря интереса — ключевой симптом депрессии/тревоги; сильный коррелят с уровнем тревожности/депрессии.
    
    Информация по колонкам датасета…
    
- **Оценка важности:** **9/10**
    
- **Как спросить в Google Form:** «За последние 2 недели, сколько дней вы испытывали снижение интереса или удовольствия от обычных дел?» — ответ: `0–14` (число).
    

---

2. `ADDOWN` — Days felt down, depressed or hopeless

- **Тип/шкала:** Num — 1–14 days / 88 None.
    
- **Почему важно:** центральный симптом настроения, тесно связан с тревожностью/депрессией.
    
    Информация по колонкам датасета…
    
- **Оценка важности:** **9/10**
    
- **Вопрос для формы:** «За последние 2 недели, в скольких днях вы чувствовали подавленность, безнадёжность?» — `0–14`.
    

---

3. `ADSLEEP` — Days had trouble with sleep (falling/staying/asleep too much)

- **Тип/шкала:** Num — 1–14 days / 88 None.
    
- **Почему важно:** нарушение сна — сильный предиктор тревожности; легко измеряется и валидно.
    
    Информация по колонкам датасета…
    
- **Оценка важности:** **9/10**
    
- **Вопрос для формы:** «За последние 2 недели, сколько дней у вас были проблемы со сном (засыпание/поддержание сна/слишком много сна)?» — `0–14`.
    

---

4. `ADENERGY` — Days were tired or had little energy

- **Тип/шкала:** Num — 1–14 days / 88 None.
    
- **Почему важно:** утомляемость — компонент депрессивно-тревожного синдрома.
    
    Информация по колонкам датасета…
    
- **Оценка важности:** **8/10**
    
- **Вопрос для формы:** «За последние 2 недели, в скольких днях вы чувствовали усталость или отсутствие энергии?» — `0–14`.
    

---
 5. `ADEAT1` — Days ate too little or too much (appetite changes)

- **Тип/шкала:** Num — 1–14 days / 88 None.
    
- **Почему важно:** изменения аппетита — симптом эмоционального расстройства.
    
    Информация по колонкам датасета…
    
- **Оценка важности:** **7/10**
    
- **Форма:** «За последние 2 недели, в скольких днях у вас были изменения аппетита (меньше/больше еды)?» — `0–14`.
    

---

 6. `ADFAIL` — Days felt like failure or let family down

- **Тип/шкала:** Num — 1–14 days / 88 None.
    
- **Почему важно:** чувство вины/бесполезности — маркер тяжелых эмоциональных состояний.
    
    Информация по колонкам датасета…
    
- **Оценка важности:** **7/10**
    
- **Форма:** «За последние 2 недели, в скольких днях вы чувствовали, что подводите себя или семью?» — `0–14`.
    

---

7. `ADTHINK` — Days had trouble concentrating

- **Тип/шкала:** Num — 1–14 days / 88 None.
    
- **Почему важно:** снижение концентрации часто сопровождает тревогу/депрессию; полезная объективная шкала.
    
    Информация по колонкам датасета…
    
- **Оценка важности:** **7/10**
    
- **Форма:** «За последние 2 недели, в скольких днях вы испытывали трудности с концентрацией?» — `0–14`.
    

---

8. `ADMOVE` — Days moved or spoke abnormally slow or fidgety/restless

- **Тип/шкала:** Num — 1–14 days / 88 None.
    
- **Почему важно:** психомоторные симптомы — индикатор тяжести; тревога часто даёт беспокойство/психомоторное возбуждение.
    
    Информация по колонкам датасета…
    
- **Оценка важности:** **7/10**
    
- **Форма:** «За последние 2 недели, в скольких днях вы были заметно вялыми или, напротив, очень беспокойными/неусидчивыми?» — `0–14`.
    

---

9. `ADANXEV` — Ever told you had an anxiety disorder

- **Тип/шкала:** Categorical (1 = Yes; 2 = No; 7/9 DK/Refused).
    
- **Почему важно:** прямой маркер наличия диагноза (история тревожного расстройства) — сильный признак.
    
    Информация по колонкам датасета…
    
- **Оценка важности:** **10/10**
    
- **Форма (если спрашивать отдельно):** «Вам когда-либо ставили диагноз тревожного расстройства (например, GAD, паника, ПТСР, ОКР и т.п.)?» — варианты: Да / Нет / Не знаю.
    

---

10. `MISTMNT` — Are you currently taking medicine or receiving treatment for a mental health condition?

- **Тип/шкала:** Categorical (1 = Yes; 2 = No; 7/9).
    
- **Почему важно:** текущее лечение — индикатор предшествующей/текущей проблемы и её тяжести.
    
    Информация по колонкам датасета…
    
- **Оценка важности:** **9/10**
    
- **Форма:** «Сейчас вы принимаете лекарства или получаете лечение у специалиста по психическому здоровью?» — Да/Нет.
    

---

11. `MENTHLTH` — Number of days mental health not good (past 30 days)

- **Тип/шкала:** Num — 1–30 (days) / 88 None.
    
- **Почему полезно:** широко используемый HRQoL индикатор — удобный непрямой target (если GAD-7 нет).
    
    Информация по колонкам датасета…
    
- **Оценка важности:** **9/10**
    
- **Форма:** «За последние 30 дней, в сколько дней ваше психическое состояние было плохим (стресс/депрессия/эмоциональные проблемы)?» — `0–30`.
    

---

12. `POORHLTH` — Days poor physical or mental health kept you from activities (past 30 days)

- **Тип/шкала:** Num — 1–30.
    
- **Почему важно:** функциональное ограничение — индикатор тяжёлости состояния.
    
    Информация по колонкам датасета…
    
- **Оценка важности:** **7/10**
    
- **Форма:** «За последние 30 дней, в сколько дней плохое физическое или психическое здоровье мешало вашей обычной деятельности?» — `0–30`.
    

---

13. `LSATISFY` — Life satisfaction

- **Тип/шкала:** Ordinal categorical (1 Very satisfied → 4 Very dissatisfied).
    
- **Почему важно:** низкая удовлетворённость связана с депрессией/тревогой.
    
    Информация по колонкам датасета…
    
- **Оценка важности:** **7/10**
    
- **Форма:** «В целом, насколько вы удовлетворены своей жизнью?» — Варианты: Очень доволен / Доволен / Недоволен / Очень недоволен.
    

---

14. `GENHLTH` — General self-rated health (Excellent → Poor)

- **Тип/шкала:** Ordinal (1–5).
    
- **Почему важно:** общее самочувствие коррелирует с ментальным здоровьем.
    
    Информация по колонкам датасета…
    
- **Оценка важности:** **6/10**
    
- **Форма:** «Как вы оцениваете ваше общее здоровье?» — Отлично / Очень хорошо / Хорошо / Удовлетворительно / Плохо.
    

---

15. `PA1MIN_ / _MINAC11` — Minutes of physical activity per week (calculated)

- **Тип/шкала:** Num — минуты/неделя.
    
- **Почему важно:** физическая активность уменьшает тревогу; объективный поведенческий признак.
    
- **Оценка важности:** **6/10**
    
- **Форма:** «Сколько минут физической активности (умеренной/интенсивной) вы делаете в среднем за неделю?» — число минут.
    

---

16. `SMOKE100` / `SMOKDAY2` — Smoking status (ever smoked / current smoker)

- **Тип/шкала:** Categorical / Num.
    
- **Почему важно:** курение связано с повышенной тревожностью/здоровьем.
    
    Информация по колонкам датасета…
    
- **Оценка важности:** **5/10**
    
- **Форма:** «Вы курите в настоящий момент?» — Да/Нет / «Курили ли вы когда-либо 100 сигарет?» — Да/Нет.
    

---

17. `ALCDAY5` / Alcohol variables — Alcohol consumption frequency

- **Тип/шкала:** Num / categorical (drinks/week or frequency).
    
- **Почему важно:** злоупотребление алкоголем часто связано с тревожностью.
    
- **Оценка важности:** **5/10**
    
- **Форма:** «Сколько стандартных напитков вы употребляете в среднем в неделю?» — число.
    

---

18. `AGE`, `SEX`, `EDUCA`, `INCOME2`, `MARITAL` — демография

- **Тип/шкала:** Num / Categorical (в зависимости).
    
- **Почему важно:** возраст/пол/образование/доход/сем. статус — важные контролирующие переменные; усиливают модели.
    
- **Оценки важности:** AGE **8/10**, SEX **7/10**, EDUCA/INCOME/MARITAL **6/10**
    
- **Форма:** Стандартные демографические вопросы.
    

---

19. `CHRONIC CONDITIONS` (diabetes, heart disease, asthma и т.д.)

- **Тип/шкала:** Binary/Num.
    
- **Почему важно:** хронические болезни повышают риск тревоги.
    
    Информация по колонкам датасета…
    
- **Оценка важности:** **5–7/10** в зависимости от состояния.


### Итог

Я пришел к тому, что меня не устраивает этот датасет. Тут нет подходящей целевой переменной вроде Anxiety Level или GAD-7 results. Поэтому я отказался

Но насчет работы с большими данными у меня появился свой гайд [[Как лучше всего импортировать большие датасеты]]


## Researching Mind-O-Meter

Я проверил распределение и увидел, что с целевой переменной - *AnxietyScore_GAD7*
Нигде (в количественных и номинативных переменных) нет визуальной корреляции.

![[Pasted image 20251002231732.png]]
Примерно такая же картина с корреляцией номинативных переменных. 


## Reseacrhing Anxiety and Indicators - Clean Dataset

В этом датасете находились достаточно скоррелированные данные.
В основном все признаки числовые, кроме как Occupation. 
Для проверки корреляций была построена матрица корреляций. Target - Anxiety Level (1-10)
![[Pasted image 20251003112112.png]]

Построен график boxplot по признаку Occupation:
![[Pasted image 20251003112307.png]]
Значимых различий тут не выявлено.

### Построение модели

Разделил данные точно также как и для Social Anxiety Dataset:
```Python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% данных пойдут в тест
    random_state=42,    # для воспроизводимости
    shuffle=True        # перемешать данные перед разбиением
)
```

Построена регрессионная модель на основе случайного леса, использовался RandomGridSearch и широкая сетка гиперпараметров.
```Python
  rf_regr=RandomForestRegressor(random_state=42 )

param_dist = {
    'n_estimators': np.arange(50,100,10),
    'max_depth': np.arange(2,10,2),
    'min_samples_split': np.arange(2, 16),
    'min_samples_leaf': np.arange(1, 8)
}

from sklearn.model_selection import RandomizedSearchCV
rand_search = RandomizedSearchCV(
    rf_regr,
    param_distributions=param_dist,
    n_iter=40,
    cv=5,
    n_jobs=-1,
    random_state=42,
)
```

```Python
My_best_params=rand_search.best_params_
My_best_params
```

Output:
{'n_estimators': np.int64(90),
 'min_samples_split': np.int64(5),
 'min_samples_leaf': np.int64(3),
 'max_depth': np.int64(8)}

```Python
mae = mean_absolute_error(y_test, Y_predict)
mse = mean_squared_error(y_test, Y_predict)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, Y_predict)
adj_r2_score = 1 - (1 - r2_score(y_test, Y_predict)) * (n - 1) / (n - p - 1)

print(f"🔹 Средняя абсолютная ошибка (MAE): {mae:.3f}")
print(f"🔹 Среднеквадратичная ошибка (MSE): {mse:.3f}")
print(f"🔹 Корень из MSE (RMSE): {rmse:.3f}")
print(f"🔹 Коэффициент детерминации (R²): {r2:.3f}")
print(f"🔹 Скорректированный R²: {adj_r2_score:.3f}")
```

Output:
🔹 Средняя абсолютная ошибка (MAE): 0.675 
🔹 Среднеквадратичная ошибка (MSE): 0.688 
🔹 Корень из MSE (RMSE): 0.830 
🔹 Коэффициент детерминации (R²): 0.851 
🔹 Скорректированный R²: 0.850

# Вопросы на Google Form + GAD-7

## GAD-7

GAD-7 (**шкала генерализованного тревожного расстройства из 7 пунктов**)
Я думаю включить в гугл форму вопросы из этого теста, т.к. это звучит более научно, чем самооценка тревожности от 0 до 10. Однако я оставлю этот признак для исследования, ведь интересно, будут ли в большинстве случаев совпадать баллы тревожности.

Gad-7 даст балл в пределах от 0 до 21, но для корректировки в диапазон от 0 до 10 можно просто поделить результат на 1.05.


**Шкала GAD-7** (Generalized Anxiety Disorder 7-item) состоит из **7 вопросов**, каждый оценивается 0–3 балла (0 = никогда, 1 = несколько дней, 2 = более половины дней, 3 = почти каждый день). Обычно считается в последнее время = за последние 14 дней.
Вопросы такие:

1. Чувство нервозности, тревожности, «на взводе».
    
2. Невозможность остановить или контролировать беспокойство.
    
3. Слишком сильное беспокойство по разным поводам.
    
4. Трудность расслабиться.
    
5. Сильное беспокойство/неусидчивость.
    
6. Лёгкая раздражительность.
    
7. Чувство страха, что что-то плохое может случиться.

## Вопросы для Google Form

- [ ] Создай гугл форму с вопросами ниже. Собери как можно больше опросов среди любых социальных групп.


**Раздел с GAD-7** - Target Value - Anxiety Level 

- Вы сильно нервничали, тревожились или беспокоились о чем-то
	- Ни разу
	- Несколько дней
	- Более половины дней
	- Практически ежедневно

- Вы были неспособны справиться с волнением и успокоиться
	- Ни разу
	- Несколько дней
	- Более половины дней
	- Практически ежедневно

- Вы сильно волновались по разным поводам
	- Ни разу
	- Несколько дней
	- Более половины дней
	- Практически ежедневно

- Вам было трудно расслабиться
	- Ни разу
	- Несколько дней
	- Более половины дней
	- Практически ежедневно

- Вы были настолько неспокойны, что Вам было тяжело усидеть на месте
	- Ни разу
	- Несколько дней
	- Более половины дней
	- Практически ежедневно

- Вы легко злились и раздражались
	- Ни разу
	- Несколько дней
	- Более половины дней
	- Практически ежедневно

- Вы испытывали страх, словно должно произойти что-то плохое
	- Ни разу
	- Несколько дней
	- Более половины дней
	- Практически ежедневно

- По вашему субъективному мнению, как вы оцените свой уровень тревожности от 1 до 10 баллов
	- От 1 до 10 (ответ)

**Раздел с предикторами**

- Ваш пол
	- Мужской
	- Женский
	- Не скажу (мб исключить опрошенных с этим вариантом поможет легче отсеять выбросы)

- Возраст 
	- От 12 до 70 пусть будет

- Часы сна в сутки (в среднем за неделю)
	- от 2 до 14

- Физическая активность (часов в неделю)
	- Число от 0 до 35

- Употребление алкоголя
	- Никогда
	- Реже одного раза в месяц
	- 1–3 раза в месяц
	- 1–2 раза в неделю
	- 3–4 раза в неделю
	- Почти каждый день

- Вы курите?
	- Да
	- Нет

- Качество вашего питания (самооценка)
	- от 1 до 10

- Количество посещений психолога за последний месяц
	- Я не посещаю психолога
	- от 1 до 5



# Вопросы на обсуждение

- [ ] Может быть сделать более глобальную выборку, не только среди студентов, а среди всех просто кого мы сможем собрать? Т.е. друзей-друзей-друзей...

- [x] Почему НАСТОЛЬКО плохо модель предсказывает? 
т.к. это был стресс. Тревога лучше подходит для предсказания субъективного признака, самооценки.
- [ ] Проблема с целевой переменной по датасету Behavioral Risk Factor Surveillance System. Что взять за цель. В идеале конечно Anxiety, от 1 до 10, но такого нет.
- [ ] + может быть можно сделать гугл форму с опросником GAD-7, преобразовать его в целевую переменную, а затем на основе *других* критериев из гугл формы научить модель предсказывать балл по GAD-7. И оставить самооценку тревоги от 1 до 10. Потом сравнить еще самооценку и балл по GAD-7, установить, насколько точно обычно люди оценивают свой стресс.


# Планы на проект

- [x] Определить итоговые датасеты
	- Behavioral Risk Factor Surveillance System
	- Social Anxiety Dataset

- [ ] Построить бейзлайн на них и выбрать лучшие с учетом особенностей исследовательской работы
	- [ ] удаление признаков из секции идентификаторов
	- [ ] Собрать набор обязательных фич. То что напрямую связано с тревогой
	- [ ] Переводи ordinal variables в порядковые числа; one-hot для небольших категорий; агрегируй даты/месяцы если хочешь сезонность
	- [ ] Создать пары/отношения 
	- [ ] Можно (но не обязательно) проверить переменные на мультиколлинеарность (Мультиколлинеарность - наличие линейной зависимости между объясняющими переменными регрессионной модели.) (+Случайный лес устойчив к мультиколлинеарности в плане предсказания, но плохо будет при интерпретации)
	- [ ] Построй бейзлайн
		- [ ] Сформируй сетку параметров для случайного леса
		- [ ] Выбери лучшие параметры
		- [ ] Сохрани лучшие значения и запиши их

- [ ] Очистка данных (базовая)

- [ ] Слияние датасетов