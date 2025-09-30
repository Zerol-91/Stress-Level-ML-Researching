
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

# [General Social Survey](https://www.kaggle.com/datasets/norc/general-social-survey?select=gss.csv)

Данные 9/10

Здесь 10.000 (не наблюдений) **колонок**

Вес 2ГБ

Минус - сложные данные. ОЧЕНЬ много надо убрать.

# [Behavioral Risk Factor Surveillance System](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system)

Данные 9-9.5/10

Данные по 0.5ГБ, разделенные по годам 2011-2015

По разным годам разное кол-во колонок: от 279 до 453


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
![[Pasted image 20250927225357.png]]
Anxiety Level Стал оказался на удивление очень предсказуемым

***Делаю на этом моменте первый комит***
- **Скачать из Colab:** Получить файлы проекта локально (на компьютер).
    
- **Настроить GitHub Desktop:** Подтянуть пустой репозиторий с GitHub.
    
- **Добавить файлы:** Положить скачанные из Colab файлы в папку репозитория.
    
- **Закоммитить и запушить:** Сделать коммит в Desktop и опубликовать на GitHub.

[[Как сделать комит из колаба в гитхаб через десктоп]]


# Вопросы на обсуждение

- [ ] Может быть сделать более глобальную выборку, не только среди студентов, а среди всех просто кого мы сможем собрать? Т.е. друзей-друзей-друзей...

- [x] Почему НАСТОЛЬКО плохо модель предсказывает? 
т.к. это был стресс. Странно, но именно тревога лучше идет


