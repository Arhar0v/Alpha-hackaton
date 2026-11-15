import os
import sys
import asyncio
import logging
from pathlib import Path
import functools
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, StateFilter
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import LinkPreviewOptions
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.enums import ParseMode
import config
from config import settings
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from calendar import monthrange
from date_adaptation import date_adaptation, seasons

import numpy as np

from llm_client import OpenAILikeLLM, InputMessage
from typing import List, Literal, Any
from string import Template

from recomendation import RecomendEmbedModel, FormatCourses
from sentence_transformers import SentenceTransformer, util


### -------------> Настройки <-------------
bot_is_on = True  # по дефолту выключено, т.к. на одного бота может быть не более 1 подключения (1 человек может запускать бота)
ml_emoji = "🚀"
profile_emoji = "💼"
recomendation_emoji = "🎓"
tl_emoji = "📌"
rl_emoji = "⏰"
interval_check_reminders = 60



### --------> Планировщик задач <--------
scheduler = AsyncIOScheduler(timezone="Europe/Moscow")



### -------------> Диспетчер <-------------
bot = Bot(token=settings["token"])
dp = Dispatcher()



### ---------> Подключение к БД <---------
def create_connection():
    connection = psycopg2.connect(
        database=settings["database"],
        user=settings["user"],
        password=settings["password"],
        host=settings["host"]
    )
    connection.autocommit = True
    return connection

@contextmanager
def db_connection():
    connection = create_connection()
    try:
        yield connection
    finally:
        connection.close()

try:
    with db_connection() as connection:
        if connection.closed:
            raise ConnectionError("Соединение закрыто")
        print(f"Подключение к базе данных PostgreSQL установлено")
except Exception as e:
    print(f"Ошибка при подключении к базе данных PostgreSQL:", e)


### ----------> Создание таблиц <----------
with db_connection() as connection:
    with connection.cursor() as cursor:
        cursor.execute("""CREATE TABLE IF NOT EXISTS users(
                    user_id BIGINT PRIMARY KEY,
                    profile VARCHAR(500))""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS chat_history(
                            qa_id SERIAL PRIMARY KEY,
                            user_id BIGINT NOT NULL,
                            question TEXT,
                            answer TEXT,
                            created_at TIMESTAMPTZ DEFAULT NOW())""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS tasks(
                        task_id SERIAL PRIMARY KEY,
                        task VARCHAR(256),
                        user_id BIGINT)""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS reminders(
                        reminder_id SERIAL PRIMARY KEY,
                        task_id INT REFERENCES tasks(task_id) ON DELETE CASCADE,
                        remind_at TIMESTAMPTZ,
                        daily BOOL,
                        yearly BOOLEAN,
                        user_id BIGINT)""")


### ----------------> main <----------------
### ---> Хранение временных данных <---
class User(StatesGroup):  # создание дочернего класса к StatesGroup
    profile = State()


### --------------> Колбэки <--------------
@dp.callback_query(F.data == "set_profile")
async def set_profile_1st_step(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    await state.set_state(User.profile)
    await callback.message.answer("Расскажите о себе, вашем деле, целях и возможных трудностях, с которыми вы столкнулись.\n"
                                  "Ваш текст не должен превышать 500 символов, иначе он будет обрезан.")

@dp.message(User.profile)
async def set_profile_2nd_step(message: Message, state: FSMContext):
    author = message.from_user
    profile_text = message.text
    await state.clear()
    if len(profile_text) > 500:
        profile_text = profile_text[:500]
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM users WHERE user_id = %s", (author.id,))
            is_exists = cursor.fetchone() is not None
            if is_exists:
                cursor.execute("UPDATE users SET profile = %s WHERE user_id = %s", (profile_text, author.id))
            else:
                cursor.execute("INSERT INTO users (user_id, profile) VALUES (%s, %s)", (author.id, profile_text))
    await profile(message)

@dp.callback_query(F.data == "delete_profile")
async def delete_profile_1st_step(callback: CallbackQuery):
    await callback.answer("")
    author = callback.from_user
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM users WHERE user_id = %s", (author.id,))
            is_exists = cursor.fetchone()
            if is_exists is not None:
                cursor.execute("UPDATE users SET profile = NULL WHERE user_id = %s", (author.id,))
    await profile(callback.message, author=author, edit_msg=True)



### --------> Основные команды <--------
@dp.message(Command("start"))
async def start(message: Message):
    author = message.from_user
    member = await get_member(author)
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM users WHERE user_id = %s LIMIT 1", (author.id,))
            if cursor.fetchone() is None:
                cursor.execute(f"INSERT INTO users VALUES (%s, NULL)", (author.id,))
    await message.answer(f"Здравствуйте {member}! Я являюсь персональным ИИ-помощником для предпринимателей от компании Альфа-Банк. \n\n"
                         f"Если вам нужен список моих команд, то используйте /help.\n\n"
                         f"Чтобы мои ответы были точнее и полезнее, для начала расскажите о себе и вашем деле, воспользовавшись командой /profile.\n\n"
                         f"После этого смело продолжайте диалог, я буду отвечать на интересующие вас вопросы.")

@dp.message(Command("help"))
async def help(message: Message):
    await message.answer(f"{ml_emoji} Команды:\n"
                         "/start — регистрация в системе\n"
                         "/help — список команд\n"
                         "/profile — информация о вас и вашем деле\n"
                         "/study — рекомендации по обучающим материалам\n\n"
                         f"{tl_emoji} Команды списка задач:\n"
                         "/task_list (/tl) — отобразить список задач\n"
                         "/add_task (/at) — добавить задачу\n"
                         "/edit_task (/et) — изменить задачу\n"
                         "/delete_task (/dt) — удалить задачу\n"
                         "/clear_task_list (/ctl) — очистить список задач\n\n"
                         f"{rl_emoji} Команды напоминаний:\n"
                         "/reminder_list (/rl) — отобразить список напоминаний у задачи\n"
                         "/add_reminder (/ar) — добавить напоминание к задаче\n"
                         "/edit_reminder (/er) — изменить напоминание в задаче\n"
                         "/delete_reminder (/dr) — удалить напоминание из задачи\n"
                         "/clear_reminder_list (/crl) — очистить список напоминаний у задачи\n\n")


@dp.message(Command("profile"))
async def profile(message: Message, author=None, edit_msg=False):
    if author is None:
        author = message.from_user
    reply_method = message.edit_text if edit_msg else message.answer
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"Редактировать", callback_data="set_profile")],
        [InlineKeyboardButton(text=f"Стереть", callback_data="delete_profile")]
    ])
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT profile FROM users WHERE user_id = %s", (author.id,))
            profile = cursor.fetchone()
    profile = profile[0] if profile is not None and profile[0] is not None else "Отсутствует"
    await reply_method(f"{profile_emoji} Информация о вас и вашем деле:\n"
                       f"`{profile}`", reply_markup=kb, parse_mode=ParseMode.MARKDOWN)

@dp.message(Command("study"))
async def study(message: Message):
    description_courses = []  # описание курсов в формате FormatCourses
    model = RecomendEmbedModel()
    history_rows = await get_history(message.from_user)
    question_history = []
    for row in history_rows:
        if row["role"] == "user":
            question_history.append(row["content"])
    history = "\n".join(question_history)
    with db_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            for course in ['Articles', 'Courses']:
                cursor.execute(f'SELECT * FROM "{course}" LIMIT {config.TOP_K};')
                rows = cursor.fetchall()
                for row in rows:
                    desc_tag = '' if 'DescriptionTag' not in row or row['DescriptionTag'] is None else row['DescriptionTag']
                    tit_navig = '' if 'TitleNavigation' not in row or row['TitleNavigation'] is None else row['TitleNavigation']
                    description_courses.append(FormatCourses(category=course.lower(),
                                                             title=row['Title'],
                                                             description=desc_tag + tit_navig,
                                                             url=f"https://kurs.alfabank.ru/{course.lower()}/{row['Url']}/",
                                                             embedding=model.embed_query(f'{row["Title"]}. {desc_tag + tit_navig}')))
    def get_courses(query: str, items: List[FormatCourses], model: SentenceTransformer, k: int = 3, category: str = None):
        if category:
            items = [i for i in items if i.category == category]
        if not items:
            return []

        matrix = np.vstack([i.embedding for i in items])
        q_emb = model.embed_query(query)
        scores = util.cos_sim(q_emb, matrix)[0]
        top_idx = np.argsort(scores.numpy())[::-1][:k]
        res = [[items[i].title, items[i].description, items[i].category, items[i].url] for i in top_idx]
        return res

    recomendation_text = []
    recomendations = get_courses(history, description_courses, model)
    for i, recomendation in enumerate(recomendations):
        title, description, category, url = recomendation
        recomendation_text.append(f"*{i+1}) {title}:*\n"
                                  f"{description}\n"
                                  f"[Ссылка на материал]({url})")
    await message.answer(f"{recomendation_emoji} Список рекомендуемых образовательных материалов:\n" +
                          "\n\n".join(recomendation_text), parse_mode=ParseMode.MARKDOWN,
                            link_preview_options=LinkPreviewOptions(is_disabled=True))

# @dp.message(Command("show_history"))
# async def show_history(message: Message):
#     author = message.from_user
#     history = await get_history(author)
#     print(history)


# ЗДЕСЬ РАНЬШЕ БЫЛ HANDLE_ANY_TEXT()




### ----> Вспомогательные функции <----
async def get_member(author):
    first_name = author.first_name if author.first_name else ""
    last_name = author.last_name if author.last_name else ""
    username = author.username
    sep = "" if first_name == "" or last_name == "" else " "
    member = f"@{username}" if username else f"{first_name}{sep}{last_name}"  # обращение к пользователю
    return member


async def get_history(author):  # история диалога с ИИ-помощником
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT question, answer FROM chat_history WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
                (author.id, config.N_MSG_HISTORY))
            history_rows = cursor.fetchall()
            history_rows = history_rows[::-1]
    history = []
    for i, row in enumerate(history_rows):
        history.append({"role": "user", "content": row[0]})
        history.append({"role": "assistant", "content": row[1]})
    return history

async def generate_answer(query: str, context, user_profile: str = None, history: list = None):
    llm = OpenAILikeLLM(
        base_url='https://api.groq.com/openai/v1',
        api_key=config.GROQ_API_KEY,
        model='openai/gpt-oss-120b',
        common_parameters={'temperature': 0}
    )
    # system_prompt = "Ты полезный ассистент. Используй русский язык для ответа, отвечай строго с использованием контекста, " \
    #                 "если ответа нет, то скажи, что не можешь ответить, придумывать ничего не надо."
    system_prompt = (
        "Ты полезный ИИ-ассистент. Используй русский язык для ответа.\n"
        "ПРАВИЛА ОТВЕТА:\n"
        "1. Если вопрос касается фактов, бизнеса, услуг или терминов — отвечай СТРОГО на основе предоставленного 'Контекста'. "
        "Если ответа там нет, скажи, что не можешь ответить по этому вопросу.\n"
        "2. Если вопрос касается вашего диалога (например, 'о чем мы говорили?', 'что я спрашивал ранее?', 'повтори свой ответ') — "
        "ОТВЕЧАЙ, ИСПОЛЬЗУЯ ИСТОРИЮ СООБЩЕНИЙ (предыдущие сообщения в диалоге), даже если этой информации нет в 'Контексте'.\n"
        "3. Не придумывай факты, которых нет в Контексте."
    )

    if user_profile:
        system_prompt += f"\n\nИНФОРМАЦИЯ О ПОЛЬЗОВАТЕЛЕ (учитывай её для персонализации ответа, "\
                         f"но не бери отсюда факты, если их нет в контексте):\n{user_profile}"

    messages = [
        InputMessage(role="system", content=system_prompt)
    ]

    if history:
        for msg in history:
            messages.append(InputMessage(role=msg["role"], content=msg["content"]))

    USER_PROMPT = Template("Вопрос: $query\n\nКонтекст (единственные источники правды):\n$context")
    user_prompt = USER_PROMPT.substitute({"query": query, "context": "\n\n".join(context)})
    messages.append(InputMessage(role="user", content=user_prompt))


    answer = await llm.generate_answer(messages)
    return answer.content





### ---------------> task_list <---------------
### ---> Хранение временных данных <---
class New_task(StatesGroup):  # создание дочернего класса к StatesGroup
    task_id = State()
    text = State()
    edit_text = State()
    rl = State()  # пропустить часть функционала для /rl
    ar = State()  # пропустить часть функционала для /ar
    er = State()  # пропустить часть функционала для /er
    dr = State()  # пропустить часть функционала для /dr
    crl = State()  # пропустить часть функционала для /crl
    back_to_edit_reminder = State()  # для add_reminder
    reminder_id = State()  # для edit_reminder и delete_reminder
    year = State()
    month = State()
    day = State()
    hour = State()
    minute = State()
    unit = State()
    i_unit = State()



### ---------------> Кнопки <---------------
tlkb = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="Добавить задачу", callback_data="add_task"),
     InlineKeyboardButton(text="Редактировать задачу", callback_data="edit_task")],
    [InlineKeyboardButton(text="Удалить задачу", callback_data="delete_task"),
     InlineKeyboardButton(text="Очистить список задач", callback_data="clear_task_list")]
])

reminder_keyboard_first_step = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="Да", callback_data="add_reminder"),
     InlineKeyboardButton(text="Нет", callback_data="dont_add_reminder")]
])

timer_keyboard_first_step = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="Минута", callback_data="add_reminder_timer_unit_minute"),
     InlineKeyboardButton(text="Час", callback_data="add_reminder_timer_unit_hour"),
     InlineKeyboardButton(text="День", callback_data="add_reminder_timer_unit_day")],
    [InlineKeyboardButton(text="Неделя", callback_data="add_reminder_timer_unit_week"),
     InlineKeyboardButton(text="Месяц", callback_data="add_reminder_timer_unit_month"),
     InlineKeyboardButton(text="Год", callback_data="add_reminder_timer_unit_year")],
    [InlineKeyboardButton(text="« Назад", callback_data="add_reminder")]
])


### --------------> Колбэки <--------------
@dp.callback_query(F.data == "add_task")  # после нажатия кнопки "Добавить задачу"
async def add_task_first_step(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    await state.set_state(New_task.text)  # объявление о готовности начала записи текста в поле text у класса New_task
    await callback.message.answer("Напишите новое задание")

@dp.message(New_task.text)  # записывает в поле у класса New_task text текст пользователя
async def add_task_second_step(message: Message, state: FSMContext):
    await state.update_data(text=message.text)
    text = message.text
    author = message.from_user
    if 1 <= len(text) <= 256:
        with db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("INSERT INTO tasks (task, user_id) VALUES (%s, %s) RETURNING task_id", (text, author.id))
                task_id = cursor.fetchone()[0]
            await state.set_state(New_task.task_id)
            await state.update_data(task_id=task_id)
        await message.answer("Добавить напоминание об этом задании?", reply_markup=reminder_keyboard_first_step)
    else:
        await message.answer("Ваше задание должно быть описано в пределах от 1 до 256 символов включительно")
        await state.clear()  # так как точка выхода без продолжения

@dp.callback_query(F.data == "add_task_back_to_second_step")
async def add_task_back_to_second_step(callback: CallbackQuery):
    await callback.message.edit_text("Добавить напоминание об этом задании?", reply_markup=reminder_keyboard_first_step)

@dp.callback_query(F.data == "add_reminder")
async def add_task_third_step_yes(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    data = await state.get_data()
    if "reminder_id" not in data.keys():
        if "back_to_edit_reminder" not in data.keys():
            if "ar" not in data.keys():
                callback_text = "add_task_back_to_second_step"
            else:
                callback_text = "edit_task"
        else:
            callback_text = "edit_task.reminder"
    else:
        callback_text = "edit_task.reminder/edit_reminder"
    if {"year", "month", "day", "hour", "minute", "unit", "i_unit"} & data.keys():
        await state_remove_keys(state, ["year", "month", "day", "hour", "minute", "unit", "i_unit"])
    # data1 = await state.get_data()                                  # отладка
    # print(f"При возврате в блок add_reminder:\n{data1}")            # отладка
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Дата и время", callback_data="add_reminder_datetime"),
         InlineKeyboardButton(text="Таймер", callback_data="add_reminder_timer")],
        [InlineKeyboardButton(text="Ежедневное", callback_data="add_daily_task"),
         InlineKeyboardButton(text="« Назад", callback_data=f"{callback_text}")]
    ])
    await callback.message.edit_text("Какой вид напоминания добавить?", reply_markup=keyboard)

@dp.callback_query(F.data == "add_reminder_datetime")
async def add_task_fourth_step_yes_datetime_year(callback: CallbackQuery, state: FSMContext, l_year=None):  # альтернативная точка входа для добавления напоминания в блоке edit_task
    await callback.answer("")
    l_limit = datetime.today().year
    r_limit = 2150
    columns = 6
    years = columns - 2
    keyboard = InlineKeyboardBuilder()
    if l_year is None:  # первый вывод клавиатуры с нынешнего года
        l_year = l_limit
    if l_year == l_limit:  # если левый год соответствует нынешнему году, то колбэк скипается
        keyboard.add(InlineKeyboardButton(text="<<", callback_data=f"skip_callback"))
    else:
        keyboard.add(InlineKeyboardButton(text="<<", callback_data=f"add_reminder/change_year_<<{years}{l_year}"))
    for i in range(years):  # заполнение клавиатуры годами
        year = l_year + i
        keyboard.add(InlineKeyboardButton(text=f"{year}", callback_data=f"add_year_{year}"))
    if l_year + years < r_limit:  # встраивание заглушки при превышении лимита справа
        keyboard.add(InlineKeyboardButton(text=">>", callback_data=f"add_reminder/change_year_>>{years}{l_year}"))
    else:
        keyboard.add(InlineKeyboardButton(text=">>", callback_data=f"skip_callback"))
    keyboard.add(InlineKeyboardButton(text="« Назад", callback_data="add_reminder"))
    keyboard_year = keyboard.adjust(columns).as_markup()
    await callback.message.edit_text("Укажите год:", reply_markup=keyboard_year)

@dp.callback_query(F.data.startswith("add_reminder/change_year_"))
async def add_task_fourth_step_yes_datetime_change_year(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    years = int(callback.data[27:28])
    l_year = int(callback.data[28:])
    if callback.data[25:27] == ">>":
        await add_task_fourth_step_yes_datetime_year(callback, state, l_year + years)
    if callback.data[25:27] == "<<":
        await add_task_fourth_step_yes_datetime_year(callback, state, l_year - years)

@dp.callback_query(F.data.startswith("add_year_"))
async def add_task_fifth_step_yes_datetime_month(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    year = int(callback.data[9:])
    await state.set_state(New_task.year)
    await state.update_data(year=year)
    keyboard = InlineKeyboardBuilder()
    if year > datetime.today().year:
        years = 12
    else:
        years = 13 - datetime.today().month
    for i in range(years):  # доступные месяцы
        if years != 12:
            new_date = datetime.today() + relativedelta(months=i)  # новое время с прибавлением числа равного i к месяцу
        else:
            new_date = date(year, i + 1, 1)
        keyboard.add(InlineKeyboardButton(text=f"{seasons[new_date.strftime('%B')]}", callback_data=f"add_month_{new_date.month}"))
    keyboard.add(InlineKeyboardButton(text="« Назад", callback_data="add_reminder_datetime"))
    keyboard_month = keyboard.adjust(4).as_markup()
    await callback.message.edit_text("Укажите месяц:", reply_markup=keyboard_month)

@dp.callback_query(F.data.startswith("add_month_"))
async def add_task_sixth_step_yes_datetime_day(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    data = await state.get_data()
    year = data["year"]
    month = int(callback.data[10:])
    await state.set_state(New_task.month)
    await state.update_data(month=month)
    keyboard = InlineKeyboardBuilder()
    days = monthrange(year, month)[1]
    first_day = 1
    last_days = days
    if month == datetime.today().month and year == datetime.today().year:  # проверка на текущий месяц
        first_day = datetime.today().day
        last_days = days + 1 - datetime.today().day
    for i in ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]:  # Заполнение метками
        keyboard.add(InlineKeyboardButton(text=f"{i}", callback_data="skip_callback"))
    for i in range(date(year, month, first_day).weekday()):  # заполнение пустыми ячейками до возможных дней
        keyboard.add(InlineKeyboardButton(text=" ", callback_data="skip_callback"))
    for i in range(last_days):  # возможные для выбора дни
        day = first_day + i
        keyboard.add(InlineKeyboardButton(text=f"{day}", callback_data=f"add_day_{day}"))
    for i in range(6 - date(year, month, day).weekday()):  # заполнение пустыми ячейками после возможных дней
        keyboard.add(InlineKeyboardButton(text=" ", callback_data="skip_callback"))
    keyboard.add(InlineKeyboardButton(text="« Назад", callback_data=f"add_year_{year}"))
    keyboard_day = keyboard.adjust(7).as_markup()
    await callback.message.edit_text("Укажите день:", reply_markup=keyboard_day)

@dp.callback_query(F.data.startswith("add_day_"))
async def add_task_fifth_step_yes_datetime_hour(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    data = await state.get_data()
    year = data["year"]
    month = data["month"]
    day = int(callback.data[8:])
    await state.set_state(New_task.day)
    await state.update_data(day=day)
    keyboard = InlineKeyboardBuilder()
    columns = 8
    first_hour = 0
    hours = 24
    if datetime.today().year == year and datetime.today().month == month and datetime.today().day == day:
        first_hour = datetime.today().hour
        hours = hours - first_hour
    for i in range(first_hour % columns):
        keyboard.add(InlineKeyboardButton(text=" ", callback_data="skip_callback"))
    for i in range(hours):  # возможные для выбора часы
        hour = first_hour + i
        keyboard.add(InlineKeyboardButton(text=f"{hour}", callback_data=f"add_hour_{hour}"))
    keyboard.add(InlineKeyboardButton(text="« Назад", callback_data=f"add_month_{month}"))
    keyboard_hour = keyboard.adjust(columns).as_markup()
    await callback.message.edit_text("Выберите час:", reply_markup=keyboard_hour)

@dp.callback_query(F.data.startswith("add_hour_"))
async def add_task_fifth_step_yes_datetime_minute(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    data = await state.get_data()
    year = data["year"]
    month = data["month"]
    day = data["day"]
    hour = int(callback.data[9:])
    await state.set_state(New_task.hour)
    await state.update_data(hour=hour)
    keyboard = InlineKeyboardBuilder()
    columns = 8
    first_minute = 0
    minutes = 60
    if datetime.today().year == year and datetime.today().month == month and datetime.today().day == day and datetime.today().hour == hour:
        first_minute = datetime.today().minute
        minutes = minutes - first_minute
    for i in range(minutes):  # доступные для выбора минуты
        minute = first_minute + i
        keyboard.add(InlineKeyboardButton(text=f"{minute}", callback_data=f"add_minute_{minute}"))
    keyboard.add(InlineKeyboardButton(text="« Назад", callback_data=f"add_day_{day}"))
    keyboard_minute = keyboard.adjust(columns).as_markup()
    await callback.message.edit_text("Выберите минуту:", reply_markup=keyboard_minute)

@dp.callback_query(F.data.startswith("add_minute_"))
async def add_task_sixth_step_yes_datetime_insert(callback: CallbackQuery, state: FSMContext):
    minute = int(callback.data[11:])
    await state.set_state(New_task.minute)
    await state.update_data(minute=minute)
    await reminder_insert_update(callback = callback, state = state, insert_type = "datetime")

@dp.callback_query(F.data == "add_reminder_timer")
async def add_task_fourth_step_yes_timer(callback: CallbackQuery):
    await callback.answer("")
    await callback.message.edit_text("Выберите единицу измерения таймера:", reply_markup=timer_keyboard_first_step)

@dp.callback_query(F.data.startswith("add_reminder_timer_unit_"))
async def add_task_fifth_step_yes_timer_unit(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    unit = callback.data[24:]
    await state.set_state(New_task.unit)
    await state.update_data(unit=unit)
    keyboard = InlineKeyboardBuilder()
    size = 28
    columns = 7
    text = date_adaptation(-1, unit)
    for i in range(size):
        keyboard.add(InlineKeyboardButton(text=f"{i + 1}", callback_data=f"add_reminder_timer_number_{i + 1}"))
    keyboard.add(InlineKeyboardButton(text="« Назад", callback_data="add_reminder_timer"))
    keyboard_unit = keyboard.adjust(columns).as_markup()
    await callback.message.edit_text(f"Укажите количество {text}:", reply_markup=keyboard_unit)

@dp.callback_query(F.data.startswith("add_reminder_timer_number_"))
async def add_task_fifth_step_yes_timer_insert(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    i_unit = int(callback.data[26:])
    await state.set_state(New_task.i_unit)
    await state.update_data(i_unit=i_unit)
    await reminder_insert_update(callback=callback, state=state, insert_type="timer")

@dp.callback_query(F.data == "add_daily_task")
async def add_daily_task_first_step(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    keyboard = InlineKeyboardBuilder()
    columns = 8
    first_hour = 0
    hours = 24
    for i in range(hours):  # возможные для выбора часы
        hour = first_hour + i
        keyboard.add(InlineKeyboardButton(text=f"{hour}", callback_data=f"add_daily_hour_{hour}"))
    keyboard.add(InlineKeyboardButton(text="« Назад", callback_data="add_reminder"))
    keyboard_hour = keyboard.adjust(columns).as_markup()
    await callback.message.edit_text("Выберите час:", reply_markup=keyboard_hour)

@dp.callback_query(F.data.startswith("add_daily_hour_"))
async def add_daily_task_second_step(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    hour = int(callback.data[15:])
    await state.update_data(hour=hour)
    keyboard = InlineKeyboardBuilder()
    columns = 8
    first_minute = 0
    minutes = 60
    for i in range(minutes):  # возможные для выбора минуты
        minute = first_minute + i
        keyboard.add(InlineKeyboardButton(text=f"{minute}", callback_data=f"add_daily_minute_{minute}"))
    keyboard.add(InlineKeyboardButton(text="« Назад", callback_data="add_daily_task"))
    keyboard_minute = keyboard.adjust(columns).as_markup()
    await callback.message.edit_text("Выберите минуту:", reply_markup=keyboard_minute)

@dp.callback_query(F.data.startswith("add_daily_minute_"))
async def add_daily_task_third_step(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    minute = int(callback.data[17:])
    await state.set_state(New_task.minute)
    await state.update_data(minute=minute)
    await reminder_insert_update(callback=callback, state=state, insert_type="daily")

@dp.callback_query(F.data == "dont_add_reminder")
async def add_task_third_step_no(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    await state.clear()  # так как возврат к task_list
    await task_list(callback.message, callback.from_user, edit_msg=True)

@dp.callback_query(F.data == "edit_task")
async def edit_task_first_step(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    await edit_task(callback.message, state=state, author=callback.from_user, edit_msg=True)

@dp.callback_query(F.data.startswith("edit_task_"))  # редактор задачи (Кнопки: Текст/Напоминание/«Назад)
async def edit_task_second_step(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    task_id = int(callback.data[10:])
    await state.set_state(New_task.task_id)
    await state.update_data(task_id=task_id)
    data = await state.get_data()
    if "rl" in data.keys():
        await edit_task_reminder_third_step(callback, state)
    elif "ar" in data.keys():
        await add_task_third_step_yes(callback, state)
    elif "er" in data.keys():
        await edit_task_edit_reminder_fourth_step(callback, state)
    elif "dr" in data.keys():
        await edit_task_delete_reminder_fourth_step(callback, state)
    elif "crl" in data.keys():
        await edit_task_clear_reminder_list(callback, state)
    else:
        with db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT task FROM tasks WHERE task_id = %s", (task_id,))
                task = cursor.fetchone()[0]
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Текст", callback_data="edit_task.text"),
             InlineKeyboardButton(text="Напоминание", callback_data="edit_task.reminder")],
            [InlineKeyboardButton(text="« Назад", callback_data="edit_task")]
        ])
        await callback.message.edit_text(f"Задача: {task}\n\nЧто редактировать в задаче?", reply_markup=keyboard)

@dp.callback_query(F.data =='edit_task.text')  # подготовка к сохранению нового текста у задачи
async def edit_task_edit_text_third_step(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    await state.set_state(New_task.edit_text)
    await callback.message.edit_text("Напишите новое задание")

@dp.message(New_task.edit_text)  # изменение текста у задачи
async def edit_task_edit_text_fourth_step(message: Message, state: FSMContext):
    await state.update_data(edit_text=message.text)
    data = await state.get_data()
    await state.clear()  # так как возврат к task_list
    author = message.from_user
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("UPDATE tasks SET task = %s WHERE task_id = %s and user_id = %s", (data["edit_text"], data["task_id"], author.id))
    await task_list(message)

@dp.callback_query(F.data =="edit_task.reminder")  # вывод всех напоминаний у задачи (кнопки: Добавить/Редактировать/Удалить/Очистить список/«Назад)
async def edit_task_reminder_third_step(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    data = await state.get_data()
    task_id = data["task_id"]
    if "back_to_edit_reminder" in data.keys():
        await state_remove_keys(state, ["task_id"], all_without=True)
    # print(f"edit_task.reminder (только task_id, rl, ar, er, dr, crl):\n{data}")  # отладка
    if {"rl", "ar", "er", "dr", "crl"} & data.keys():
        callback_text = "edit_task"
    else:
        callback_text = f"edit_task_{task_id}"
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT task FROM tasks WHERE task_id = %s", (task_id,))
            task = cursor.fetchone()[0]
            cursor.execute("SELECT reminder_id, remind_at, daily FROM reminders WHERE task_id = %s ORDER BY remind_at", (task_id,))
            reminders_tuple = cursor.fetchall()
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Добавить", callback_data="edit_task.reminder/add_reminder"),
         InlineKeyboardButton(text="Редактировать", callback_data="edit_task.reminder/edit_reminder")],
        [InlineKeyboardButton(text="Удалить", callback_data="edit_task.reminder/delete_reminder"),
         InlineKeyboardButton(text="Очистить список", callback_data="edit_task.reminder/clear_reminder_list")],
        [InlineKeyboardButton(text="« Назад", callback_data=f"{callback_text}")]
    ])
    if reminders_tuple:
        reminders = []
        for i in range(len(reminders_tuple)):
            if bool(reminders_tuple[i][2]):
                text = f"{i + 1}) {reminders_tuple[i][1].strftime('%H:%M')} — ежедневное"
            else:
                text = f"{i + 1}) {reminders_tuple[i][1].strftime('%d.%m.%y %H:%M')}"
            reminders.append(text)
        await callback.message.edit_text(f"{rl_emoji} Список напоминаний для задания `{task}`:\n" + "\n".join(reminders), reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
    else:
        await callback.message.edit_text(f"{rl_emoji} У данного задания нет напоминаний", reply_markup=keyboard)

@dp.callback_query(F.data == "edit_task.reminder/add_reminder")  # добавление новой задачи
async def edit_task_add_reminder_fourth_step(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    await state.set_state(New_task.back_to_edit_reminder)
    await state.update_data(back_to_edit_reminder=True)
    await add_task_third_step_yes(callback=callback, state=state)

@dp.callback_query(F.data == "edit_task.reminder/edit_reminder")
async def edit_task_edit_reminder_fourth_step(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    data = await state.get_data()
    task_id = data["task_id"]
    if "reminder_id" in data.keys():
        await state_remove_keys(state, ["task_id", "er"], all_without=True)
    if "er" in data.keys():
        callback_text = "edit_task"
    else:
        callback_text = "edit_task.reminder"
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT task FROM tasks WHERE task_id = %s", (task_id,))
            task = cursor.fetchone()[0]
            cursor.execute("SELECT reminder_id, remind_at, daily FROM reminders WHERE task_id = %s ORDER BY remind_at", (task_id,))
            reminders_tuple = cursor.fetchall()
    keyboard = InlineKeyboardBuilder()
    reminders = []
    for i in range(len(reminders_tuple)):
        if bool(reminders_tuple[i][2]):
            text = f"{i + 1}) {reminders_tuple[i][1].strftime('%H:%M')} — ежедневное"
        else:
            text = f"{i + 1}) {reminders_tuple[i][1].strftime('%d.%m.%y %H:%M')}"
        reminders.append(text)
        keyboard.add(InlineKeyboardButton(text=f"{i + 1}", callback_data=f"edit_task.reminder/edit_reminder_{reminders_tuple[i][0]}"))
    keyboard.add(InlineKeyboardButton(text=f"« Назад", callback_data=callback_text))
    await callback.message.edit_text(f"{rl_emoji} Список напоминаний для задания `{task}`:\n" + "\n".join(reminders) + "\n\nВыберите номер редактируемого напоминания:", reply_markup=keyboard.adjust(5).as_markup(), parse_mode=ParseMode.MARKDOWN)

@dp.callback_query(F.data.startswith("edit_task.reminder/edit_reminder_"))
async def edit_reminder_task_fifth_step(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    await state.set_state(New_task.reminder_id)
    await state.update_data(reminder_id = int(callback.data[33:]))
    await add_task_third_step_yes(callback = callback, state = state)

@dp.callback_query(F.data == "edit_task.reminder/delete_reminder")
async def edit_task_delete_reminder_fourth_step(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    data = await state.get_data()
    task_id = data["task_id"]
    if "dr" in data.keys():
        callback_text = "edit_task"
    else:
        callback_text = "edit_task.reminder"
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT task FROM tasks WHERE task_id = %s", (task_id,))
            task = cursor.fetchone()[0]
            cursor.execute("SELECT reminder_id, remind_at FROM reminders WHERE task_id = %s ORDER BY remind_at", (task_id,))
            reminders_tuple = cursor.fetchall()
    keyboard = InlineKeyboardBuilder()
    reminders = []
    for i in range(len(reminders_tuple)):
        if bool(reminders_tuple[i][2]):
            text = f"{i + 1}) {reminders_tuple[i][1].strftime('%H:%M')} — ежедневное"
        else:
            text = f"{i + 1}) {reminders_tuple[i][1].strftime('%d.%m.%y %H:%M')}"
        reminders.append(text)
        keyboard.add(InlineKeyboardButton(text=f"{i + 1}", callback_data=f"edit_task.reminder/delete_reminder_{reminders_tuple[i][0]}"))
    keyboard.add(InlineKeyboardButton(text=f"« Назад", callback_data=callback_text))
    text = f"{rl_emoji} Список напоминаний для задания `{task}`:\n" + "\n".join(reminders) + "\n\nВыберите номер удаляемого напоминания:"
    await callback.message.edit_text(f"{text}", reply_markup=keyboard.adjust(5).as_markup(), parse_mode=ParseMode.MARKDOWN)

@dp.callback_query(F.data.startswith("edit_task.reminder/delete_reminder_"))
async def edit_task_delete_datetime_fifth_step(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    reminder_id = int(callback.data[35:])
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM reminders WHERE reminder_id = %s", (reminder_id,))
    await edit_task_reminder_third_step(callback = callback, state = state)

@dp.callback_query(F.data == "edit_task.reminder/clear_reminder_list")
async def edit_task_clear_reminder_list(callback: CallbackQuery, state: FSMContext):
    await callback.answer("")
    data = await state.get_data()
    task_id = data["task_id"]
    if callback.message.text != f"{rl_emoji} У данного задания нет напоминаний":
        with db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM reminders WHERE task_id = %s", (task_id,))
        await edit_task_reminder_third_step(callback=callback, state=state)

@dp.callback_query(F.data == "delete_task")
async def delete_task_first_step(callback: CallbackQuery):
    await callback.answer("")
    await delete_task(callback.message, author=callback.from_user, edit_msg=True)

@dp.callback_query(F.data.startswith("delete_task_"))
async def delete_task_second_step(callback: CallbackQuery):
    await callback.answer("")
    task_id = int(callback.data[12:])
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"DELETE FROM tasks WHERE task_id = {task_id}")
    await task_list(callback.message, callback.from_user, edit_msg=True)

@dp.callback_query(F.data == "clear_task_list")
async def clear_task_list_first_step(callback: CallbackQuery):
    await callback.answer("")
    await clear_task_list(message=callback.message, author=callback.from_user, edit_msg=True)

@dp.callback_query(F.data == "skip_callback")
async def callback_skip(callback: CallbackQuery):
    await callback.answer("")

@dp.callback_query(F.data == "back_to_tl")
async def back_to_task_list(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await task_list(callback.message, author=callback.from_user, edit_msg=True)

@dp.callback_query(F.data.startswith("reminder_delete_task_"))
async def reminder_delete_task(callback: CallbackQuery):
    await callback.answer("")
    task_id = int(callback.data[21:])
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM tasks WHERE task_id = %s", (task_id,))
    await task_list(callback.message, author=callback.from_user, edit_msg=True)

@dp.callback_query(F.data == "reminder_dont_delete_task")
async def reminder_dont_delete_task(callback: CallbackQuery):
    await callback.answer("")
    await task_list(callback.message, author=callback.from_user, edit_msg=True)


### ----> Вспомогательные функции <----
async def get_member(author):
    first_name = author.first_name if author.first_name else ""
    last_name = author.last_name if author.last_name else ""
    username = author.username
    sep = "" if first_name == "" or last_name == "" else " "
    member = f"@{username}" if username else f"{first_name}{sep}{last_name}"  # обращение к пользователю
    return member

async def reminder_insert_update(callback: CallbackQuery, state: FSMContext, insert_type):
    data = await state.get_data()
    # print(f"insert_update до очистки:\n{data}")
    if {"back_to_edit_reminder", "reminder_id", "year", "month", "day", "hour", "minute", "unit", "i_unit"} & data.keys():
        await state_remove_keys(state, ["back_to_edit_reminder", "reminder_id", "year", "month", "day", "hour", "minute", "unit", "i_unit"])
    # data1 = await state.get_data()                             # отладка
    # print(f"insert_update после очистки:\n{data1}")            # отладка
    task_id = data["task_id"]
    await callback.answer("")
    if insert_type == "datetime":
        current_datetime = datetime(data["year"], data["month"], data["day"], data["hour"], data["minute"])
        text = f'Напоминание установлено на {data["day"]:02}.{data["month"]:02}.{data["year"]} {data["hour"]:02}:{data["minute"]:02}'
        daily = False
        yearly = False
    elif insert_type == "timer":
        unit = data["unit"]
        i = data["i_unit"]
        units = {"minute": relativedelta(minutes=i),
                 "hour": relativedelta(hours=i),
                 "day": relativedelta(days=i),
                 "week": relativedelta(weeks=i),
                 "month": relativedelta(months=i),
                 "year": relativedelta(years=i)}
        current_datetime = datetime.today() + units[unit]
        text = f"Таймер сработает через {date_adaptation(i, unit)}"
        daily = False
        yearly = False
    elif insert_type == "daily":
        hour = data["hour"]
        minute = data["minute"]
        current_datetime = datetime(datetime.today().year, datetime.today().month, datetime.today().day, hour, minute)
        if current_datetime < datetime.today():
            current_datetime += relativedelta(days=1)
        text = f"Ежедневное напоминание установлено на {hour:02}:{minute:02}"
        daily = True
        yearly = False
    if "reminder_id" not in data.keys():
        with db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("INSERT INTO reminders (task_id, remind_at, daily, yearly, user_id) VALUES (%s, %s, %s, %s, %s)",
                    (task_id, current_datetime, daily, yearly, callback.from_user.id))
        if "back_to_edit_reminder" not in data.keys() and "ar" not in data.keys():
            await state.clear()  # так как возврат к task_list
            await task_list(callback.message, callback.from_user, edit_msg=True)
            await callback.message.answer(f"{text}")
        else:
            await edit_task_reminder_third_step(callback=callback, state=state)
    else:
        reminder_id = data["reminder_id"]
        with db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("UPDATE reminders SET task_id = %s, remind_at = %s, daily = %s, yearly = %s, user_id = %s WHERE reminder_id = %s",
                    (task_id, current_datetime, daily, yearly, callback.from_user.id, reminder_id))
        await edit_task_reminder_third_step(callback=callback, state=state)

async def state_remove_keys(state: FSMContext, keys: list, all_without=False):
    data = await state.get_data()
    if all_without is False:
        await state.set_data({k: v for k, v in data.items() if k not in keys})
    else:
        await state.set_data({k: v for k, v in data.items() if k in keys})


### ------> Работа планировщика <------
async def check_reminders():  # обработчик
    try:
        with db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT reminder_id, task_id, remind_at, daily, yearly, user_id FROM reminders WHERE remind_at <= NOW()")
                reminders = cursor.fetchall()
                if reminders is not None:
                    for remind in reminders:
                        reminder_id, task_id, remind_at, daily, yearly, user_id = remind[0], remind[1], remind[2],\
                                                                                  bool(remind[3]), bool(remind[4]), remind[5]
                        cursor.execute("SELECT task FROM tasks WHERE task_id = %s", (task_id,))
                        task = cursor.fetchone()[0]
                        if daily:
                            await bot.send_message(user_id, f"{task}")
                            next_data = remind_at + relativedelta(days=1)
                            cursor.execute("UPDATE reminders SET remind_at = %s WHERE reminder_id = %s",
                                           (next_data, reminder_id))
                        elif yearly:
                            cursor.execute("SELECT task FROM tasks WHERE task_id = %s", (task_id,))
                            await bot.send_message(user_id, f"{task}")
                            next_data = remind_at + relativedelta(years=1)
                            cursor.execute("UPDATE reminders SET remind_at = %s WHERE reminder_id = %s", (next_data, reminder_id))
                        else:
                            reminder_keyboard_delete_task = InlineKeyboardMarkup(inline_keyboard=[
                                [InlineKeyboardButton(text="Да", callback_data=f"reminder_delete_task_{task_id}"),
                                 InlineKeyboardButton(text="Нет", callback_data="reminder_dont_delete_task")]
                            ])
                            await bot.send_message(user_id, f"Напоминание на {remind_at.hour:02}:{remind_at.minute:02}:\n{task}\n\n"
                                                   f"Удалить задание?", reply_markup=reminder_keyboard_delete_task)
                            cursor.execute("DELETE FROM reminders WHERE reminder_id = %s", (reminder_id,))
    except Exception as e:
        print(f"Ошибка в блоке check_reminders: {e}")


### --------> Основные команды <--------
@dp.message(Command(commands=["task_list", "tl"]))
async def task_list(message: Message, author=None, edit_msg=False):
    if author is None:
        author = message.from_user
    reply_method = message.edit_text if edit_msg is True else message.answer
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT task FROM tasks WHERE user_id = %s ORDER BY task_id", (author.id,))
            task_tuple = cursor.fetchall()
    if task_tuple:
        task_list = []
        for i in range(len(task_tuple)):
            task_list.append(f"{i + 1}) {task_tuple[i][0]}")
        await reply_method(f"{tl_emoji} Список задач:\n" + '\n'.join(task_list), reply_markup=tlkb)
    else:
        await reply_method(f"{tl_emoji} Ваш список задач пуст", reply_markup=tlkb)

@dp.message(Command(commands=["add_task", "at"]))
async def add_task(message: Message, state: FSMContext):
    await state.set_state(New_task.text)  # объявление о готовности начала записи текста в поле text у класса New_task
    await message.answer("Напишите новое задание")

@dp.message(Command(commands=["edit_task", "et"]))
async def edit_task(message: Message, state: FSMContext, author=None, edit_msg=False):
    if author is None:
        author = message.from_user
    reply_method = message.edit_text if edit_msg is True else message.answer
    data = await state.get_data()
    if "rl" in data.keys():
        lower_text = "Выберите номер задачи, у которой нужно отобразить все напоминания:"
    elif "ar" in data.keys():
        lower_text = "Выберите номер задачи, к которой нужно добавить напоминание:"
    elif "er" in data.keys():
        lower_text = "Выберите номер задачи, у которой нужно редактировать напоминание:"
    elif "dr" in data.keys():
        lower_text = "Выберите номер задачи, у которой нужно удалить напоминание:"
    elif "crl" in data.keys():
        lower_text = "Выберите номер задачи, у которой нужно удалить все напоминания:"
    else:
        lower_text = "Выберите номер редактируемой задачи:"
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT task_id, task FROM tasks WHERE user_id = %s ORDER BY task_id", (author.id,))
            task_tuple = cursor.fetchall()
    if task_tuple:
        task_list = []
        keyboard = InlineKeyboardBuilder()
        for i in range(len(task_tuple)):
            task_list.append(f"{i + 1}) {task_tuple[i][1]}")  # форматирование списка задач
            keyboard.add(InlineKeyboardButton(text=f"{i + 1}", callback_data=f"edit_task_{task_tuple[i][0]}"))
        keyboard.add(InlineKeyboardButton(text=f"« Назад", callback_data="back_to_tl"))
        keyboard_et = keyboard.adjust(5).as_markup()  # форматирование позиций кнопок
        await reply_method(f"{tl_emoji} Список задач:\n" + '\n'.join(task_list) + "\n\n" + lower_text, reply_markup=keyboard_et)
    else:
        await reply_method(f"{tl_emoji} Ваш список задач пуст")

@dp.message(Command(commands=["delete_task", "dt"]))
async def delete_task(message: Message, author=None, edit_msg=False):
    if author is None:
        author = message.from_user
    reply_method = message.edit_text if edit_msg is True else message.answer
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT task_id, task FROM tasks WHERE user_id = %s ORDER BY task_id", (author.id,))
            task_tuple = cursor.fetchall()
    if task_tuple:
        task_list = []
        keyboard = InlineKeyboardBuilder()
        for i in range(len(task_tuple)):
            task_list.append(f"{i + 1}) {task_tuple[i][1]}")  # форматирование списка задач
            keyboard.add(InlineKeyboardButton(text=f"{i + 1}", callback_data=f"delete_task_{task_tuple[i][0]}"))  # добавление кнопок
        keyboard.add(InlineKeyboardButton(text=f"« Назад", callback_data="back_to_tl"))
        keyboard_dt = keyboard.adjust(5).as_markup()  # форматирование позиций кнопок
        await reply_method(f"{tl_emoji} Список задач:\n" + '\n'.join(task_list) + "\n\n" + "Выберите номер удаляемой задачи:", reply_markup=keyboard_dt)
    else:
        await reply_method(f"{tl_emoji} Ваш список задач пуст", reply_markup=tlkb)

@dp.message(Command(commands=["clear_task_list", "ctl"]))
async def clear_task_list(message: Message, author=None, edit_msg=False):
    if author is None:
        author = message.from_user
    reply_method = message.edit_text if edit_msg is True else message.answer
    if message.text != f"{tl_emoji} Ваш список задач пуст":
        with db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM tasks WHERE user_id = %s LIMIT 1", (author.id,))
                if cursor.fetchone() is not None:
                    cursor.execute(f"DELETE FROM tasks WHERE user_id = %s", (author.id,))
                await reply_method(f"{tl_emoji} Ваш список задач пуст", reply_markup=tlkb)



### -----------> reminder_list <-----------
@dp.message(Command(commands=["reminder_list", "rl"]))
async def reminder_list(message: Message, state: FSMContext):
    await state_remove_keys(state, ["ar", "er", "dr", "crl"])
    await state.set_state(New_task.rl)
    await state.update_data(rl=True)
    await edit_task(message=message, state=state)

@dp.message(Command(commands=["add_reminder", "ar"]))
async def add_reminder(message: Message, state: FSMContext):
    await state_remove_keys(state, ["rl", "er", "dr", "crl"])
    await state.set_state(New_task.ar)
    await state.update_data(ar=True)
    await edit_task(message=message, state=state)

@dp.message(Command(commands=["edit_reminder", "er"]))
async def edit_reminder(message: Message, state: FSMContext):
    await state_remove_keys(state, ["rl", "ar", "dr", "crl"])
    await state.set_state(New_task.er)
    await state.update_data(er=True)
    await edit_task(message=message, state=state)

@dp.message(Command(commands=["delete_reminder", "dr"]))
async def delete_reminder(message: Message, state: FSMContext):
    await state_remove_keys(state, ["rl", "ar", "er", "crl"])
    await state.set_state(New_task.dr)
    await state.update_data(dr=True)
    await edit_task(message=message, state=state)

@dp.message(Command(commands=["clear_reminder_list", "crl"]))
async def clear_reminder_list(message: Message, state: FSMContext):
    await state_remove_keys(state, ["rl", "ar", "er", "dr"])
    await state.set_state(New_task.crl)
    await state.update_data(crl=True)
    await edit_task(message=message, state=state)



@dp.message(F.text, StateFilter(None))  # ловит все сообщения пользователя, не начинающиеся на / и не имеющие активного state
async def handle_any_text(message: Message):
    if not message.text.startswith("/"):
        author = message.from_user
        user_profile = None
        with db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT profile FROM users WHERE user_id = %s", (author.id,))
                row = cursor.fetchone()
                if row:
                    user_profile = row[0]
        history = await get_history(author)
        query = message.text
        results = config.get_collection().query(query_texts=[query], n_results=config.TOP_K)
        documents = results["documents"][0]
        distances = results["distances"][0]
        context = []
        for doc, dist in zip(documents, distances):
            if dist <= config.MAX_DISTANCE:  # если расстояние меньше порога, берем
                context.append(doc)
        answer = await generate_answer(query, context, user_profile, history)
        answer = answer.replace("**", "*")
        answer = answer.replace("   -", "   •")
        with db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("INSERT INTO chat_history (user_id, question, answer) VALUES (%s, %s, %s)", (author.id, query, answer))
        answer_char_limit = 4000
        answer_parts = []
        for i in range(0, len(answer), answer_char_limit):
            part = answer[i:i + answer_char_limit]
            answer_parts.append(part.strip())
            if i + answer_char_limit >= len(answer):
                break
        for answer_part in answer_parts:
            try:
                await message.answer(answer_part, parse_mode=ParseMode.MARKDOWN)
            except Exception as e:
                await message.answer(answer_part)




### -----------> Запуск бота <-----------
async def main():
    if bot_is_on:
        try:
            print("Бот включен")
            scheduler.add_job(check_reminders, 'interval', seconds=interval_check_reminders)  # Запускаем проверку каждую минуту
            scheduler.start()
            await dp.start_polling(bot)
        finally:
            print("Бот выключен")
            pass
    else:
        print("Бот выключен: bot_is_on == False")

if __name__ == "__main__":
    asyncio.run(main())