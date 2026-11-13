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
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.enums import ParseMode
import config
from config import settings
import psycopg2
from contextlib import contextmanager

from llm_client import OpenAILikeLLM, InputMessage
from typing import List
from string import Template
import asyncio


### -------------> Настройки <-------------
bot_is_on = True  # по дефолту выключено, т.к. на одного бота может быть не более 1 подключения (1 человек может запускать бота)
profile_emoji = "💼"


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
                cursor.execute("DELETE FROM users WHERE user_id = %s", (author.id,))
    await profile(callback.message, author=author, edit_msg=True)



### --------> Основные команды <--------
@dp.message(Command("start"))
async def start(message: Message):
    author = message.from_user
    member = await get_member(author)
    await message.answer(f"Здравствуйте {member}! Я являюсь персональным ИИ-помощником для предпринимателей от компании Альфа-Банк. \n\n"
                         f"Чтобы мои ответы были точнее и полезнее, для начала расскажите о себе и вашем деле, воспользовавшись командой /profile.\n\n"
                         f"После этого смело продолжайте диалог, я буду отвечать на интересующие вас вопросы.")


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
    profile = profile[0] if profile is not None else "Отсутствует"
    await reply_method(f"{profile_emoji} Информация о вас и вашем деле:\n"
                       f"`{profile}`", reply_markup=kb, parse_mode=ParseMode.MARKDOWN)


@dp.message(F.text, StateFilter(None))  # ловит все сообщения пользователя, не начинающиеся на /
async def handle_any_text(message: Message):
    if not message.text.startswith("/"):
        user_profile = None
        with db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT profile FROM users WHERE user_id = %s", (message.from_user.id,))
                row = cursor.fetchone()
                if row:
                    user_profile = row[0]
        query = message.text
        results = config.get_collection().query(query_texts=[query], n_results=config.TOP_K)
        documents = results["documents"][0]
        distances = results["distances"][0]
        context = []
        for doc, dist in zip(documents, distances):
            if dist <= config.MAX_DISTANCE:  # если расстояние меньше порога, берем
                context.append(doc)
        answer = await generate_answer(query, context, user_profile)
        answer = answer.replace("**", "*")
        answer = answer.replace("   -", "   •")
        await message.answer(answer, parse_mode=ParseMode.MARKDOWN)



### ----> Вспомогательные функции <----
async def get_member(author):
    first_name = author.first_name if author.first_name else ""
    last_name = author.last_name if author.last_name else ""
    username = author.username
    sep = "" if first_name == "" or last_name == "" else " "
    member = f"@{username}" if username else f"{first_name}{sep}{last_name}"  # обращение к пользователю
    return member


### -----------> generation <-----------
async def generate_answer(query: str, context, user_profile):
    llm = OpenAILikeLLM(
        base_url='https://api.groq.com/openai/v1',
        api_key=config.GROQ_API_KEY,
        model='openai/gpt-oss-120b',
        common_parameters={'temperature': 0}
    )
    system_prompt = "Ты полезный ассистент. Используй русский язык для ответа, отвечай строго с использованием контекста, " \
                    "если ответа нет, то скажи, что не можешь ответить, придумывать ничего не надо."

    if user_profile:
        system_prompt += f"\n\nИНФОРМАЦИЯ О ПОЛЬЗОВАТЕЛЕ (учитывай её для персонализации ответа, "\
                         f"но не бери отсюда факты, если их нет в контексте):\n{user_profile}"

    USER_PROMPT = Template("Вопрос: $query\n\nКонтекст (единственные источники правды):\n$context")
    user_prompt = USER_PROMPT.substitute({"query": query, "context": "\n\n".join(context)})

    messages = [
        InputMessage(role="system", content=system_prompt),
        InputMessage(role="user", content=user_prompt)
    ]
    answer = await llm.generate_answer(messages)
    return answer.content



### -----------> Запуск бота <-----------
async def main():
    if bot_is_on:
        try:
            print("Бот включен")
            await dp.start_polling(bot)
        finally:
            print("Бот выключен")
            pass
    else:
        print("Бот выключен: bot_is_on == False")

if __name__ == "__main__":
    asyncio.run(main())