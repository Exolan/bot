from aiogram import Router, Bot
from aiogram.types import Message
from aiogram.filters import Command
from keyboards import main_keyboard
from database import Database

start_router = Router()

@start_router.message(Command("start"))
async def start_command(message: Message, db: Database):
    query = """
        INSERT INTO users (telegram_id)
        VALUES (%s)
        ON DUPLICATE KEY UPDATE telegram_id=telegram_id;
    """
    await db.execute(query, (message.chat.id))

    await message.answer("📚 <b>Приветствуем!</b>\n\n"\
                        "🤖 Данный чат-бот предназначен для помощи в изучении бережливого мышления\n\n"\
                        "✅ Выбрав соответствующее поле в меню, вы сможете найти ответы на возникающие вопросы\n\n"\
                        "🎯 Успешного пользования!", reply_markup=await main_keyboard(db))
