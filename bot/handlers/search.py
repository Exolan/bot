import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from aiogram import Router, Bot
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from database import Database
from keyboards import back_buttons
from states import MenuState
from utils import delete_old_mes
from scipy.spatial.distance import cosine

search_router = Router()

# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct")

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def embed_text(text: str) -> list:
    instruct_text = f'Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery: {text}'
    inputs = tokenizer(instruct_text, max_length=512, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding[0].tolist()  # возвращаем как обычный список для совместимости

@search_router.message()
async def search_info(message: Message, state: FSMContext, bot: Bot, db: Database):
    await delete_old_mes(bot, message.chat.id, message.message_id)
    await state.set_state(MenuState.search_query)

    data = await state.get_data()
    search_text = data.get("search_text")

    if not search_text:
        search_text = message.text.strip()

    if len(search_text) < 5:
        await message.answer("🤔 Я вас не понимаю. Напишите более подробно", reply_markup=back_buttons())
        return

    try:
        # Преобразуем запрос в вектор через новую модель
        query_vector = embed_text(search_text)

        # Запрашиваем темы с векторными представлениями
        query_themes = "SELECT theme_id, theme_name, theme_vector FROM themes WHERE theme_vector IS NOT NULL"
        query_subthemes = "SELECT subtheme_id, subtheme_name, subtheme_vector FROM subthemes WHERE subtheme_vector IS NOT NULL"

        themes = await db.fetch_all(query_themes)
        subthemes = await db.fetch_all(query_subthemes)

        if not themes and not subthemes:
            await message.answer("😕 Я ничего не нашел", reply_markup=back_buttons())
            return

        similarities = []

        for theme in themes:
            theme_vector = json.loads(theme["theme_vector"])
            similarity = 1 - cosine(query_vector, theme_vector)
            if similarity > 0.7:
                similarities.append((theme["theme_name"], similarity, f'select_theme_{theme["theme_id"]}'))

        for subtheme in subthemes:
            subtheme_vector = json.loads(subtheme["subtheme_vector"])
            similarity = 1 - cosine(query_vector, subtheme_vector)
            if similarity > 0.7:
                similarities.append((subtheme["subtheme_name"], similarity, f'select_subtheme_{subtheme["subtheme_id"]}'))

        if len(similarities) == 0:
            await message.answer("😕 Я ничего не нашел", reply_markup=back_buttons())
            return

        best_matches = sorted(similarities, key=lambda x: x[1], reverse=True)

        buttons = []

        if best_matches:
            for name, sim, call in best_matches[:3]:
                buttons.append([InlineKeyboardButton(text=name, callback_data=call)])
        else:
            buttons.append([InlineKeyboardButton(text="Ничего не найдено", callback_data="menu")])

        buttons.append([InlineKeyboardButton(text="Главное меню", callback_data="menu")])

        await message.answer("🔎 Вот что я нашел", reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons))

        await state.set_state(MenuState.search_query)
        await state.update_data(search_text=search_text)

    except Exception as e:
        await message.answer("⚠ Произошла ошибка. Попробуйте позже", reply_markup=back_buttons())
        print(f"Ошибка поиска: {e}")

@search_router.callback_query(lambda call: call.data == "search_results")
async def return_to_search(call: CallbackQuery, state: FSMContext, bot: Bot, db: Database):
    await call.message.delete()

    data = await state.get_data()
    search_text = data.get("search_text")

    if not search_text:
        await call.message.answer("⚠ Ошибка. Повторите поиск позже", reply_markup=back_buttons())
        return
    
    await search_info(call.message, state, bot, db)
