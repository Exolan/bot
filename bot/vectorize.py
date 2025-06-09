import json
import asyncio
import aiomysql
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from config import DB_HOST, DB_NAME, DB_PASSWORD, DB_USER

# Загрузка модели
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct")

# Подключение к БД
DB_CONFIG = {
    "host": DB_HOST,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "db": DB_NAME,
}

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def embed_text(text: str) -> list:
    instruct_text = f'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {text}'
    inputs = tokenizer(instruct_text, max_length=512, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.squeeze(0).tolist()  # убираем размерность (1, 1024) → (1024,)

async def update_vectors(pool, table_name, id_column, name_column, text_column, vector_column, exclude_names=None):
    """ Кодирует объединённый текст (name + text) из БД и обновляет его векторное представление """
    exclude_names = exclude_names or []
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            # Изменённый запрос: выбираем только записи без вектора или с пустым вектором
            query = f"""
                SELECT {id_column}, {name_column}, {text_column} 
                FROM {table_name} 
                WHERE {text_column} IS NOT NULL
                AND ({vector_column} IS NULL OR JSON_LENGTH({vector_column}) = 0)
            """
            if exclude_names:
                placeholders = ', '.join(['%s'] * len(exclude_names))
                query += f" AND {name_column} NOT IN ({placeholders})"

            await cursor.execute(query, exclude_names)
            rows = await cursor.fetchall()

            if not rows:
                print(f"⏩ Нет новых записей для обработки в таблице {table_name}")
                return

            print(f"🔍 Найдено {len(rows)} записей для обработки в {table_name}")

            for row_id, name, text in rows:
                full_text = f"{name}. {text}" if name else text
                try:
                    vector = embed_text(full_text)
                    vector_json = json.dumps(vector)

                    await cursor.execute(
                        f"UPDATE {table_name} SET {vector_column} = %s WHERE {id_column} = %s",
                        (vector_json, row_id)
                    )
                    print(f"✅ Обновлён {table_name}: ID {row_id}")
                except Exception as e:
                    print(f"❌ Ошибка при обработке {table_name} ID {row_id}: {str(e)}")

            await conn.commit()

async def main():
    print("⚡ Начинаем кодирование текстов...")

    pool = await aiomysql.create_pool(**DB_CONFIG)

    await update_vectors(pool, "themes", "theme_id", "theme_name", "theme_text", "theme_vector")
    await update_vectors(pool, "subthemes", "subtheme_id", "subtheme_name", "subtheme_text", "subtheme_vector")

    pool.close()
    await pool.wait_closed()

    print("🎉 Кодирование завершено!")

if __name__ == "__main__":
    asyncio.run(main())
