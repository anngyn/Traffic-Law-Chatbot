import os
import sys
from dotenv import load_dotenv

# Thêm đường dẫn để import được module trong src
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src', 'domain'))

from src.domain.Retrieval.chatbot import ChatBot

def test_chatbot_api():
    # Load environment variables
    load_dotenv()

    # Khởi tạo các đường dẫn y hệt như trong main.py
    root_dir = current_dir
    stopwords_path = os.path.join(root_dir, 'data', 'vietnamese-stopwords-dash.txt')
    folder_path = os.path.join(root_dir, 'data') 
    keyword_file = os.path.join(root_dir, 'data', 'top_keywords.txt')
    processed_json_file = os.path.join(root_dir, 'data', 'output.json')

    print("Đang khởi tạo cấu trúc ChatBot...")
    try:
        chatbot = ChatBot(
            stopwords_path=stopwords_path,
            folder_path=folder_path,
            keyword_file=keyword_file,
            processed_json_file=processed_json_file,
        )
        print("Khởi tạo ChatBot thành công!\n")
    except Exception as e:
        print(f"Lỗi khởi tạo ChatBot: {e}")
        return

    # Chạy một vài câu test
    test_queries = [
        "người 18 tuổi được lái ô tô chưa?",
    ]

    for query in test_queries:
        print("-" * 50)
        print(f"User: {query}")
        try:
            response = chatbot.process_query(query)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Bot Lỗi: {e}")
            
if __name__ == "__main__":
    test_chatbot_api()

