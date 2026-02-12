import sys
import os

# Ensure rag_core is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_core.engine import answer_question

def main():
    print("\n💬 Financial RAG Chatbot (Type 'exit' to quit)")
    print("-" * 50)

    while True:
        try:
            user_query = input("\n❓ Ask a question: ").strip()
        except EOFError:
            break
        
        if user_query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye! 👋")
            break
            
        if not user_query:
            continue

        print("Thinking...")
        try:
            result = answer_question(user_query)

            print("\n💡 Answer:")
            print("="*60)
            print(result["answer"])
            print()
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
