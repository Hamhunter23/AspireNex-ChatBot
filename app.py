from flask import Flask, render_template, request, Response
import weaviate
import ollama
import warnings
import json

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

app = Flask(__name__)

class_name = "AspireTest2"

intents = {
    "greeting": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
    "farewell": ["bye", "goodbye", "see you", "take care", "have a nice day", "until next time"],
    "assistance": ["help", "assist", "support", "can you help", "need assistance", "how do I"],
    "thanks": ["thank you", "thanks", "appreciate it", "grateful", "much appreciated"],
    "weather": ["what's the weather", "is it sunny", "will it rain", "temperature today"],
    "time": ["what time is it", "current time", "clock"],
    "joke": ["tell me a joke", "say something funny", "make me laugh"],
    "compliment": ["you're smart", "good job", "well done", "you're helpful"],
    "complaint": ["this isn't helpful", "you're not understanding", "that's wrong"],
    "smalltalk": ["how are you", "what's up", "how's it going"],
}

responses = {
    "greeting": "Hello! How can I assist you today?",
    "farewell": "Goodbye! Have a great day!",
    "assistance": "Sure, I am here to help. What do you need assistance with?",
    "thanks": "You're welcome! I'm glad I could help.",
    "weather": "I can't provide weather updates right now, but you can check a weather website or app.",
    "time": "I don't have the ability to tell the current time, but you can check a clock or your device.",
    "joke": "Why don't scientists trust atoms? Because they make up everything!",
    "compliment": "Thank you! I appreciate your kind words.",
    "complaint": "I'm sorry to hear that. Could you please provide more details so I can assist you better?",
    "smalltalk": "I'm just a bot, but I'm here to help! How can I assist you today?",
}

def identify_intent(user_input):
    user_input_lower = user_input.lower()
    for intent, keywords in intents.items():
        for keyword in keywords:
            if keyword in user_input_lower:
                return intent
    return None

def handle_user_input(user_input, use_context):
    intent = identify_intent(user_input)
    
    if intent:
        return responses[intent]
    else:
        return rag_with_llm_response(user_input, use_context)

def get_ollama_embedding(text):
    embedding = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return embedding['embedding']

def rag_with_llm_response(user_input, use_context):
    if use_context:
        weaviate_client = weaviate.Client("http://localhost:8080")
        try:
            query = user_input
            query_embedding = get_ollama_embedding(query)

            search_results = (weaviate_client.query
                              .get(class_name, ["content", "source"])
                              .with_near_vector({"vector": query_embedding})
                              .with_limit(5)
                              .do())

            context = "\n".join([result['content'] for result in search_results['data']['Get'][class_name]])
            prompt = f"Based on the following context, \n\nContext: {context}\n\nAnswer: {query}"
            
            yield " "
            for chunk in ollama.generate(model="llama3:8b", prompt=prompt, stream=True):
                yield chunk['response']
        
        finally:
            weaviate_client = None
    else:
        prompt = f"Answer the following query directly: {user_input}"
        yield " "
        for chunk in ollama.generate(model="llama3:8b", prompt=prompt, stream=True):
            yield chunk['response']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    use_context = request.json['useContext']
    intent = identify_intent(user_message)
    
    if intent:
        return Response(json.dumps({'response': responses[intent]}), mimetype='application/json')
    else:
        return Response(rag_with_llm_response(user_message, use_context), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)