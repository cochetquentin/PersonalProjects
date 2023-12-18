from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)

api_key = "YOUR_API_KEY_HERE"
client = OpenAI(api_key=api_key)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/traduction', methods=['POST'])
def traduction():
    try:
        data = request.get_json()
        prompt = data['prompt']

        def gpt_translate(prompt:str):
            initial_prompt = """
                Vous êtes un traducteur professionnel du français au japonais. 
                Je vais vous donner des phrases en français et vous devrez les traduire dans le japonais le plus naturel. 
                En plus de la traduction, vous expliquerez les différents points grammaticaux utilisés pour m'aider à comprendre comment la phrase a été construite.
                Vous me renverrez toujours la même structure de réponses. La première ligne sera la traduction japonaise et la suivante la liste des points de grammaire.
                Toutes les réponses seront en français.
            """

            messages=[
                {"role": "system", "content": initial_prompt},
                {"role": "user", "content": prompt}
            ]
            completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
            return completion.choices[0].message.content
        
        def split_translatation_explanation(result:str):
            translation = result.split("\n")[0].strip()
            explanation = result[len(translation):].strip()

            return {"translation": translation, "explanations": explanation}
        
        def gpt_translate_and_explain(prompt:str):
            result = gpt_translate(prompt)
            return split_translatation_explanation(result)

        translation = gpt_translate_and_explain(prompt)
        return jsonify(translation)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
