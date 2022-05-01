from flask import Flask, request, jsonify
# import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import T5ForConditionalGeneration, T5Tokenizer
from flask_cors import CORS, cross_origin

# define a variable to hold you app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
    
def parsedFullTranscript(videoID):
  listText = YouTubeTranscriptApi.get_transcript(videoID)
  textInVideo = ""
  for i in listText:
    textInVideo += i['text'] + " "
  # print(textInVideo)
  return textInVideo[:-1]

# parsedFullTranscript('Y5_qH99_lLI')

# initialize the model architecture and weights
model = T5ForConditionalGeneration.from_pretrained("t5-base")
# initialize the model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def summarizeTranscript(fullTranscript):
  inputs = tokenizer.encode("summarize: " + fullTranscript, return_tensors="pt", max_length=512, truncation=True)
  # generate the summarization output
  outputs = model.generate(
      inputs, 
      max_length=150, 
      min_length=40, 
      length_penalty=2.0, 
      num_beams=4, 
      early_stopping=True)
  # just for debugging
  # print(outputs)
  return tokenizer.decode(outputs[0])

# print(summarizeTranscript(parsedFullTranscript('Nv4Nk4AAgk8')))

# define your resource endpoints
@app.route('/')
def index_page():
    return "Hello world"

@app.route('/api/summarize', methods=['GET'])
@cross_origin()
def get_transcript():
    id = request.args['youtube_url'][-11:]
    return jsonify({'fullTranscript':parsedFullTranscript(id),
                    'summarizeTranscript':summarizeTranscript(parsedFullTranscript(id))})
    # return summarizeTranscript(parsedFullTranscript(id))

# server the app when this file is run
if __name__ == '__main__':
    app.run(debug=True)