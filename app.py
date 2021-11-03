from flask import Flask, render_template, request, redirect
import string
import re
from forms import ImageCaptionForm
#from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
import os 
from image_cap import encode, greedy_search, beam_search_predictions
from PIL import Image 
from gensim import corpora, models
#from summarizer import Abstractive_Summarizer


app = Flask("__name__")
app.secret_key = "its_secret"
stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
exclude = set(string.punctuation)
exclude.add('–')
def preprocessing_text(x):
    x = " ".join([i for i in x.lower().split() if i not in stop_words])
        
    # remove number and words containing no.
    x = re.sub('\d+','',x)
    x = re.sub('\W*\d\w*','',x)
  
    #remove punctuation
    x = ''.join(ch for ch in x if ch not in exclude)
    
    #remove stopword
    x = " ".join(lemma.lemmatize(word) for word in x.split())  
    return x

def save_picture(form_picture):
    filename = "detectimg.png"
    #form_picture.save(filename)
    picture_path = os.path.join(app.root_path, 'static/imgs', filename)
    img = Image.open(form_picture)
    img.save(picture_path)
    return picture_path

def topic_model(text, ntopics):
    text = preprocessing_text(text)
    lst_of_tokens = [text.split()]
    dict_ = corpora.Dictionary(lst_of_tokens)
    doc_term_matrix = [dict_.doc2bow(i) for i in lst_of_tokens]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=ntopics, id2word = dict_, passes=15, random_state=0, eval_every=None)
    topics = ldamodel.print_topics(num_topics=1, num_words=ntopics)
    all_topics = set()
    for top in topics:
        lst = top[1].split('+')
        for i in lst:
            word = i.split("*")[1]
            word = re.sub('"', '', word)
            word = re.sub(' +', '', word)
            all_topics.add(word)
    return list(all_topics)
 

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summary")
def summarize():
    return render_template("summary.html")

# @app.route("/get_summary", methods=['POST'])
# def get_summary():
#     if request.method == "POST":
#         article = request.form['article']
#         summary = Abstractive_Summarizer(article)
#         return render_template("result_summary.html", Summary=summary)
#     else:
#         return render_template("summary.html")

@app.route("/topics")
def topics():
    return render_template("topics.html") 

@app.route("/get_topics", methods=['POST'])
def get_topics():
    if request.method == "POST":
        article = request.form['article']
        ntopics=int (request.form['num_topics'])
        topics=topic_model(article, ntopics)
        print(topics)
        return render_template("result_topics.html", Topics=topics)
    else:
        return render_template("topics.html") 

@app.route("/generateCaption", methods=['GET','POST'])
def generateCaption():
    form = ImageCaptionForm()
    if form.validate_on_submit():
        if form.picture.data:
            img_path = save_picture(form.picture.data)
            
        #Greedy search
        X = img_path
        image = encode(X).reshape((1, 2048))
        gred_capt = greedy_search(image)
        
        #Beam Search
        beam_capt = beam_search_predictions(image)
        return render_template("cap_generate.html", GreedCap = gred_capt, BeamCap = beam_capt)
    return render_template("image_caption.html", form=form)
if __name__ == "__main__":
    app.run(debug=True)