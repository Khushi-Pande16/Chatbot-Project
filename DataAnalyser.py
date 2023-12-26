import nltk #to encode human language into computer language 
import warnings
from tkinter import Tk, Frame, Label, Entry, Button, Scrollbar, Text, messagebox
from ttkbootstrap import Style  #style is a class, for theme in gui

warnings.filterwarnings("ignore")

import numpy as np 
import random  # to generate random values
import string 

f = open("VKdataset.txt", "r", errors="ignore") 
raw = f.read() 
raw = raw.lower()
nltk.download("punkt") #a tokenizer comes in nltk ,splits the text into individual sentences

nltk.download("wordnet") # has so many words which converts the in dataset into natural language
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw) # split text into words

sent_tokens[:2]
word_tokens[:5]

lemmer = nltk.stem.WordNetLemmatizer() #preprocessing of text,reducing words in dict form


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text): #for input from user
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = (
    "hello",
    "oye",
    "hi",
    "greetings",
    "sup",
    "what's up",
    "hey",
)
GREETING_RESPONSES = [
    "hi",
    "hoye",
    "umhumm",
    "*nods*",
    "hi there",
    "hello",
    "namaskar",
    "namastey",
    "I am glad! You are talking to me",
]


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


from sklearn.feature_extraction.text import TfidfVectorizer #convert text into numerical feature
from sklearn.metrics.pairwise import cosine_similarity #measurment between two vectors
  

def response(user_response):
    FanBot_response = ""
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten() #converting  in one dimensional array
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        FanBot_response = "I am sorry! I don't understand you, Please elaborate your question or you can ask something else about Virat kohli"
        return FanBot_response
    else:
        FanBot_response = (
            "as per my knowledge, "
            + FanBot_response
            + sent_tokens[idx]
            + " Any thing else? Was it helpful? you can thank me..or just say bye.. ;)\n"
        )
        return FanBot_response


def send_message(event=None):
    user_input = user_entry.get()
    user_entry.delete(0, "end")

    if user_input.lower() == "bye":
        chat_area.insert("end", "FanBot: Bye! Take care...\n")
        messagebox.showinfo("FanBot", "Bye! Take care...")
        root.destroy()
        return

    if user_input.lower() == "thanks" or user_input.lower() == "thank you":
        chat_area.insert("end", "FanBot: You are welcome..\n")
        messagebox.showinfo("FanBot", "You are welcome..")
        return

    if greeting(user_input) != None:
        bot_response = greeting(user_input)
    else:
        bot_response = response(user_input)

    chat_area.insert("end", "You: " + user_input + "\n")
    chat_area.insert("end", "FanBot: " + bot_response + "\n")
    chat_area.see("end")


root = Tk()
root.title("Virat Kohli FanBot")
style = Style(theme="cyborg")

frame = Frame(root)
frame.pack(pady=20)

title_label = Label(
    frame,
    text="Virat Kohli FanBot",
    font=("Helvetica", 18),
)
title_label.pack()

user_entry = Entry(
    frame,
    font=("Helvetica", 12),
    width=30,
)
user_entry.pack(pady=10)
user_entry.bind("<Return>", send_message)

chat_area = Text(
    frame,
    width=40,
    height=20,
    font=("Helvetica", 12),
    bg="white",
    fg="black",
)
chat_area.pack(pady=10)

send_button = Button(
    frame,
    text="Send",
    font=("Helvetica", 12),
    bg="blue",
    fg="white",
    command=send_message,
)
send_button.pack()

root.mainloop()

