import streamlit as st
import numpy as np
import pandas as pd

import spacy
import en_core_web_md
#nlp = en_core_web_sm.load()
nlp = spacy.load("en_core_web_md")

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import stem_text

import spacy_streamlit
from spacy_streamlit import visualize_ner

import re

#### Dictionary of meta-discourse markers

PersonMarkers = ["i", "we", "me", "mine", "our", "my", "us", "we", 
   "you", "your", "yours", "your's", "ones", "one's", "their"]
AnnounceGoals = ["purpose", "aim", "intend", "seek", "wish", "argue", 
   "propose", "suggest", "discuss", "like", "focus", "emphasize", 
   "goal", "this", "do", "will"]# contains two instances of "this"
CodeGloss = ["example", "instance", "e.g", "e.g.", "i.e", "i.e.", 
   "namely", "other", "means", "specifically", "known", "such", 
   "define", "call"]
AttitudeMarkers = ["admittedly", "agree", "amazingly", "correctly", 
   "curiously", "disappointing", "disagree", "even", "fortunate", 
   "hope", "hopeful", "hopefully", "important", "interest", "prefer", 
   "must", "ought", "remarkable", "surprise", "surprisingly", 
   "unfortunate", "unfortunately", "unusual", "unusually", 
   "understandably"]
Endophorics = ["see", "note", "noted", "above", "below", "section", 
   "chapter", "discuss", "e.g.", "e.g", "example", "chapter", 
   "figure", "fig", "plot", "chart"]
Hedges = ["almost", "apparent", "apparently", "assume", "assumed", 
   "believe", "believed", "certain", "extent", "level", "amount", 
   "could", "couldnt", "couldn't", "doubt", "essentially", "estimate",
    "frequent", "frequently", "general", "generally", "indicate", 
   "largely", "likely", "mainly", "may", "maybe", "mostly", "might", 
   "often", "perhaps", "possible", "probable", "probably", "relative",
    "seem", "seems", "sometime", "sometimes", "somewhat", "suggest", 
   "suspect", "unlikely", "uncertain", "unclear", "usual", "usually", 
   "would", "wouldnt", "wouldn't", "little", "bit"]
Emphatics = ["actually", "always", "certainly", "certainty", "clear", 
   "clearly", "conclusively", "decidedly", "demonstrate", 
   "determine", "doubtless", "essential", "establish", "indeed", 
   "know", "must", "never", "obvious", "obviously", "prove", "show", 
   "sure", "true", "absolutely", "undoubtedly", "very"]
FrameMarkersStages = ["start", "first", "firstly", "second", 
   "secondly", "third", "thirdly", "fourth", "fourthly", "fifth", 
   "fifthly", "next", "last", "begin", "lastly", "finally", 
   "subsequently", "one", "two", "three", "four", "five"]
Evidentials = ["according", "cite", "cites", "quote", "establish",
   "established", "said", "say", "says", "argue", "argues", "claim",
   "claims", "believe", "believes", "suggest", "suggests", "show",
   "shows", "prove", "proves", "demonstrate", "demonstrates", 
   "literature", "study", "studys", "research"]

####


# EMBELLISHMENTS
st.set_page_config(page_title='Metadiscourse marker identifier',layout='wide')

st.title("Discourse Analysis")
st.sidebar.markdown("##### This python web app was created by [Aneet Narendranath Ph.D.](mailto:dnaneet@mtu.edu)  This code is governed under the GPL 3.0 license.")
st.sidebar.write("##### The development of this app was partial supported by ASEE Grant No. ....")
st.sidebar.markdown("\n")



#MAIN

#models = ["en_core_web_sm", "en_core_web_md"]

def marker_count(markerlist=[]):
    """
    :param markerlist: a list of discourse markers
    :return: unique markers with their number of occurrences.
    """
    markers = []
    marker_types = set(markerlist)
    for marker_type in marker_types:
        markers.append(':'.join([marker_type, str(markerlist.count(marker_type))]))
    return '; '.join(markers)

def marker_count_data_frame(markerlist=[]):
    """
    :param markerlist: a list of discourse markers
    :return: unique markers with their number of occurrences in dataframe
    """
    numbers = []
    marker_types = list(set(markerlist))
    for marker_type in marker_types:
        numbers.append(markerlist.count(marker_type))
    data = {
        "Markers": marker_types,
        "Frequency": numbers
    }

    return pd.DataFrame(data)


selection=st.sidebar.radio(label=' ',options=['Meta-discourse count', 'Named entity recognition', 'AI integration'])


if selection == "Meta-discourse count":
    st.markdown("## Meta-discourse count")
    st.markdown("This page defines and uses a limited dictionary of metadiscursive markers.  Details of the dictionary may be found in our ASEE Paper... *link goes here.* Barring any logical errors in this code, the accuracy of the output is validated as per the details described in our paper.")
    st.markdown("----")
    st.write("#### Enter the text you wish to count metadiscursive markers into this textbox.")
    txt = st.text_area('', """  """)
    words = strip_multiple_whitespaces(strip_punctuation(txt.lower())).split()

    epsilon = 0.000000000001;

    nWords = len(txt.split())
    st.write("Word count: ", nWords)    
    
    person_markers = [i for i in words if i in PersonMarkers]
    n_person_markers = len(person_markers)
    st.write("Number of Person Markers: ", n_person_markers, "Percentage: ", np.round(float(n_person_markers/(nWords+epsilon))*100,3))
    st.write("Person Markers: ", marker_count(person_markers))
    # st.table(marker_count_data_frame(person_markers))
    st.dataframe(marker_count_data_frame(person_markers),
                 column_config={
                     "Markers": st.column_config.Column(width='medium')
                 }
                 )

    announce_goals = [i for i in words if i in AnnounceGoals]
    n_announce_goals = len(announce_goals)
    st.write("Number of 'Goal announcements': ", n_announce_goals, "Percentage: ", np.round(float(n_announce_goals/(nWords+epsilon))*100,3))
    st.write("Goal announcements: ", marker_count(announce_goals))
    st.dataframe(marker_count_data_frame(announce_goals),
                 column_config={
                     "Markers":st.column_config.Column(width='medium')
                 }
                 )

    code_gloss = [i for i in words if i in CodeGloss]
    n_code_gloss = len(code_gloss)
    st.write("Number of 'Code Gloss': ", n_code_gloss, "Percentage: ", np.round(float(n_code_gloss/(nWords+epsilon))*100,3))
    st.write("Code Gloss: ", marker_count(code_gloss))
    st.dataframe(marker_count_data_frame(code_gloss),
                 column_config={
                     "Markers":st.column_config.Column(width='medium')
                 }
                 )

    att_markers = [i for i in words if i in AttitudeMarkers]
    n_att_markers = len(att_markers)
    st.write("Number of 'Attitude Markers': ", n_att_markers, "Percentage: ", np.round(float(n_att_markers/(nWords+epsilon))*100,3))
    st.write("Attitude Markers: ", marker_count(att_markers))
    st.dataframe(marker_count_data_frame(att_markers),
                 column_config={
                     "Markers":st.column_config.Column(width='medium')
                 }
                 )

    endophorics = [i for i in words if i in Endophorics]
    n_endophorics = len(endophorics)
    st.write("Number of 'Endophorics': ", n_endophorics, "Percentage: ", np.round(float(n_endophorics/(nWords+epsilon))*100,3))
    st.write("Endophorics: ", marker_count(endophorics))
    st.dataframe(marker_count_data_frame(endophorics),
                 column_config={
                     "Markers":st.column_config.Column(width='medium')
                 }
                 )

    hedges = [i for i in words if i in Hedges]
    n_hedges = len(hedges)
    st.write("Number of 'Hedges': ", n_hedges, "Percentage: ", np.round(float(n_hedges/(nWords+epsilon))*100,3))
    st.write("Hedges: ", marker_count(hedges))
    st.dataframe(marker_count_data_frame(hedges),
                 column_config={
                     "Markers":st.column_config.Column(width='medium')
                 }
                 )

    emphatics = [i for i in words if i in Emphatics]
    n_emphatics = len(emphatics)
    st.write("Number of 'Emphatics': ", n_emphatics, "Percentage: ", np.round(float(n_emphatics/(nWords+epsilon))*100,3))
    st.write("Emphatics: ", marker_count(emphatics))
    st.dataframe(marker_count_data_frame(emphatics),
                 column_config={
                     "Markers":st.column_config.Column(width='medium')
                 }
                 )

    frm_markers_stage = [i for i in words if i in FrameMarkersStages]
    n_frm_markers_stgs = len(frm_markers_stage)
    st.write("Number of 'Frame Markers': ", n_frm_markers_stgs, "Percentage: ", np.round(float(n_frm_markers_stgs/(nWords+epsilon))*100,3))
    st.write("Frame Markers: ", marker_count(frm_markers_stage))
    st.dataframe(marker_count_data_frame(frm_markers_stage),
                 column_config={
                     "Markers":st.column_config.Column(width='medium')
                 }
                 )

    evidentials = [i for i in words if i in Evidentials]
    n_evidentials = len(evidentials)
    st.write("Number of 'Evidentials': ", n_evidentials, "Percentage: ", np.round(float(n_evidentials/(nWords+epsilon))*100,3))        
    st.write("Evidentials: ", marker_count(evidentials))
    st.dataframe(marker_count_data_frame(evidentials),
                 column_config={
                     "Markers":st.column_config.Column(width='medium')
                 }
                 )

elif selection == "Named entity recognition":
    st.write("Named entity recognition (NER) is an automated natural language processing technique that segments the words in a passage of text into various classes.  These classes could be names of people, organizations, cardinal numbers, seasons, etc.  NER assists a discourse analyst or linguist to visualize different classes of information and extract intelligence.")   
    
    st.write("This app uses Named Entity Recognition (NER) to identify and classify named entities in text. NER is a statistical method that identifies and classifies named entities in text, such as people, places, organizations, and dates. While NER is a powerful tool, it is important to note that it is not perfect. NER can sometimes make mistakes.")
    st.write("#### Enter the text you wish to perform NER into this textbox.")
    txt_ner = st.text_area('', """  """)
    doc = nlp(txt_ner)
    visualize_ner(doc, labels=nlp.get_pipe("ner").labels)
elif selection == "AI integration":
    st.markdown("## AI integration")    
    st.write("This page is under construction")
    st.markdown("# 🚧")
    

#eof