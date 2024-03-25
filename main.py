import sys
import heapq
import numpy as np
import re
import itertools
import requests
from googleapiclient.discovery import build
import spacy
from bs4 import BeautifulSoup
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs
from gemini_helper_6111 import get_gemini_completion
from gemini_prompt_generator import gemini_prompt_generate
from difflib import SequenceMatcher


exclude_filetype = ['jpg', 'jpeg', 'png', 'gif', 'tiff', 'psd', 'pdf', 'eps', 'ai', 'indd', 'raw']
extraction_method = api_key = engine_id = gemini_api_key = None
r = t = q = k = None
# r : relation to extract
# t : extraction confidence threshold
# q : seed query
# k : number of tuples requested in output

X = dict()  # dictionary of extracted tuples, {(tuple): score}
visited_urls = []   # list of visited links
used_query = []     # list of tuples already used for query
relations = {
    # dictionary of relations: (Name, Internal Name, Subject, Object)
    '1': ['Schools_Attended', 'per:schools_attended', ['PERSON'], ['ORGANIZATION']],
    '2': ['Work_For', 'per:employee_of', ['PERSON'], ['ORGANIZATION']],
    '3': ['Live_In', 'per:cities_of_residence', ['PERSON'], ['LOCATION', 'CITY', 'STATE_OR_PROVINCE', 'COUNTRY']],
    '4': ['Top_Member_Employees', 'org:top_members/employees', ['ORGANIZATION'], ['PERSON']]
}

# ===== varaibles for Spacy ====
nlp = None  # tokenizer
entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
# ==============================


# ===== parameters for Gemini API =====
model_name = 'gemini-pro'
max_tokens = 4096
temperature = 0.2
top_p = 1
top_k = 32
# =====================================


def run_query(key, cx, query):
    """
    Call Google API and receive list of results.

    :param key: Google API key
    :param cx: Google Engine id
    :param query: query in string
    :return: list of filtered results
    """
    service = build("customsearch", "v1", developerKey=key)
    res = (service.cse().list(q=query, cx=cx).execute())

    return res['items']


def process_query_results(results):
    """
    Loop through search results and filter out unwanted file type.
    Call get_website_text() to fetch website text.
    Call extract_relation() on website text, if website text is not None.

    :param results: list of search results returned from Google
    """

    global visited_urls
    
    for i, res in enumerate(results):
        if res['link'] not in visited_urls:
            print()
            print()
            print('URL ( ' + str(i+1) + ' / ' + str(len(results)) + '): ' + res['link'])
            
            print('     Fetching text from url ...')
            
            # Fetch text from website if file type is valid
            if (res['link'].split('.')[-1] in exclude_filetype) or ('fileFormat' in res):
                fetched_text = None
            else:
                fetched_text = get_website_text(res['link'])
            
            # Extract relation if fetched text is valid
            if fetched_text:
                extract_relation(fetched_text)
            else:
                print('Unable to fetch URL. Continuing.')
            visited_urls += [res['link']]


def get_website_text(url):
    """
    Use BeautifulSoup to extract website text.

    :param url: a link from a search result
    """

    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(separator=" ", strip=True)

        if len(text) > 10000:
            print('     Trimming webpage content from ' + str(len(text)) + ' to 10000 characters')
            text = text[:10000]
        print('     Webpage length (num characters): ' + str(len(text)))
        text = text.replace(r"\n", ' ').replace(r"\r", ' ').replace(r"\t", ' ')
        text = re.sub(' +', ' ', text)
        return text
    except Exception as ex:
        print('Unable to fetch URL. Continuing.')
        return None


def extract_relation(text):
    """
    Extract sentence on parameter 'text'.
    Identify entity pairs for each sentence and then apply filter based on requested relation.
    Finally, extract relations by SpanBERT or Gemini.

    :param text: fetched text from website
    """
    
    print('     Annotating the webpage using spacy...')
    doc = nlp(text)

    num_sentence = len(list(doc.sents))
    print('     Extracted {} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...'.format(num_sentence))

    num_extracted_sentence = num_relation = num_extracted_relation = 0
    for i, sentence in enumerate(doc.sents):

        # Create entity pairs
        candidate_pairs = []
        sentence_entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        for ep in sentence_entity_pairs:
            candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  # e1=Subject, e2=Object
            candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})  # e1=Object, e2=Subject
        
        # Filter candidate pairs based on given relation
        candidate_pairs = [p for p in candidate_pairs if ((p["subj"][1] in relations[r][2]) & (p["obj"][1] in relations[r][3]))]

        if len(candidate_pairs) > 0:
            
            relation_preds = None
            if extraction_method == '-spanbert':
                
                num_extracted_sentence += 1

                # Classify Relations for all Candidate Entity Pairs using SpanBERT
                relation_preds = spanbert.predict(candidate_pairs)  # get predictions: list of (relation, confidence) pairs
                
                for ex, pred in list(zip(candidate_pairs, relation_preds)):
                    num_relation += 1
                    # extract tuple with confidence above threshold
                    if pred[0] == relations[r][1]:
                       num_extracted_relation += evaluate_relation(ex, pred, sentence.text)

            elif extraction_method == '-gemini':

                relation_preds =  get_gemini_completion(
                    gemini_prompt_generate(relations[r][0], sentence.text), 
                    gemini_api_key,
                    model_name, 
                    max_tokens, 
                    temperature, 
                    top_p, 
                    top_k
                    )

                if len(relation_preds): num_extracted_sentence += 1

                for ex in relation_preds:
                    num_relation += 1
                    # Note that we assume the extracted gemini relations always have confidence 1.
                    # This is because gemini does not return confidences for the extracted tuples.
                    pred = (0, 1)
                    num_extracted_relation += evaluate_relation(ex, pred, sentence.text)

        i += 1
        if i % 5 == 0:
            print()
            print('     Processed {} / {} sentences'.format(i, num_sentence))
    
    print()
    print('     Extracted annotations for  {}  out of total  {}  sentences'.format(num_extracted_sentence, num_sentence))
    print('     Relations extracted from this website: {} (Overall: {})'.format(num_extracted_relation, num_relation))


def evaluate_relation(extract, prediction, sentence):
    """
    Evaluate extracted relation, based on threshold / duplication.
    Add to extracted tuple set (X) only if conditions fulfilled.

    :param extract: extracted relation from Spacy
    :param prediction: 
        for spanbert, prediction is (relation, confidence) of the extracted tuple
        for gemini, prediction is always (0, 1)
    :param sentence: sentence under evaluation
    """
    print()
    print('          === Extracted Relation ===')

    flag_add_relation = False

    if extraction_method == '-spanbert':
        print('          Input tokens: ', extract['tokens'])
    else:
        print('          Sentence: {}'.format(sentence))
    print('          Output Confidence: {} ; Subject: {} ; Object: {} ;'.format(prediction[1], extract['subj'][0], extract['obj'][0]))
    this_tuple = (extract['subj'][0], extract['obj'][0])
    
    # For spanbert, add this_tuple to X if it is not a duplication or has higher confidence
    # For gemini, add this_tuple to X if it is not a duplication
    if prediction[1] < t:
        # Note that for gemini results, prediction confidence will always be 1 i.e. this block never executes
        print('          Confidence is lower than threshold confidence. Ignoring this.')
    elif this_tuple in X:
        # Handle duplication
        if extraction_method == '-gemini':
           print('          Duplicate. Ignoring this.') 
        elif X[this_tuple] > prediction[1]:
            print('          Duplicate with lower confidence than existing record. Ignoring this.')
        else:
            flag_add_relation = True
    else:
        flag_add_relation = True
    
    if flag_add_relation:
        print('          Adding to set of extracted relations')
        X[this_tuple] = prediction[1]
    print('          ==========')
    return int(flag_add_relation)
	

def generate_next_query(last_query):
    """
    Generate next query based on extracted tuple.
    For spanbert method, select unused tuple with highest confidence among the extracted tuples.
    For gemini method, select unused tuple with least similarity to the last used query.

    :param last_query: last query used
    """
    # Choose next query
    if extraction_method == '-spanbert':
        sorted_X = sorted(X.items(), key=lambda item: item[1], reverse=True)
        for item in sorted_X:
            if ' '.join(item[0]) not in used_query:
                return ' '.join(item[0]) 
    elif extraction_method == '-gemini':
        # For gemini, we are going to use the extracted tuple that has a 
        # subject with the least similarity to the last tuple. This will 
        # ensure that we gain as much information as possible.
        sorted_X = sorted(X.items(), key=lambda item: SequenceMatcher(None, last_query, item[0][0]).ratio())
        for item in sorted_X:
            if ' '.join(item[0]) not in used_query:
                return ' '.join(item[0]) 
    
    # Return None if next query cannot be generated.
    return None


def print_extracted_relations():
    """
    Print all extracted relations (X)
    """
    print()
    print()
    print('================== ALL RELATIONS for {} ( {} ) ================='.format(relations[r][1], len(X)))
    sorted_X = sorted(X.items(), key=lambda item: item[1], reverse=True)
    if extraction_method == '-spanbert':
        for item in sorted_X:
            print('Confidence: {:.10f}       | Subject: {}       | Object: {}'.format(item[1], item[0][0], item[0][1]))
    elif extraction_method == '-gemini':
        for item in sorted_X:
            print('Subject: {}       | Object: {}'.format(item[0][0], item[0][1]))


def return_extraction_result():
    """
    For SpanBERT method, return top-k results
    For Gemini method, return all results
    """
    if extraction_method == '-spanbert':
        sorted_X = sorted(X.items(), key=lambda item: item[1], reverse=True)
        return [item for item in sorted_X[:k]] 
    elif extraction_method == '-gemini':
        return X.keys()


if __name__ == "__main__":
    extraction_method, api_key, engine_id, gemini_api_key = sys.argv[1:5]
    r, t, q, k = sys.argv[5], float(sys.argv[6]), sys.argv[7], int(sys.argv[8]) 

    if extraction_method == '-spanbert':
        # Load pre-trained spanbert model
        spanbert = SpanBERT("./pretrained_spanbert")
    elif extraction_method == '-gemini':
        # Nothing to do here
        pass 

    print('____')
    print('Parameters:')
    print('Client Key	= ' + api_key)
    print('Engine Key	= ' + engine_id)
    print('Gemini Key	= ' + gemini_api_key)
    print('Method		= ' + extraction_method)
    print('Relation	= ' + relations[r][0])
    print('Threshold	= ' + str(t))
    print('Query		= ' + q)
    print('# of Tuples	= ' + str(k))

    print('Loading necessary libraries; This should take a minute or so ...')
    nlp = spacy.load("en_core_web_lg")

    iteration_count = 0
    this_query = q
    while this_query is not None:
        print('=========== Iteration: {} - Query: {} ==========='.format(iteration_count, this_query))
        
        results = run_query(api_key, engine_id, this_query)
        used_query += [this_query]
        process_query_results(results)
        print_extracted_relations()

        if len(X) >= k:
            break
        
        iteration_count = iteration_count + 1
        this_query = generate_next_query(this_query)
    
    print('Total # of iterations = ' + str(iteration_count + 1))
    return_extraction_result()