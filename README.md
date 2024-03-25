NAME: Bevis Cheung (mc5312), Sebastian Hereu (smh2278)

------
Files| 
------

(1) main.py - contains the logic for our Iterative Set Expansion (ISE) system. Uses the code contained in the files listed below.    

(2) gemini_prompt_generator.py - contains the function gemini_prompt_generate(), which returns a gemini prompt for an extraction task. 

(3) spacy_help_functions.py - contains helping parser functions to be used with spaCy

(4) spanbert.py - contains code for interacting with a pretrained spanBERT classifier 

(5) gemini_helper_6111.py - contains code for calling the google gemini API

(6) README.md - this file


---------
API Keys|
---------

Google Custom Search Engine JSON API Key: AIzaSyC8EbccVhwPEcQ-oFeFgRTZ1DgfVAOg6-8

Google Custom Search Engine JSON API Engine ID: 9604022c4b8b04d7e 

-------------------
External Libraries|
-------------------

(1) Beautiful Soup - toolkit to extract plain text from a given webpage, ignoring HTML tags, links, images, etc.
	to install, run 'pip3 install beautifulsoup4'

(2) Google Custom Search API - Used for searching the web. Note that this API was used in Project1 
	to install, run 'pip3 install --upgrade google-api-python-client'

(3) spaCy - Library to process and annotate text with linguistic analysis. 
	to install, run:
		sudo apt-get update
		pip3 install -U pip setuptools wheel
		pip3 install -U spacy
		python3 -m spacy download en_core_web_lg

(4) Google Gemini - Currently free API for which 60 queries per minute can be submitted. To use API, a Gemini API key is required (see https://ai.google.dev/?utm_source=google&utm_medium=cpc&utm_campaign=brand_core_brand&gad_source=1&gclid=CjwKCAiAiP2tBhBXEiwACslfnumu6TJncwJXVvBseKBNGy9K21w8BbRBQVKoERZE_O6hlfD4pdLhQxoCAkoQAvD_BwE)
	to install, run: 
		pip install -q -U google-generativeai

(5) SpanBERT - classifer to extract four types of relations from text documents: 1) Schools_Attended, 2) Work_For, 3) Live_In 4) Top_Member_Employees 
	In the directory that includes all of the files included in the submission (see section: 'Files' above),
	run:
		git clone https://github.com/larakaracasu/SpanBERT
		cd SpanBERT
		pip3 install -r requirements.txt
		bash download_finetuned.sh
	
	VERY IMPORTANT: After running the above commands, you should be in the SpanBERT directory. Go one directory up (run 'cd ../'), so that
	you are back in the directory containing the submission code. Now run the command 'cp -rT SpanBERT ./'. This will 
	copy all files in the SpanBERT directory with the submission code.

(6) numpy - numerical python library
	to install, run:
		pip install numpy

Standard Libraries

(7) difflib - provides classes and functions for comparing sequences. We use the library for selecting the next query for gemini. 

(8) sys, heapq, itertools, requests, re - standard library

------------------
How to Run Program|
-------------------

IMPORTANT: Files (1-5) and the provided folder "pretrained_spanbert" have to be placed in the same directory.

python3 main.py [-spanbert|-gemini] <google api key> <google engine id> <google gemini api key> <r> <t> <q> <k>

Note that the parameters are given in the project 2 specification.  

For example, to run the ISE algorithm using the SpanBERT classifier, starting with seed query 'sergey brin stanford', confidence threshold 0.7,
relation Schools_Attended, and 10 requested tuples, run:

python3 main.py -spanbert AIzaSyC8EbccVhwPEcQ-oFeFgRTZ1DgfVAOg6-8 9604022c4b8b04d7e AIzaSyB98XWvUOx6i_S29uu3jhoV6rbhA2CGuRo 1 0.7 'sergey brin stanford' 10

To execute the same ISE task, but using gemini, run:

python3 main.py -gemini AIzaSyC8EbccVhwPEcQ-oFeFgRTZ1DgfVAOg6-8 9604022c4b8b04d7e AIzaSyB98XWvUOx6i_S29uu3jhoV6rbhA2CGuRo 1 0.7 'sergey brin stanford' 10

***Note that we have pre-included our <google api key>, <google engine id>, and <google gemini api key> for convenience*** 

-------------------
General Description|
-------------------

We will first delve into the general structure of main.py. 

When the program starts, the main function parses the command line arguments, initializes spaCy (in addition to SpanBERT if -spanbert is specfied), sets the seed query, and starts the main while loop.

The main while loop executes the core logic of the ISE algorithm: use a query tuple q to retrieve documents, parse those documents for more tuples, query
for more documents using one of the retrived tuples, and repeat this process until k tuples have been extracted. If the ISE algorithm does retrieve k tuples after the 
previous iteration, we use generate_next_query() to obtain the tuple to be used for querying in the next iteration. If gemini is being used, we pick the next query tuple
based on its dissimilarity with the current query tuple. We can sort the tuples based on dissimilarity with the current query using difflib's SequenceMatcher(). SequenceMathcher()
returns a score that estimates how similar two strings are. Thus, using this function, we can choose the next tuple that will approximately provide the most different search results 
from the current tuple. If SpanBERT is being used, we use the unused tuple with the higest confidence. 
 
When the main loop issues a query to the web, run_query() is used. The raw query results are passed to process_query_reults(), which cleans the search results, filtering out unwanted file types along the way. process_query_results() also calls get_website_text(), which uses BeautifulSoup to extract and clean the website text. Finally, this website text is passed
to extract_relation(), which does all of the heavy lifting of extracting tuples. 

extract_relation() is essential to the program, as it performs the actual extraction for the requested relation. First, using the create_candidate_pairs() helper function from spacy_help_functions.py, entity pairs are extracted for the current sentence. For each of these pairs, two symmetrical candidate_pairs are created where the subject and object have their roles reversed. Once the
intial set of candiate_pairs is generated, we filter this set so that it only includes those candidate_pairs that have an object and subject corresponding to the requested relation. For example, if the user requested the relation Work_For, we make sure that the subject is PERSON and the Object is ORGANIZATION. If this step is not taken, then too many sentences will wastefully be passed
to either SpanBERT or gemini, which is computationally inefficient and perhaps economically wasteful.

This final set of candidate_pairs are passed to either SpanBERT or gemini for a prediction. 

If gemini is used, the function get_gemini_completion() is called to retrieve tuples. Note
that unlike SpanBERT, gemini simply takes the current sentence as a string, not pre-procecesed candidate pairs. The reason why we perform the filtering on the cadidate_pairs
before calling gemini is because we still only want to call the API if we have identified a possible relationship in the current sentence. get_gemini_completion(), defined in 
gemini_helper_6111.py, calls the gemini API with a prompt given as an argument, and formats the results. Instead of using different prompts for each relation extraction task, we used
one large prompt that contains a small number of examples for each realtion type. We found that one large prompt that lists each extraction task performs better than several seperate prompts. 
It appears that gemini learns how to extract certain relations more effectively when it also sees different relation tasks as a part of the prompt. We believe that this occurs because it
learns how to differentiate between different extraction tasks. To format the prompt for the current sentence, we use gemini_prompt_generator.py's gemini_prompt_generate().

If SpanBERT is used, the candidate_pairs are passed to spanbert.predict() which takes input in the exact format of candidate_pairs and provides a predicted confidence for each of the candiates.
Note that gemini provides no confidence estimate of the extracted tuples, so we always assume a confidence of 1 for gemini.

After obtaining predictions from either SpanBERT or gemini, we pass the candidate_pairs, or tuples, that were deemed matches by the classifier to evaluate_relation(). evaluate_relation()
checks if a tuple has a confidence above the desired threshold and if the extracted tuple is a duplicate, i.e. we already have the tuple in our set of extracted relations. If these checks pass,
the tuple is added to the set of extracted relations.

If more than k tuples are not extracted after the current iteration of the ISE algorithm, we select a tuple, that hasn't been previously used for querying, from the set of extracted tuples for the next query.
