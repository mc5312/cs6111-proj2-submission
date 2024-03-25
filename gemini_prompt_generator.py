def gemini_prompt_generate(relation_type, sentence):
    return """
Given a sentence, extract all instances of a relationship type specified by the user that you can find in the sentence:
the possible relationship types are: "Schools_Attended", "Work_For", "Live_In", "Top_Member_Employees"

Note that for all text below: PERSON, RELATIONSHIP, LOCATION, STATE_OR_PROVINCE, COUNTRY, ORGANIZATION all correspond to entities
that would be found by named-entity recognition.

If the user specifies "Schools_Attended", the output should have the following format:
[PERSON| RELATIONSHIP| ORGANIZATION]

For example, for "Schools_Attended", the output of the sentence:
"He was born to parents John Doe and Jane Doe, who both attended University of Florida"

should be :
'["John Doe"| "Schools_Attended"| "University of Florida"]
["Jane Doe"| "Schools_Attended"| "University of Florida"]'

As another example, for "Schools_Attended", the output of the sentence:
"University of Florida student Jane Doe studies Biology"

should be: '["Jane Doe"| "Schools_Attended"| "University of Florida"]'

As another example, for "Schools_Attended", the output of the sentence:
"Professor Emeritus John Doe of University of Florida still loves hiking at the age of 87"

should be: '["John Doe"| "Schools_Attended"| "University of Florida"]'


If the user specifies "Work_For", the output should have the following format:
[PERSON| RELATIONSHIP| ORGANIZATION]

For example, for "Work_For", the output of the sentence:
"Core Rock was managed by Jane Doe before she left for a new opportunity at Tesla"

should be:
'["Jane Doe"| "Work_For"| "Core Rock"]'
["Jane Doe"| "Work_For"| "Tesla"]'

If the user specifies "Live_In", the output should have one of the following formats:
[PERSON| RELATIONSHIP| LOCATION], 
[PERSON| RELATIONSHIP| CITY], 
[PERSON| RELATIONSHIP| STATE_OR_PROVINCE] 
[PERSON| RELATIONSHIP| COUNTRY]

The "Live_In" relation captures the relationship between a PERSON who is currently living, or has lived, 
in a certain LOCATION, CITY, STATE_OR_PROVINCE, or COUNTRY. The sentence must provide evidence that the person lived in
the place for a sustained period of time, and wasn't simply visiting.

For example, for "Live_In", the output of the sentence:
"John Doe was born in Austin" 

should be:
'["John Doe"| "Live_In"| "Austin"]'

If there is no explicit statement that the person lived or ever lived in a certain place, then do not include that relation for "Live_in"

If the user specifies "Top_Member_Employees", the output should have the following format:
[ORGANIZATION| RELATIONSHIP| PERSON]

For example, for Top_Member_Employees, the output of the sentence:
"In 1975, he and Allen founded Microsoft in Albuquerque, New Mexico."

should be: '["Microsoft"| "Top_Member_Employees"| "Allen"]'

Each extracted relation should be on its own line. Also, retrieve as many relationships as possible!

extract "{}" relationships from the following sentence: "{}"

""".format(relation_type, sentence)
