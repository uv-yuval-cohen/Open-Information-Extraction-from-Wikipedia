import wikipedia
import spacy
import google.generativeai as genai


def find_proper_noun_sequences(nlp, text):
    doc = nlp(text)
    proper_noun_sequences = []
    current_sequence = []

    for token in doc:
        if token.pos_ == "PROPN":  # If the token is a proper noun
            current_sequence.append(token.text)
        else:
            if current_sequence:  # End of a proper noun sequence
                proper_noun_sequences.append(" ".join(current_sequence))
                current_sequence = []

    # Capture any remaining sequence
    if current_sequence:
        proper_noun_sequences.append(" ".join(current_sequence))

    return proper_noun_sequences


def build_triplet(subject_tokens, relation_tokens, object_tokens):
    """
    Build a (Subject, Relation, Object) triplet, with the Relation
    filtered to only include VERB or ADP tokens.
    """
    subject = " ".join(subject_tokens)
    obj = " ".join(object_tokens)
    # Filter only the tokens with POS == VERB or ADP
    filtered_relation = [t.text for t in relation_tokens if t.pos_ in ("VERB", "ADP")]
    relation = " ".join(filtered_relation)
    return subject, relation, obj


def contains_verb(seq):
    """
    Return True if any token in `tokens` has the POS tag 'VERB';
    otherwise, return False.
    """
    return any(token.pos_ == "VERB" for token in seq)


def extract_pos_triplets(nlp, text):
    doc = nlp(text)
    triplets = []

    # Iterate through tokens to find consecutive proper nouns
    first_proper_noun_seq = []
    last_proper_noun_seq = []
    intermediate_seq = []
    for i, token in enumerate(doc):

        # If the token is punctuation and the last proper noun sequence is empty, reset the sequences
        if token.pos_ == "PUNCT" and last_proper_noun_seq == []:
            first_proper_noun_seq = []
            last_proper_noun_seq = []
            intermediate_seq = []

        # If the token is a proper noun, add it to the proper noun sequence
        elif token.pos_ == "PROPN":
            if intermediate_seq:
                last_proper_noun_seq.append(token.text)
            else:
                first_proper_noun_seq.append(token.text)

        # If the token is not a proper noun, check if the proper noun sequence is complete or add the token to the intermediate sequence
        else:
            if first_proper_noun_seq and last_proper_noun_seq:
                if contains_verb(intermediate_seq):
                    triplet = build_triplet(first_proper_noun_seq, intermediate_seq, last_proper_noun_seq)
                    triplets.append(triplet)

                if token.pos_ == "PUNCT":
                    first_proper_noun_seq = []
                    last_proper_noun_seq = []
                    intermediate_seq = []
                else:
                    first_proper_noun_seq = last_proper_noun_seq
                    last_proper_noun_seq = []
                    intermediate_seq = [token]

            elif first_proper_noun_seq:
                intermediate_seq.append(token)

    if first_proper_noun_seq and last_proper_noun_seq:
        if contains_verb(intermediate_seq):
            triplet = build_triplet(first_proper_noun_seq, intermediate_seq, last_proper_noun_seq)
            triplets.append(triplet)

    return triplets


def find_proper_noun_heads(doc):
    """
    Return all tokens that are 'proper-noun heads':
    i.e., tokens with POS == 'PROPN' and dep_ != 'compound'.
    """
    proper_noun_heads = [
        token for token in doc
        if token.pos_ == "PROPN" and token.dep_ != "compound"
    ]
    return proper_noun_heads


def extract_dependencies_triplets(nlp, text):
    doc = nlp(text)
    triplets = []

    proper_noun_heads = find_proper_noun_heads(doc)
    head_to_set = {}
    for token in proper_noun_heads:
        proper_noun_set = []
        for child in token.children:
            if child.dep_ == "compound":
                proper_noun_set.append(child.text)
        proper_noun_set.append(token.text)
        head_to_set[token] = proper_noun_set

    # Iterate through pairs of proper noun heads
    for h1 in proper_noun_heads:
        for h2 in proper_noun_heads:
            if h1 == h2:
                continue  # Skip self-pairs

            # Condition #1: h1 and h2 share the same head, with specific dependencies
            if h1.head == h2.head and h1.dep_ == "nsubj" and h2.dep_ == "dobj":
                relation = h1.head.text  # The shared head token
                subject = " ".join(head_to_set[h1])
                obj = " ".join(head_to_set[h2])
                triplets.append((subject, relation, obj))

            # Condition #2: h1's parent is h2's grandparent, with specific dependencies
            elif h1.head == h2.head.head and h1.dep_ == "nsubj" and h2.head.dep_ == "prep" and h2.dep_ == "pobj":
                relation = f"{h1.head.text} {h2.head.text}"  # Concatenation of parent and grandparent tokens
                subject = " ".join(head_to_set[h1])
                obj = " ".join(head_to_set[h2])
                triplets.append((subject, relation, obj))

    return triplets


def extract_llm_triplets(page_name):
    genai.configure(api_key='AIzaSyDKV8hJkVf-BYWlX6L9Mw8l8Z9Skq-8GUM')
    model = genai.GenerativeModel('gemini-pro')

    prompt = f"""
    You are an advanced information extraction model. Your task is to extract triplets of the form (Subject, Relation, Object) from {page_name} Wikipedia page. Follow these guidelines strictly:

    Subject and Object should be proper nouns (e.g., names of people, places, or organizations).
    Relation should consist of:
    - A verb or
   -  A verb combined with a preposition (e.g., "appointed to").
    If no valid triplet exists in a sentence, skip it.

    Here is an example to guide you:    
    Input: "Brad Pitt married Angelina Jolie."
    Output: ("Brad Pitt", "married", "Angelina Jolie")
    Input: "Neil Gorsuch was appointed to the Supreme Court."
    Output: ("Neil Gorsuch", "was appointed to", "Supreme Court")

    Provide the output as a list of triplets in this format:
    ("Subject1", "Relation1", "Object1")
    ("Subject2", "Relation2", "Object2")
    ...

    Include all valid triplets from the Wikipedia page in the response. Do not include additional commentary or explanations, only the list of triplets.
    """

    return model.generate_content(prompt)


def main():
    nlp = spacy.load("en_core_web_sm")
    wiki_pages = {"Donald Trump": wikipedia.page("Donald Trump", auto_suggest=False).content,
                  "Ruth Bader Ginsburg": wikipedia.page("Ruth Bader Ginsburg", auto_suggest=False).content,
                  "J.K. Rowling": wikipedia.page("J.K. Rowling", auto_suggest=False).content}

    for page_name, page in wiki_pages.items():
        pos_triplets = extract_pos_triplets(nlp, page)
        dep_triplets = extract_dependencies_triplets(nlp, page)
        print(f"--- {page_name} ---")
        print(f"Number of POS Based Triplets: {len(pos_triplets)}")
        print(f"Number of Dependency Based Triplets: {len(dep_triplets)}\n")

    print("LLM Triplets:")
    for page_name in wiki_pages:
        print(f"--- {page_name} ---")
        response = extract_llm_triplets(page_name)
        llm_triplets = [line.strip() for line in response.text.split('\n')]
        print(f"Number of LLM Triplets: {len(llm_triplets)}\n")


if __name__ == "__main__":
    main()



