def extract_main_reason(text):
    doc = nlp(text)
    main_reason = []
    
    # Identify the main verb and complement
    for token in doc:
        if token.dep_ in ('ROOT', 'prep', 'xcomp', 'advcl'):
            # Extract the most relevant phrases connected to the root
            phrase = ' '.join([child.text for child in token.subtree if child.dep_ not in ('punct', 'cc', 'conj')])
            main_reason.append(phrase)
    
    # Return the simplified main reason
    return ' '.join(main_reason).strip()