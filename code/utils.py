import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.metrics import precision_score, recall_score, f1_score
import spacy
import networkx as nx

# download required nltk packages
# required for tokenization
nltk.download('punkt', quiet=True)
# required for parts of speech tagging
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')

def get_head_word(constituent):
    global nlp
    sentence = u' '.join(constituent)
    doc = nlp(sentence)
    head = list(doc.sents)[0]
    return head.root.text, head.root.pos_, head.root.ent_type_

def get_shortest_path(words, trigger, head):
    global nlp
    doc = nlp(u''+' '.join(words))
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token),
                          '{0}'.format(child)))
    graph = nx.Graph(edges)
    entity1 = trigger
    entity2 = head
    if entity1 not in graph.nodes or entity2 not in graph.nodes:
        return -1
    return nx.shortest_path_length(graph, source=entity1, target=entity2)

def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_word(word, nltk_tag):
    tag = nltk_pos_tagger(nltk_tag)
    return lemmatizer.lemmatize(word, tag)

def pos_sentence(sentence):
    return nltk.pos_tag(sentence)

def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(sentence)
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
    lemmatized_sentence = []

    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return lemmatized_sentence

def extract_noun(sentence):
    pos_words = pos_sentence(sentence)
    nouns = []
    for tag_word in pos_words:
        if tag_word[1].startswith('N'):
            nouns.append(lemmatize_word(tag_word[0], tag_word[1]))
    return nouns

def score_trigger_detection(preds, all_tags):
    tp, tn, fp, fn = 0, 0, 0, 0
    for pred_tags, tags in zip(preds, all_tags):
        for pred_tag, tag in zip(pred_tags, tags):
            tp += (pred_tag==1 and tag==1)
            tn += (pred_tag==0 and tag==0)
            fp += (pred_tag==1 and tag==0)
            fn += (pred_tag==0 and tag==1)
    if (tp==0):
        recall=0
    else:
        recall = tp/(tp+fn)
    if (tp==0):
        precision=0
    else:
        precision = tp/(tp+fp)
    if recall==0 or precision==0:
        f1=0
    else:
        f1 = 2*recall*precision/(recall+precision)
    return recall, precision, f1

def score_trigger_classification(preds, all_tags):
    pred_trigger = []
    for pred in preds:
        pred_trigger.extend([tag for tag in pred if tag!='None'])
    true_trigger = []
    for tags in all_tags:
        true_trigger.extend([tag for tag in tags if tag!='None'])
    recall = recall_score(y_true=true_trigger, y_pred=pred_trigger, average='macro', zero_division=0)
    precision = precision_score(y_true=true_trigger, y_pred=pred_trigger, average='macro', zero_division=0)
    f1 = f1_score(y_true=true_trigger, y_pred=pred_trigger, average='macro', zero_division=0)
    return recall, precision, f1

def score_augment_classification(preds, all_roles_lst):
    pred_roles = []
    for roles_lst in preds:
        for roles in roles_lst:
            pred_roles.extend(roles)
    true_roles = []
    for roles_lst in all_roles_lst:
        for roles in roles_lst:
            true_roles.extend(roles)
    recall = recall_score(y_true=true_roles, y_pred=pred_roles, average='macro', zero_division=0)
    precision = precision_score(y_true=true_roles, y_pred=pred_roles, average='macro', zero_division=0)
    f1 = f1_score(y_true=true_roles, y_pred=pred_roles, average='macro', zero_division=0)
    return recall, precision, f1

def score_argument_classification(all_preds, all_events):
    tp, fp, fn =0, 0, 0
    for preds, events in zip(all_preds, all_events):
        for pred, event in zip(preds, events):
            pred_dict = pred[1]
            event_dict = event[1]
            for item in event_dict.items():
                if item in pred_dict.items():
                    tp+=1
                else:
                    fn+=1
            for item in pred_dict.items():
                if item not in event_dict.items():
                    fp+=1
    if tp+fp == 0:
        precision = 0
    else:
        precision = tp/(tp+fp)
    if tp+fn==0:
        recall = 0
    else:
        recall = tp/(tp+fn)
    if precision + recall ==0:
        f1=0
    else:
        f1 = (2*precision*recall)/(precision+recall)
    return recall, precision, f1

if __name__ == '__main__':
    from dataset import argument_classification_set
    set = argument_classification_set('./data/valid.json')
    print(score_argument_classification(set.all_events, set.all_events))
    '''
    print(lemmatize_word('voting', 'V'))
    dataset = trigger_set('./data/train.json')
    present = time.time()
    for sentence in dataset.sentences:
        lemmatize_sentence(sentence)
    end = time.time()
    print("total time (s):", end-present)
    print("avg time (s):", (end-present)/len(dataset.sentences))
    '''