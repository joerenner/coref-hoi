# pre-processing the data
# finding the markables
import os
import sys
from spacy_mention_extraction.reader import *
import spacy
from spacy.tokens import Doc

class WhitespaceTokenizer(object):
        def __init__(self, vocab):
                self.vocab = vocab

        def __call__(self, text):
                words = text.split(' ')
                # All tokens 'own' a subsequent space character in this tokenizer
                spaces = [True] * len(words)
                return Doc(self.vocab, words=words, spaces=spaces)

#nlp = spacy.load("en_core_web_trf")
nlp = spacy.load("en_core_web_md")
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def check_extended_span(doc, chunk):
    stree_indices = []
    new_words = []
    for w in doc:
        if w.head==chunk.root and w.dep_ in ['prep', 'relcl']:
            for el in w.subtree:
                stree_indices.append(el.i)
                new_words.append(el.text)
            return stree_indices[-1], new_words
    return -1, new_words

def has_markable(markables, m):
    for m2 in markables:
        if m.start==m2.start and m.end==m2.end:
            return True 
    return False
    
# finding chunks (markables) with SpaCy
# doc_name is a String
# doc_sentences is a list of Sentence objects
# (1) Sentence: object with the following fields: sent_id, text, tokens: (token_id, token_str)
# defined in reader.py
# (2) Markable: object with the following fields: doc_name, start, end, MIN, is_referring, words, is_split_antecedent, split_antecedent_members
# defined in markable.py
def extract_chunks_as_markables(doc_name, doc_sentences):
    doc_markables = []
    tokens_so_far = 0
    for sentence in doc_sentences:
        pretokenized = ' '.join([t for (tidx, t) in sentence.tokens])
        doc = nlp(pretokenized)
        processed_singleton_ids = []
        for chunk in doc.noun_chunks:
            words = chunk.text.split()
            start = chunk[0].i+tokens_so_far
            # expand the span if PPs were not extracted
            old_span_end = chunk[-1].i
            new_span_end, new_words = check_extended_span(doc, chunk)
            if (new_span_end > 0) and (new_span_end > old_span_end):
                end = new_span_end+tokens_so_far
                words += new_words
            else:
                end = old_span_end+tokens_so_far
            MIN = (chunk.root.i+tokens_so_far, chunk.root.i+tokens_so_far)
            is_referring = None
            m = markable.Markable(doc_name, start, end, MIN, is_referring, words, is_split_antecedent=False,split_antecedent_members=set())
            doc_markables.append(m)
            if len(chunk)==1:
                processed_singleton_ids.append(chunk[0].i)
        # special cases that SpaCy doesn't cosider as NPs and
        # if we have inside the chunk a pronoun: my, your etc.        
        for w in doc:
            if w.text.lower() in ['here', 'there', 'now', 'then'] or w.pos_=='PRON' and not w.i in processed_singleton_ids: # 'this', 'that' make f-score lower!
                start = w.i+tokens_so_far
                end = start
                MIN = (w.i+tokens_so_far, w.i+tokens_so_far)
                is_referring = None
                m = markable.Markable(doc_name, start, end, MIN, is_referring, [w.text],is_split_antecedent=False,split_antecedent_members=set())
                doc_markables.append(m)                    
        tokens_so_far+=len(sentence.tokens)
    return doc_markables

# tokens: list of tokens
# returns: list of sent mentions (list of (start, end) tuples)
def extract_token_chunks_as_markables(doc_tokens):
    doc_markables = []
    pretokenized = ' '.join(doc_tokens)
    doc = nlp(pretokenized)
    processed_singleton_ids = []
    for chunk in doc.noun_chunks:
        words = chunk.text.split()
        start = chunk[0].i
        # expand the span if PPs were not extracted
        old_span_end = chunk[-1].i
        new_span_end, new_words = check_extended_span(doc, chunk)
        if (new_span_end > 0) and (new_span_end > old_span_end):
            end = new_span_end
            words += new_words
        else:
            end = old_span_end
        doc_markables.append((start, end))
        if len(chunk)==1:
            processed_singleton_ids.append(chunk[0].i)
    # special cases that SpaCy doesn't cosider as NPs and
    # if we have inside the chunk a pronoun: my, your etc.
    for w in doc:
        if w.text.lower() in ['here', 'there', 'now', 'then'] or w.pos_=='PRON' and not w.i in processed_singleton_ids: # 'this', 'that' make f-score lower!
            start = w.i
            end = start
            doc_markables.append((start, end))
    return doc_markables

def print_mention_extraction_stats(doc_name, debug_show=False):
    gold_docs = get_all_docs(doc_name)
    doc2sentences = dict()
    doc2markables = dict()
    doc2clusters = dict()
    for doc in gold_docs:
        doc_lines, doc_sentence_lines = gold_docs[doc]
        sentences = get_doc_sentences(doc, doc_sentence_lines)
        doc2sentences[doc] = sentences
        clusters, bridging_pairs, id2markable = get_doc_markables(doc, doc_lines, extract_MIN=True, keep_bridging=False, word_column=1, markable_column=10, bridging_column=11, print_debug=False)
        doc2markables[doc] = [id2markable[i] for i in id2markable] 
        doc2clusters[doc] = clusters
        #for id in id2markable:
        #    markable_tokens, sentence = find_tokens(id2markable[id], sentences)
        #    print(markable_tokens, '***', sentence.text) # doc, id, id2markable[id], 
    macrof1 = 0
    totaltp = 0
    totalfp = 0
    totalfn = 0 
    doc_n = 0
    for doc in doc2sentences:
        doc_sentences = doc2sentences[doc]
        doc_markables = doc2markables[doc]
        extracted_markables = extract_chunks_as_markables(doc, doc_sentences)
        tp = 0
        fp = 0
        fn = 0
        gold_mentions = []
        for m in doc_markables:
            gold_mentions.append((' '.join(m.words),m.start,m.end))
            if has_markable(extracted_markables, m):
                #print('Correct:', ' '.join(m.words))
                tp+=1
            else:
                #print('False negative:', ' '.join(m.words))
                fn+=1        
        threshold = 20
        if debug_show:
            print('Golden mentions:', gold_mentions[:threshold])
            print('*****')
            print('SpaCy mentions:', [(' '.join(m.words),m.start,m.end) for m in extracted_markables][:threshold])
        
        for m in extracted_markables:
            if not has_markable(doc_markables, m):
                #print('False positive:', ' '.join(m.words))
                fp+=1
        totaltp+=tp
        totalfp+=fp
        totalfn+=fn
        prec = 0
        rec = 0
        if tp+fp>0:
            prec = round(tp/(tp+fp),2)        
        if tp+fn>0:
            rec = round(tp/(tp+fn),2)
        if prec+rec>0:
            fscore = round(2*prec*rec/(prec+rec),2)
            
        if debug_show:
            print('Doc:', doc, 'tp:', tp, 'fp:', fp, 'fn:', fn)
            print('precision:', prec)
            print('recall:', prec)
            print('f-score:', fscore)
            print()
        macrof1+=fscore
        doc_n += 1
        #for cluster_id in doc2clusters[doc]:
        #    print(cluster_id, [' '.join(m.words) for m in doc2clusters[doc][cluster_id][0]])       
        #for sentence in doc_sentences:
        #    print(sentence.speaker, ':', sentence.text)     
    microprec = totaltp/(totaltp+totalfp)
    microrec = totaltp/(totaltp+totalfn)
    microf1 = 2*microprec*microrec/(microprec+microrec)
    print(doc_name)
    print('Micro Precision:', round(microprec,3))
    print('Micro Recall:', round(microrec,3))
    print('Micro F1:', round(microf1,3))

def main():
	input_dir = sys.argv[1] #'CRAC_data_2022/'
	for doc_name in os.listdir(input_dir):
		print_mention_extraction_stats(input_dir+doc_name)
	#../../CRAC_data_2022/original/light_dev.2022.CONLLUA
	#spacy
	#Micro Precision: 0.902
	#Micro Recall: 0.793
	#Micro F1: 0.844

if __name__ == '__main__':
    main()
