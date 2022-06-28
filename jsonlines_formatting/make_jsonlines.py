# pre-processing the data
# writing as jsonl file

from jsonlines_formatting.reader import *
from jsonlines_formatting.utils_mapping import *
from transformers import BertTokenizerFast
from spacy_mention_extraction.collect_markables_only_spacy import extract_chunks_as_markables

from util import flatten
import json


def create_doc_obj(doc_id, doc_prefix, doc_tokens, doc_sentences, doc_speakers, doc_clusters, doc_sentence_map, doc_subtoken_map):
    doc_obj = {'doc_key':doc_prefix+'/'+doc_id, 'tokens':doc_tokens, 'sentences':doc_sentences, 'speakers':doc_speakers, 'constituents':[], 'ner':[], 'clusters':doc_clusters, 'sentence_map':doc_sentence_map, 'subtoken_map':doc_subtoken_map, 'pronouns':[]}
    return doc_obj


def get_token_id2offset(subtoken_map):
    token_id2offset = dict()        
    s = 0
    e = 0
    prev = subtoken_map[0]
    for i, t in enumerate(subtoken_map):
        if t!=prev:
            token_id2offset[prev] = (s, e)
            s = i
            e = i
        e = i
        prev = t
    if subtoken_map[-1]==prev:
        token_id2offset[prev] = (s, e)   
    return token_id2offset         


def split_docs_into_segments(doc_name, doc_prefix, doc_list):
    gold_docs = get_all_docs(doc_name)
    doc_object_list = []
    for doc in gold_docs:        
        doc_lines, doc_sentence_lines = gold_docs[doc]
        doc_sentences = get_doc_sentences(doc, doc_sentence_lines)
        pred_markables = extract_chunks_as_markables(doc_name, doc_sentences)
        doc_clusters, _, id2markable = get_doc_markables(doc, doc_lines, extract_MIN=True, keep_bridging=False, word_column=1, markable_column=10, bridging_column=11, print_debug=False)      

        segment_bert_tokens = []
        segment_speakers = []
        segment_sentence_map = []
        segment_subtoken_map = []

        doc_clusters_remapped = []
        doc_bert_tokens = []
        doc_tokens = []
        doc_speakers = []
        doc_sentence_map = []
        doc_subtoken_map = []
        global_offset = 0
        segment_offset = 0
        
        for sentence_map_id, s in enumerate(doc_sentences):
            stokenized = tokenizer.tokenize(s.text)
            if len(stokenized) + 2 > max_segment_length:
                print("WARNING: sentence longer than segment, need to break up mid-sentence")
            if len(stokenized)+segment_offset+2>max_segment_length:
                doc_bert_tokens.append(['[CLS]']+segment_bert_tokens+['[SEP]'])
                doc_speakers.append(['[SPL]']+segment_speakers+['[SPL]'])
                segment_sentence_map = [segment_sentence_map[0]]+segment_sentence_map+[segment_sentence_map[-1]]
                doc_sentence_map.extend(segment_sentence_map)
                segment_subtoken_map = [segment_subtoken_map[0]]+segment_subtoken_map+[segment_subtoken_map[-1]]
                doc_subtoken_map.extend(segment_subtoken_map)
                
                segment_bert_tokens = []
                segment_speakers = []
                segment_sentence_map = []
                segment_subtoken_map = []
                segment_offset = 0

            segment_bert_tokens.extend(stokenized)
            cur_speaker = s.speaker
            str_tokens = [t[1] for t in s.tokens]
                        
            doc_tokens.extend(str_tokens)
            segment_speakers.extend([cur_speaker]*len(stokenized))
            segment_sentence_map.extend([sentence_map_id]*len(stokenized))
            sentence_subtoken_map = []

            bert_token_mapping = map_orig_tokens_to_bert_tokens(stokenized, str_tokens) 
            for token_num, tmap in enumerate(bert_token_mapping):
                sentence_subtoken_map.extend([token_num+global_offset]*len(tmap))
            """
            subtoken_offsets = tokenizer(str_tokens, add_special_tokens=False, is_split_into_words=True,
                                         return_offsets_mapping=True)["offset_mapping"]
            full_tok_idx = -1
            for start, _ in subtoken_offsets:
                if start == 0:
                    full_tok_idx += 1
                sentence_subtoken_map.append(full_tok_idx+global_offset)
            """
            segment_subtoken_map.extend(sentence_subtoken_map)

            segment_offset+=len(stokenized)
            global_offset+=len(str_tokens)

        if len(segment_bert_tokens)>0:
            doc_bert_tokens.append(['[CLS]']+segment_bert_tokens+['[SEP]'])
            doc_speakers.append(['[SPL]']+segment_speakers+['[SPL]'])
            segment_sentence_map = [segment_sentence_map[0]]+segment_sentence_map+[segment_sentence_map[-1]]            
            doc_sentence_map.extend(segment_sentence_map)
            segment_subtoken_map = [segment_subtoken_map[0]]+segment_subtoken_map+[segment_subtoken_map[-1]]
            doc_subtoken_map.extend(segment_subtoken_map)

        token_id2offset = get_token_id2offset(doc_subtoken_map)           
        for cluster_i, doc_cluster in doc_clusters.items():
            subtokens_cluster = []
            for cluster_mention in doc_cluster[0]:
                start_token = cluster_mention.start
                end_token = cluster_mention.end
                if not start_token in token_id2offset:
                    raise Exception('Wrong start token:', start_token, token_id2offset)                    
                start = token_id2offset[start_token][0]
                if end_token+1 in token_id2offset:
                    end = token_id2offset[end_token+1][0]-1 # for split cases like "I pre #sume ..."
                else:
                    end = token_id2offset[end_token][1]
                mention_subtokens_span = (start, end)
                subtokens_cluster.append(mention_subtokens_span)
            if len(subtokens_cluster)>0:
                doc_clusters_remapped.append(subtokens_cluster)

        mentions = [(m.start, m.end) for m in pred_markables]

        # create word -> [subtokens] dict
        tok_to_subtokens = {}
        all_subtokens = flatten(doc_bert_tokens)
        for subtok_idx in range(len(doc_subtoken_map)):
            if all_subtokens[subtok_idx] in {tokenizer.cls_token, tokenizer.sep_token}:
                continue
            if doc_subtoken_map[subtok_idx] not in tok_to_subtokens:
                tok_to_subtokens[doc_subtoken_map[subtok_idx]] = [subtok_idx]
            else:
                tok_to_subtokens[doc_subtoken_map[subtok_idx]].append(subtok_idx)
        # map mentions to subtokens
        subtok_mentions = []
        for mention in mentions:
            subtok_mentions.append((tok_to_subtokens[mention[0]][0], tok_to_subtokens[mention[1]][-1]))
        doc_obj = create_doc_obj(doc, doc_prefix, doc_tokens, doc_bert_tokens, doc_speakers, doc_clusters_remapped, doc_sentence_map, doc_subtoken_map)
        doc_obj["mentions"] = subtok_mentions
        doc_object_list.append(doc_obj)
    doc_list.extend(doc_object_list)                


def save_as_json_doc(doc_name, doc_list):
    with open(doc_name+'.jsonl', 'w') as f:
        for doc_obj in doc_list:
            json_string = json.dumps(doc_obj)
            f.write(json_string+'\n')


#####
# (1) Markable: object with the following fields: doc_name, start, end, MIN, is_referring, words, is_split_antecedent, split_antecedent_mebers
# (2) Sentence: object with the following fields: speaker, sent_id, text, tokens [list of (token_id, token_str)], original_lines

# Example of a markable 'the cash' span: (7313,7314) - inclusive!

if len(sys.argv)<5:
    print('You need to define the data mode: test, train or dev!')
    print('Also, make sure that you specified the data directory. E.g., ../CRAC_data/')
    print('3rd arg: tokenizer name')
    print('4th arg: segment length')
    sys.exit(3)
mode = sys.argv[1] # train, dev, test
input_dir = sys.argv[2] # ../CRAC_data/
data_prefixes = ['2022_AMI_'+mode+'_v0', 'light_'+mode+'.2022', 'Persuasion_'+mode+'.2022',
                 "RST_DTreeBank_"+mode+".ARRAU3.0", "Switchboard_"+mode+".2022"]
doc_prefixes = ['ami', 'light', 'persuasion', "arrau", "switchboard"]
tokenizer = BertTokenizerFast.from_pretrained(sys.argv[3])
max_segment_length = int(sys.argv[4]) #512 or 384
json_doc_name = '2022_SharedTask_'+mode+"_"+str(max_segment_length)# '2022_SharedTask_train' '2022_SharedTask_dev' '2022_SharedTask_test'

doc_list = []
for i, data_prefix in enumerate(data_prefixes):
    input_file = input_dir+data_prefix+'.CONLLUA'
    print('Processing:', input_file)
    doc_prefix = doc_prefixes[i]
    split_docs_into_segments(input_file, doc_prefix, doc_list) 
save_as_json_doc(json_doc_name, doc_list)

