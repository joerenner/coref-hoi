from copy import deepcopy
# this code is needed to map between the original tokens and the output of BERT Tokenizer

def align(bert_str, original_str):
    bert2orig = []
    orig2bert = []
    bert_idx = 0
    orig_idx = 0
    bert_len = len(bert_str)
    orig_len = len(original_str)
    while bert_idx<bert_len and orig_idx<orig_len:
        bert_chr = bert_str[bert_idx]
        orig_chr = original_str[orig_idx]
        # [UNK] replacement will not work
        # if the next token in the bert string
        # is part of the original string replaced by [UNK]
        # in bert tokenization
        if bert_chr=='[':
            unk_len = 0
            unk_idx = bert_idx
            while bert_str[unk_idx:unk_idx+5]=='[UNK]':
               unk_len += 5
               unk_idx += 5
            if unk_len > 0:
                for i in range(bert_idx, bert_idx+unk_len):
                    bert2orig.append((i,orig_idx))
                    orig2bert.append((orig_idx,i))
                bert_idx+=unk_len
                bert_chr = bert_str[bert_idx]
                while orig_idx<len(original_str) and orig_chr!=bert_chr:                 
                    orig_idx+=1
                    orig_chr = original_str[orig_idx]
                continue
                
        if bert_chr==orig_chr:
            bert2orig.append((bert_idx,orig_idx))
            orig2bert.append((orig_idx,bert_idx))
            orig_idx+=1
            bert_idx+=1
        else:
            i = bert_idx
            j = orig_idx
            while i<bert_len:
                if bert_str[i]==original_str[j]:
                    bert2orig.append((i,j))
                    orig2bert.append((j,i))
                    bert_idx = i
                    orig_idx = j
                    break
                i+=1
            orig_idx+=1
            bert_idx+=1
    return bert2orig, orig2bert

def join_tokens(tokens):
    joined = ''.join(tokens)
    split_points = []
    point = 0
    for t in tokens:
        split_points.append((point, point+len(t)))
        point += len(t)
    return joined, split_points

def collect_tuples_in_span(input_tuples, span):
    list_of_tuples = []
    for tpl in input_tuples:
        if tpl[0]>=span[0] and tpl[0]<span[1]:
            list_of_tuples.append(tpl)
    return list_of_tuples

def get_spans_given_positions(positions, input_spans):
    spans = []
    token_indices = []
    idx = 0
    for sp in input_spans:
        for position in positions:
            if position>=sp[0] and position<sp[1]:
                spans.append(sp)
                token_indices.append(idx)
                break
        idx+=1
    return spans, token_indices

def map_orig_tokens_to_bert_tokens(bert_tokens, orig_tokens, orig_span):
    bert_indices = []
    bert_str, bert_split_points = join_tokens(bert_tokens)
    orig_str, orig_split_points = join_tokens(orig_tokens)
    bert2orig, orig2bert = align(bert_str, orig_str)
    
    # iterate over all tokens in span
    for orig_tkn_idx in range(orig_span[0],orig_span[1]):
        # find the original token span
        tkn_span_orig = orig_split_points[orig_tkn_idx]
        # find the mapping between the orig and bert chars
        # that belong to the same token 
        tpl_list = collect_tuples_in_span(orig2bert, tkn_span_orig)
        # extract the corresponding bert chars
        positions_bert = [el[1] for el in tpl_list]
        # get bert token spans that correspond to the orig token
        # and bert token ids that correspond to the orig token
        bert_spans, bert_tkn_idx = get_spans_given_positions(positions_bert, bert_split_points)
        bert_indices.extend(bert_tkn_idx)
    return bert_indices


def map_orig_tokens_to_bert_tokens(bert_tokens, orig_tokens):
    bert_indices = []
    bert_str, bert_split_points = join_tokens(bert_tokens)
    orig_str, orig_split_points = join_tokens(orig_tokens)
    bert2orig, orig2bert = align(bert_str, orig_str)

    # iterate over all tokens in span
    for orig_tkn_idx in range(len(orig_tokens)):
        # find the original token span
        tkn_span_orig = orig_split_points[orig_tkn_idx]
        # find the mapping between the orig and bert chars
        # that belong to the same token 
        tpl_list = collect_tuples_in_span(orig2bert, tkn_span_orig)
        # extract the corresponding bert chars
        positions_bert = [el[1] for el in tpl_list]
        # get bert token spans that correspond to the orig token
        # and bert token ids that correspond to the orig token
        bert_spans, bert_tkn_idx = get_spans_given_positions(positions_bert, bert_split_points)
        bert_indices.append(bert_tkn_idx)
    return bert_indices


