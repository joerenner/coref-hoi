# source: https://github.com/juntaoy/universal-anaphora-scorer/blob/main/coval/ua/reader.py
# with some modifications:
# added Sentence class
# return id2markable in get_doc_markables()

from os import walk
from os.path import isfile, join
from collections import deque
import sys

import markable as markable

__author__ = 'ns-moosavi; juntaoy'

def get_doc_markables(doc_name, doc_lines, extract_MIN, keep_bridging, word_column=1,
    markable_column=10, bridging_column=11, print_debug=False):
  markables_cluster = {}
  markables_animacy = {}
  markables_start = {}
  markables_end = {}
  markables_MIN = {}
  markables_coref_tag = {}
  markables_split = {} # set_id: [markable_id_1, markable_id_2 ...]
  bridging_antecedents = {}
  all_words = []
  stack = []
  for word_index, line in enumerate(doc_lines):
    columns = line.split()
    all_words.append(columns[word_column])

    if columns[markable_column] != '_':
      markable_annotations = columns[markable_column].split("(")
      nom_sem_annotations = columns[14].split("(")[1:]
      if markable_annotations[0]:
        #the close bracket
        for _ in range(len(markable_annotations[0])):
          markable_id = stack.pop()
          markables_end[markable_id] = word_index

      for idx, markable_annotation in enumerate(markable_annotations[1:]):
        if markable_annotation.endswith(')'):
          single_word = True
          markable_annotation = markable_annotation[:-1]
        else:
          single_word = False
        markable_animacy = False  
        if idx<len(nom_sem_annotations):
            nom_sem_annotation = nom_sem_annotations[idx]
            if nom_sem_annotation.endswith(')'):
               nom_sem_annotation = nom_sem_annotation[:-1]
            #print(nom_sem_annotation, nom_sem_annotations, idx)
            nom_sem_info = {p[:p.find('=')]:p[p.find('=')+1:] for p in nom_sem_annotation.split('|')}

            if nom_sem_info['Entity_Type']=='person' or nom_sem_info['Entity_Type']=='animate':
                markable_animacy = True

        markable_info = {p[:p.find('=')]:p[p.find('=')+1:] for p in markable_annotation.split('|')}
        markable_id = markable_info['MarkableID']
        cluster_id = markable_info['EntityID']
        markables_animacy[markable_id] = markable_animacy
        markables_cluster[markable_id] = cluster_id
        markables_start[markable_id] = word_index
        if single_word:
          markables_end[markable_id] = word_index
        else:
          stack.append(markable_id)

        markables_MIN[markable_id] = None
        if extract_MIN and 'Min' in markable_info:
          MIN_Span = markable_info['Min'].split(',')
          if len(MIN_Span) == 2:
            MIN_start = int(MIN_Span[0]) - 1
            MIN_end = int(MIN_Span[1]) - 1
          else:
            MIN_start = int(MIN_Span[0]) - 1
            MIN_end = MIN_start
          markables_MIN[markable_id] = (MIN_start,MIN_end)

        markables_coref_tag[markable_id] = 'referring'
        if cluster_id.endswith('-Pseudo'):
          markables_coref_tag[markable_id] = 'non_referring'

        if 'ElementOf' in markable_info:
          element_of = markable_info['ElementOf'].split(',') # for markable participate in multiple plural using , split the element_of, e.g. ElementOf=1,2
          for ele_of in element_of:
            if ele_of not in markables_split:
              markables_split[ele_of] = []
            markables_split[ele_of].append(markable_id)
    if keep_bridging and columns[bridging_column] != '_':
      bridging_annotations = columns[bridging_column].split("(")
      for bridging_annotation in bridging_annotations[1:]:
        if bridging_annotation.endswith(')'):
          bridging_annotation = bridging_annotation[:-1]
        bridging_info = {p[:p.find('=')]:p[p.find('=')+1:] for p in bridging_annotation.split('|')}
        bridging_antecedents[bridging_info['MarkableID']] = bridging_info['MentionAnchor']

  clusters = {}
  id2markable = {}
  for markable_id in markables_cluster:
    m = markable.Markable(
        doc_name, markables_start[markable_id],
        markables_end[markable_id], markables_MIN[markable_id],
        markables_coref_tag[markable_id],
        all_words[markables_start[markable_id]:
            markables_end[markable_id] + 1])
    m.animacy = markables_animacy[markable_id]
    id2markable[markable_id] = m
    if markables_cluster[markable_id] not in clusters:
      clusters[markables_cluster[markable_id]] = (
          [], markables_coref_tag[markable_id],doc_name,[markables_cluster[mid] for mid in markables_split.get(markables_cluster[markable_id],[])])
    clusters[markables_cluster[markable_id]][0].append(m)

  bridging_pairs = {}
  for anaphora, antecedent in bridging_antecedents.items():
    if not anaphora in id2markable or not antecedent in id2markable:
      print('Skip bridging pair ({}, {}) as markable_id does not exist in identity column!'.format(antecedent,anaphora))
      continue
    bridging_pairs[id2markable[anaphora]] = id2markable[antecedent]

  #print([(str(ana),str(ant)) for ana,ant in bridging_pairs.items()])
  # for cid in clusters:
  #   cl = clusters[cid]
  #   print(cid,[str(m) for m in cl[0]],cl[1],cl[2],cl[3] )
  return clusters, bridging_pairs, id2markable


def process_clusters(clusters, keep_singletons, keep_non_referring,keep_split_antecedent):
  removed_non_referring = 0
  removed_singletons = 0
  processed_clusters = []
  processed_non_referrings = []

  for cluster_id, (cluster, ref_tag, doc_name, split_cid_list) in clusters.items():
    #recusively find the split singular cluster
    if split_cid_list and keep_split_antecedent:
      # if using split-antecedent, we shouldn't remove singletons as they might be used by split-antecedents
      assert keep_singletons
      split_clusters = set()
      queue = deque()
      queue.append(cluster_id)
      while queue:
        curr = queue.popleft()
        curr_cl, curr_ref_tag, doc_name, curr_cid_list = clusters[curr]
        #non_referring shouldn't be used as split-antecedents
        # if curr_ref_tag != 'referring':
        #   print(curr_ref_tag, doc_name, curr_cid_list)
        if curr_cid_list:
          for c in curr_cid_list:
            queue.append(c)
        else:
          split_clusters.add(tuple(curr_cl))
      split_m = markable.Markable(
        doc_name, -1,
        -1, None,
        'referring',
        '',
        is_split_antecedent=True,
        split_antecedent_members=split_clusters)

      cluster.append(split_m) #add the split_antecedents

    if ref_tag == 'non_referring':
      if keep_non_referring:
        processed_non_referrings.append(cluster[0])
      else:
        removed_non_referring += 1
      continue

    if not keep_singletons and len(cluster) == 1:
      removed_singletons += 1
      continue

    processed_clusters.append(cluster)

  if keep_split_antecedent:
    #step 2 merge equivalent split-antecedents clusters
    merged_clusters = []
    for cl in processed_clusters:
      existing = None
      for m in cl:
        if m.is_split_antecedent:
        #only do this for split-antecedents
          for c2 in merged_clusters:
            if m in c2:
              existing = c2
              break
      if existing:
        # print('merge cluster ', [str(m) for m in cl], ' and ', [str(m) for m in existing])
        existing.update(cl)
      else:
        merged_clusters.append(set(cl))
    merged_clusters = [list(cl) for cl in merged_clusters]
  else:
    merged_clusters = processed_clusters

  return (merged_clusters, processed_non_referrings,
      removed_non_referring, removed_singletons)


def get_coref_infos(key_file,
    sys_file,
    keep_singletons,
    keep_split_antecedent,
    keep_bridging,
    keep_non_referring,
    evaluate_discourse_deixis,
    use_MIN,
    print_debug=False):
  key_docs = get_all_docs(key_file)
  sys_docs = get_all_docs(sys_file)

  doc_coref_infos = {}
  doc_non_referrig_infos = {}
  doc_bridging_infos = {}

  for doc in key_docs:

    if doc not in sys_docs:
      print('The document ', doc,
          ' does not exist in the system output.')
      continue
    markable_column = 12 if evaluate_discourse_deixis else 10
    key_clusters, key_bridging_pairs, _ = get_doc_markables(doc, key_docs[doc][0], use_MIN, keep_bridging,markable_column=markable_column)
    sys_clusters, sys_bridging_pairs, _ = get_doc_markables(doc, sys_docs[doc][0], False, keep_bridging,markable_column=markable_column)

    (key_clusters, key_non_referrings, key_removed_non_referring,
        key_removed_singletons) = process_clusters(
        key_clusters, keep_singletons, keep_non_referring,keep_split_antecedent)
    (sys_clusters, sys_non_referrings, sys_removed_non_referring,
        sys_removed_singletons) = process_clusters(
        sys_clusters, keep_singletons, keep_non_referring,keep_split_antecedent)

    sys_mention_key_cluster = get_markable_assignments(key_clusters)
    key_mention_sys_cluster = get_markable_assignments(sys_clusters)

    doc_coref_infos[doc] = (key_clusters, sys_clusters,
          key_mention_sys_cluster, sys_mention_key_cluster)
    doc_non_referrig_infos[doc] = (key_non_referrings, sys_non_referrings)
    doc_bridging_infos[doc] = (key_bridging_pairs, sys_bridging_pairs, sys_mention_key_cluster)

    if print_debug and not keep_non_referring:
      print('%s and %s non-referring markables are removed from the '
          'evaluations of the key and system files, respectively.'
          % (key_removed_non_referring, sys_removed_non_referring))

    if print_debug and not keep_singletons:
      print('%s and %s singletons are removed from the evaluations of '
          'the key and system files, respectively.'
          % (key_removed_singletons, sys_removed_singletons))

  return doc_coref_infos, doc_non_referrig_infos, doc_bridging_infos



def get_markable_assignments(clusters):
  markable_cluster_ids = {}
  for cluster_id, cluster in enumerate(clusters):
    for m in cluster:
      markable_cluster_ids[m] = cluster_id
  return markable_cluster_ids


def get_all_docs(path):
  all_docs = {}
  doc_lines = []
  doc_sentence_lines = []
  doc_name = None
  for line in open(path):
    line = line.strip()
    if line.startswith('# newdoc'):
      if doc_name and doc_lines:
        all_docs[doc_name] = (doc_lines, doc_sentence_lines)
        doc_lines = []
        doc_sentence_lines = []
      doc_name = line[len('# newdoc id = '):]
    elif line.startswith('#') or len(line) == 0:
      if line.startswith('# speaker') or line.startswith('# sent_id') or line.startswith('# text'):
          doc_sentence_lines.append(line)
      continue
    else:
      doc_lines.append(line)
      doc_sentence_lines.append(line)
  if doc_name and doc_lines:
    all_docs[doc_name] = (doc_lines, doc_sentence_lines)
  return all_docs


class Sentence:
    def __init__(self, speaker, sent_id, text, tokens, original_lines):
        self.speaker = speaker
        self.sent_id = sent_id
        self.text = text
        self.tokens = tokens # list of (token_id, token_str)
        self.original_lines = original_lines
    
    def __eq__(self, other):
        if self.sent_id==other.sent_id:
            return True
        else:
            return False
            
    def __str__(self):            
        res = str(self.sent_id)+' '
        res += self.text
        for idx, token in self.tokens:
            res+=' '+str(idx)+' '+token
        return res            
            
# returns list of Sentence objects
# Sentence: object with the following fields: speaker, sent_id, text, tokens: (token_id, token_str)
def get_doc_sentences(doc, doc_lines_splitted):
    sentences = []
    SPEAKER = '# speaker = '
    SENTID = '# sent_id = '
    TEXT = '# text = '
    speaker = ''
    sentence_id = '?'
    text = ''
    sentence_lines = []
    tokens = []
    for line in doc_lines_splitted:
        sentence_lines.append(line.strip())
        if not(line.startswith(SPEAKER) or line.startswith(SENTID) or line.startswith(TEXT)):
            splitted = line.split()
            token_id = int(splitted[0])
            token = splitted[1]
            tokens.append((token_id, token))            
        else:
            if len(tokens)>0:
                sentence = Sentence(speaker, sentence_id, text, tokens[:], sentence_lines[:])
                tokens = []
                sentence_lines = []
                sentences.append(sentence)
            if line.startswith(SPEAKER):
                speaker = line[len(SPEAKER):]
            elif line.startswith(SENTID):
                sentence_id = line[len(SENTID):]
            elif line.startswith(TEXT):
                text = line[len(TEXT):]

    if len(tokens)>0: # for the last sentence in the file
        sentence = Sentence(speaker, sentence_id, text, tokens[:], sentence_lines[:])
        sentences.append(sentence)
    return sentences


def find_tokens(markable, sentences):
    start = markable.start+1
    end = markable.end+1
    markable_tokens = []
    markable_sentence = None
    for s in sentences:
        if s.tokens[0][0]>end:
            break
        if s.tokens[0][0]<=start and s.tokens[-1][0]>=end:
            markable_sentence = s
            for t in s.tokens:
                if t[0]>=start and t[0]<=end:
                    markable_tokens.append(t[1])
    return ' '.join(markable_tokens), markable_sentence            
            
