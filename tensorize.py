import util
import numpy as np
import random
import os
from os.path import join
import json
import pickle
import logging
import torch
import torch.nn.functional as F

from kb.knowbert_utils import KnowBertBatchifier

logger = logging.getLogger(__name__)


class CorefDataProcessor:
    def __init__(self, config, language='english'):
        self.config = config
        self.language = language

        self.max_seg_len = config['max_segment_len']
        self.max_training_seg = config['max_training_sentences']
        self.data_dir = config['data_dir']

        self.tokenizer = util.get_tokenizer(config['bert_tokenizer_name'])
        self.tensor_samples, self.stored_info = None, None  # For dataset samples; lazy loading

    def get_tensor_examples_from_custom_input(self, samples):
        """ For interactive samples; no caching """
        tensorizer = Tensorizer(self.config, self.tokenizer)
        tensor_samples = [tensorizer.tensorize_example(sample, False) for sample in samples]
        tensor_samples = [(doc_key, self.convert_to_torch_tensor(*tensor)) for doc_key, tensor in tensor_samples]
        return tensor_samples, tensorizer.stored_info

    def get_tensor_examples(self):
        """ For dataset samples """
        cache_path = self.get_cache_path()
        if os.path.exists(cache_path):
            # Load cached tensors if exists
            with open(cache_path, 'rb') as f:
                self.tensor_samples, self.stored_info = pickle.load(f)
                logger.info('Loaded tensorized examples from cache')
        else:
            # Generate tensorized samples
            self.tensor_samples = {}
            tensorizer = Tensorizer(self.config, self.tokenizer)
            paths = {
                'trn': join(self.data_dir, f'train.{self.language}.{self.max_seg_len}.jsonlines'),
                'dev': join(self.data_dir, f'dev.{self.language}.{self.max_seg_len}.jsonlines'),
                'tst': join(self.data_dir, f'test.{self.language}.{self.max_seg_len}.jsonlines')
            }
            for split, path in paths.items():
                logger.info('Tensorizing examples from %s; results will be cached)' % path)
                is_training = (split == 'trn')
                with open(path, 'r') as f:
                    samples = [json.loads(line) for line in f.readlines()]
                tensor_samples = [tensorizer.tensorize_example(sample, is_training) for sample in samples]
                self.tensor_samples[split] = [(doc_key, self.convert_to_torch_tensor(*tensor)) for doc_key, tensor
                                              in tensor_samples]
            self.stored_info = tensorizer.stored_info
            # Cache tensorized samples
            with open(cache_path, 'wb') as f:
                pickle.dump((self.tensor_samples, self.stored_info), f)
        return self.tensor_samples['trn'], self.tensor_samples['dev'], self.tensor_samples['tst']

    def get_stored_info(self):
        return self.stored_info

    @classmethod
    def convert_to_torch_tensor(cls, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                                is_training, gold_starts, gold_ends, gold_mention_cluster_map, kb_candidates=None):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)
        sentence_len = torch.tensor(sentence_len, dtype=torch.long)
        genre = torch.tensor(genre, dtype=torch.long)
        sentence_map = torch.tensor(sentence_map, dtype=torch.long)
        is_training = torch.tensor(is_training, dtype=torch.bool)
        gold_starts = torch.tensor(gold_starts, dtype=torch.long)
        gold_ends = torch.tensor(gold_ends, dtype=torch.long)
        gold_mention_cluster_map = torch.tensor(gold_mention_cluster_map, dtype=torch.long)
        if kb_candidates is not None:
            return input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, \
                   is_training, gold_starts, gold_ends, gold_mention_cluster_map, kb_candidates
        else:
            return input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, \
                   is_training, gold_starts, gold_ends, gold_mention_cluster_map,

    def get_cache_path(self):
        cache_path = join(self.data_dir, f'cached.tensors.{self.language}.{self.max_seg_len}.{self.max_training_seg}.bin')
        return cache_path


def pad_candidates(candidates, wn_max_ent, wn_max_cand, wk_max_ent, wk_max_cand):
    for i in range(len(candidates)):
        priors_shape = candidates[i]["wordnet"]["candidate_entity_priors"].shape
        candidates[i]["wordnet"]["candidate_entity_priors"] = \
            F.pad(candidates[i]["wordnet"]["candidate_entity_priors"],
                  [0, wn_max_cand - priors_shape[1], 0, wn_max_ent - priors_shape[0]],
                  "constant", 0.0)
        candidates[i]["wordnet"]["candidate_entities"]["ids"] = \
            F.pad(candidates[i]["wordnet"]["candidate_entities"]["ids"],
                  [0, wn_max_cand - priors_shape[1], 0, wn_max_ent - priors_shape[0]],
                  "constant", 0)
        candidates[i]["wordnet"]["candidate_spans"] = \
            F.pad(candidates[i]["wordnet"]["candidate_spans"],
                  [0, 0, 0, wn_max_ent - priors_shape[0]],
                  "constant", -1)
        candidates[i]["wordnet"]["candidate_segment_ids"] = \
            F.pad(candidates[i]["wordnet"]["candidate_segment_ids"],
                  [0, wn_max_ent - priors_shape[0]],
                  "constant", 0)
        priors_shape = candidates[i]["wiki"]["candidate_entity_priors"].shape
        candidates[i]["wiki"]["candidate_entity_priors"] = \
            F.pad(candidates[i]["wiki"]["candidate_entity_priors"],
                  [0, wk_max_cand - priors_shape[1], 0, wk_max_ent - priors_shape[0]],
                  "constant", 0.0)
        candidates[i]["wiki"]["candidate_entities"]["ids"] = \
            F.pad(candidates[i]["wiki"]["candidate_entities"]["ids"],
                  [0, wk_max_cand - priors_shape[1], 0, wk_max_ent - priors_shape[0]],
                  "constant", 0)
        candidates[i]["wiki"]["candidate_spans"] = \
            F.pad(candidates[i]["wiki"]["candidate_spans"],
                  [0, 0, 0, wk_max_ent - priors_shape[0]],
                  "constant", -1)
        candidates[i]["wiki"]["candidate_segment_ids"] = \
            F.pad(candidates[i]["wiki"]["candidate_segment_ids"],
                  [0, wk_max_ent - priors_shape[0]],
                  "constant", 0)

    aggregated_cand = {"wordnet": {}, "wiki": {}}
    aggregated_cand["wordnet"]["candidate_entities"] = {}
    aggregated_cand["wiki"]["candidate_entities"] = {}
    aggregated_cand["wordnet"]["candidate_entity_priors"] = torch.cat([candidates[i]["wordnet"]["candidate_entity_priors"].unsqueeze(0) for i in range(len(candidates))], dim=0)
    aggregated_cand["wordnet"]["candidate_entities"]["ids"] = torch.cat([candidates[i]["wordnet"]["candidate_entities"]["ids"].unsqueeze(0) for i in range(len(candidates))], dim=0)
    aggregated_cand["wordnet"]["candidate_spans"] = torch.cat([candidates[i]["wordnet"]["candidate_spans"].unsqueeze(0) for i in range(len(candidates))], dim=0)
    aggregated_cand["wordnet"]["candidate_segment_ids"] = torch.cat([candidates[i]["wordnet"]["candidate_segment_ids"].unsqueeze(0) for i in range(len(candidates))], dim=0)
    aggregated_cand["wiki"]["candidate_entity_priors"] = torch.cat([candidates[i]["wiki"]["candidate_entity_priors"].unsqueeze(0) for i in range(len(candidates))], dim=0)
    aggregated_cand["wiki"]["candidate_entities"]["ids"] = torch.cat([candidates[i]["wiki"]["candidate_entities"]["ids"].unsqueeze(0) for i in range(len(candidates))], dim=0)
    aggregated_cand["wiki"]["candidate_spans"] = torch.cat([candidates[i]["wiki"]["candidate_spans"].unsqueeze(0) for i in range(len(candidates))], dim=0)
    aggregated_cand["wiki"]["candidate_segment_ids"] = torch.cat([candidates[i]["wiki"]["candidate_segment_ids"].unsqueeze(0) for i in range(len(candidates))], dim=0)

    return aggregated_cand


class Tensorizer:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        # Will be used in evaluation
        self.stored_info = {}
        self.stored_info['tokens'] = {}  # {doc_key: ...}
        self.stored_info['subtoken_maps'] = {}  # {doc_key: ...}; mapping back to tokens
        self.stored_info['gold'] = {}  # {doc_key: ...}
        self.stored_info['genre_dict'] = {genre: idx for idx, genre in enumerate(config['genres'])}
        if "knowbert" in self.config.bert_pretrained_name_or_path:
            self.stored_info['kb_candidates'] = {}  # {doc_key: ...}
            self.batcher = KnowBertBatchifier(self.config.bert_pretrained_name_or_path, batch_size=1)

    def _tensorize_spans(self, spans):
        if len(spans) > 0:
            starts, ends = zip(*spans)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def _tensorize_span_w_labels(self, spans, label_dict):
        if len(spans) > 0:
            starts, ends, labels = zip(*spans)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[label] for label in labels])

    def _get_speaker_dict(self, speakers):
        speaker_dict = {'UNK': 0, '[SPL]': 1}
        for speaker in speakers:
            if len(speaker_dict) > self.config['max_num_speakers']:
                pass  # 'break' to limit # speakers
            if speaker not in speaker_dict:
                speaker_dict[speaker] = len(speaker_dict)
        return speaker_dict

    def tensorize_example(self, example, is_training):
        # Mentions and clusters
        clusters = example['clusters']
        gold_mentions = sorted(tuple(mention) for mention in util.flatten(clusters))
        gold_mention_map = {mention: idx for idx, mention in enumerate(gold_mentions)}
        gold_mention_cluster_map = np.zeros(len(gold_mentions))  # 0: no cluster
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                gold_mention_cluster_map[gold_mention_map[tuple(mention)]] = cluster_id + 1

        # Speakers
        speakers = example['speakers']
        speaker_dict = self._get_speaker_dict(util.flatten(speakers))

        # Sentences/segments
        sentences = example['sentences']  # Segments
        sentence_map = example['sentence_map']
        num_words = sum([len(s) for s in sentences])
        max_sentence_len = self.config['max_segment_len']
        sentence_len = np.array([len(s) for s in sentences])

        # Bert input
        wn_max_cand = 0
        wn_max_ent = 0
        wk_max_cand = 0
        wk_max_ent = 0
        input_ids, input_mask, speaker_ids = [], [], []
        kb_candidates = []
        for idx, (sent_tokens, sent_speakers) in enumerate(zip(sentences, speakers)):
            if "knowbert" in self.config.bert_pretrained_name_or_path:
                token_start = 0 if idx == 0 else example["subtoken_map"][sum([len(sentences[i]) for i in range(idx)])] + 1
                sent_lengths = sum([len(sentences[i]) for i in range(idx+1)])
                token_end = example["subtoken_map"][sent_lengths-1] + 1
                text = ' '.join(example["tokens"][token_start:token_end])
                single_batch = next(self.batcher.iter_batches([text], verbose=False))
                if single_batch["candidates"]["wordnet"]["candidate_entity_priors"].shape[1] > wn_max_ent:
                    wn_max_ent = single_batch["candidates"]["wordnet"]["candidate_entity_priors"].shape[1]
                if single_batch["candidates"]["wordnet"]["candidate_entity_priors"].shape[2] > wn_max_cand:
                    wn_max_cand = single_batch["candidates"]["wordnet"]["candidate_entity_priors"].shape[2]
                if single_batch["candidates"]["wiki"]["candidate_entity_priors"].shape[1] > wk_max_ent:
                    wk_max_ent = single_batch["candidates"]["wiki"]["candidate_entity_priors"].shape[1]
                if single_batch["candidates"]["wiki"]["candidate_entity_priors"].shape[2] > wk_max_cand:
                    wk_max_cand = single_batch["candidates"]["wiki"]["candidate_entity_priors"].shape[2]
                ex_candidates = {}
                ex_candidates["wordnet"] = {"candidate_entities": {"ids": single_batch["candidates"]["wordnet"]["candidate_entities"]["ids"].squeeze(0)},
                                            "candidate_entity_priors": single_batch["candidates"]["wordnet"]["candidate_entity_priors"].squeeze(0),
                                            "candidate_spans": single_batch["candidates"]["wordnet"]["candidate_spans"].squeeze(0),
                                            "candidate_segment_ids": single_batch["candidates"]["wordnet"]["candidate_segment_ids"].squeeze(0)}
                ex_candidates["wiki"] = {"candidate_entities": {"ids": single_batch["candidates"]["wiki"]["candidate_entities"]["ids"].squeeze(0)},
                                         "candidate_entity_priors": single_batch["candidates"]["wiki"]["candidate_entity_priors"].squeeze(0),
                                         "candidate_spans": single_batch["candidates"]["wiki"]["candidate_spans"].squeeze(0),
                                         "candidate_segment_ids": single_batch["candidates"]["wiki"]["candidate_segment_ids"].squeeze(0)}
                kb_candidates.append(ex_candidates)
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
            sent_input_mask = [1] * len(sent_input_ids)
            sent_speaker_ids = [speaker_dict[speaker] for speaker in sent_speakers]
            while len(sent_input_ids) < max_sentence_len:
                sent_input_ids.append(0)
                sent_input_mask.append(0)
                sent_speaker_ids.append(0)
            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
            speaker_ids.append(sent_speaker_ids)
        if "knowbert" in self.config.bert_pretrained_name_or_path:
            kb_candidates = pad_candidates(kb_candidates, wn_max_ent, wn_max_cand, wk_max_ent, wk_max_cand)
        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        speaker_ids = np.array(speaker_ids)
        assert num_words == np.sum(input_mask), (num_words, np.sum(input_mask))

        # Keep info to store
        doc_key = example['doc_key']
        self.stored_info['subtoken_maps'][doc_key] = example.get('subtoken_map', None)
        self.stored_info['gold'][doc_key] = example['clusters']
        if "knowbert" in self.config.bert_pretrained_name_or_path:
            self.stored_info['kb_candidates'][doc_key] = kb_candidates
        # self.stored_info['tokens'][doc_key] = example['tokens']

        # Construct example
        genre = self.stored_info['genre_dict'].get(doc_key[:2], 0)
        gold_starts, gold_ends = self._tensorize_spans(gold_mentions)
        if "knowbert" in self.config.bert_pretrained_name_or_path:
            example_tensor = (input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, is_training,
                              gold_starts, gold_ends, gold_mention_cluster_map, kb_candidates)
        else:
            example_tensor = (input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, is_training,
                              gold_starts, gold_ends, gold_mention_cluster_map)

        if is_training and len(sentences) > self.config['max_training_sentences']:
            if "knowbert" in self.config.bert_pretrained_name_or_path:
                return doc_key, self.truncate_example(*example_tensor[:-1], kb_candidates=example_tensor[-1])
            else:
                return doc_key, self.truncate_example(*example_tensor)
        else:
            return doc_key, example_tensor

    def truncate_example(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, is_training,
                         gold_starts, gold_ends, gold_mention_cluster_map, sentence_offset=None, kb_candidates=None):
        max_sentences = self.config["max_training_sentences"]
        num_sentences = input_ids.shape[0]
        assert num_sentences > max_sentences

        sent_offset = sentence_offset
        if sent_offset is None:
            sent_offset = random.randint(0, num_sentences - max_sentences)
        word_offset = sentence_len[:sent_offset].sum()
        num_words = sentence_len[sent_offset: sent_offset + max_sentences].sum()

        input_ids = input_ids[sent_offset: sent_offset + max_sentences, :]
        input_mask = input_mask[sent_offset: sent_offset + max_sentences, :]
        speaker_ids = speaker_ids[sent_offset: sent_offset + max_sentences, :]
        sentence_len = sentence_len[sent_offset: sent_offset + max_sentences]
        if kb_candidates is not None:
            kb_candidates["wordnet"]["candidate_entity_priors"] = kb_candidates["wordnet"]["candidate_entity_priors"][sent_offset: sent_offset + max_sentences]
            kb_candidates["wordnet"]["candidate_entities"]["ids"] = kb_candidates["wordnet"]["candidate_entities"]["ids"][sent_offset: sent_offset + max_sentences]
            kb_candidates["wordnet"]["candidate_spans"] = kb_candidates["wordnet"]["candidate_spans"][sent_offset: sent_offset + max_sentences]
            kb_candidates["wordnet"]["candidate_segment_ids"] = kb_candidates["wordnet"]["candidate_segment_ids"][sent_offset: sent_offset + max_sentences]
            kb_candidates["wiki"]["candidate_entity_priors"] = kb_candidates["wiki"]["candidate_entity_priors"][sent_offset: sent_offset + max_sentences]
            kb_candidates["wiki"]["candidate_entities"]["ids"] = kb_candidates["wiki"]["candidate_entities"]["ids"][sent_offset: sent_offset + max_sentences]
            kb_candidates["wiki"]["candidate_spans"] = kb_candidates["wiki"]["candidate_spans"][sent_offset: sent_offset + max_sentences]
            kb_candidates["wiki"]["candidate_segment_ids"] = kb_candidates["wiki"]["candidate_segment_ids"][sent_offset: sent_offset + max_sentences]
        sentence_map = sentence_map[word_offset: word_offset + num_words]
        gold_spans = (gold_starts < word_offset + num_words) & (gold_ends >= word_offset)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        gold_mention_cluster_map = gold_mention_cluster_map[gold_spans]
        if kb_candidates is not None:
            example_tensor = (input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, is_training,
                              gold_starts, gold_ends, gold_mention_cluster_map, kb_candidates)
        else:
            example_tensor = (input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, is_training,
                              gold_starts, gold_ends, gold_mention_cluster_map)
        return example_tensor
