import re
import tempfile
import subprocess
import operator
import collections
import logging
import json

logger = logging.getLogger(__name__)

BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")  # First line at each document
COREF_RESULTS_REGEX = re.compile(r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)


def get_doc_key(doc_id, part):
    return "{}_{}".format(doc_id, int(part))


def output_conll(input_file, output_file, predictions, subtoken_map):
    prediction_map = {}
    for doc_key, clusters in predictions.items():
        start_map = collections.defaultdict(list)
        end_map = collections.defaultdict(list)
        word_map = collections.defaultdict(list)
        for cluster_id, mentions in enumerate(clusters):
            for start, end in mentions:
                start, end = subtoken_map[doc_key][start], subtoken_map[doc_key][end]
                if start == end:
                    word_map[start].append(cluster_id)
                else:
                    start_map[start].append((cluster_id, end))
                    end_map[end].append((cluster_id, start))
        for k,v in start_map.items():
            start_map[k] = [cluster_id for cluster_id, end in sorted(v, key=operator.itemgetter(1), reverse=True)]
        for k,v in end_map.items():
            end_map[k] = [cluster_id for cluster_id, start in sorted(v, key=operator.itemgetter(1), reverse=True)]
        prediction_map[doc_key] = (start_map, end_map, word_map)

    word_index = 0
    for line in input_file.readlines():
        row = line.split()
        if len(row) == 0:
            output_file.write("\n")
        elif row[0].startswith("#"):
            begin_match = re.match(BEGIN_DOCUMENT_REGEX, line)
            if begin_match:
                doc_key = get_doc_key(begin_match.group(1), begin_match.group(2))
                start_map, end_map, word_map = prediction_map[doc_key]
                word_index = 0
            output_file.write(line)
            output_file.write("\n")
        else:
            assert get_doc_key(row[0], row[1]) == doc_key
            coref_list = []
            if word_index in end_map:
                for cluster_id in end_map[word_index]:
                    coref_list.append("{})".format(cluster_id))
            if word_index in word_map:
                for cluster_id in word_map[word_index]:
                    coref_list.append("({})".format(cluster_id))
            if word_index in start_map:
                for cluster_id in start_map[word_index]:
                    coref_list.append("({}".format(cluster_id))

            if len(coref_list) == 0:
                row[-1] = "-"
            else:
                row[-1] = "|".join(coref_list)

            output_file.write("   ".join(row))
            output_file.write("\n")
            word_index += 1


def convert_coref_json_to_ua(JSON_PATH, UA_PATH, MODEL="coref-hoi"):
    data = []
    ua_all_lines = []

    if MODEL == "coref-hoi":
        convert_coref_json_to_ua_doc_fn = convert_coref_json_to_ua_doc_coref_hoi
    else:
        raise NotImplementedError

    with open(JSON_PATH, "r") as f:
        for r in f.readlines():
            json_doc = json.loads(r.strip())
            ua_all_lines += convert_coref_json_to_ua_doc_fn(json_doc) + ["\n"]

        with open(UA_PATH, "w") as f:
            for line in ua_all_lines:
                f.write(line + "\n")


def convert_coref_json_to_ua_doc_coref_hoi(json_doc):
    # TODO: Include metadata
    # TODO: Include sentence breaks

    print(json_doc['doc_key'])

    pred_clusters = [tuple(tuple(m) for m in cluster) for cluster in json_doc['clusters']]
    men_to_pred = {m: clus for c, clus in enumerate(pred_clusters) for m in clus}

    lines = []
    lines.append(
        "# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC IDENTITY BRIDGING DISCOURSE_DEIXIS REFERENCE NOM_SEM")
    lines.append("# newdoc id = " + json_doc['doc_key'])
    #     lines.append("turn_id = " + json_doc['doc_key'].split()[1] + "-t1")
    #     lines.append("speaker = -")
    #     lines.append("sent_id = " + json_doc['doc_key'].split()[1] + "-1")
    markable_id = 1
    entity_id = 1

    coref_strs = [""] * len(json_doc['tokens'])

    for clus in pred_clusters:
        for (start, end) in clus:
            start = json_doc['subtoken_map'][start]
            end = json_doc['subtoken_map'][end]

            coref_strs[start] += "(EntityID={}|MarkableID=markable_{}".format(entity_id, markable_id)
            markable_id += 1
            if start == end:
                coref_strs[end] += ")"
            else:
                coref_strs[end] = ")" + coref_strs[end]

        entity_id += 1

    for _id, token in enumerate(json_doc['tokens']):
        if coref_strs[_id] == "":
            coref_strs[_id] = "_"
        sentence = "{}  {}  _  _  _  _  _  _  _  _  {}  _  _  _  _".format(_id, token, coref_strs[_id])
        lines.append(sentence)

    return lines


def official_conll_eval(gold_path, predicted_path, metric, official_stdout=True):
    cmd = ["conll-2012/scorer/v8.01/scorer.pl", metric, gold_path, predicted_path, "none"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    if stderr is not None:
        logger.error(stderr)

    if official_stdout:
        logger.info("Official result for {}".format(metric))
        logger.info(stdout)

    coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
    recall = float(coref_results_match.group(1))
    precision = float(coref_results_match.group(2))
    f1 = float(coref_results_match.group(3))
    return {"r": recall, "p": precision, "f": f1}


def evaluate_conll(gold_path, predictions, subtoken_maps, official_stdout=True):
    with tempfile.NamedTemporaryFile(delete=True, mode="w") as prediction_file:
        with open(gold_path, "r") as gold_file:
            output_conll(gold_file, prediction_file, predictions, subtoken_maps)
        # logger.info("Predicted conll file: {}".format(prediction_file.name))
        results = {m: official_conll_eval(gold_file.name, prediction_file.name, m, official_stdout) for m in ("muc", "bcub", "ceafe") }
    return results
