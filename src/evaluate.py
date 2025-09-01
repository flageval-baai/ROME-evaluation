from typing import Dict, List, Tuple, Union
from collections import defaultdict
import re
from prompt import LLM_EVAL_PROMPT
from openai import OpenAI
import os
import json
import argparse
from typing import Optional
from utils import normalize_string, extract_answer, process_multiple_choice

def key_items_matching(pred: Dict, key_items: List[List[str]], remove_space=False) -> int:
    def process(x):
        if remove_space:
            x = x.replace(' ', '')
        return x.lower()
    # Check if any answer variant matches the prediction as a substring
    def match_any(pred_str, answer_variants):
        assert isinstance(answer_variants, list)
        return int(any(process(answer) in pred_str for answer in answer_variants))

    processed_pred = process(pred["answer"])
    if isinstance(key_items[0], list):
        return int(
            all(
                match_any(processed_pred, item_variants)
                for item_variants in key_items
            )
        )
    elif isinstance(key_items[0], str):
        return int(match_any(processed_pred, key_items))
    else:
        raise ValueError(f"Unsupported key_items type: {type(key_items[0])}")

def choices_matching(pred: Dict, label: str) -> int:
    pred["answer"] = process_multiple_choice(pred["answer"])
    label = label.upper()
    if len(label) > 1:
        label = ''.join(sorted(label))
        pred["answer"] = "".join(sorted(set(pred["answer"])))
    elif len(pred["answer"]) > 1:
        pred["answer"] = pred["answer"][0]
    return int(label == pred["answer"])

def ordered_list_matching(pred: Dict, order:Union[str, List[str]]) -> int:    # Is label a subsequence of pred_ans
    if isinstance(order, list):
        order = ",".join(order)
    pred_ans = pred["answer"].lower()
    order = order.lower().replace(" ", "")
    
    # Use two pointers approach to check if label is a subsequence of pred_ans
    i, j = 0, 0
    while i < len(pred_ans) and j < len(order):
        if pred_ans[i] == order[j]:
            j += 1
        i += 1
    
    return int(j == len(order))
    

def number_matching(pred: Dict, value_to_match: Union[int, float]) -> int:
    # extract number from pred_ans
    matches = re.findall(r'-?\d+(?:\.\d+)?', pred["answer"])
    result = matches[-1] if matches else None
    if result is None:
        return 0
    pred_ans = float(result)  
    if isinstance(value_to_match, float):
        relative_error = abs(value_to_match) * 0.1
    else:
        relative_error = 1e-3
    return int(abs(pred_ans - value_to_match) < relative_error)

def location_matching(pred: Dict, location_fine_grained: List[str], location_coarse_grained: List[str]=[],
        fine_grained_score: float = 1.0, coarse_grained_score: float = 0.5) -> int:
    pred_ans = pred["answer"].lower()
    location_fine_grained = [location_fine_grained.lower() for location_fine_grained in location_fine_grained]
    location_coarse_grained = [location_coarse_grained.lower() for location_coarse_grained in location_coarse_grained]
    if any(location_fine_grained in pred_ans for location_fine_grained in location_fine_grained):
        return fine_grained_score
    elif any(location_coarse_grained in pred_ans for location_coarse_grained in location_coarse_grained):
        return coarse_grained_score
    return 0

class ROMEEvaluator():
    def __init__(
        self,
        tracker_type,
        tracker_subtype=None,
        use_llm_evaluator=False,
        **kwargs,
    ):
        self.tracker_type = tracker_type
        self.tracker_subtype = tracker_subtype
        self.use_llm_evaluator = use_llm_evaluator
        if self.use_llm_evaluator:
            api_key = 'OPENAI_API_KEY'
            base_url = 'OPENAI_BASE_URL'
            if not api_key or not base_url:
                raise ValueError(f"OPENAI_API_KEY and OPENAI_BASE_URL must be set")
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_score(self, gt: Dict, pred: Dict) -> Union[float, List[float]]:
        evaluator = gt["evaluator"]
        pred["raw_answer"] = pred["answer"]
        pred["answer"] = normalize_string(extract_answer(pred))
        registed_evaluator = set(["key_items_matching", "choices_matching", "ordered_list_matching", "number_matching", "location_matching", "interval_matching", "multi_interval_matching"])
        if evaluator not in registed_evaluator:
            raise ValueError(f"Unsupported evaluator: {evaluator}")
        return eval(evaluator)(pred, **gt["evaluator_kwargs"])

    def get_score_by_llm(self, gt: Dict, pred: Dict, model: Optional[str] = None) -> Tuple[str, int]:
        """Grade with an LLM via OpenAI-compatible API.

        Returns (raw_response_text, score_int_in_{0,1}).
        """
        # Build prompt
        prompt = (
            LLM_EVAL_PROMPT
            .replace("{{question}}", gt.get("question", ""))
            .replace("{{answer}}", gt.get("reference", ""))
            .replace("{{extracted_answer}}", pred.get("answer", ""))
        )

        # Choose model
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

        # Call API
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
        )
        content = response.choices[0].message.content or ""

        # Parse score from the assistant message content
        compare_result = content.replace("Judgement:", "").strip()
        if '\n\n' in compare_result:
            parts = compare_result.split('\n\n')
            score_part = parts[-1].strip()
        else:
            score_part = compare_result
        score_match = re.search(r'(\d+(?:\.\d+)?)', score_part)
        score_str = score_match.group(1) if score_match else "0"
        return content, int(float(score_str) > 0.5)

    def cal_accuracy(
        self, annotations: Dict, predictions: List[Dict], *args, llm_model: Optional[str] = None, **kwargs
    ) -> Dict:
        class ScoreTracker:
            def __init__(self):
                self.total_score = 0
                self.count = 0
                self.accuracy = 0
                self.subtypes = defaultdict(
                    lambda: [0, 0, 0]
                )  # [score_sum, count, accuracy]
            def update(self, score, sub_type):
                self.total_score += score
                self.count += 1
                self.subtypes[sub_type][0] += score
                self.subtypes[sub_type][1] += 1

        results = {}
        scores_by_type = defaultdict(ScoreTracker)
        for pred in predictions:
            question_id = str(pred["question_id"])
            gt = annotations[question_id]
            if self.use_llm_evaluator:
                judgement_response, score = self.get_score_by_llm(gt, pred, model=llm_model)
            else:
                score = self.get_score(gt, pred)
            pred.update(gt)
            pred["correct"] = score
            pred["judgement_response"] = judgement_response if self.use_llm_evaluator else None
            # Update scores
            bucket_key = pred.get(self.tracker_type) or gt.get(self.tracker_type) or "all"
            tracker = scores_by_type[bucket_key]
            if self.tracker_subtype is not None:
                sub_bucket_key = pred.get(self.tracker_subtype) or gt.get(self.tracker_subtype) or "all"
                tracker.update(score, sub_bucket_key)
            else:
                tracker.update(score, bucket_key)
        # Calculate accuracy
        for tracker in scores_by_type.values():
            tracker.accuracy = round(tracker.total_score / tracker.count, 3)
            for sub_type in tracker.subtypes:
                tracker.subtypes[sub_type][2] = round(
                    tracker.subtypes[sub_type][0] / tracker.subtypes[sub_type][1], 3
                )
        final_score = sum(tracker.total_score for tracker in scores_by_type.values())
        results["final_score"] = [final_score, len(predictions)]
        results["accuracy"] = round(final_score / len(predictions) * 100, 3)

        # Convert ScoreTracker objects to the expected format
        for qtype, tracker in scores_by_type.items():
            results[qtype] = [
                tracker.total_score,
                tracker.count,
                tracker.accuracy,
                dict(tracker.subtypes),
            ]

        return results


def _load_jsonl_or_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return []
        # Try JSON Lines
        if "\n" in text and text.split("\n")[0].strip().startswith("{") and not text.strip().startswith("["):
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        return json.loads(text)


def _ensure_predictions_list(obj) -> List[Dict]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # Case 1: Single prediction object
        if "question_id" in obj and "answer" in obj:
            return [obj]
        # Case 2: Mapping from question_id -> prediction object
        if all(isinstance(v, dict) for v in obj.values()):
            out = []
            for qid, item in obj.items():
                merged = {"question_id": qid, **item}
                if "answer" not in merged:
                    raise ValueError(f"Prediction for question_id {qid} must include 'answer'.")
                out.append(merged)
            return out
        raise ValueError("Unsupported predictions dict format. Provide a single object with 'question_id'/'answer', a list of such objects, or a mapping from id to object.")
    raise ValueError("Unsupported predictions format; expected list or dict.")


def run_cli():
    parser = argparse.ArgumentParser(description="Simple, pluggable evaluator for various task types.")
    parser.add_argument("--annotations", required=True, help="Path to ground-truth annotations JSON/JSONL (dict keyed by question_id).")
    parser.add_argument("--predictions", required=True, help="Path to model predictions JSON/JSONL (list or dict keyed by question_id).")
    parser.add_argument("--tracker_type", default="task", help="Top-level bucket key; falls back to 'all' if missing.")
    parser.add_argument("--tracker_subtype", default=None, help="Optional sub-bucket key; falls back to 'all' if missing.")
    parser.add_argument("--use_llm_evaluator", action="store_true", help="Use LLM-based semantic grader (OpenAI-compatible API).")
    parser.add_argument("--llm_model", default=None, help="Model name for LLM grading (e.g., gpt-4.1-mini). Default from OPENAI_MODEL.")
    parser.add_argument("--output", default=None, help="Where to write aggregated metrics JSON. Default:scores.json")
    args = parser.parse_args()

    annotations = _load_jsonl_or_json(args.annotations)
    if isinstance(annotations, list):
        # Expect list of objects with question_id
        annotations_dict = {}
        for item in annotations:
            qid = str(item.get("question_id"))
            if not qid:
                raise ValueError("Each annotation entry must contain question_id.")
            annotations_dict[qid] = item
        annotations = annotations_dict
    elif isinstance(annotations, dict):
        annotations = {str(k): v for k, v in annotations.items()}
    else:
        raise ValueError("Unsupported annotations format; expected list or dict.")

    predictions_raw = _load_jsonl_or_json(args.predictions)
    predictions = _ensure_predictions_list(predictions_raw)

    # Validate presence of required fields per question
    for p in predictions:
        qid = str(p.get("question_id"))
        if qid not in annotations:
            raise KeyError(f"Prediction question_id {qid} not found in annotations.")
        gt = annotations[qid]
        if "evaluator" not in gt or "evaluator_kwargs" not in gt:
            raise KeyError(f"Annotation for question_id {qid} must include 'evaluator' and 'evaluator_kwargs'.")
        if "answer" not in p:
            raise KeyError(f"Prediction for question_id {qid} must include 'answer'.")

    evaluator = ROMEEvaluator(
        tracker_type=args.tracker_type,
        tracker_subtype=args.tracker_subtype,
        use_llm_evaluator=args.use_llm_evaluator,
    )
    results = evaluator.cal_accuracy(annotations, predictions, llm_model=args.llm_model)

    output_path = args.output or (os.path.join(os.path.dirname(args.predictions), "scores.json"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    updated_pre_path = args.updated or os.path.join(os.path.dirname(args.predictions), "updated_predictions.json")
    with open(updated_pre_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_cli()
