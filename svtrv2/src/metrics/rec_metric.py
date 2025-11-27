"""Recognition Metric"""
import string

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:
    # Fallback if rapidfuzz is not available
    def Levenshtein():
        pass
    Levenshtein.normalized_distance = lambda s1, s2: 0.0


class RecMetric:
    """Recognition Metric"""

    def __init__(
        self,
        main_indicator="acc",
        is_filter=False,
        is_lower=True,
        ignore_space=True,
        **kwargs,
    ):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.is_lower = is_lower
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()

    def _normalize_text(self, text):
        """Normalize text by keeping only alphanumeric characters"""
        text = "".join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text)
        )
        return text

    def reset(self):
        """Reset metric state"""
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0.0

    def __call__(self, pred_label, batch=None, training=False, *args, **kwargs):
        """Update metric with predictions and labels"""
        return self.eval_metric(pred_label)

    def eval_metric(self, pred_label, *args, **kwargs):
        """Evaluate metric"""
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            if self.is_lower:
                pred = pred.lower()
                target = target.lower()
            
            try:
                norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            except Exception:
                # Fallback if rapidfuzz is not available
                norm_edit_dis += 0.0 if pred == target else 1.0
            
            if pred == target:
                correct_num += 1
            all_num += 1
        
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        
        return {
            "acc": correct_num / (all_num + self.eps),
            "norm_edit_dis": 1 - norm_edit_dis / (all_num + self.eps),
        }

    def get_metric(self, training=False):
        """Get current metric values"""
        if self.all_num == 0:
            return {self.main_indicator: 0.0, "acc": 0.0, "norm_edit_dis": 0.0}
        
        acc = self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        
        return {
            self.main_indicator: acc,
            "acc": acc,
            "norm_edit_dis": norm_edit_dis,
        }

