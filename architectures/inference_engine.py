import torch
import numpy as np
from datetime import datetime

def run_sliding_window_inference(gamma_dose, model, device, train_std, num_classes, window_size=512, stride=10, batch_size=256):
    """
    Runs sliding window inference over the continuous data using batching, accumulating probabilities.
    Returns the accumulated probabilities array.
    """
    segment_size = len(gamma_dose)
    probabilities = np.zeros((segment_size, num_classes))
    counts = np.zeros(segment_size)

    num_windows = (segment_size - window_size) // stride + 1
    start_time = datetime.now()

    windows = []
    indices = []

    # 1. Extract and normalize windows
    for start in range(0, segment_size - window_size + 1, stride):
        end = start + window_size
        window_data = gamma_dose[start:end]
        window_mean = np.mean(window_data)
        window_scaled = (window_data - window_mean) / train_std
        windows.append(window_scaled)
        indices.append(start)

    # 2. Batched inference
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i + batch_size]
            batch_indices = indices[i:i + batch_size]

            x_tensor = torch.tensor(np.array(batch_windows), dtype=torch.float32).view(len(batch_windows), 1, window_size).to(device)
            output = model(x_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()

            for b_idx, start_idx in enumerate(batch_indices):
                end_idx = start_idx + window_size
                probabilities[start_idx:end_idx] += probs[b_idx]
                counts[start_idx:end_idx] += 1

    # 3. Average the accumulated probabilities
    for i in range(len(counts)):
        if counts[i] > 0:
            probabilities[i] /= counts[i]
        else:
            probabilities[i] = np.nan

    duration = datetime.now() - start_time
    return probabilities, counts, num_windows, duration


def evaluate_detection_events(probabilities, is_anomaly, num_classes, bg_threshold=0.3, tolerance=50):
    """
    Performs event-level evaluation by extracting Predicted Events and True Events,
    and matching them via overlap.
    """
    segment_size = len(is_anomaly)
    
    # 1. Extract True Events from Ground Truth
    true_events = []
    idx = 0
    while idx < segment_size:
        label = int(is_anomaly[idx])
        if label > 0:
            evt_start = idx
            while idx < segment_size and int(is_anomaly[idx]) == label:
                idx += 1
            true_events.append({'start': evt_start, 'end': idx, 'class': label})
        else:
            idx += 1
            
    # 2. Extract Predicted Events (Detection-driven)
    pred_events = []
    idx = 0
    prob_c0 = np.nan_to_num(probabilities[:, 0], nan=1.0)
    is_pred_anomaly = prob_c0 < bg_threshold
    
    while idx < segment_size:
        if is_pred_anomaly[idx]:
            evt_start = idx
            evt_end = idx
            
            while idx < segment_size:
                if is_pred_anomaly[idx]:
                    evt_end = idx
                    idx += 1
                else:
                    # Tolerance gap bridging
                    next_anom = -1
                    for lookahead in range(1, tolerance + 1):
                        if idx + lookahead < segment_size and is_pred_anomaly[idx + lookahead]:
                            next_anom = idx + lookahead
                            break
                    if next_anom != -1:
                        idx = next_anom
                        evt_end = idx
                    else:
                        break
            
            # Integral-based classification
            event_probs = probabilities[evt_start:evt_end + 1]
            valid_mask = ~np.isnan(event_probs[:, 0])
            if valid_mask.sum() > 0:
                # Sum the probabilities for anomaly classes (1 to num_classes-1)
                sum_probs = np.sum(event_probs[valid_mask], axis=0)
                # The prediction is the anomaly class with the highest integral
                pred_class = int(np.argmax(sum_probs[1:]) + 1)
                confidence = float(np.mean(event_probs[valid_mask, pred_class]))
                pred_events.append({
                    'start': evt_start, 
                    'end': evt_end + 1, 
                    'class': pred_class,
                    'conf': confidence,
                    'avg_probs': np.mean(event_probs[valid_mask], axis=0)
                })
        else:
            idx += 1
            
    # 3. Match Events (Overlap)
    tp = 0
    fp = 0
    fn = 0
    
    event_details = [] 
    event_true_list = []
    event_pred_list = []
    event_conf_list = []
    event_probs_all = []
    
    # Check all true events
    for te_idx, te in enumerate(true_events):
        overlaps = []
        for pe in pred_events:
            overlap_start = max(te['start'], pe['start'])
            overlap_end = min(te['end'], pe['end'])
            if overlap_start < overlap_end:
                overlaps.append(pe)
                
        if not overlaps:
            # False Negative (missed entirely)
            event_true_list.append(te['class'])
            event_pred_list.append(0)
            event_conf_list.append(0.0)
            event_probs_all.append(np.zeros(num_classes))
            event_details.append({
                'type': 'FN (Missed)',
                'true': te['class'], 'pred': 0, 'conf': 0.0,
                'start': te['start'], 'end': te['end'], 'length': te['end'] - te['start']
            })
            fn += 1
        else:
            # Find best overlapping predicted event (max overlap)
            best_pe = None
            max_overlap = 0
            for pe in overlaps:
                o_start = max(te['start'], pe['start'])
                o_end = min(te['end'], pe['end'])
                if o_end - o_start > max_overlap:
                    max_overlap = o_end - o_start
                    best_pe = pe
            
            event_true_list.append(te['class'])
            event_pred_list.append(best_pe['class'])
            event_conf_list.append(best_pe['conf'])
            event_probs_all.append(best_pe['avg_probs'])
            
            if best_pe['class'] == te['class']:
                tp += 1
                event_details.append({
                    'type': 'TP (Hit)',
                    'true': te['class'], 'pred': best_pe['class'], 'conf': best_pe['conf'],
                    'start': te['start'], 'end': te['end'], 'length': te['end'] - te['start']
                })
            else:
                fn += 1 # Misclassified, so it's a FN for the true class
                event_details.append({
                    'type': 'FN (Misclassified)',
                    'true': te['class'], 'pred': best_pe['class'], 'conf': best_pe['conf'],
                    'start': te['start'], 'end': te['end'], 'length': te['end'] - te['start']
                })

    # Check predicted events that didn't overlap with ANY true event (Pure False Positives)
    for pe in pred_events:
        overlap_with_any = False
        for te in true_events:
            if max(te['start'], pe['start']) < min(te['end'], pe['end']):
                overlap_with_any = True
                break
        
        if not overlap_with_any:
            fp += 1
            event_true_list.append(0)
            event_pred_list.append(pe['class'])
            event_conf_list.append(pe['conf'])
            event_probs_all.append(pe['avg_probs'])
            event_details.append({
                'type': 'FP (Hallucination)',
                'true': 0, 'pred': pe['class'], 'conf': pe['conf'],
                'start': pe['start'], 'end': pe['end'], 'length': pe['end'] - pe['start']
            })

    # Calculate Macro F1 and Per-Class Metrics
    event_true = np.array(event_true_list)
    event_pred = np.array(event_pred_list)
    event_conf = np.array(event_conf_list)
    
    per_class_metrics = {}
    e_precs = []
    e_recs = []
    e_f1s = []
    
    all_classes = sorted(set(event_true.tolist() + event_pred.tolist()))
    for c in all_classes:
        if c == 0:
            continue
        c_tp = int(np.sum((event_pred == c) & (event_true == c)))
        c_fp = int(np.sum((event_pred == c) & (event_true != c)))
        c_fn = int(np.sum((event_pred != c) & (event_true == c)))
        support = int(np.sum(event_true == c))
        
        prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        per_class_metrics[c] = {
            'tp': c_tp, 'fp': c_fp, 'fn': c_fn,
            'precision': prec, 'recall': rec, 'f1': f1, 'support': support
        }
        
        e_precs.append(prec)
        e_recs.append(rec)
        e_f1s.append(f1)
        
    macro_f1 = np.mean(e_f1s) if e_f1s else 0.0
    macro_precision = np.mean(e_precs) if e_precs else 0.0
    macro_recall = np.mean(e_recs) if e_recs else 0.0
    
    return {
        'total_true_events': len(true_events),
        'total_pred_events': len(pred_events),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'per_class_metrics': per_class_metrics,
        'event_true': event_true,
        'event_pred': event_pred,
        'event_conf': event_conf,
        'event_probs_all': np.array(event_probs_all),
        'event_details': event_details
    }
