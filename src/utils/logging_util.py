import time

def print_evaluation_results(epoch, current_mAP, best_mAP):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp}, {epoch} epoch, mAP {current_mAP:.2f}%, bestmAP {best_mAP:.2f}%")

def log_evaluation_results(score_file, epoch, lr, loss, current_mAP, best_mAP):
    score_file.write(
    f"{epoch} epoch, LR {lr:.6f}, Loss {loss:.6f}, mAP {current_mAP:.2f}%, bestmAP {best_mAP:.2f}%\n"
    )
    score_file.flush()


def log_model(score_file, model):
    score_file.write(str(model) + "\n")
    param_size = sum(param.numel() for param in model.parameters()) / 1024 / 1024
    score_file.write(f"Model parameter number = {param_size:.4f}\n")
    score_file.flush()

