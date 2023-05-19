def smape(y_pred, y_test):
    n = len(y_pred)
    acc = 0
    for i in range(n):
        a = abs(y_pred[i][0] - y_test[i][0])
        b = abs(y_pred[i][1] - y_test[i][1])
        c = y_test[i][0] + y_test[i][1]
        d = y_pred[i][0] + y_pred[i][1]
        acc += abs((a + b) / ((abs(c) + abs(d)) / 2))
    return acc / n

def sq_loss_fn(gt, pred, output1_w = 0.5, output2_w = 0.5):
    count = len(gt)
    losses = 0
    for i in range(count):
        losses += ((gt[i][0] - pred[i][0]) * output1_w) ** 2
        losses += ((gt[i][1] - pred[i][1]) * output2_w) ** 2
    losses /= count
    return losses

def abs_loss_fn(gt, pred, output1_w = 0.5, output2_w = 0.5):
    count = len(gt)
    losses = 0
    for i in range(count):
        losses += abs((gt[i][0] - pred[i][0]) * output1_w)
        losses += abs((gt[i][1] - pred[i][1]) * output2_w)
    losses /= count
    return losses