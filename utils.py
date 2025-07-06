import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score, roc_curve, auc, precision_recall_curve, roc_auc_score
from PIL import Image

class DiceLoss(nn.Module):
    def __init__(self, ):
        super(DiceLoss, self).__init__()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    
    def forward(self, inputs, target):
        bs = target.shape[0]
        inputs = torch.sigmoid(inputs.view(bs,-1))
        target = target.view(bs,-1)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        dice = self._dice_loss(inputs, target)     
        return dice
            
class Mutil_DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
def str2bool(in_str):
    if in_str in [1, "1", "t", "True", "true"]:
        return True
    elif in_str in [0, "0", "f", "False", "false", "none"]:
        return False


def metric_img_score(pd, gt):
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = float(np.logical_and(pd, gt_inv).sum())
    false_neg = float(np.logical_and(seg_inv, gt).sum())
    true_neg = float(np.logical_and(seg_inv, gt_inv).sum())

    acc = (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos + 1e-6)
    sen = true_pos / (true_pos + false_neg + 1e-6)
    spe = true_neg / (true_neg + false_pos + 1e-6)
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)#single class used
    #f1 = 2 * sen * spe / (sen + spe + 1e-6) #now negative, so spe=0, f1=0
    return acc, sen, spe, f1, true_pos, true_neg, false_pos, false_neg
    
def metric_pixel_score(premask, gt):
    if np.max(premask)==np.max(gt) and np.max(premask)==0:
        f1, iou = 1.0, 1.0
        return f1, iou, 0.0, 0.0
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(gt)
    true_pos = float(np.logical_and(premask, gt).sum())  # float for division
    #true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
    false_pos = float(np.logical_and(premask, gt_inv).sum())
    false_neg = float(np.logical_and(seg_inv, gt).sum())
    cross = float(np.logical_and(premask, gt).sum())
    union = float(np.logical_or(premask, gt).sum())
    
    f1 = 2 * true_pos / ( 2 * true_pos + false_pos + false_neg + 1e-6)
    iou = cross / (union + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)

    return f1, iou,  precision, recall

def test_single_sample(image, label, net,
                       num_class=1,
                       test_save_path=None,
                       case=None,
                       is_save = False,
                       th=0.5):
    bs = image.shape[0]
    assert bs==1 and image.shape[0]==label.shape[0],'single sample test,please input one image per batch'
    net.eval()
    
    with torch.no_grad():
        output = net(image)
        if num_class > 1 :
            out = torch.argmax(torch.softmax(output,dim=1),dim=1).squeeze()
        else:
            out = torch.sigmoid(output.squeeze())
        out = out.cpu().detach().numpy()

        if np.isnan(out).any() or np.isinf(out).any():
            max_score = 0.0
        else:
            max_score = np.max(out)

    label = label.squeeze().cpu().detach().numpy()
    assert out.shape == label.shape

    # save premask
    if is_save:
        outputImg = Image.fromarray(np.uint8(out * 255), mode='L')
        outputImg.save(test_save_path+'/'+case.strip('.jpg')+'_pred.png')

    fpr, tpr, thre = roc_curve(label.astype(int).flatten(),
                                out.flatten(), drop_intermediate=False)
    pixel_auc = auc(fpr, tpr)

    
    #out[out>=th] = 1 # thresholds fixed equal th=0.5
    #out[out<th] = 0
    #f1, iou, _, _ = metric_pixel_score(out, label)
   
    precision, recall, th = precision_recall_curve(label.astype(int).flatten(),out.flatten(),)
    f1 = (2 * precision * recall) / (precision + recall+ 1e-6)
    f1 = np.max(f1[np.isfinite(f1)])
    return f1, 0, 0, 0, pixel_auc, max_score
