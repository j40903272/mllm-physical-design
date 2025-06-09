from feature_extract_func import *
import models
import warnings
from argparse import ArgumentParser
import re
import pandas as pd
from tqdm.auto import tqdm
import os
import torch
import multiprocessing
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import heapq

plt.style.use(['science','grid','retro'])

warnings.filterwarnings("ignore")

train_design = ["RISCY-a", "RISCY-b", "RISCY-FPU-a", "RISCY-FPU-b"]
test_design_a = ["zero-riscy-a"]
test_design_b = ["zero-riscy-b"]
tile_size = 16
image_size = 256


def get_drc_violations(image, threshold=0.1):
    """
    Get DRC violations from the image based on a threshold.
    """
    violations = np.where(image > threshold, 1, 0).sum()
    return violations


def get_all_features(images):    
    final_features = {}
    
    for feat_func in feat_func_list:
        feat = feat_func(images)
        final_features.update(feat)
        
    return final_features

def replaceWithRank(arr):
    n = len(arr)
    res = [0] * n
    pq = []
    for i in range(n):
        heapq.heappush(pq, (arr[i], i))

    rank = 0
    lastNum = float('inf')

    while pq:
        curr, index = heapq.heappop(pq)

        if lastNum == float('inf') or curr != lastNum:
            rank += 1
            
        res[index] = rank - 1
        lastNum = curr

    return res


def evalute_corr(congestion_set, predicted, corr_metrics):
    x = np.array(list(congestion_set.values()))
    x_label = list(congestion_set.keys())
    y = np.array([predicted[id] for id in x_label])
    results = {}
    if "PLCC" in corr_metrics:
        results["PLCC"] = stats.pearsonr(x, y)
    if "SRCC" in corr_metrics:
        results["SRCC"] = stats.spearmanr(x, y)
    if "KRCC" in corr_metrics:
        results["KRCC"] = stats.kendalltau(x, y)
    
    return results


def evaluate_design(df, save_path, baseline=False, design_name="zero-riscy-a"):
    congestion_set = dict(zip(df["id"], df["label"]))
    congestion_set = dict(sorted(congestion_set.items(), key=lambda x: x[1]))
    corr_metrics = ["PLCC", "SRCC", "KRCC"]
    if baseline:
        predicted_gpdl = dict(zip(df["id"], df["prediction_gpdl"]))
        predicted = dict(zip(df["id"], df["prediction"]))
        results = evalute_corr(congestion_set, predicted, corr_metrics)
        gpdl_results = evalute_corr(congestion_set, predicted_gpdl, corr_metrics)
        x = list(congestion_set.keys())[::10]
        x = [name.split("-")[0] for name in x]
        x_label = list(range(0,len(x)))
        y = [predicted[file_path] for file_path in congestion_set.keys()][::10]
        y_gpdl = [predicted_gpdl[file_path] for file_path in congestion_set.keys()][::10]
        y_label = replaceWithRank(y)
        y_gpdl_label = replaceWithRank(y_gpdl)
        plt.figure(figsize=(10,5))
        plt.plot(x, y_label, linewidth="2", marker="o")
        plt.plot(x, y_gpdl_label, linewidth="2", marker="o")
        plt.plot(x, x_label, linewidth="2", marker="o")
        plt.xticks(ticks=x_label, labels=x, rotation=90)
        plt.xlabel("Images")
        plt.ylabel("Rank Order")
        plt.title(f"Congestion Prediction Rank Order on {design_name}")
        plt.legend(["GARF", "RouteNet", "Ground Truth"])
        plt.savefig(save_path + f"congestion_rank_order_{design_name}.png")
        plt.close()
        return {"GPDL": gpdl_results,"feats": results}
    else:
        predicted = dict(zip(df["id"], df["prediction"]))
        results = evalute_corr(congestion_set, predicted, corr_metrics)
        x = list(congestion_set.keys())[::10]
        x = [name.split("-")[0] for name in x]
        x_label = list(range(0,len(x)))
        y = [predicted[file_path] for file_path in congestion_set.keys()][::10]
        y_label = replaceWithRank(y)
        plt.figure(figsize=(10,5))
        plt.plot(x, y_label, linewidth="2", marker="o")
        plt.plot(x, x_label, linewidth="2", marker="o")
        plt.xticks(ticks=x_label, labels=x, rotation=90)
        plt.xlabel("Images")
        plt.ylabel("Rank Order")
        plt.title(f"Congestion Prediction Rank Order on {design_name}")
        plt.legend(["GARF", "RouteNet", "Ground Truth"])
        plt.savefig(save_path + f"congestion_rank_order_{design_name}.png")
        plt.close()
        return {"feats": results}



def get_gpdl_prediction(design, feature_path, model, device):
     feature_path = feature_path + design + "/"
     gpdl_prediction = {}
     
     for filename in tqdm(os.listdir(feature_path)):
          file_path = os.path.join(feature_path, filename)
          numpy_image = np.load(file_path)
          batch_image = numpy_image.transpose(2,0,1)
          with torch.no_grad():
               input_image = torch.tensor(batch_image).unsqueeze(0).float().to(device)
               output_image = model(input_image)
               prediction = get_drc_violations(output_image.cpu().numpy().squeeze())
          
          gpdl_prediction[filename] = prediction
          
     return gpdl_prediction


def single_extractor(design, label_path, feature_path, baseline=False):
        feature_path = feature_path + design + "/"
        label_path = label_path + design + "/"

        labels = []
        ids = []

        for filename in tqdm(os.listdir(label_path)):
            file_path = os.path.join(label_path, filename)
            label_image = np.load(file_path).squeeze()
            label = get_drc_violations(label_image)
            ids.append(filename)
            labels.append(label)
            
        df = pd.DataFrame({"id": ids,})

        for filename in tqdm(os.listdir(feature_path)):
            file_path = os.path.join(feature_path, filename)
            numpy_image = np.load(file_path)
            batch_image = numpy_image.transpose(2,0,1)
            image_features = []
            for i, image in enumerate(batch_image):
                image_features.append(image)
            
            index = (df["id"] == filename)
            
            all_features = get_all_features(image_features)
            for key, value in all_features.items():
                df.loc[index, key] = value
                
            if baseline:
                df.loc[index, "prediction_gpdl"] = gpdl_prediction[design][filename]

        
        df['label'] = labels
        return df


def dataset_setting(designs, label_path, feature_path, baseline=False):
    pool = multiprocessing.Pool()
    df_list = pool.starmap(single_extractor, zip(designs, [label_path]*len(designs), [feature_path]*len(designs), [baseline]*len(designs)))
    pool.close()
    return pd.concat(df_list)



def main():
    parser = ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--pretrained", type=str, default="/home/felixchaotw/CircuitNet/model/DRC.pth")
    parser.add_argument("--feature_path", type=str, default="/data2/NVIDIA/CircuitNet-N28/Dataset/DRC/feature/")
    parser.add_argument("--label_path", type=str, default="/data2/NVIDIA/CircuitNet-N28/Dataset/DRC/label/")
    parser.add_argument("--save_path", type=str, default="./")
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    baseline = args.baseline
    feature_path = args.feature_path
    label_path = args.label_path
    print(f"Using device: {device}")
    
    if baseline:
        global gpdl_prediction
        opt = {'task': 'drc_routenet', 'save_path': 'work_dir/drc_routenet/', 'pretrained': '/home/felixchaotw/CircuitNet/model/DRC.pth', 'max_iters': 200000, 'plot_roc': False, 'arg_file': None, 'cpu': False, 'dataroot': '../../training_set/DRC', 'ann_file_train': './files/train_N28.csv', 'ann_file_test': './files/test_N28.csv', 'dataset_type': 'DRCDataset', 'batch_size': 16, 'aug_pipeline': ['Flip'], 'model_type': 'RouteNet', 'in_channels': 9, 'out_channels': 1, 'lr': 0.0002, 'weight_decay': 0, 'loss_type': 'MSELoss', 'eval_metric': ['NRMS', 'SSIM', 'EMD'], 'ann_file': './files/test_N28.csv', 'test_mode': True}
        model = models.__dict__["RouteNet"](**opt)
        model.init_weights(**opt)
        model.to(device)
        gpdl_prediction = {}
        gpdl_prediction["zero-riscy-a"] = get_gpdl_prediction("zero-riscy-a", feature_path, model, device)
        gpdl_prediction["zero-riscy-b"] = get_gpdl_prediction("zero-riscy-b", feature_path, model, device)
        

    train_df = dataset_setting(train_design, label_path, feature_path)
    test_df_a = single_extractor(test_design_a[0], label_path, feature_path, baseline)
    test_df_b = single_extractor(test_design_b[0], label_path, feature_path, baseline)
    
    print("Train dataset shape:", train_df.shape)
    print("Test dataset A shape:", test_df_a.shape)
    print("Test dataset B shape:", test_df_b.shape)
    
    train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df_a.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df_b.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    
    if baseline:
        x = train_df.drop(columns=(["id", "label"]))
        y = train_df["label"]
        x_test_a = test_df_a.drop(columns=(["id", "label", "prediction_gpdl"]))
        x_test_b = test_df_b.drop(columns=(["id", "label", "prediction_gpdl"]))
    else:
        x = train_df.drop(columns=(["id", "label"]))
        y = train_df["label"]
        x_test_a = test_df_a.drop(columns=(["id", "label"]))
        x_test_b = test_df_b.drop(columns=(["id", "label"]))

    rf_reg = RandomForestRegressor(random_state=42, max_depth=50, max_features='sqrt', min_samples_leaf=1, n_estimators=300)
    rf_reg.fit(x, y)
    y_pred_test_a = rf_reg.predict(x_test_a)
    y_pred_test_b = rf_reg.predict(x_test_b)
    test_df_a["prediction"] = y_pred_test_a
    test_df_b["prediction"] = y_pred_test_b
    
    test_df_a = test_df_a.drop_duplicates(subset=["label"])
    test_df_b = test_df_b.drop_duplicates(subset=["label"])
    
    results_a = evaluate_design(test_df_a, args.save_path, baseline, design_name="zero-riscy-a")
    results_b = evaluate_design(test_df_b, args.save_path, baseline, design_name="zero-riscy-b")
    
    if baseline:
        print("==== Baseline Results ====")
        print("Results for zero-riscy-a with baseline:")
        print("PLCC: {0:.3f}".format(results_a["GPDL"]["PLCC"].statistic))
        print("SRCC: {0:.3f}".format(results_a["GPDL"]["SRCC"].statistic))
        print("KRCC: {0:.3f}".format(results_a["GPDL"]["KRCC"].statistic))
        print("Results for zero-riscy-b with baseline:")
        print("PLCC: {0:.3f}".format(results_b["GPDL"]["PLCC"].statistic))
        print("SRCC: {0:.3f}".format(results_b["GPDL"]["SRCC"].statistic))
        print("KRCC: {0:.3f}".format(results_b["GPDL"]["KRCC"].statistic))
        
    print("==== New Features Results ====")
    print("Results for zero-riscy-a with new feats:")
    print("PLCC: {0:.3f}".format(results_a["feats"]["PLCC"].statistic))
    print("SRCC: {0:.3f}".format(results_a["feats"]["SRCC"].statistic))
    print("KRCC: {0:.3f}".format(results_a["feats"]["KRCC"].statistic))
    print("Results for zero-riscy-b with new feats:")
    print("PLCC: {0:.3f}".format(results_b["feats"]["PLCC"].statistic))
    print("SRCC: {0:.3f}".format(results_b["feats"]["SRCC"].statistic))
    print("KRCC: {0:.3f}".format(results_b["feats"]["KRCC"].statistic))


if __name__ == "__main__":
    main()

