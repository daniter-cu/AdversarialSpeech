import csv

_data_path = 'asset/data/'
set_name = "valid"

# load metadata
label, mfcc_file = [], []
mfcc_to_filename = {}
with open(_data_path + 'preprocess/meta/%s.csv' % set_name) as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        # mfcc file
        mfcc_file.append(_data_path + 'preprocess/mfcc/' + row[0] + '.npy')
        # label info ( convert to string object for variable-length support )
        label.append(tuple(row[1:]))
        assert tuple(row[1:]) not in mfcc_to_filename
        mfcc_to_filename[tuple(row[1:])] = row[0] 

# load predictions
preds_file = "preds_vs_labels.tsv"
f = open("preds_vs_labels_filename.tsv", "wb")
preds = [line.rstrip().split("\t") for line in open(preds_file)]
preds = preds[1:] # cut off header
for i, p in enumerate(preds):
  same_diff, pred_on_orig, pred_on_adv, target,	\
    num_pred_on_orig, num_pred_on_adv, num_target = p
  target_tup = tuple(target.split())
  filename = mfcc_to_filename[target_tup]
  f.write("\t\n".join[same_diff, pred_on_orig, pred_on_adv, target, \
    num_pred_on_orig, num_pred_on_adv, num_target, filename])
