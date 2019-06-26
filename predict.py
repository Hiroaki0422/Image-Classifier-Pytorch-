import argparse
from ml_utils import load_checkpoint, predict, getNameMap

parser = argparse.ArgumentParser()

parser.add_argument('image_path', default="./flowers/test/1/image_06743.jpg")
parser.add_argument('model_path', default="./models/mymodel.pth")
parser.add_argument('--category_names', default="./cat_to_name.json")
parser.add_argument('--topk', default=5, type=int)

args = parser.parse_args()
mymodel = load_checkpoint(args.model_path)
name_map = getNameMap(args.category_names)
probs, labels = predict(args.image_path, mymodel, args.topk)

print('the most likely classes:')
for i, x in enumerate(zip(labels, probs)):
    print(i+1, ':', name_map[str(x[0])], ' ', x[1]*100, '%')

