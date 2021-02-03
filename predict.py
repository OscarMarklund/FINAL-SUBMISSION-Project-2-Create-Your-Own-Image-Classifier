# Programmer: Oscar Marklund
# Submission date: 03-Feb-2021
# Last altered: 03-Feb-2021

import argparse

parser = argparse.ArgumentParser('Arguments for the predict.py file')

parser.add_argument('-chpath', type=str, default='/home/workspace/ImageClassifier/checkpoint.pth', 
                    help="Input your or another checkpoint filepath - model checkpoint. Enclose in ''. DEFAULT IS '/home/workspace/ImageClassifier/checkpoint.pth'")
parser.add_argument('-impath', type=str, default='/home/workspace/ImageClassifier/flowers/test/21/image_06805.jpg', 
                    help="Input filepath for image - image to be classified. Enclose in ''. DEFAULT IS '/home/workspace/ImageClassifier/flowers/test/21/image_06805.jpg'")
parser.add_argument('-gpu', action='store_true', 
                    help='No input. Attempts to use GPU to train if readily available. DEFAULT IS False')
parser.add_argument('-topk', type=int, default=5, 
                    help='Input an integer - number of top classifications for image. DEFAULT IS 5')
parser.add_argument('-catnames', type=str, default='/home/workspace/ImageClassifier/cat_to_name.json', 
                    help="Input a JSON file path - sets names to map against categories. Enclose directory within ''. DEFAULT IS '/home/workspace/ImageClassifier/cat_to_name.json'")


args = parser.parse_args()

ch_path = args.chpath
image_path = args.impath
topk = args.topk
cat_names = args.catnames
gpu = args.gpu


if __name__ == "__main__":
    from predict_function import loadandpredict
    loadandpredict(ch_path, image_path, topk, cat_names, gpu)