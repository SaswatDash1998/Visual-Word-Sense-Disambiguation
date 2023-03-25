from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast, CLIPFeatureExtractor, AutoProcessor, AutoModel
import torch
import os
import h5py
import pickle


model_name = "openai/clip-vit-base-patch32"
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModel.from_pretrained(model_name)
tokenizer = CLIPTokenizerFast.from_pretrained(model_name)


FEATURES_PATH = "./data/features"
IMAGE_EMBEDDING_PATHWAY = "./data/embeddings/image_embeddings"
TEXT_EMBEDDING_PATHWAY = "./data/embeddings/text_embeddings"
TRAIN_IMAGE_PATH = "./data/train/train_images_v1"
TRAIN_DATA_PATH = "./data/train/train.data.v1.txt"
TRAIN_GOLD_PATH = "./data/train/train.gold.v1.txt"
TRIAL_IMAGE_PATH = "./data/trial/trial_images_v1"
TRIAL_DATA_PATH = "./data/trial/trial.data.v1.txt"
TRIAL_GOLD_PATH = "./data/trial/trial.gold.v1.txt"

WRAPPER_PATH = "./data/embeddings/wrapper"

def encode_text(data, gold, text_embeddings, tokenizer, model):
    """
    Creates text embeddings and stores them individually in .h5 files.
    Args:
        data (str): string describing file pathway to the data
        gold (str): string  describing file pathway to the gold choices
        text_embeddings (str): string describing destination file pathway to
        where the text embeddings will be saved
        tokenizer (CLIPTokenizerFast): tokenizer object
        model (CLIPModel): CLIP model
    Return:
        None
    """
    print("Encoding text...")
    word_list, _, _ = prepare_text(data, gold)
    phrases = [item[1] for item in word_list]
    progress_bar = tqdm(total=len(phrases))

    for index, phrase in enumerate(phrases):
        text_encodings = tokenizer(phrase, truncation=True, padding=True, return_tensors="pt")
        text_features = model.get_text_features(**text_encodings)
        text_features = text_features.detach().numpy()

        with h5py.File(f"{text_embeddings}/{index}.h5", "w") as file:
            dataset = file.create_dataset("text", np.shape(text_features), data=text_features)

        progress_bar.update(1)

    progress_bar.close()

def encode_images(image_pathway, image_emb_pathway, model, processor):
    """
    Creates image embeddings and saves them as .h5 files.
    Args:
        image_pathway (str): contains file pathway to images
        image_emb_pathway (str): contains destination file pathway for image embeddings
        model (CLIPModel): CLIP model
        processor: CLIP processor
    Return:
        None
    """
    print("Encoding images...")
    all_images = os.listdir(image_pathway)
    progress_bar = tqdm(total=len(all_images))

    for img in all_images:
        image = Image.open(f"{image_pathway}/{img}").convert("RGB").resize((100, 100))
        exif_data = image.info.get("exif")
        image.save(f"{img}", exif=exif_data)
        input = processor(images=image, return_tensors="pt")
        image_feature = model.get_image_features(**input)
        image_feature = image_feature.detach().numpy()

        with h5py.File(f"{image_emb_pathway}/{img}.h5", "w") as file:
            dataset = file.create_dataset("image", np.shape(image_feature), data=image_feature)

        progress_bar.update(1)

    progress_bar.close()

def prepare_text(data, gold):
    """
    Takes the data and gold file pathways and returns three numpy arrays, one 
    containing the target word isolated and in context, one containing the gold 
    choice, and one containing the image choices for each trial.
    :param data: String containing the pathway to the data .txt file
    :param gold: String containing the pathway to the gold .txt file
    :return: Tuple of three numpy arrays
    """
    raw_data = []
    raw_gold = []
    image_list = []

    # Opens files and stores them line-by-line in lists
    with open(data,encoding = 'utf-8') as file:
        for line in file:
            raw_data.append(line.split('\t'))

    with open(gold) as file:
        for line in file:
            raw_gold.append(line.split('\t'))

    # Formats and orders the words and gold choices into lists
    word_list = [[raw_data[i][0], raw_data[i][1]] for i in range(len(raw_data))]
    gold_list = [[gold_image[0][:-1]] for gold_image in raw_gold]

    # Formats and orders the image choices into a list
    for index, line in enumerate(raw_data):
        image_list.append([])
        for ind, image in enumerate(line):
            if ind >= 2:
                if '\n' in image:
                    image_list[index].append(image[:-1])
                else:
                    image_list[index].append(image)

    # Creates and returns arrays from the three lists
    return np.array(word_list), np.array(gold_list), np.array(image_list)

def text_encoder_h5(data, gold):
  word_list, _, _ = prepare_text(data, gold)
  phrases = [item[1] for item in word_list]
  pbar = tqdm(total=len(phrases))
  for index, phrase in enumerate(phrases):

    text_encodings = tokenizer(phrase, truncation = True, padding = True, return_tensors = 'pt')
    text_features = model.get_text_features(**text_encodings)
    text_features = text_features.detach().numpy()
    with h5py.File(f'{TEXT_EMBEDDING_PATHWAY}/{index}.h5',"w") as f:
        dset = f.create_dataset("text", np.shape(text_features), data = text_features)
    pbar.update(1)

  print('text encoder done!')
  print('')
  pbar.close()

def img_encoder_h5(IMAGE_PATHWAY):
  #takes a list of all images in the directory
  all_images = os.listdir(IMAGE_PATHWAY)
  #opens each image
    
  pbar = tqdm(total=len(all_images))

  for img in all_images:
    #opens as RGB, resize to 100,100
    image = Image.open(f'{IMAGE_PATHWAY}/{img}').convert('RGB').resize((100,100))
    #get vectors
    input = processor(images = image, return_tensors = "pt")
    #get features
    image_feature = model.get_image_features(**input)
    #convert to numpy
    image_feature = image_feature.detach().numpy()
    #create a h5py file
    with h5py.File(f'{IMAGE_EMBEDDING_PATHWAY}/{img}.h5',"w") as f:
      dset = f.create_dataset("image", np.shape(image_feature), data = image_feature)
    pbar.update(1)

  pbar.close()
  print('Img encoder done!')
        
def create_wrapper(wrapper_pathway):
    file = h5py.File(f"{wrapper_pathway}/image_wrapper.h5", "w")
    text_file = h5py.File(f"{wrapper_pathway}/text_wrapper.h5", "w")
    file.close()
    text_file.close()
    print("Wrappers created.")
    """
    Takes two arrays containing the images and the gold choices, compares them
    and returns a list containing the index number of the correct choice in the
    image array.
    Args:
        gold_list (np.ndarray): numpy array containing the gold choices
        image_list (np.ndarray): numpy array containing the potential images
    Return:
        list: contains the index number of the correct choices in image array
    """
    target_images = []

    for i in range(len(gold_list)):
        index = 0
        for j in range(len(image_list[i])):
            if gold_list[i] != image_list[i][j]:
                index += 1

        target_images.append(index)

    return target_images

def wrap_image_files(image_emb_pathway, wrapper_pathway):
    embedded_image_list = os.listdir(image_emb_pathway)

    if os.path.isfile(f"{wrapper_pathway}/image_wrapper.h5"):
        file = h5py.File(f"{wrapper_pathway}/image_wrapper.h5", "a")

        for embedding in embedded_image_list:
            file[f"{str(embedding).removesuffix('.h5')}"] = h5py.ExternalLink(f"{image_emb_pathway}/{embedding}", "image")

        file.close()
    else:
        print("File not found.")

    print("Image embeddings wrapped.")

def wrap_text_files(text_emb_pathway, wrapper_pathway):
    embedded_text_list = os.listdir(text_emb_pathway)

    if os.path.isfile(f"{wrapper_pathway}/text_wrapper.h5"):
        file = h5py.File(f"{wrapper_pathway}/text_wrapper.h5", "a")

        for embedding in embedded_text_list:
            file[f"{str(embedding).removesuffix('.h5')}"] = h5py.ExternalLink(f"{text_emb_pathway}/{embedding}", "text")

        file.close()

    else:
        print("File not found.")

    print('Text embeddings wrapped.')
    

def get_image_embeddings(data, gold, wrapper_pathway):
    """
    Returns a dictionary with index numbers and all tensors with image embeddings.
    Args:
        data (str): file pathway to data file
        gold (str): file pathway to gold file
        wrapper_pathway (str): destination file pathway to wrapper file
    Return:
        dict: index numbers and tensors with image embeddings
    """
    _, _, image_list = prepare_text(data, gold)
    wrapper_file = h5py.File(f"{wrapper_pathway}/image_wrapper.h5", "r+")
    embedding_dictionary = {}
    progress_bar = tqdm(total=len(image_list))

    for index, item in enumerate(image_list):
        embedding_dictionary[index] = torch.stack([torch.from_numpy(wrapper_file[image][0]) for image in item])
        progress_bar.update(1)

    wrapper_file.close()
    progress_bar.close()

    print("Image embeddings retrieved.")
    return embedding_dictionary


def get_text_embeddings(text_emb_pathway, wrapper_pathway):
    """
    Creates a tensor containing all text embeddings.
    Args:
        text_emb_pathway (str): file pathway to text embeddings
        wrapper_pathway (str): file pathway to wrapper
    Return:
        torch.Tensor: filled with the text embeddings
    """
    embedding_list = []
    for item in os.listdir(text_emb_pathway):
        if not item.startswith(".") and os.path.isfile(os.path.join(text_emb_pathway, item)):
            embedding_list.append(item)
    wrapper_file = h5py.File(f"{wrapper_pathway}/text_wrapper.h5", "r+")
    text_embeddings = [torch.from_numpy(wrapper_file[f"{item.removesuffix('.h5')}"][0]) for item in embedding_list]
    text_embeddings = torch.stack(text_embeddings)

    wrapper_file.close()

    print("Text embeddings retrieved.")
    return text_embeddings

def save_features(text_features: torch.Tensor, image_features, destination_pathway):
    """
    Description
    Args:
        text_features (torch.Tensor):
        image_features (dict):
        destination_pathway (str):
    Return:
        Text and image feature tuples
    """
    torch.save(text_features, f"{destination_pathway}/text_features.pt")

    with open(f"{destination_pathway}/image_features.pickle", "wb") as file:
        pickle.dump(image_features, file)
        file.close()


    return text_features, image_features

def get_files():

    encode_text(TRIAL_DATA_PATH, TRIAL_GOLD_PATH, TEXT_EMBEDDING_PATHWAY, tokenizer, model)
    encode_images(TRIAL_IMAGE_PATH, IMAGE_EMBEDDING_PATHWAY, model, processor)

    create_wrapper(WRAPPER_PATH)
    wrap_image_files(IMAGE_EMBEDDING_PATHWAY, WRAPPER_PATH)
    wrap_text_files(TEXT_EMBEDDING_PATHWAY, WRAPPER_PATH)

    text_features = get_text_embeddings(TEXT_EMBEDDING_PATHWAY, WRAPPER_PATH)
    image_features = get_image_embeddings(TRIAL_DATA_PATH, TRIAL_GOLD_PATH, WRAPPER_PATH)
    text_features, image_features = save_features(text_features, image_features, FEATURES_PATH)