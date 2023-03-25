import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt

#Text and Image embeddings generated from CLIP are stored here, so 
#which are used for further processing without generating them every time 
#we run the code.
TEXT_EMB_PATH = './embeddings_pkl_files/text_emb_new.pt'
IMAGE_EMB_PATH = './embeddings_pkl_files/image_features.pickle'

class ContrastiveCosineLoss(nn.Module):
    """
    This class helps us to calculate the Contrastive Cosine Loss
    function. Positive loss is calculated by mean sqaured error between 1 
    and cosine similarity of embeddings. Negetive Loss is calculated by 
    mean sqaured error between margin and cosine similarity of embeddings 
    clamped to 0. Loss is obtained by adding Positive loss and negetive loss.
    
    """
    def __init__(self, margin=0.2):
        super(ContrastiveCosineLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        #cosine_similarity calculation between predction array and 
        #target_images list(gold_label)
        cos_sim = F.cosine_similarity(output, target)
        pos_loss = torch.mean(torch.pow(1 - cos_sim, 2))
        neg_loss = torch.mean(torch.clamp(torch.pow(self.margin - cos_sim, 2), min=0.0))
        loss = pos_loss + neg_loss
        return loss


def gold_position_search(image_list, gold_list):
  """ 
    Args:
        image_list :list where each row depicts the 10 images assigned to the 
                    text at that index in text_list 
        gold_list  :list that contain the image out of the image_list which 
                    is most relevant to the text
    
    Return: 
        target_images  : list which conatins the gold image index in the 
                         image_list 
  """
  target_images = []

  for i in range(len(gold_list)):
    #pos_idx stores the position of gold_image in image_list
    pos_idx = 0
    for j in range(len(image_list[i])):
      if gold_list[i] != image_list[i][j]:
        pos_idx += 1

    target_images.append(pos_idx)
  return target_images

def normalize_features(text_features,image_features):
  """ 
    Args:
        text_features : tensor with text_embeddings tensors
        
        image_features  : list of image_embeddings tensors(10 tensors to form one 
                                                          row that correspond to one text)
    
    Return: 
        normalized text and image features 
  """

  #normalizing text features
  text_features = torch.nn.functional.normalize(text_features, dim = 0)

  #normalizing image features
  img_features = []
  for emb in range(len(image_features)):
    emb_n = torch.nn.functional.normalize(image_features[emb], dim = 0)
    img_features.append(emb_n)

  return text_features, img_features

def load_dataset(image_list,gold_list):
    """ 
      Args:
          image_list :list where each row depicts the 10 images assigned to the 
                      text at that index in text_list 
          gold_list  :list that contain the image out of the image_list which 
                      is most relevant to the text
      
      Return: 
          Normalized ext and image features along with target images
    """

    #Load the pretrained clip text embeddings for all the texts in word_list
    #Embeddings are stored in a pt file.
    #To request rerun to regenrate these embedding files, please make the "prepare" argumnet as false
    text_features = torch.load(TEXT_EMB_PATH)

    #Load the pretrained clip image embeddings for all the image in the same format as image_list.
    #Embeddings are stored in a pickle file due to nature of the data(dictionary).
    #To request rerun to regenrate these embedding files, please make the "prepare" argumnet as false
    with open(IMAGE_EMB_PATH, 'rb') as f:
        image_features = pickle.load(f)
        f.close()

    #A list that contains the index of gold_image per trial
    target_images = gold_position_search(image_list,gold_list)

    #Normalize the features
    text_features, image_features = normalize_features(text_features,image_features)
    
    return text_features, image_features, target_images


def plot_loss_graph(epoch_loss,epoch_hit,epoch_mrr):
    """ 
      Args:
          epoch_loss :list that contains the loss values for each epoch
          epoch_hit  :list that contains the hit@1 values for each epoch
          epoch_mrr  :list that contains the mrr values for each epoch
                      
      
      Return: 
          Saves the loss,hit@1 and mrr values with number of epochs graph
          and plots them 
    """
    #defining the figure the size
    plt.figure(figsize=(15,5))
    # The graphs are plotted suing 3 subplots
    plt.subplot(1,3,1)
    plt.plot(epoch_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss_value")
    plt.title("loss vs epoch graph")
  
    plt.subplot(1,3,2)
    plt.plot(epoch_hit)
    plt.xlabel("Epochs")
    plt.ylabel("hit@1_value")
    plt.title("hit@1 vs epoch graph")
  
    plt.subplot(1,3,3)
    plt.plot(epoch_mrr)
    plt.xlabel("Epochs")
    plt.ylabel("MRR_values")
    plt.title("mrr vs epoch graph")
    plt.savefig("clip.png")
    plt.show()


def similarity_score_calc(text_features, image_features):
  """ 
    Args:
        text_features : tensor with text_embeddings tensors
        
        image_features  : list of image_embeddings tensors(10 tensors to form one 
                                                          row that correspond to one text)
    
    Return: 
        cosine similarity calculation between the text embedding 
        and the embeddings of 10 images assigned to that text

  """
  logits_per_image = []
  i = 0

  #calculates cosine similarity between images and texts
  for embbedings in image_features:
    logits_per_image.append(text_features[i] @ embbedings.t())
    i+=1

  return logits_per_image

