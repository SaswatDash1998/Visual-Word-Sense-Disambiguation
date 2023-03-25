import torch
from torch.utils.data import Dataset, DataLoader

class DataSetForCLIP(Dataset):
  """
  DataSetForCLIP class takes the text_data, image_data and gold_labels as input.
  At each index of the objects of this class, we get a text embeddings, 
  the corresponding image embeddings and the target image index 
  from the target array.
  """
  
  def __init__(self,text_data,image_data, target):
    self.text_data = text_data
    self.image_data = image_data
    self.target = target
    
  def __len__(self):
    return len(self.text_data)

  def __getitem__(self,idx):
    text_emb = self.text_data[idx]
    image_emb = self.image_data[idx]
    
    label = torch.zeros(10)
    index = self.target[idx]
    label[index] = 1

    return text_emb, image_emb, label, index


def train_test_split(image_features, text_features, target_images):
  """ 
    Args:
        image_features : list of image_embeddings tensors(10 tensors to form one 
                                                          row that correspond to one text)
        text_features  : tensor with text_embeddings tensors
        
        target_images  : list where each index represents the gold_image
                         position in the image_list array
    
    Return: 
        train and test split for all the three above arguments
  """
  #This function enables our data is divided such as train_set gets 75% data
  #and text_set gets the remaining 25%.
  train_size = int(len(image_features) * 0.75)

  train_text = text_features[:train_size]
  test_text = text_features[train_size:]


  train_image = image_features[:train_size]
  test_image = image_features[train_size:]

  train_targets = target_images[:train_size]
  test_targets = target_images[train_size:]

  return train_text, test_text, train_image, test_image, train_targets, test_targets

def get_dataloaders( text_features, image_features, target_images):
    """ 
    This function enables us separate our data into batches of 32 which shuffle enabled for training data.
    The above functionality is enabled after the data is formed in accordance with the DataSetForCLIP class.
    
      Args:
          image_features : list of image_embeddings tensors(10 tensors to form one 
                                                            row that correspond to one text)
          text_features  : tensor with text_embeddings tensors
        
          targets_images : list where each index represents the gold_image
                           position in the image_list array
      
      Return: 
          train and test_dataloader
    """
    train_text, test_text, train_image, test_image, train_targets, test_targets = train_test_split(image_features, text_features, target_images)
    
    #The train_text and train_image are combined to 
    #get train_data and feed to the dataloader to get batches of 32.
    #The same thing is done for the test set.
    train_data =  DataSetForCLIP(train_text, train_image, train_targets)
    test_data = DataSetForCLIP(test_text, test_image, test_targets)
    train_dataloader = DataLoader(train_data, batch_size = 32, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = 32)
    
    return train_dataloader,test_dataloader

