import numpy as np

def hit_score(pred,gold):
  """ 
    Args:
        pred : prediction array from the model
        gold : gold values
    
    Return: 
        hit_score(number of times the predicted image is equal to the
                  gold image.)
  """
  
  hit_e = 0

  for i in range(len(gold)):
    if (np.array(np.argmax(pred[i]))==np.array(gold[i])):
      hit_e = hit_e+1

  return hit_e

def mrr_score(pred,gold):
  """ 
    Args:
        pred : prediction array from the model
        gold : gold values
    
    Return: 
        mrr_score(sum over reciprocal of rank of the position of the gold image 
                  in the predicted array )
  """
  
  mrr_e = 0

  for i,j in zip(pred,gold):
    idx = np.where(np.argsort(-i)==int(j))[0]
    mrr_e += 1/(idx+1)

  return mrr_e

def evaluation_using_logits(logits_per_image, target_images):
  """ 
    Args:
        logits_per_image : cosine similaries between the texts and 
                           the 10 images assigned to that text
        target_images    : list where each index represents the gold_image
                           position in the image_list array
    
    Return: 
        hit_score and mrr_score just based on CLIP 
        embeddings and their cosine similarities
  """
    
  hit = hit_score(logits_per_image, target_images)
  mrr = mrr_score(logits_per_image, target_images)
  return hit/len(target_images),mrr/len(target_images)