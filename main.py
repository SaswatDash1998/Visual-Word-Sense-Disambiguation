import argparse

from language_model import get_eval_scores
from evaluation import evaluation_using_logits
from utils import load_dataset,similarity_score_calc
from data_preparation import get_dataloaders
from helper import prepare_text,get_files


def main():
    
  #Distribute the data into three parts
  #Using prepare_text function to get three lists
  trial_text_data = './data/train.data.v1.txt'
  trial_gold_data = './data/train.gold.v1.txt'
  word_list, gold_list, image_list = prepare_text(trial_text_data,trial_gold_data)
  
  parser = argparse.ArgumentParser(
          description='Visual Word Sense Ambiguation'
      )

  parser.add_argument(
      "--prepare", dest="prepare",
      help="Prepares the data",
      action="store_true",
      default = False
  )

  parser.add_argument(
      "--choose_model", dest="CLIP_train",
      help="Calls the CLIP model selected here",
      action="store",
      default = "CLIP_3",
      choices = ["CLIP_0", "CLIP_1", "CLIP_2", "CLIP_3"]
  )

  parser.add_argument(
      "--loss_function", dest="loss_function",
      help="Selects the loss function to be used",
      action="store",
      default = "contrastive_cosine_loss",
      choices = ["cross_entropy_loss", "contrastive_cosine_loss"]
  )


  args = parser.parse_args()
  
  text_features, image_features, target_images = load_dataset(image_list, gold_list)
  train_dataloader, test_dataloader = get_dataloaders(text_features, image_features, target_images)

  #This flag is set to True only if the pretrained clip need to be recalculated 
  #and stored in the respective files "torch_emb_new.pt" and "image_features.pickle".
  if args.prepare:
    get_files()

  #Evaluation of model by using just the pretrained clip embeddings for texts and images
  #and finding the cosine similarity between them.
  if args.CLIP_train == "CLIP_0":
    logits_per_image  = similarity_score_calc(text_features, image_features)
    hit_at_1,mrr = evaluation_using_logits(logits_per_image, target_images)
    print("Hit@1 value is: ", hit_at_1)
    print("MRR value is: ", mrr)


  #Using variations of finetuned clip models for training of new data
  #Need to choose the loss function to use as well(defaullt loss is contrastive cosine loss)
  #The piece of code below prints the "loss vs epoch", "mrr vs epoch" and "hit@1 vs epoch"
  #for the model and loss function selected from the argumemts.
  #This is the case if CLIP_1 model is selected for performance evaluation
  if args.CLIP_train == "CLIP_1":
    if args.loss_function == "cross_entropy_loss":
      get_eval_scores(train_dataloader,
                      test_dataloader,
                      choose_model = "clip_1",
                      loss_function ="cross entropy loss")


    if args.loss_function == "contrastive_cosine_loss":
      get_eval_scores(train_dataloader,
                      test_dataloader,
                      choose_model = "clip_1",
                      loss_function ="contrastive cosine loss")


  
  #This is the case if CLIP_2 model is selected for performance evaluation
  if args.CLIP_train == "CLIP_2":
    if args.loss_function == "cross_entropy_loss":
      get_eval_scores(train_dataloader,
                      test_dataloader,
                      choose_model = "clip_2",
                      loss_function ="cross entropy loss")


    if args.loss_function == "contrastive_cosine_loss":
      get_eval_scores(train_dataloader,
                      test_dataloader,
                      choose_model = "clip_2",
                      loss_function ="contrastive cosine loss")



  #This is the case if CLIP_3 model is selected for performance evaluation
  if args.CLIP_train == "CLIP_3":
    if args.loss_function == "cross_entropy_loss":
      get_eval_scores(train_dataloader,
                      test_dataloader,
                      choose_model = "clip_3",
                      loss_function ="cross entropy loss")


    if args.loss_function == "contrastive_cosine_loss":
      print('q')
      get_eval_scores(train_dataloader,
                      test_dataloader,
                      choose_model = "clip_3",
                      loss_function ="contrastive cosine loss")


if __name__ == "__main__":
    main()