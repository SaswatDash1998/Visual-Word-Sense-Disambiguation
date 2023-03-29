import torch
import torch.nn as nn
from tqdm import tqdm

from utils import ContrastiveCosineLoss, plot_loss_graph
from finetune_clip_models import CLIP_1, CLIP_2, CLIP_3
from evaluation import hit_score, mrr_score


def training(train_dataloader,choose_model = "clip_3", 
             loss_function="contrastive cosine loss"):
  """ 
    Args:
        train_dataloader : Pytorch dataloader object which segments data 
                           into batches of 32 samples for train data
                         
        choose_model     : Model that is used for training out 
                           of (CLIP_1, CLIP_2 and CLIP_3)
                        
        loss_function    : The loss function which has to be used for 
                           training(either cross_entropy or contrastive_cosine)
    
    Return: 
        trained_model along with lists of recoreded loss, mrr and 
        hit@1 values over the number of epochs
  """

  num_epochs = 25
  input_size = 512
  hidden_size = 512
  output_size = 512
  print("Start training process for " + choose_model + " with the loss function as "+ loss_function)
  if(loss_function == "contrastive cosine loss"):
    loss_f = ContrastiveCosineLoss()
  elif(loss_function == "cross entropy loss"):
    loss_f = nn.CrossEntropyLoss()
  
  if(choose_model == "clip_1"):
    model = CLIP_1(input_size, output_size)
  elif(choose_model == "clip_2"):
    model = CLIP_2(input_size, output_size)
  elif(choose_model == "clip_3"):
    model = CLIP_3(input_size, hidden_size, output_size)

  #AdamW optimizer defined with learning rate of 1e-4.
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
  
  #lists to store model evaluation metrics per epoch
  epoch_loss = []
  epoch_mrr = []
  epoch_hit = []
  
  for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}")
        avg_loss = 0
        model.train()
        hit_ = 0
        mrr_ = 0

        for batch in tqdm(train_dataloader):

          text, img, target,label_idx = batch
          optimizer.zero_grad()
          text_logit, img_logit = model(img,text)
          
          #Performs matrix multiplication of 2 tensors given the dimensions.
          #This method is called Einstien's summation.
          sim = torch.einsum('ijk,ik->ij', text_logit, img_logit)
          loss = loss_f(sim, target)
          loss.backward()
          optimizer.step()


          avg_loss += loss.item()*img.size(0)
          
          #label_idx and sim are converted to numpy arrays since compnenets with 
          #required_grad as True can be referenced before next epoch.
          x = label_idx.detach().numpy()
          y = sim.detach().numpy()

          hit_ += hit_score(y,x)
          mrr_ += mrr_score(y,x)

        
        epoch_loss.append(avg_loss/len(train_dataloader.dataset))
        epoch_hit.append(hit_/len(train_dataloader.dataset))
        epoch_mrr.append(mrr_/len(train_dataloader.dataset))
        print("Train_loss:",avg_loss/len(train_dataloader.dataset))
        print("Train_mrr:",mrr_/len(train_dataloader.dataset))
        print("Train_hit:",hit_/len(train_dataloader.dataset))


  return model,epoch_loss,epoch_hit, epoch_mrr

def testing(model,test_dataloader):
  """ 
   Args:
      
       test_dataloader  : Pytorch dataloader object which segments data 
                            into batches of 32 samples for test data
                        
       model            : trained model obtained after training

   
   Return: 
       Returns the MRR and Hit@1 values for the test set
  """ 
  hit = 0
  mrr = 0
  print("Start testing process for the trained model")
  with torch.no_grad():
      for batch in tqdm(test_dataloader):
          images, text, _, label_idx = batch

          
          text_logit, img_logit = model(images, text)

          sim = torch.einsum('ijk,ik->ij', text_logit, img_logit)

          hit += hit_score(sim,label_idx)
          mrr += mrr_score(sim,label_idx)

  return hit,mrr

def get_eval_scores(train_dataloader,
                    test_dataloader,
                    choose_model,
                    loss_function):
    
    """ 
      Args:
          train_dataloader : Pytorch dataloader object which segments data 
                             into batches of 32 samples for train data
          
          test_dataloader  : Pytorch dataloader object which segments data 
                               into batches of 32 samples for test data
                           
          choose_model     : Model that is used for training out 
                             of (CLIP_1, CLIP_2 and CLIP_3)
                          
          loss_function    : The loss function which has to be used for 
                             training(either cross_entropy or contrastive_cosine)
      
      Return: 
          Prints the hit@1 and MRR values when trained models are 
          subjected to test set
    """
  
    model,epoch_loss,epoch_hit,epoch_mrr = training(train_dataloader,choose_model,
                                                    loss_function)
    print("Show the metrics vs number of epochs graphs")
    plot_loss_graph(epoch_loss,epoch_hit,epoch_mrr)
    hit_at_1,mrr = testing(model,test_dataloader)
    print("Hit@1 value for the test set= "+hit_at_1/len(test_dataloader.dataset))
    print("MRR value for the test set= "+mrr/len(test_dataloader.dataset))
